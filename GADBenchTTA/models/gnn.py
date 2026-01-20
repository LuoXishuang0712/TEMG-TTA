import dgl.data
import dgl.udf
import torch
import torch.nn.functional as F
import dgl.function as fn
import sympy
import scipy
import dgl.nn.pytorch.conv as dglnn
import dgl
from torch import nn
from scipy.special import comb
import math
import copy
import numpy as np
from typing import Literal
from functools import lru_cache
import torch_scatter
import tqdm
# import numba as nb

from models.dgagnn import DGAGNNWrapped as DGAGNN, DGAGNNWrapped_M as DGAGNN_M
from models.secgfd import SECGFDWrapped as SECGFD
from models.space_gnn import SpaceGNNWrapped as SpaceGNN
from models.arc import ARCWrapped as ARC

from models.feature_fusion import wrap_class_instance_for_feature_fusion, wrap_class_for_feature_fusion, FeatureSimpleFusion

class PolyConv(nn.Module):
    def __init__(self, theta):
        super(PolyConv, self).__init__()
        self._theta = theta
        self._k = len(self._theta)

    def forward(self, graph, feat):
        def unnLaplacian(feat, D_invsqrt, graph):
            """ Operation Feat * D^-1/2 A D^-1/2 """
            graph.ndata['h'] = feat * D_invsqrt
            graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
            return feat - graph.ndata.pop('h') * D_invsqrt

        with graph.local_scope():
            D_invsqrt = torch.pow(graph.in_degrees().float().clamp(
                min=1), -0.5).unsqueeze(-1).to(feat.device)
            h = self._theta[0]*feat
            for k in range(1, self._k):
                feat = unnLaplacian(feat, D_invsqrt, graph)
                h += self._theta[k]*feat
        return h


class BernConv(nn.Module):
    def __init__(self, orders=2):
        super().__init__()
        self.K = orders
        self.weight = nn.Parameter(torch.ones(orders+1))

    def forward(self, graph, feat):
        def unnLaplacian1(feat, D_invsqrt, graph):
            """ \hat{L} X """
            graph.ndata['h'] = feat * D_invsqrt
            graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
            return feat - graph.ndata.pop('h') * D_invsqrt

        def unnLaplacian2(feat, D_invsqrt, graph):
            """ (2I - \hat{L}) X """
            graph.ndata['h'] = feat * D_invsqrt
            graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
            return feat + graph.ndata.pop('h') * D_invsqrt

        with graph.local_scope():
            tmp = [feat]
            weight = nn.functional.relu(self.weight)
            D_invsqrt = torch.pow(graph.in_degrees().float().clamp(
                min=1), -0.5).unsqueeze(-1).to(feat.device)
            for i in range(self.K):
                feat = unnLaplacian2(feat, D_invsqrt, graph)
                tmp.append(feat)

            out_feat = (comb(self.K, 0)/(2**self.K))*weight[0]*tmp[self.K]
            for i in range(self.K):
                x = tmp[self.K-i-1]
                for j in range(i+1):
                    x = unnLaplacian1(feat, D_invsqrt, graph)
                out_feat = out_feat+(comb(self.K, i+1)/(2**self.K))*weight[i+1]*x
        return out_feat


class BernNet(nn.Module):
    def __init__(self, in_feats, h_feats=32, num_classes=2, orders=2, mlp_layers=1, dropout_rate=0, activation='ReLU',
                 **kwargs):
        super().__init__()
        self.bernconv = BernConv(orders=orders)
        self.linear1 = nn.Linear(in_feats, h_feats)
        self.linear2 = nn.Linear(h_feats, h_feats)
        self.mlp = MLP(h_feats, h_feats, num_classes, mlp_layers, dropout_rate=dropout_rate)
        self.act = getattr(nn, activation)()

    def forward(self, graph):
        in_feat = graph.ndata['feature']
        h = self.linear1(in_feat)
        h = self.act(h)
        h = self.linear2(h)
        h = self.act(h)
        h = self.bernconv(graph, h)
        h = self.act(h)
        h = self.mlp(h, False)
        return h


class AMNet(nn.Module):
    def __init__(self, in_feats, h_feats=32, num_classes=2, orders=2, num_layers=3, mlp_layers=1, dropout_rate=0,
                 activation='ReLU', **kwargs):
        super().__init__()
        self.linear1 = nn.Linear(in_feats, h_feats)
        self.linear2 = nn.Linear(h_feats, h_feats)
        self.mlp_layers = nn.Sequential()
        if mlp_layers > 0:
            for i in range(mlp_layers-1):
                self.mlp_layers.append(nn.Linear(h_feats, h_feats))
            self.mlp_layers.append(nn.Linear(h_feats, num_classes))
        self.act = getattr(nn, activation)()
        self.attn = nn.Tanh()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
        self.num_layers = num_layers
        self.layers = nn.Sequential()
        for i in range(num_layers):
            self.layers.append(BernConv(orders))
        self.linear_transform_in = nn.Sequential(self.linear1, self.act, self.linear2)
        self.W_f = nn.Sequential(self.linear2, self.attn)
        self.W_x = nn.Sequential(self.linear2, self.attn)
        self.linear_cls_out = nn.Sequential(self.mlp_layers)

    def forward(self, graph):
        in_feat = graph.ndata['feature']
        h = self.linear_transform_in(in_feat)
        h_list = []
        for i, layer in enumerate(self.layers):
            h_ = layer(graph, h)
            h_list.append(h_)
        h_filters = torch.stack(h_list, dim=1)
        h_filters_proj = self.W_f(h_filters)
        x_proj = self.W_x(h).unsqueeze(-1)
        score_logit = torch.bmm(h_filters_proj, x_proj)
        soft_score = F.softmax(score_logit, dim=1)
        score = soft_score
        res = h_filters[:, 0, :] * score[:, 0]
        for i in range(1, self.num_layers):
            res += h_filters[:, i, :] * score[:, i]
        y_hat = self.linear_cls_out(res)
        return y_hat


def calculate_theta(d):
    thetas = []
    x = sympy.symbols('x')
    for i in range(d+1):
        f = sympy.poly((x/2) ** i * (1 - x/2) ** (d-i) / (scipy.special.beta(i+1, d+1-i)))
        coeff = f.all_coeffs()
        inv_coeff = []
        for i in range(d+1):
            inv_coeff.append(float(coeff[d-i]))
        thetas.append(inv_coeff)
    return thetas


class BWGNN(nn.Module):
    def __init__(self, in_feats, h_feats=32, num_classes=2, num_layers=2, mlp_layers=2, dropout_rate=0,
                 activation='ReLU', **kwargs):
        super(BWGNN, self).__init__()
        self.thetas = calculate_theta(d=num_layers)
        self.conv = []
        for i in range(len(self.thetas)):
            self.conv.append(PolyConv(self.thetas[i]))
        self.linear = nn.Linear(in_feats, h_feats)
        self.linear2 = nn.Linear(h_feats, h_feats)
        self.mlp = MLP(h_feats*len(self.conv), h_feats, num_classes, mlp_layers, dropout_rate)
        self.act = getattr(nn, activation)()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, graph):
        in_feat = graph.ndata['feature']
        h = self.linear(in_feat)
        h = self.act(h)
        h = self.linear2(h)
        h = self.act(h)
        h_final = torch.zeros([len(in_feat), 0], device=h.device)

        for conv in self.conv:
            h0 = conv(graph, h)
            h_final = torch.cat([h_final, h0], -1)
        h_final = self.dropout(h_final)
        h = self.mlp(h_final, False)
        return h


class GCN(nn.Module):
    def __init__(self, in_feats, h_feats=32, num_classes=2, num_layers=2, mlp_layers=1, dropout_rate=0,
                 activation='ReLU', **kwargs):
        super().__init__()
        self.h_feats = h_feats
        self.layers = nn.ModuleList()
        self.act = getattr(nn, activation)()
        self.layers.append(dglnn.GraphConv(in_feats, h_feats, activation=self.act, allow_zero_in_degree=True))
        for i in range(num_layers-1):
            self.layers.append(dglnn.GraphConv(h_feats, h_feats, activation=self.act, allow_zero_in_degree=True))
        self.mlp = MLP(h_feats, h_feats, num_classes, mlp_layers, dropout_rate)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, graph, edge_weight=None):
        h, emb = self.get_emb(graph, edge_weight)
        return h

    def get_emb(self, graph, edge_weight=None):
        h = graph.ndata['feature']
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(graph, h, edge_weight=edge_weight)
        emb = h
        h = self.mlp(h, False)
        return h, emb


class GCN_M(GCN):
    def __init__(self, in_feats, h_feats=32, num_classes=2, num_layers=2, mlp_layers=1, dropout_rate=0, activation='ReLU', **kwargs):
        super().__init__(in_feats, h_feats, num_classes, num_layers, mlp_layers, dropout_rate, activation, **kwargs)
        self.feature_fusion = FeatureSimpleFusion(i_dim=in_feats, h_dim=in_feats, o_dim=in_feats)

    def get_emb(self, graph, edge_weight=None):
        if 'feature' in graph.ndata and 'motif' in graph.ndata:
            if 'og_feature' not in graph.ndata:
                print(f"load motif for {self.__class__.__name__} with {graph}")
                graph.ndata['og_feature'] = graph.ndata['feature'].clone()
            graph.ndata['feature'] = self.feature_fusion(graph.ndata['og_feature'], graph.ndata['motif'])
        return super().get_emb(graph, edge_weight)


class RGCN(nn.Module):
    def __init__(self, in_feats, h_feats=32, num_classes=2, num_layers=2, mlp_layers=1, dropout_rate=0, activation='ReLU', etypes=None, **kwargs):
        super().__init__()
        self.h_feats = h_feats
        self.layers = nn.ModuleList()
        self.act = getattr(nn, activation)()
        self.layers.append(dgl.nn.HeteroGraphConv({
            (st, et, dt) : dglnn.GraphConv(in_feats, h_feats, activation=self.act)
            for (st, et, dt) in etypes
        }, aggregate='mean'))
        self.type = etypes[0][0]
        for i in range(num_layers-1):
            self.layers.append(dgl.nn.HeteroGraphConv({
                (st, et, dt) : dglnn.GraphConv(h_feats, h_feats, activation=self.act)
                for (st, et, dt) in etypes
            }, aggregate='mean'))
        self.mlp = MLP(h_feats, h_feats, num_classes, mlp_layers, dropout_rate)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, graph):
        h = {self.type: graph.ndata['feature']}
        for i, layer in enumerate(self.layers):
            # print(layer)
            if i != 0:
                h[self.type] = self.dropout(h[self.type])
            # print(graph, h)
            h = layer(graph, h)
        h = self.mlp(h[self.type], False)
        return h


class HGT(nn.Module):
    def __init__(self, in_feats, h_feats=32, num_heads=1, num_classes=2, num_layers=2, mlp_layers=1, dropout_rate=0.2, activation='ReLU', etypes=None, **kwargs):
        super().__init__()
        self.h_feats = h_feats
        self.layers = nn.ModuleList()
        self.act = getattr(nn, activation)()
        self.input_linear = nn.Linear(in_feats, h_feats)
        self.type = etypes[0][0]
        for i in range(num_layers):
            self.layers.append(dgl.nn.HGTConv(h_feats, h_feats // num_heads,
                    num_heads, 1, len(etypes), dropout=dropout_rate))
        self.mlp = MLP(h_feats, h_feats, num_classes, mlp_layers, dropout_rate)
        self.graph = None

    def forward(self, graph):
        if self.graph is None:
            self.graph = dgl.to_homogeneous(graph, ndata=['feature'])
        h = self.graph.ndata['feature']
        graph = self.graph
        h = self.input_linear(h)
        for i, layer in enumerate(self.layers):
            h = layer(graph, h, graph.ndata[dgl.NTYPE], graph.edata[dgl.ETYPE])
            h = self.act(h)
        h = self.mlp(h, False)
        # print( graph.ndata[dgl.NTYPE], graph.edata[dgl.ETYPE])
        return h


class GIN(nn.Module):
    def __init__(self, in_feats, h_feats=32, num_classes=2, num_layers=2, agg='mean', dropout_rate=0,
                 activation='ReLU', **kwargs):
        super().__init__()
        self.layers = nn.ModuleList()
        self.act = getattr(nn, activation)()
        self.layers.append(dglnn.GINConv(nn.Linear(in_feats, h_feats), activation=self.act, aggregator_type=agg))
        for i in range(1, num_layers-1):
            self.layers.append(dglnn.GINConv(nn.Linear(h_feats, h_feats), activation=self.act, aggregator_type=agg))
        self.layers.append(dglnn.GINConv(nn.Linear(h_feats, num_classes),  activation=None, aggregator_type=agg))
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, graph):
        h = graph.ndata['feature']
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(graph, h)
        return h


class GIN_noparam(nn.Module):
    def __init__(self, num_layers=2, agg='mean', init_eps=-1, **kwargs):
        super().__init__()
        self.gnn = dglnn.GINConv(None, activation=None, init_eps=init_eps,
                                 aggregator_type=agg)
        self.num_layers = num_layers

    def forward(self, graph):
        h = graph.ndata['feature']
        h_final = h.detach().clone()
        for i in range(self.num_layers):
            h = self.gnn(graph, h)
            h_final = torch.cat([h_final, h], -1)
        print(h_final)
        return h_final


class ChebNet(nn.Module):
    def __init__(self, in_feats, h_feats=32, num_classes=2, num_layers=2, dropout_rate=0, activation='ReLU', **kwargs):
        super().__init__()
        self.input_linear = nn.Linear(in_feats, h_feats)
        self.act = getattr(nn, activation)()
        self.chebconv = dglnn.ChebConv(h_feats, h_feats, num_layers, activation=self.act)
        self.output_linear = nn.Linear(h_feats, num_classes)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, graph):
        h = graph.ndata['feature']
        h = self.input_linear(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.chebconv(graph, h, lambda_max=[2])
        h = self.dropout(h)
        h = self.output_linear(h)
        return h


class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats=32, num_classes=2, num_layers=2, agg='mean', dropout_rate=0,
                 activation='ReLU', **kwargs):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()
        self.act = getattr(nn, activation)()
        self.layers.append(dglnn.SAGEConv(in_feats, h_feats, agg, activation=self.act))
        for i in range(num_layers-1):
            self.layers.append(dglnn.SAGEConv(h_feats, h_feats, agg, activation=self.act))
        self.output_linear = nn.Linear(h_feats, num_classes)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, graph):
        h, emb = self.get_emb(graph)
        return h

    def get_emb(self, graph):
        h = graph.ndata['feature']
        for layer in self.layers:
            h = self.dropout(h)
            h = layer(graph, h)
        emb = h
        h = self.output_linear(h)
        return h, emb


class GraphSAGE_M(GraphSAGE):
    def __init__(self, in_feats, h_feats=32, num_classes=2, num_layers=2, agg='mean', dropout_rate=0,
                 activation='ReLU', **kwargs):
        super().__init__(in_feats, h_feats, num_classes, num_layers, agg, dropout_rate, activation, **kwargs)
        self.feature_fusion = FeatureSimpleFusion(i_dim=in_feats, h_dim=in_feats, o_dim=in_feats)
    
    def get_emb(self, graph):
        if 'feature' in graph.ndata and 'motif' in graph.ndata:
            if 'og_feature' not in graph.ndata:
                print(f"load motif for {self.__class__.__name__} with {graph}")
                graph.ndata['og_feature'] = graph.ndata['feature'].detach()
            feat = graph.ndata['og_feature']
            motif = graph.ndata['motif']
            feat = self.feature_fusion(feat, motif)
            graph.ndata['feature'] = feat
        return super().get_emb(graph)



class GraphSAGEMean(GraphSAGE):
    def __init__(self, *args, **kwargs):
        if "agg" in kwargs:
            del kwargs['agg']
        super().__init__(*args, agg="mean", **kwargs)
        
        
class GraphSAGEPool(GraphSAGE):
    def __init__(self, *args, **kwargs):
        if "agg" in kwargs:
            del kwargs['agg']
        super().__init__(*args, agg="pool", **kwargs)


class GraphSAGELSTM(GraphSAGE):
    def __init__(self, *args, **kwargs):
        if "agg" in kwargs:
            del kwargs['agg']
        super().__init__(*args, agg="lstm", **kwargs)


class GraphSAGEGCN(GraphSAGE):
    def __init__(self, *args, **kwargs):
        if "agg" in kwargs:
            del kwargs['agg']
        super().__init__(*args, agg="gcn", **kwargs)


class MLP(nn.Module):
    def __init__(self, in_feats, h_feats=32, num_classes=2, num_layers=2, dropout_rate=0, activation='ReLU', **kwargs):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.act = getattr(nn, activation)()
        if num_layers == 0:
            return
        if num_layers == 1:
            self.layers.append(nn.Linear(in_feats, num_classes))
        else:
            self.layers.append(nn.Linear(in_feats, h_feats))
            for i in range(1, num_layers-1):
                self.layers.append(nn.Linear(h_feats, h_feats))
            self.layers.append(nn.Linear(h_feats, num_classes))
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, h, is_graph=True):
        if is_graph:
            h = h.ndata['feature']
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(h)
            if i != len(self.layers)-1:
                h = self.act(h)
        return h


class SGC(nn.Module):
    def __init__(self, in_feats, h_feats=32, num_classes=2, k=2, mlp_layers=1, dropout_rate = 0, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList()
        self.mlp = None
        if mlp_layers==1:
            self.sgc = dglnn.SGConv(in_feats, num_classes, k=k)
        else:
            self.sgc = dglnn.SGConv(in_feats, h_feats, k=k)
            self.mlp = MLP(h_feats, num_classes, num_classes, mlp_layers-1, dropout_rate)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, graph):
        h = graph.ndata['feature']
        h = self.dropout(h)
        h = self.sgc(graph, h)
        if self.mlp is not None:
            h = self.mlp(h, False)
        return h


class DGI_GIN(nn.Module):
    def __init__(self, in_feats, h_feats=32, num_layers=2, agg='mean', dropout_rate=0, **kwargs):
        super().__init__()
        self.h_feats = h_feats
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.GINConv(nn.Linear(in_feats, h_feats), activation=F.relu, aggregator_type=agg))
        for i in range(num_layers-1):
            self.layers.append(dglnn.GINConv(nn.Linear(h_feats, h_feats), activation=F.relu, aggregator_type=agg))
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

    def forward(self, graph, h=None):
        if h is None:
            h = graph.ndata['feature']
        for i, layer in enumerate(self.layers):
            if i != 0 and self.dropout:
                h = self.dropout(h)
            h = layer(graph, h)
        return h


class DCI_Encoder(nn.Module):
    def __init__(self, in_feats, h_feats=32, num_classes=2, num_layers=2, mlp_layers=2, dropout_rate=0, agg='mean',  **kwargs):
        super().__init__()
        self.conv = DGI_GIN(in_feats, h_feats, num_layers, agg, dropout_rate)
        self.mlp = MLP(h_feats, h_feats, num_classes, mlp_layers, dropout_rate)

    def forward(self, graph, corrupt=False, use_mlp=False):
        h = graph.ndata['feature']
        if corrupt:
            perm = torch.randperm(graph.num_nodes())
            h = h[perm]
        h = self.conv(graph, h)
        if use_mlp:
            h = self.mlp(h, is_graph=False)
        return h


class Discriminator(nn.Module):
    def __init__(self, h_feats, **kwargs):
        super(Discriminator, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(h_feats, h_feats))
        self.reset_parameters()

    def uniform(self, size, tensor):
        bound = 1.0 / math.sqrt(size)
        if tensor is not None:
            tensor.data.uniform_(-bound, bound)

    def reset_parameters(self):
        size = self.weight.size(0)
        self.uniform(size, self.weight)

    def forward(self, features, summary):
        features = torch.matmul(features, torch.matmul(self.weight, summary))
        return features


class DCI(nn.Module):
    def __init__(self, in_feats, h_feats=32, num_classes=2, num_layers=2, mlp_layers=1, dropout_rate=0, **kwargs):
        super().__init__()
        self.encoder = DCI_Encoder(in_feats, h_feats, num_classes, num_layers, mlp_layers, dropout_rate)
        self.discriminator = Discriminator(h_feats)
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, graph, cluster_info, cluster_num):
        positive = self.encoder(graph, corrupt=False)
        negative = self.encoder(graph, corrupt=True)
        loss = 0
        for i in range(cluster_num):
            node_idx = cluster_info[i]

            positive_block = torch.unsqueeze(positive[node_idx], 0)
            negative_block = torch.unsqueeze(negative[node_idx], 0)
            summary = torch.sigmoid(positive.mean(dim=0))

            negative_block = self.discriminator(negative_block, summary)
            positive_block = self.discriminator(positive_block, summary)

            l1 = self.loss(positive_block, torch.ones_like(positive_block))
            l2 = self.loss(negative_block, torch.zeros_like(negative_block))

            loss_tmp = l1 + l2
            loss += loss_tmp

        return loss / cluster_num

    def get_emb(self, graph):
        h = self.encoder(graph, corrupt=False)
        return h
    

class PrincipalAggregate(nn.Module):
    def __init__(self, aggregators=['mean', 'max', 'min', 'std'], h_feats=32, act='ReLU', dropout=0.):
        super().__init__()
        self.aggregators = aggregators
        self.agg_funcs = [getattr(self, f"agg_{agg}") for agg in aggregators]
        self.linear = nn.Linear(len(self.agg_funcs)*h_feats, h_feats)
        self.act =getattr(nn, act)()

    def forward(self, mfg, feat):
        h = [agg(mfg, feat) for agg in self.agg_funcs]
        return self.act(self.linear(torch.cat(h, dim=1)))

    def agg_mean(self, mfg, X):
        mfg.srcdata['h'] = X
        mfg.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'h'))
        return mfg.dstdata['h']

    def agg_min(self, mfg, X):
        mfg.srcdata['h'] = X
        mfg.update_all(fn.copy_u('h', 'm'), fn.min('m', 'h'))
        return mfg.dstdata['h']

    def agg_max(self, mfg, X):
        mfg.srcdata['h'] = X
        mfg.update_all(fn.copy_u('h', 'm'), fn.max('m', 'h'))
        return mfg.dstdata['h']

    def agg_std(self, mfg, X):
        diff = self.agg_mean(mfg, X ** 2) - self.agg_mean(mfg, X) ** 2
        return torch.sqrt(F.relu(diff) + 1e-5)

    def agg_sum(self, mfg, X):
        mfg.srcdata['h'] = X
        mfg.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
        return mfg.dstdata['h']

    def __repr__(self):
        return f"""PrincipalAggregate(
            aggregators={self.aggregators}
            linear={self.linear}
        )"""


class PNA(nn.Module):
    def __init__(self, in_feats, h_feats=32, num_classes=2, num_layers=2, activation='ReLU', dropout_rate=0, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList()
        # self.layers.append(nn.Linear(in_feats, h_feats))
        self.input_linear = nn.Linear(in_feats, h_feats)

        self.act = getattr(nn, activation)()
        self.output_linear = nn.Linear(h_feats, num_classes)
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

        for i in range(0, num_layers):
            self.layers.append(PrincipalAggregate(h_feats=h_feats, act=activation, dropout=dropout_rate))


    def forward(self, graph):
        h = graph.ndata['feature']
        h = self.input_linear(h)
        for i, layer in enumerate(self.layers):
            if i != 0 and self.dropout:
                h = self.dropout(h)
            h = layer(graph, h)
        h = self.output_linear(h)
        return h


class CAREConv(nn.Module):
    def __init__(self,in_feats, h_feats, num_classes=2, activation=None, step_size=0.02, **kwargs):
        super().__init__()
        self.activation = activation
        self.step_size = step_size
        self.in_feats = in_feats
        self.h_feats = h_feats
        self.num_classes = num_classes
        self.dist = {}
        self.linear = nn.Linear(self.in_feats, self.h_feats)
        self.MLP = nn.Linear(self.in_feats, self.num_classes)
        self.p = {}
        self.last_avg_dist = {}
        self.f = {}
        self.cvg = {}

    def _calc_distance(self, edges):
        # formula 2
        d = torch.norm(torch.tanh(self.MLP(edges.src["h"]))
            - torch.tanh(self.MLP(edges.dst["h"])), 1, 1,)
        return {"d": d}

    def _top_p_sampling(self, graph, p):
        # Compute the number of neighbors to keep for each node
        in_degrees = graph.in_degrees()
        num_neigh = torch.ceil(in_degrees.float() * p).int()

        # Fetch all edges and their distances
        all_edges = graph.edges(form="eid")
        dist = graph.edata["d"]

        # Create a prefix sum array for in-degrees to use for indexing
        prefix_sum = torch.cat([torch.tensor([0]).cuda(), in_degrees.cumsum(0)[:-1]])

        # Get the edges for each node using advanced indexing
        selected_edges = []
        for i, node_deg in enumerate(num_neigh):
            start_idx = prefix_sum[i]
            end_idx = start_idx + node_deg
            sorted_indices = torch.argsort(dist[start_idx:end_idx])[:node_deg]
            selected_edges.append(all_edges[start_idx:end_idx][sorted_indices])
        return torch.cat(selected_edges)

    def forward(self, graph, epoch=0):
        feat = graph.ndata['feature']
        edges = graph.canonical_etypes
        if epoch == 0:
            for etype in edges:
                self.p[etype] = 0.5
                self.last_avg_dist[etype] = 0
                self.f[etype] = []
                self.cvg[etype] = False

        with graph.local_scope():
            graph.ndata["h"] = feat

            hr = {}
            for i, etype in enumerate(edges):
                graph.apply_edges(self._calc_distance, etype=etype)
                self.dist[etype] = graph.edges[etype].data["d"]
                sampled_edges = self._top_p_sampling(graph[etype], self.p[etype])

                # formula 8
                graph.send_and_recv(
                    sampled_edges,
                    fn.copy_u("h", "m"),
                    fn.mean("m", "h_%s" % etype[1]),
                    etype=etype,
                )
                hr[etype] = graph.ndata["h_%s" % etype[1]]
                if self.activation is not None:
                    hr[etype] = self.activation(hr[etype])

            # formula 9 using mean as inter-relation aggregator
            p_tensor = (
                torch.Tensor(list(self.p.values())).view(-1, 1, 1).to(graph.device)
            )
            h_homo = torch.sum(torch.stack(list(hr.values())) * p_tensor, dim=0)
            h_homo += feat
            if self.activation is not None:
                h_homo = self.activation(h_homo)

            return self.linear(h_homo)


class CAREGNN(nn.Module):
    def __init__(self, in_feats, num_classes=2, h_feats=64, edges=None, num_layers=1, activation=None, step_size=0.02, **kwargs):
        super().__init__()
        self.in_feats = in_feats
        self.h_feats = h_feats
        self.num_classes = num_classes
        self.activation = None if activation is None else getattr(nn, activation)()
        self.step_size = step_size
        self.num_layers = num_layers
        self.output_linear = nn.Linear(h_feats, num_classes)
        self.layers = nn.ModuleList()
        self.layers.append(          # Input layer
            CAREConv(self.in_feats, self.num_classes, self.num_classes, activation=self.activation, step_size=self.step_size,))
        for i in range(self.num_layers - 1):  # Hidden layers with n - 2 layers
            self.layers.append(CAREConv(self.h_feats, self.h_feats, self.num_classes, activation=self.activation, step_size=self.step_size,))
            # self.layers.append(   # Output layer
                # CAREConv(self.h_feats, self.num_classes, self.num_classes, activation=self.activation, step_size=self.step_size,))

    def forward(self, graph, epoch=0):            
        for layer in self.layers:
            feat = layer(graph, epoch)
        return feat

    def RLModule(self, graph, epoch, idx):
        for layer in self.layers:
            for etype in graph.canonical_etypes:
                if not layer.cvg[etype]:
                    # formula 5
                    eid = graph.in_edges(idx, form='eid', etype=etype)
                    avg_dist = torch.mean(layer.dist[etype][eid])

                    # formula 6
                    if layer.last_avg_dist[etype] < avg_dist:
                        if layer.p[etype] - self.step_size > 0:
                            layer.p[etype] -=   self.step_size
                        layer.f[etype].append(-1)
                    else:
                        if layer.p[etype] + self.step_size <= 1:
                            layer.p[etype] += self.step_size
                        layer.f[etype].append(+1)
                    layer.last_avg_dist[etype] = avg_dist

                    # formula 7
                    if epoch >= 9 and abs(sum(layer.f[etype][-10:])) <= 2:
                        layer.cvg[etype] = True


class H2FDetector_layer(nn.Module):
    def __init__(self, in_feats, h_feats, head, relation_aware, etype, dropout_rate, if_sum=False):
        super().__init__()
        self.etype = etype
        self.head = head
        self.hd = h_feats
        self.if_sum = if_sum
        self.relation_aware = relation_aware
        self.w_liner = nn.Linear(in_feats, h_feats*head)
        self.atten = nn.Linear(2*self.hd, 1)
        self.relu = nn.ReLU()
        self.leakyrelu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, graph, h):
        with graph.local_scope():
            graph.ndata['feat'] = h
            graph.apply_edges(self.sign_edges, etype=self.etype)
            h = self.w_liner(h)
            graph.ndata['h'] = h
            graph.update_all(message_func=self.message, reduce_func=self.reduce, etype=self.etype)
            out = graph.ndata['out']
            return out

    def message(self, edges):
        src = edges.src
        src_features = edges.data['sign'].view(-1,1)*src['h']
        src_features = src_features.view(-1, self.head, self.hd)
        z = torch.cat([src_features, edges.dst['h'].view(-1, self.head, self.hd)], dim=-1)
        alpha = self.atten(z)
        alpha = self.leakyrelu(alpha)
        return {'atten':alpha, 'sf':src_features}

    def reduce(self, nodes):
        alpha = nodes.mailbox['atten']
        sf = nodes.mailbox['sf']
        alpha = self.softmax(alpha)
        out = torch.sum(alpha*sf, dim=1)
        if not self.if_sum:
            out = out.view(-1, self.head*self.hd)
        else:
            out = out.sum(dim=-2)
        return {'out':out}

    def sign_edges(self, edges):
        src = edges.src['feat']
        dst = edges.dst['feat']
        score = self.relation_aware(src, dst)
        return {'sign':torch.sign(score)}


class RelationAware(nn.Module):
    def __init__(self, in_feats, h_feats, dropout_rate):
        super().__init__()
        self.d_liner = nn.Linear(in_feats, h_feats)
        self.f_liner = nn.Linear(3*h_feats, 1)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

    def forward(self, src, dst):
        src = self.d_liner(src)
        dst = self.d_liner(dst)
        diff = src-dst
        e_feats = torch.cat([src, dst, diff], dim=1)
        if self.dropout is not None:
            e_feats = self.dropout(e_feats)
        score = self.f_liner(e_feats).squeeze()
        score = self.tanh(score)
        return score


def hinge_loss(labels, scores):
    margin = 1
    ls = labels*scores
    
    loss = F.relu(margin-ls)
    loss = loss.mean()
    return loss


class MultiRelationH2FDetectorLayer(nn.Module):
    def __init__(self, in_feats, h_feats, head, dataset, dropout_rate, if_sum=False):
        super().__init__()
        self.relation = copy.deepcopy(dataset.etypes)
        self.n_relation = len(self.relation)
        if not if_sum:
            self.liner = nn.Linear(self.n_relation*h_feats*head, h_feats*head)
        else:
            self.liner = nn.Linear(self.n_relation*h_feats, h_feats)
        self.relation_aware = RelationAware(in_feats, h_feats*head, dropout_rate)
        self.minelayers = nn.ModuleDict()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        for e in self.relation:
            self.minelayers[e] = H2FDetector_layer(in_feats, h_feats, head, self.relation_aware, e, dropout_rate, if_sum)
    
    
    def forward(self, graph, h):
        hs = []
        for e in self.relation:
            he = self.minelayers[e](graph, h)
            hs.append(he)
        h = torch.cat(hs, dim=1)
        if self.dropout is not None:
            h = self.dropout(h)

        h = self.liner(h)
        return h
    
    def loss(self, graph, h):
        with graph.local_scope():
            graph.ndata['feat'] = h
            agg_h = self.forward(graph, h)
            
            graph.apply_edges(self.score_edges, etype='homo')
            edges_score = graph.edges['homo'].data['score']
            edge_train_mask = graph.edges['homo'].data['train_mask'].bool()
            edge_train_label = graph.edges['homo'].data['label'][edge_train_mask]
            edge_train_pos = edge_train_label == 1
            edge_train_neg = edge_train_label == -1
            edge_train_pos_index = edge_train_pos.nonzero().flatten().detach().cpu().numpy()
            edge_train_neg_index = edge_train_neg.nonzero().flatten().detach().cpu().numpy()
            edge_train_pos_index = np.random.choice(edge_train_pos_index, size=len(edge_train_neg_index))
            index = np.concatenate([edge_train_pos_index, edge_train_neg_index])
            index.sort()
            edge_train_score = edges_score[edge_train_mask]
            # hinge loss
            edge_diff_loss = hinge_loss(edge_train_label[index], edge_train_score[index])

            train_mask = graph.ndata['train_mask'].bool()
            train_h = agg_h[train_mask]
            train_label = graph.ndata['label'][train_mask]
            train_pos = train_label==1
            train_neg = train_label==0
            train_pos_index = train_pos.nonzero().flatten().detach().cpu().numpy()
            train_neg_index = train_neg.nonzero().flatten().detach().cpu().numpy()
            train_neg_index = np.random.choice(train_neg_index, size=len(train_pos_index))
            node_index = np.concatenate([train_neg_index, train_pos_index])
            node_index.sort()
            pos_prototype = torch.mean(train_h[train_pos], dim=0).view(1,-1)
            neg_prototype = torch.mean(train_h[train_neg], dim=0).view(1,-1)
            train_h_loss = train_h[node_index]
            pos_prototypes = pos_prototype.expand(train_h_loss.shape)
            neg_prototypes = neg_prototype.expand(train_h_loss.shape)
            diff_pos = - F.pairwise_distance(train_h_loss, pos_prototypes)
            diff_neg = - F.pairwise_distance(train_h_loss, neg_prototypes)
            diff_pos = diff_pos.view(-1,1)
            diff_neg = diff_neg.view(-1,1)
            diff = torch.cat([diff_neg, diff_pos], dim=1)
            diff_loss = F.cross_entropy(diff, train_label[node_index])

            return agg_h, edge_diff_loss, diff_loss
        
    def score_edges(self, edges):
        src = edges.src['feat']
        dst = edges.dst['feat']
        score = self.relation_aware(src, dst)
        return {'score':score}
        
        
class H2FD(nn.Module):
    def __init__(self, in_feats, graph, n_layer=1, intra_dim=16, n_class=2, gamma1=1.2, gamma2=2, head=2, dropout_rate=0.1, **kwargs):
        super().__init__()
        self.in_feats = in_feats
        self.n_layer = n_layer 
        self.intra_dim = intra_dim 
        self.n_class = n_class
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.head = head
        self.dropout_rate = dropout_rate
        self.mine_layers = nn.ModuleList()
        if n_layer == 1:
            self.mine_layers.append(MultiRelationH2FDetectorLayer(self.in_feats, self.n_class, head, graph, dropout_rate, if_sum=True))
        else:
            self.mine_layers.append(MultiRelationH2FDetectorLayer(self.in_feats, self.intra_dim, head, graph, dropout_rate))
            for _ in range(1, self.n_layer-1):
                self.mine_layers.append(MultiRelationH2FDetectorLayer(self.intra_dim*head, self.intra_dim, head, graph, dropout_rate))
            self.mine_layers.append(MultiRelationH2FDetectorLayer(self.intra_dim*head, self.n_class, head, graph, dropout_rate, if_sum=True))
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.relu = nn.ReLU()
    
    def forward(self, graph):
        feats = graph.ndata['feature'].float()
        train_mask = graph.ndata['train_mask'].bool()
        train_label = graph.ndata['label'][train_mask]
        train_pos = train_label == 1
        train_neg = train_label == 0
        
        pos_index = train_pos.nonzero().flatten().detach().cpu().numpy()
        neg_index = train_neg.nonzero().flatten().detach().cpu().numpy()
        neg_index = np.random.choice(neg_index, size=len(pos_index), replace=False)
        index = np.concatenate([pos_index, neg_index])
        index.sort()
        h, edge_loss, prototype_loss = self.mine_layers[0].loss(graph, feats)
        if self.n_layer > 1:
            h = self.relu(h)
            h = self.dropout(h)
            for i in range(1, len(self.mine_layers)-1):
                h, e_loss, p_loss = self.mine_layers[i].loss(graph, h)
                h = self.relu(h)
                h = self.dropout(h)
                edge_loss += e_loss
                prototype_loss += p_loss
            h, e_loss, p_loss = self.mine_layers[-1].loss(graph, h)
            edge_loss += e_loss
            prototype_loss += p_loss
        model_loss = F.cross_entropy(h[train_mask][index], train_label[index])
        loss = model_loss + self.gamma1*edge_loss + self.gamma2*prototype_loss
        return loss, h


class MySAGEModule(nn.Module):
    def __init__(self, in_feat, out_feat, agg='mean', act='ReLU'):
        super().__init__()
        
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.linear = nn.Linear(in_feat, out_feat, bias=False)
        if agg != 'gcn':
            self.res_linear = nn.Linear(in_feat, out_feat, bias=True)
        self.act = getattr(nn, act)()
        self.agg = agg
        
        if agg == 'pool':
            self.pool = nn.AdaptiveMaxPool1d(1)
            self.pool_linear = nn.Linear(in_feat, in_feat)
        
        if agg == 'lstm':
            self.lstm = nn.LSTM(in_feat, in_feat, batch_first=True)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        if self.agg == 'pool':
            nn.init.xavier_uniform_(self.pool_linear.weight, gain=gain)
        if self.agg == 'lstm':
            self.lstm.reset_parameters()
        if self.agg != 'gcn':
            nn.init.xavier_uniform_(self.res_linear.weight, gain=gain)
        nn.init.xavier_uniform_(self.linear.weight, gain=gain)
    
    def forward(self, graph: dgl.DGLGraph, h: torch.Tensor):
        def sage_lstm_agg_reducer(nodes: dgl.udf.NodeBatch):
            f: torch.Tensor = nodes.mailbox['src']
            B, _, _ = f.shape
            h_s, c_s = (  # hidden_state, cell_state
                # Tensor.new_zeros功能同torch.zeros, 但拷贝被调用Tensor的dtype与device
                f.new_zeros((1, B, self.in_feat)),  # (L, B, H)
                f.new_zeros((1, B, self.in_feat)),
            )
            _, (h_s, _) = self.lstm(f, (h_s, c_s))  # 输出output, (hidden_state, cell_state)
            # output: (B, L, H)
            # hidden_state&cell_state: (L, B, H)
            return {'f': h_s.squeeze(0)}  # 合并Layer维度
            
        with graph.local_scope():
            h_res = h
            if self.agg == 'mean':
                graph.ndata['f_last'] = h
            
                graph.update_all(
                    fn.copy_u('f_last', 'src'),  # 将边源节点的特征发送到目标节点mailbox的src中
                    fn.mean('src', 'f')  # 对目标节点收到的src特征及当前节点特征进行平均，写入目标节点f中
                )
                h = graph.ndata['f']
                h = self.act(self.linear(h))
            elif self.agg == 'pool':
                graph.ndata['f_last'] = F.relu(self.pool_linear(h))
            
                graph.update_all(
                    fn.copy_u('f_last', 'src'),
                    fn.max("src", "f")
                )
                h = graph.ndata['f']
                h = self.act(self.linear(h))
            elif self.agg == 'lstm':
                graph.ndata['f_last'] = h
                
                graph.update_all(
                    fn.copy_u('f_last', 'src'),
                    sage_lstm_agg_reducer
                )
                h = graph.ndata['f']
                h = self.act(self.linear(h))
            elif self.agg == 'gcn':
                graph.ndata['f_last'] = h
                
                graph.update_all(fn.copy_u('f_last', 'src'), fn.sum('src', 'f'))
                degs = graph.in_degrees().to(h)
                h = (graph.dstdata['f'] + graph.dstdata['f_last']) / (
                    degs.unsqueeze(-1) + 1
                )  # 特征平均
                
                h = self.act(self.linear(h))
            else:
                raise NotImplementedError
            
            if self.agg != 'gcn':
                h = self.res_linear(h_res) + h  # 残差
            
            return h


class MySAGE(nn.Module):
    def __init__(self, in_feats, h_feats=32, num_classes=2, 
                 num_layers=2,  # K
                 agg='mean',  # 聚合函数
                 act='ReLU',  # 激活函数
                 dropout=0.0,
                 **kwargs
                 ):
        super().__init__()
        
        self.dropout = nn.Dropout(dropout) if dropout != 0 else nn.Identity()
        self.layers = nn.ModuleList()
        self.out = nn.Linear(h_feats, num_classes)
        self.act = getattr(nn, act)()
        
        last_i = in_feats
        for _ in range(num_layers):
            self.layers.append(MySAGEModule(last_i, h_feats, agg, act))
            # self.layers.append(dglnn.SAGEConv(last_i, h_feats, agg, activation=act))
            last_i = h_feats
        
    def forward(self, graph):
        h = graph.ndata['feature']
        for l in self.layers:
            # print("[DEBUG]", h.shape)
            h = self.dropout(h)
            h = l(graph, h)
        h = self.out(h)
        return h


class MySAGEMean(MySAGE):
    def __init__(self, *args, **kwargs):
        if "agg" in kwargs:
            del kwargs['agg']
        super().__init__(*args, agg="mean", **kwargs)


class MySAGEPool(MySAGE):
    def __init__(self, *args, **kwargs):
        if "agg" in kwargs:
            del kwargs['agg']
        super().__init__(*args, agg="pool", **kwargs)


class MySAGELSTM(MySAGE):
    def __init__(self, *args, **kwargs):
        if "agg" in kwargs:
            del kwargs['agg']
        super().__init__(*args, agg="lstm", **kwargs)


class MySAGEGCN(MySAGE):
    def __init__(self, *args, **kwargs):
        if "agg" in kwargs:
            del kwargs['agg']
        super().__init__(*args, agg="gcn", **kwargs)


class GAGAEmbedding(nn.Embedding):
    def __init__(self, max_len, emb_dim=32):
        super().__init__(max_len, emb_dim)


class GAGA(nn.Module):  # Label Information Enhanced Fraud Detection against Low Homophily in Graphs
    def __init__(
            self,
            in_feats,
            num_classes=2,
            h_feats=32,  # mlp hid feats
            num_heads=4,
            transformer_dropout=0.1,
            mlp_dropout=0.0,
            agg : Literal['mean', 'cat'] = 'cat',
            num_edge_types=1,
            n_layers=3,
            **kwargs
        ):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(in_features=in_feats, out_features=h_feats),
            nn.ELU(),
            (nn.Dropout(mlp_dropout) if mlp_dropout > 0 else nn.Identity()),
            nn.LayerNorm(h_feats),
            nn.Linear(in_features=h_feats, out_features=h_feats)
        )
        in_feats = h_feats

        self.hop_to_sample = 3  # 2 hop + 1 local
        self.relations = 4  # maximum
        self.groups = 3  # masked, benign, and fraud

        assert num_edge_types <= self.relations, ("To match the transformer num_heads,"
                                                  f" the num of relations should in the range of {self.relations}.")

        self.hop_embeddings = GAGAEmbedding(self.hop_to_sample, in_feats)
        self.rel_embeddings = GAGAEmbedding(self.relations, in_feats)
        self.grp_embeddings = GAGAEmbedding(self.groups, in_feats)

        self.transformer_layer = nn.TransformerEncoderLayer(
            in_feats,
            num_heads,
            dropout=transformer_dropout,
            dim_feedforward=128,
        )
        self.transformer = nn.TransformerEncoder(self.transformer_layer, n_layers)
        
        self.agg = agg
        
        in_feats = in_feats if agg == 'mean' else in_feats * num_edge_types
        self.classifier = nn.Sequential(
            nn.Linear(in_features=in_feats, out_features=num_classes),
            nn.Softmax(0)
        )

    def pre_feature_sample(self, graph, label):
        h = graph.ndata['feature']

        def build_masked_feature(feature, label, target):
            f = torch.clone(feature).detach()
            f[(label != target).bool()] = torch.nan
            return f

        def deal_with_reducer_nan(feature, fill=0):
            feature[feature.isnan()] = fill
            return feature

        def message_with_label_wrapper(cur_label, next_label, init=False, labels=None):
            if labels is None:
                labels = (0, 1, 2)

            def messager_init(edges: dgl.udf.EdgeBatch):
                label = edges.src['label']
                feature = edges.src[cur_label]

                return {
                    f"{next_label}_{l}": build_masked_feature(feature, label, l) for l in labels
                }

            def messager(edges: dgl.udf.EdgeBatch):
                return {
                    f"{next_label}_{l}": edges.src[f"{cur_label}_{l}"] for l in labels
                }

            if init:
                return messager_init
            return messager

        def pre_transformer_reducer_wrapper(cur_feat, next_feat, labels=None):
            if labels is None:
                labels = (0, 1, 2)

            def reducer(nodes: dgl.udf.NodeBatch):
                res = {
                    f"{next_feat}_{l}": deal_with_reducer_nan(torch.nanmean(nodes.mailbox[f"{cur_feat}_{l}"], dim=1)) for l in labels
                }
                return res

            return reducer

        with graph.local_scope():
            graph.ndata['feature'] = h
            graph.ndata['label'] = label

            labels = torch.unique(label).cpu().tolist()

            edge_types = graph.etypes

            # multi hop sampling
            for e_id, etype in enumerate(edge_types):
                graph.update_all(
                    message_with_label_wrapper('feature', f'f_0_{e_id}', init=True, labels=labels),
                    pre_transformer_reducer_wrapper(f'f_0_{e_id}', f'h_0_{e_id}', labels=labels),
                    etype=etype
                )
                graph.update_all(
                    message_with_label_wrapper(f'h_0_{e_id}', f'f_1_{e_id}', labels=labels),
                    pre_transformer_reducer_wrapper(f'f_1_{e_id}', f'h_1_{e_id}', labels=labels),
                    etype=etype
                )

            # feature extracting and position embedding inject
            for eid, etype in enumerate(edge_types):
                graph.ndata[f'final_feature_{eid}'] = torch.stack(
                    [graph.ndata[f] for f in [
                        'feature', *[f'h_0_{eid}_{l}' for l in labels], *[f'h_1_{eid}_{l}' for l in labels]
                    ]],
                    dim=1
                )

            res = []
            for eid, etype in enumerate(edge_types):
                res.append(
                    graph.ndata[f'final_feature_{eid}']
                )

            sampled_feature = {k: v for k, v in zip(edge_types, res)}

        return sampled_feature
    
    def forward(self, graph: dgl.DGLGraph, sampled_feature):
        emb_type = {
            "hop": self.hop_embeddings,
            "rel": self.rel_embeddings,
            "grp": self.grp_embeddings,
        }
        device = graph.device

        def get_pos_emb(typ, pos, dtype=torch.float32, d=None) -> torch.Tensor:
            d = d or device
            return emb_type[typ](torch.tensor(pos).to(d)).to(d, dtype)

        edge_types = graph.etypes

        res = [sampled_feature[k] for k in edge_types]

        res_sub = []
        for r in res:
            res_sub.append(self.mlp(r))
        res = res_sub

        res_sub = []
        if len(edge_types) == 1:
            target_shape = res[0].shape[0]
            res_sub.append(
                torch.sum(torch.stack(
                (res[0],
                    torch.sum(torch.stack([
                        torch.stack((get_pos_emb("hop", 0), get_pos_emb("rel", 0), get_pos_emb("grp", 2))),
                        *[torch.stack((get_pos_emb("hop", 1), get_pos_emb("rel", 0), get_pos_emb("grp", r_i))) for r_i in range(3)],
                        *[torch.stack((get_pos_emb("hop", 2), get_pos_emb("rel", 0), get_pos_emb("grp", r_i))) for r_i in range(3)],
                    ]), dim=1).broadcast_to((target_shape, -1, -1)))
                , dim=2)
            , dim=2))
        else:
            for e_i, _ in enumerate(edge_types):
                target_shape = res[e_i].shape[0]
                res_sub.append(
                    torch.sum(torch.stack(
                        (res[e_i],
                        torch.sum(torch.stack([
                            torch.stack((get_pos_emb("hop", 0), get_pos_emb("rel", e_i), get_pos_emb("grp", 2))),
                            *[torch.stack((get_pos_emb("hop", 1), get_pos_emb("rel", e_i), get_pos_emb("grp", r_i))) for r_i in range(3)],
                            *[torch.stack((get_pos_emb("hop", 2), get_pos_emb("rel", e_i), get_pos_emb("grp", r_i))) for r_i in range(3)],
                        ]), dim=1).broadcast_to((target_shape, -1, -1)))
                    , dim=2)
                , dim=2))
        res = res_sub

        res_sub = []
        for r in res:
            res_sub.append(self.transformer(r).transpose(0, 1)[0])
        res = res_sub

        if self.agg == 'mean':
            if len(edge_types) > 1:
                res = [
                    torch.mean(torch.stack(res, dim=0), dim=0)
                ]
        elif self.agg == 'cat':
            res = [
                torch.concat(res, dim=-1)
            ]
        else:
            raise NotImplementedError(f"No such aggregation method: {self.agg}")

        final = self.classifier(res[0])
        return final


# ConsisGAD
class CustomLinear(nn.Linear):
    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight)
        nn.init.zeros_(self.bias)


class CustomBatchNorm1d(nn.BatchNorm1d):
    def forward(self, input, update_running_stats: bool=True):
        self.track_running_stats = update_running_stats
        return super(CustomBatchNorm1d, self).forward(input)


class ConsisCombinedLoss(nn.Module):  # L + L_{c} => For consistenct training  # nll_loss
    def __init__(self):
        super(ConsisCombinedLoss, self).__init__()

        self.cross_entropy_loss_labelled = nn.NLLLoss()
        self.cross_entropy_loss_consist = nn.NLLLoss()
    
    def forward(self, p_v, y_v, y_w_u, p_h_u):
        L_Cro = self.cross_entropy_loss_labelled(p_v, y_v)
        L_Con = self.cross_entropy_loss_consist(p_h_u, y_w_u)
        return L_Cro + L_Con


class ConsisDataAugumentationLoss(nn.Module):  # alpha * L_{\delta c} + L_{\delta d}
    def __init__(self):
        super(ConsisDataAugumentationLoss, self).__init__()

        # self.alpha = nn.Parameter(torch.randn((1), requires_grad=True))
        self.alpha = 1.5

        self.cross_entropy_loss_consist = nn.NLLLoss(reduction='none')
    
    def forward(self, h_u, h_h_u, y_w_u, p_h_u, i_mask):
        L_Con = (self.cross_entropy_loss_consist(p_h_u, y_w_u) * i_mask).mean()
        L_Div = (F.pairwise_distance(h_u, h_h_u) * i_mask).mean()
        return self.alpha * L_Con + L_Div


class ConsisGADCustomMLP(nn.Module):
    def __init__(self, in_dim, out_dim, p, mid_dim, fin_act=True):
        super(ConsisGADCustomMLP, self).__init__()

        mod_list = [
            CustomLinear(in_dim, mid_dim),
            nn.ELU(),
            nn.Dropout(p),
            nn.LayerNorm(mid_dim),
            CustomLinear(mid_dim, out_dim),
        ] + (
            [
                nn.ELU(),
                nn.Dropout(p)
            ] if fin_act else []
        )
        self.mlp = nn.Sequential(*mod_list)
    
    def forward(self, x):
        return self.mlp(x)


class ConsisGADGNNModule(nn.Module):
    def __init__(self, in_feats, out_feats, edge_types: list, drop_rate: 0.8,
                 mid_dim=64):
        super(ConsisGADGNNModule, self).__init__()

        self.edge_types = edge_types
        
        self.edge_mlp = nn.ModuleDict()
        for edge_type in edge_types:
            self.edge_mlp[edge_type] = ConsisGADCustomMLP(
                in_feats * 2,  # in_feat + out_feat
                out_feats, drop_rate, mid_dim
            )
        
        self.before_ret = CustomLinear(out_feats, out_feats)
        self.before_res = CustomLinear(in_feats, out_feats) if in_feats != out_feats else nn.Identity()
        
        self.edge_bn = nn.ModuleDict()
        for edge_type in edge_types:
            self.edge_bn[edge_type] = CustomBatchNorm1d(out_feats)
    
    def edge_func_wrap(self, edge_type):
        mlp_func = self.edge_mlp[edge_type]

        def func(edges):
            x = torch.cat([edges.src['feat'], edges.dst['feat']], dim=-1)
            x = mlp_func(x)
            return {'proj': x}
        return func
    
    def forward(self, g, features, update_bn=True):
        with g.local_scope():
            src_feat = dst_feat = features
            if g.is_block:
                dst_feat = src_feat[:g.num_dst_nodes()]  # keep limit features for sub-graph
            
            g.srcdata['feat'] = src_feat
            g.dstdata['feat'] = dst_feat

            for e_t in self.edge_types:
                g.apply_edges(self.edge_func_wrap(e_t), etype=e_t)
            for c_e_t in g.canonical_etypes:
                if len(self.edge_types) == 1:
                    g.edata['proj'] = self.edge_bn[self.edge_types[0]](g.edata['proj'], update_running_stats=update_bn)
                else:
                    g.edata['proj'][c_e_t] = self.edge_bn[c_e_t[1]](g.edata['proj'][c_e_t], update_running_stats=update_bn)
            
            etype_dict = {}
            for e_t in self.edge_types:
                etype_dict[e_t] = (fn.copy_e('proj', 'proj'), fn.sum('proj', 'out'))
            g.multi_update_all(etype_dict=etype_dict, cross_reducer='stack')

            out = g.dstdata.pop('out')
            
            out = torch.sum(out, dim=1)

            return self.before_ret(out) + self.before_res(dst_feat)


class ConsisGADGNN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, edge_types, 
                 input_mlp_mid_dim, mid_mlp_mid_dim,
                 input_drop=0.0, hid_drop=0.0, mlp_drop=0.0,
                 num_layers = 1):
        super(ConsisGADGNN, self).__init__()

        self.after_input = nn.Sequential(
            nn.Dropout(input_drop),
            ConsisGADCustomMLP(in_feats, hid_feats, mlp_drop, input_mlp_mid_dim),
            CustomBatchNorm1d(hid_feats),
        )
        in_feats = hid_feats  # after proj

        self.gnns = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            self.gnns.append(
                ConsisGADGNNModule(
                    in_feats, hid_feats, edge_types, mlp_drop, mid_mlp_mid_dim
                )
            )
            self.bns.append(
                CustomBatchNorm1d(hid_feats)
            )
        
        self.layer_dropout = nn.Dropout(hid_drop)
        self.act = F.selu

        self.before_ret = ConsisGADCustomMLP(
            hid_feats * (num_layers + 1), out_feats, mlp_drop, input_mlp_mid_dim, fin_act=False
        )
    
    def forward(self, blocks, update_bn=True, return_emb=False):
        final_dim = blocks[-1].num_dst_nodes()
        features = blocks[0].srcdata['feature']
        features = self.after_input(features)

        outputs = [features[:final_dim]]
        x = features
        for block, gnn, bn in zip(blocks, self.gnns, self.bns):
            x = gnn(block, x, update_bn)
            x = bn(x, update_running_stats=update_bn)
            x = self.act(x)
            x = self.layer_dropout(x)

            outputs.append(x[:final_dim])
        
        if return_emb:
            return outputs
        else:
            features = torch.stack(outputs, dim=1)
            features = features.reshape((features.shape[0], -1))
            features = self.before_ret(features)
            features = features.log_softmax(dim=-1)
            return features
    
    def predict(self, features):
        features = self.before_ret(features)
        features = features.log_softmax(dim=-1)
        return features


def l2_regularization(model):
    l2_reg = torch.tensor(0., requires_grad=True)
    for key, value in model.named_parameters():
        if len(value.shape) > 1 and 'weight' in key:
            l2_reg = l2_reg + torch.sum(value ** 2) * 0.5
    return l2_reg


class LearnableDataArugmentation(nn.Module):
    def __init__(self, gnn, device="cuda", drop_rate=0.2, lr=0.01, eps=1e-15, temp=0.0001, weight_decay=1.0):
        super(LearnableDataArugmentation, self).__init__()

        self.xi = eps
        self.temp = temp
        self.drop_rate = drop_rate
        self.weight_decay = weight_decay

        self.pos_thr = 5
        self.neg_thr = 85

        self.gnn = gnn
        self.mask_proj = CustomLinear(64, 64).to(device)

        self.con_loss = ConsisDataAugumentationLoss().to(device)
        self.optim = torch.optim.Adam(self.mask_proj.parameters(), lr=lr, weight_decay=0.0)
    
    def sharpen(self, h, chain=True):
        h_hat = torch.zeros_like(h).to(h.device)
        for _ in range(int(self.drop_rate * h.shape[0])):
            if chain:  # do not detach grad from previous nums
                m = 1 - h_hat  # reversed mask (1, 0)
            else:  # do detach
                m = torch.ones_like(h_hat)
                m[h_hat == 1] = 0
                m = m.to(h.device)
            m_hat = torch.log(m + self.xi)  # get mask (0, -/inf)
            y = torch.softmax(  # m_hat = 0 (h_hat = 0) and h => less ==> y = 1
                (-h + m_hat) / self.temp, dim=0
            )
            h_hat += y * m
        return 1 - h_hat
    
    def forward(self, u_blocks):
        self.gnn.eval()

        with torch.no_grad():
            h_u = self.gnn(copy.deepcopy(u_blocks), update_bn=False, return_emb=True)
            h_u = torch.stack(h_u, dim=1)
            h_u = h_u.reshape((h_u.shape[0], -1))
            p_w_u = self.gnn.predict(h_u).log_softmax(dim=-1).exp()[:, 1]
        
        y_w_u = torch.ones_like(p_w_u).long()
        pos_mask = (p_w_u >= (self.pos_thr / 100)).bool()
        neg_mask = (p_w_u <= (self.neg_thr / 100)).bool()
        y_w_u[pos_mask] = 1
        y_w_u[neg_mask] = 0
        masked_index = torch.logical_or(pos_mask, neg_mask)

        self.gnn.eval()
        self.mask_proj.train()  # Linear
        for param in self.gnn.parameters():
            param.requires_grad = False
        for param in self.mask_proj.parameters():
            param.requires_grad = True

        h_u_2 = self.gnn(copy.deepcopy(u_blocks), update_bn=False, return_emb=True)
        to_stack = [h_u_2[0]]
        for index in range(1, len(h_u_2)):
            x = h_u_2[index]
            x = self.mask_proj(x)
            x = self.sharpen(x, chain=False)
            to_stack.append(x)
        
        h_h_u = torch.stack(to_stack, dim=1)
        h_h_u = h_h_u.reshape((h_h_u.shape[0], -1))
        p_h_u = self.gnn.predict(h_h_u)
        p_h_u = p_h_u.log_softmax(dim=-1)

        loss = self.con_loss(h_u, h_h_u, y_w_u, p_h_u, masked_index) + self.weight_decay * l2_regularization(self.mask_proj)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        self.mask_proj.eval()
        self.gnn.train()
        for param in self.mask_proj.parameters():
            param.requires_grad = False
        for param in self.gnn.parameters():
            param.requires_grad = True

        # for gnn training

        h_u_3 = self.gnn(copy.deepcopy(u_blocks), update_bn=False, return_emb=True)
        to_stack = [h_u_3[0]]
        for index in range(1, len(h_u_3)):
            x = h_u_3[index]
            x = self.mask_proj(x)
            x = self.sharpen(x, chain=False)
            to_stack.append(x)
        
        h_h_u_3 = torch.stack(to_stack, dim=1)
        h_h_u_3 = h_h_u_3.reshape((h_h_u_3.shape[0], -1))
        p_h_u_3 = self.gnn.predict(h_h_u_3)
        p_h_u_3 = p_h_u_3.log_softmax(dim=-1)

        return p_h_u_3, y_w_u  # preedicted pseudo label, pseudo label


class KYCConv(nn.Module):
    def __init__(self, in_feats, out_feats, dropout, activation):
        super(KYCConv, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.activation = nn.ReLU() if activation == "ReLU" else nn.Tanh()
        
        self.neigh_linear = nn.Linear(in_feats, out_feats)
        self.imp_linear = nn.Linear(in_feats, out_feats)
        self.final_linear = nn.Linear(out_feats * 2, out_feats)
    
    def forward(self, graph):
        with graph.local_scope():
            h = graph.ndata["feature"]
            graph.srcdata['h'] = h
            graph.update_all(fn.copy_u('h', 'm'), fn.mean(msg='m', out='h'))
            h = graph.dstdata['h']
            g = graph.ndata['m_feature']
            g = torch.mean(g, dim=1)
            h = self.neigh_linear(h)
            h = self.dropout(h)
            h = self.activation(h)
            g = self.imp_linear(g)
            g = self.dropout(g)
            g = self.activation(g)
            h = torch.cat([h, g], dim=1)
            h = self.final_linear(h)
            h = self.dropout(h)
            h = self.activation(h)
        return h
        

# Who is Who on Ethereum? Account Labeling Using Heterophilic Graph Convolutional Network (2024 TSMC, Lin et al.)
class KYCGCN(nn.Module):
    def __init__(
        self,
        in_feats: int,
        num_classes: int = 2,
        h_feats: int = 128,
        mlp_layers: int = 2,
        n_layers: int = 2,
        dropout: float = 0.1,
        activation: str = "LeakyReLU",
        sample_size: int = 8,
        teleport_const: float = 0.1,
        residual_eps: float = 1e-8,
        **kwargs
    ):
        super().__init__()
        
        self.sample_size = sample_size
        self.teleport_const = teleport_const
        self.residual_eps = residual_eps
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.activation = getattr(nn, activation)
        self.mlp = MLP(h_feats, h_feats, num_classes, mlp_layers, dropout, activation)
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            in_feats = h_feats if i > 0 else in_feats
            out_feats = h_feats
            self.layers.append(KYCConv(in_feats, out_feats, dropout, activation))
        
    @staticmethod
    def calc_appr_full_matrix(graph: dgl.DGLGraph):
        A = graph.adjacency_matrix().to_dense()
        X = graph.ndata["feature"]
        
        with torch.no_grad():
            X_i = X.unsqueeze(0)
            X_j = X.unsqueeze(1)
            
            dist = torch.norm(X_i - X_j, dim=-1, p=2)
            # dist += 1e-8  # 1 has been added
        
        A_hat = A / (1 + dist)
        
        return A_hat
    
    @staticmethod
    def calc_appr_sparse_matrix(graph):
        A = graph.adjacency_matrix()
        X = graph.ndata['feature']

        with torch.no_grad():
            i_idx, j_idx = A.indices()
            X_i = X[i_idx]
            X_j = X[j_idx]
            dist = torch.norm(X_i - X_j, p=2, dim=1)
            new_vals = A.val / (1 + dist)
        
        return torch.sparse_coo_tensor(A.indices(), new_vals, A.shape)

    @classmethod
    def calc_appr(cls, graph: dgl.DGLGraph):
        if True or graph.adjacency_matrix().shape[0] > 10000:  # Force sparse matrix
            return cls.calc_appr_sparse_matrix(graph).coalesce()
        else:
            return cls.calc_appr_full_matrix(graph)
        
    @staticmethod
    def csr_matrix_row_slice(tensor: torch.Tensor, start: int, end: int):
        if tensor.layout != torch.sparse_csr:
            return None
        crow_indices = tensor.crow_indices()[start:end+1]
        col_indices = tensor.col_indices()[crow_indices[0]:crow_indices[-1]]
        values = tensor.values()[crow_indices[0]:crow_indices[-1]]
        new_col_indices = torch.zeros_like(col_indices)
        col_indices_map = []                # new_index: og_index
        reverse_col_indices_map = dict()    # og_index: new_index
        for i, v in enumerate(col_indices):
            v = v.item()
            ri = reverse_col_indices_map.get(v, None)
            if ri is not None:
                new_col_indices[i] = ri
                continue
            col_indices_map.append(v)
            reverse_col_indices_map[v] = len(col_indices_map) - 1
            new_col_indices[i] = reverse_col_indices_map[v]
        return torch.sparse_csr_tensor(crow_indices - crow_indices[0], new_col_indices, values, size=(end-start, len(col_indices_map))), torch.tensor(col_indices_map)
    
    @classmethod
    def sparse_topk(cls, tensor: torch.Tensor, k: int, max_mem: float = 32, sparse_adj: float = 0.1):
        if False and tensor.shape[0] < 10_000:  # As the sparse process is quick enough, it is no need to use dense trick.
            tensor = tensor.to_dense()
        if tensor.layout != torch.sparse_csr:
            tensor = tensor.to_sparse_csr()

        max_t, min_t = torch.max(tensor.values()), torch.min(tensor.values())
        v = (tensor.values() - min_t) / (max_t - min_t) * 255 # 2^8-1
        tensor = torch.sparse_csr_tensor(tensor.crow_indices(), tensor.col_indices(), v, dtype=torch.int8)
        
        sparse_adj = max(0, min(sparse_adj, 1))
        sparse_factor = (len(tensor.values()) / (tensor.shape[0] * tensor.shape[1]))
        max_row = int(max(1, (max_mem * (1024 ** 3) / tensor.dtype.itemsize) // tensor.shape[1]) * (1 / (sparse_factor ** sparse_adj)))
        vals, idxs = [], []
        for start_i in range(0, tensor.shape[0], max_row):
            end_i = min(start_i + max_row, tensor.shape[0])
            # print(start_i, end_i)
            batch_m, col_i_map = cls.csr_matrix_row_slice(tensor, start_i, end_i)
            batch_m = batch_m.to_dense()
            v, i = torch.topk(batch_m, k)
            vals.append(v)
            idxs.append(col_i_map[i])
        return torch.cat(vals), torch.cat(idxs)

    @lru_cache
    def get_sim_neighbor_cache4graph(self, graph: dgl.DGLGraph):
        if graph.device != "cpu":
            graph = graph.cpu()
        A_appr = self.calc_appr(graph)
        A = graph.adjacency_matrix()
        A = torch.sparse_coo_tensor(A.indices(), A.val, A.shape).coalesce()
        num_v = graph.number_of_nodes()
        P = torch.zeros_like(A.values(), device=graph.device, dtype=torch.float32)
        R = torch.ones_like(A.values(), device=graph.device, dtype=torch.float32)
        
        while True:
            appr_thr = (self.residual_eps * torch.sum(A_appr, dim=1)).to_dense()
            update_idx = (R >= appr_thr[A.indices()[0].to(torch.int64)])  # select \any s \in V
            if not update_idx.any():
                break
            P[update_idx] += self.teleport_const * R[update_idx]
            to_update_neigh_col = set(A.indices()[1][update_idx].unique().tolist())
            # update_neigh_idx = (A.indices()[1][..., None] == to_update_neigh_col).any(dim=1)  # WARN may oom, create a A.shape[0] * len(to_update_neigh_col) matrix
            update_neigh_idx = torch.tensor([c.item() in to_update_neigh_col for c in A.indices()[1]], dtype=torch.bool)  # Naive but safe
            appr_sum = torch.sum(A_appr.values()[update_neigh_idx])
            R[update_neigh_idx] = (1 - self.teleport_const) * R[A.indices()[0][update_neigh_idx]] * (A_appr.values()[update_neigh_idx] / appr_sum)
            R[update_idx] = 0
        
        _, idx = self.sparse_topk(torch.sparse_coo_tensor(A.indices(), P, A.shape), k=self.sample_size)  # the most important neighbors for each node
        return idx
        
    def forward(self, graph: dgl.DGLGraph):
        idx = self.get_sim_neighbor_cache4graph(graph)
        
        with graph.local_scope():
            for i, layer in enumerate(self.layers):
                imp_multi_hop_neighbors = graph.ndata['feature'][idx]
                graph.ndata['m_feature'] = imp_multi_hop_neighbors
                if i > 0:
                    graph.ndata['feature'] = self.dropout(graph.ndata['feature'])
                graph.ndata['feature'] = layer(graph)
            h = self.mlp(graph.ndata['feature'], False)
        
        return h


# Global Attribute-Association Pattern Aggregation for Graph Fraud Detection (AAAI 25' Duan et al.)
class GAAP(nn.Module):
    ...

