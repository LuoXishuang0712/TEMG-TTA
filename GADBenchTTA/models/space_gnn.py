import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import MessagePassing, global_add_pool, global_max_pool, global_mean_pool
from torch_geometric.utils import softmax
from torch_scatter import scatter
import random
import dgl

from models.feature_fusion import wrap_class_for_feature_fusion, wrap_class_instance_for_feature_fusion, FeatureSimpleFusion

def expmap(e, c):
    sqrt_c = abs(c) ** 0.5
    e_norm = torch.clamp_min(e.norm(dim=-1, p=2, keepdim=True), 1e-15)
    if float(c) < 0:
        hemb = torch.tanh(sqrt_c * e_norm) * e / (sqrt_c * e_norm)
    else:
        hemb = torch.tan(sqrt_c * e_norm) * e / (sqrt_c * e_norm)
    return hemb

def logmap(h, c):
    sqrt_c = abs(c) ** 0.5
    h_norm = h.norm(dim=-1, p=2, keepdim=True).clamp_min(1e-15)
    if float(c) < 0:
        scale = 1. / sqrt_c * torch.arctanh(sqrt_c * h_norm) / h_norm
    else:
        scale = 1. / sqrt_c * torch.atan(sqrt_c * h_norm) / h_norm
    return scale * h

def proj(x, c):
    norm = torch.clamp_min(x.norm(dim=-1, keepdim=True, p=2), 1e-15)
    maxnorm = (1 - 1e-5) / (abs(c) ** 0.5)
    cond = norm > maxnorm
    projected = x / norm * maxnorm
    return torch.where(cond, projected, x)

class CustomLinear(nn.Linear):
    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight)
        nn.init.zeros_(self.bias)

class LinearLayer(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, drop_rate, final=True):
        super(LinearLayer, self).__init__()
        self.linear1 = CustomLinear(in_dim, hid_dim)
        self.linear2 = CustomLinear(hid_dim, out_dim)
        self.act = nn.ELU()
        self.drop = nn.Dropout(drop_rate)
        self.norm = nn.LayerNorm(hid_dim)
        self.final = final

    def forward(self, h):
        h = self.linear1(h)
        h = self.act(h)
        h = self.drop(h)
        h = self.norm(h)
        h = self.linear2(h)
        if self.final:
            h = self.act(h)
            h = self.drop(h)
        return h

class CurvLayer(MessagePassing):
    def __init__(self, in_dim, hid_dim, out_dim, drop_rate):
        super(CurvLayer, self).__init__(aggr='add')  # Use "add" aggregation
        self.edge_linear = LinearLayer(in_dim * 2, hid_dim, hid_dim, drop_rate)
        self.out_linear = CustomLinear(hid_dim, out_dim)
        self.edge_bn = nn.BatchNorm1d(hid_dim)
    
    def forward(self, x, edge_index, c):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        
        if not (float(c) == 0):
            x = expmap(x, c)
            x = proj(x, c)
            x = logmap(x, c)

        # Start propagating messages
        out = self.propagate(edge_index, x=x, c=c)
        
        # Apply batch norm and linear layer
        out = self.out_linear(out)
        
        if not (float(c) == 0):
            out = expmap(out, c)
            out = proj(out, c)
            out = logmap(out, c)
            out = F.selu(out)
        
        out += x[:out.size(0)]  # Residual connection
        
        return out

    def message(self, x_i, x_j, c):
        # x_i has shape [E, in_channels] (source nodes)
        # x_j has shape [E, in_channels] (target nodes)
        
        if float(c) == 0:
            coef = 1 - torch.sigmoid((x_i - x_j).norm(dim=-1, p=2, keepdim=True))
        else:
            dist = (x_i - x_j).norm(dim=-1, p=2, keepdim=True)
            multi = (x_i * x_j).sum(dim=-1, keepdim=True)
            coef = 1 - torch.sigmoid(2 * dist - 2 * c * ((dist ** 3) / 3 + multi * (dist ** 2)))
        
        msg = torch.cat([coef * x_i + x_i, x_j], dim=-1)
        msg = self.edge_linear(msg)
        return msg

    def update(self, aggr_out: torch.Tensor):
        # Apply batch normalization to aggregated messages
        if aggr_out.shape[0] == 1:
            return aggr_out
        aggr_out = self.edge_bn(aggr_out)
        return aggr_out

class CurvGNN(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, layer_num, drop_rate):
        super(CurvGNN, self).__init__()
        self.in_linears = nn.ModuleList()
        self.gnns = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.out_linears = nn.ModuleList()
        self.layer_num = layer_num
        
        for i in range(layer_num):
            self.in_linears.append(LinearLayer(in_dim, hid_dim, hid_dim, drop_rate))
            self.gnns.append(CurvLayer(hid_dim, hid_dim, hid_dim, drop_rate))
            self.bns.append(nn.BatchNorm1d(hid_dim))
            self.out_linears.append(CustomLinear(hid_dim * 2, hid_dim))
            
        self.out_linear = CustomLinear(hid_dim * layer_num, out_dim)

        self.act = F.selu
        
    def forward(self, x, edge_index, c):
        # x: node features [num_nodes, in_dim]
        # edge_index: graph connectivity [2, num_edges]
        results = []
        
        for i in range(self.layer_num):
            inter_results = []
            h = self.in_linears[i](x)  # Using x directly
            inter_results.append(h)
            h = self.gnns[i](h, edge_index, c[i])
            if h.shape[0] > 1:
                h = self.bns[i](h)
            h = self.act(h)
            inter_results.append(h)
            h = torch.stack(inter_results, dim=1)
            h = h.reshape(h.shape[0], -1)
            h = self.out_linears[i](h)
            results.append(h)
        
        h = torch.stack(results, dim=1)
        h = h.reshape(h.shape[0], -1)
        h = self.out_linear(h)

        return h.log_softmax(dim=-1)

    def get_emb(self, x, edge_index, c):
        # x: node features [num_nodes, in_dim]
        # edge_index: graph connectivity [2, num_edges]
        results = []
        
        for i in range(self.layer_num):
            inter_results = []
            h = self.in_linears[i](x)  # Using x directly
            inter_results.append(h)
            h = self.gnns[i](h, edge_index, c[i])
            h = self.bns[i](h)
            h = self.act(h)
            inter_results.append(h)
            h = torch.stack(inter_results, dim=1)
            h = h.reshape(h.shape[0], -1)
            h = self.out_linears[i](h)
            results.append(h)
        
        h = torch.stack(results, dim=1)
        h = h.reshape(h.shape[0], -1)
        emb = h
        h = self.out_linear(h)

        return h.log_softmax(dim=-1), emb

class SpaceGNN(nn.Module):
    def __init__(self, input_dim, *args, hid_dim=None, out_dim=None, layer_num=3, drop_rate=0, stdneg=None, stdpos=None, alpha=0.5, beta=0.5, pool='mean', **kwargs):
        super(SpaceGNN, self).__init__()
        if stdneg is None:
            stdneg = random.uniform(0.01,0.02)
        if stdpos is None:
            stdpos = random.uniform(0.01,0.02)
        if hid_dim is None:
            hid_dim = int(0.618 * input_dim)  # "golden cut"
        if out_dim is None:
            out_dim = hid_dim
        self.alpha = alpha
        self.beta = beta
        self.curvgnnneg = CurvGNN(input_dim, hid_dim, out_dim, layer_num, drop_rate)
        self.curvgnnpos = CurvGNN(input_dim, hid_dim, out_dim, layer_num, drop_rate)
        self.eucgnn = CurvGNN(input_dim, hid_dim, out_dim, layer_num, drop_rate)
        cneg = torch.FloatTensor(layer_num).normal_(-0.1, stdneg)
        cpos = torch.FloatTensor(layer_num).normal_(0.1, stdpos)
        self.cneg = nn.Parameter(cneg)
        self.cpos = nn.Parameter(cpos)
        self.zeros = [0] * layer_num
        if pool == "sum":
            self.pool = global_add_pool
        elif pool == "mean":
            self.pool = global_mean_pool
        elif pool == "max":
            self.pool = global_max_pool
        # elif pool == "attention":
        #     self.pool = GlobalAttention(gate_nn=torch.nn.Linear(emb_dim, 1))
        else:
            raise ValueError("Invalid graph pooling type.")

    def forward(self, x, edge_index, batch = None, prompt = None, prompt_type = None, *args, **kwargs):
        h1 = self.curvgnnneg(x, edge_index, self.cneg)
        h2 = self.curvgnnpos(x, edge_index, self.cpos)
        h3 = self.eucgnn(x, edge_index, self.zeros)
        
        node_emb = (1 - self.beta) * ((1 - self.alpha) * h1 + self.alpha * h2) + self.beta * h3
        
        if batch == None:
            return node_emb
        else:
            if prompt_type == 'Gprompt':
                node_emb = prompt(node_emb)
            graph_emb = self.pool(node_emb, batch.long())
            return graph_emb
    
    def get_emb(self, x, edge_index, batch = None, prompt = None, prompt_type = None, *args, **kwargs):
        h1, h1_emb = self.curvgnnneg.get_emb(x, edge_index, self.cneg)
        h2, h2_emb = self.curvgnnpos.get_emb(x, edge_index, self.cpos)
        h3, h3_emb = self.eucgnn.get_emb(x, edge_index, self.zeros)
        
        mid_emb = (1 - self.beta) * ((1 - self.alpha) * h1_emb + self.alpha * h2_emb) + self.beta * h3_emb
        node_emb = (1 - self.beta) * ((1 - self.alpha) * h1 + self.alpha * h2) + self.beta * h3
        
        if batch == None:
            return node_emb, mid_emb
        else:
            if prompt_type == 'Gprompt':
                node_emb = prompt(node_emb)
            graph_emb = self.pool(node_emb, batch.long())
            return graph_emb, mid_emb
    
    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)
    
    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()

class SpaceGNNWrapped(SpaceGNN):
    def __init__(self, in_feats, num_classes=2, h_feats=32, num_layers=2, drop_rate=0.0, stdneg=None, stdpos=None, alpha=0.5, beta=0.5, pool='mean', **kwargs):
        super().__init__(in_feats, hid_dim=h_feats, out_dim=num_classes, layer_num=num_layers, drop_rate=drop_rate, stdneg=stdneg, stdpos=stdpos, alpha=alpha, beta=beta, pool=pool)
        self.feature_fusion = FeatureSimpleFusion(i_dim=in_feats, h_dim=in_feats, o_dim=in_feats)
    
    def get_new_graph(self, graph: dgl.DGLGraph):
        if 'feature' in graph.ndata and 'motif' in graph.ndata:
            if 'og_feature' not in graph.ndata:
                print(f"load motif for {self.__class__.__name__} with {graph}")
                graph.ndata['og_feature'] = graph.ndata['feature'].detach()
            feat = graph.ndata['og_feature']
            motif = graph.ndata['motif']
            feat = self.feature_fusion(feat, motif)
            graph.ndata['feature'] = feat
        return graph
    
    def forward(self, graph: dgl.DGLGraph):
        graph = self.get_new_graph(graph)
        x = graph.ndata['feature']
        edge_index = torch.stack(graph.edges())
        batch = graph.batch_num_nodes()
        return super().forward(x, edge_index)

    def get_emb(self, graph: dgl.DGLGraph):
        graph = self.get_new_graph(graph)
        x = graph.ndata['feature']
        edge_index = torch.stack(graph.edges())
        batch = graph.batch_num_nodes()
        return super().get_emb(x, edge_index)
