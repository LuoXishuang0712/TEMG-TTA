import torch
from torch import nn
import torch.nn.functional as F
import random
import numpy as np
import scipy.sparse as sp
import dgl
from dgl import nn as dglnn

from models.feature_fusion import wrap_class_for_feature_fusion, FeatureSimpleFusion


class GIN_noparam(nn.Module):
    def __init__(self, num_layers=2, agg='mean', init_eps=-1, **kwargs):
        super().__init__()
        self.gnn = dglnn.GINConv(None, activation=None, init_eps=init_eps,
                                 aggregator_type=agg)
        self.num_layers = num_layers

    def forward(self, graph):
        h = graph.ndata['feature']
        h_final = [h.detach().clone()]
        for i in range(self.num_layers):
            h = self.gnn(graph, h)
            h_final.append(h)
        return h_final

class ARC(nn.Module):
    def __init__(self, in_feats, h_feats=1024, num_layers=4, drop_rate=0, activation='ELU', num_hops=2, **kwargs):
        super(ARC, self).__init__()
        self.layers = nn.ModuleList()
        self.act = getattr(nn, activation)()
        self.num_hops = num_hops
        if num_layers == 0:
            return
        self.layers.append(nn.Linear(in_feats, h_feats))
        for i in range(1, num_layers - 1):
            self.layers.append(nn.Linear(h_feats, h_feats))
        self.dropout = nn.Dropout(drop_rate) if drop_rate > 0 else nn.Identity()
        self.cross_attn = CrossAttn(h_feats * (num_hops - 1))

        self.sampler = GIN_noparam(num_layers=num_hops)

    def forward(self, graph):
        # raise NotImplementedError("ARC forward not implemented. ::Detail : ARC use custom Dataset class with x_list attribute.")
        # x_list = h.x_list
        x_list = self.sampler(graph)
        # Z^{[l]} = MLP(X^{[l]}
        for i, layer in enumerate(self.layers):
            if i != 0:
                x_list = [self.dropout(x) for x in x_list]
            x_list = [layer(x) for x in x_list]
            if i != len(self.layers) - 1:
                x_list = [self.act(x) for x in x_list]
        residual_list = []
        # Z^{[0]}
        first_element = x_list[0]
        for h_i in x_list[1:]:
            # R^{[l]} = Z^{[l]}-Z^{[0]}
            dif = h_i - first_element
            residual_list.append(dif)
        # H = [R^{[1]} || ... || R^{[L]}]
        residual_embed = torch.hstack(residual_list)
        return residual_embed


class CrossAttn(nn.Module):
    def __init__(self, embedding_dim):
        super(CrossAttn, self).__init__()
        self.embedding_dim = embedding_dim

        self.Wq = nn.Linear(embedding_dim, embedding_dim)
        self.Wk = nn.Linear(embedding_dim, embedding_dim)

    def cross_attention(self, query_X, support_X):
        Q = self.Wq(query_X)  # query
        K = self.Wk(support_X)  # key
        attention_scores = torch.matmul(Q, K.transpose(0, 1)) / torch.sqrt(
            torch.tensor(self.embedding_dim, dtype=torch.float32))
        attention_weights = F.softmax(attention_scores, dim=1)
        weighted_query_embeddings = torch.matmul(attention_weights, support_X)
        return weighted_query_embeddings

    def get_train_loss(self, X, y, num_prompt):
        positive_indices = torch.nonzero((y == 1)).squeeze(1).tolist()
        all_negative_indices = torch.nonzero((y == 0)).squeeze(1).tolist()

        negative_indices = random.sample(all_negative_indices, len(positive_indices))
        # H_q_i, y_i == 1
        query_p_embed = X[positive_indices]
        # H_q_i, y_i == 0
        query_n_embed = X[negative_indices]
        # H_q
        query_embed = torch.vstack([query_p_embed, query_n_embed])

        remaining_negative_indices = list(set(all_negative_indices) - set(negative_indices))

        if len(remaining_negative_indices) < num_prompt:
            raise ValueError(f"Not enough remaining negative indices to select {num_prompt} support nodes.")

        support_indices = random.sample(remaining_negative_indices, num_prompt)
        support_indices = torch.tensor(support_indices).to(y.device)
        # H_k
        support_embed = X[support_indices]

        # the updated query node embeddings
        # \tilde{H_q}
        query_tilde_embeds = self.cross_attention(query_embed, support_embed)
        # tilde_p_embeds: \tilde{H_q_i}, y_i == 1; tilde_n_embeds: \tilde{H_q_i}, y_i == 0;
        tilde_p_embeds, tilde_n_embeds = query_tilde_embeds[:len(positive_indices)], query_tilde_embeds[
                                                                                              len(positive_indices):]

        yp = torch.ones([len(negative_indices)]).to(y.device)
        yn = -torch.ones([len(positive_indices)]).to(y.device)
        # cos_embed_loss(H_q_i, \tilde{H_q_i}, 1), if y_i == 0
        loss_qn = F.cosine_embedding_loss(query_n_embed, tilde_n_embeds, yp)
        # cos_embed_loss(H_q_i, \tilde{H_q_i}, -1), if y_i == 1
        loss_qp = F.cosine_embedding_loss(query_p_embed, tilde_p_embeds, yn)
        loss = torch.mean(loss_qp + loss_qn)
        return loss

    def get_test_score(self, X, prompt_mask, y):
        # prompt node indices
        negative_indices = torch.nonzero((prompt_mask == True) & (y == 0)).squeeze(1).tolist()
        n_support_embed = X[negative_indices]
        # query node indices
        query_indices = torch.nonzero(prompt_mask == False).squeeze(1).tolist()
        # H_q
        query_embed = X[query_indices]
        # \tilde{H_q}
        query_tilde_embed = self.cross_attention(query_embed, n_support_embed)
        # dis(H_q, \tilde{H_q})
        diff = query_embed - query_tilde_embed
        # score
        query_score = torch.sqrt(torch.sum(diff ** 2, dim=1))

        return query_score


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

class ARCWrapped(ARC):
    def __init__(self, in_feats, h_feats=1024, num_classes=2, num_layers=4, drop_rate=0, activation='ELU', num_hops=2, **kwargs):
        super().__init__(in_feats, h_feats, num_layers, drop_rate, activation, num_hops, **kwargs)
        self.classifier = nn.Linear(h_feats * num_hops, num_classes)
        self.feature_fusion = FeatureSimpleFusion(i_dim=in_feats, h_dim=in_feats, o_dim=in_feats)
    
    def forward(self, graph):
        h, emb = self.get_emb(graph)
        return h
    
    def get_emb(self, graph):
        if 'feature' in graph.ndata and 'motif' in graph.ndata:
            if 'og_feature' not in graph.ndata:
                print(f"load motif for {self.__class__.__name__} with {graph}")
                graph.ndata['og_feature'] = graph.ndata['feature'].detach()
            feat = graph.ndata['og_feature']
            motif = graph.ndata['motif']
            feat = self.feature_fusion(feat, motif)
            graph.ndata['feature'] = feat
        emb = super().forward(graph)
        return self.classifier(emb), emb
