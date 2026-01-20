import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import numpy as np

DIR_MAP = {
    ((0, 2), (1, 2), (0, 2)): (0, 0),
    ((0, 2), (1, 2), (2, 0)): (0, 1),
    ((0, 2), (1, 2), (0, 1)): (0, 2),
    ((0, 2), (1, 2), (1, 0)): (0, 3),
    ((0, 2), (1, 2), (2, 1)): (0, 4),
    ((0, 2), (1, 2), (1, 2)): (0, 5),
    ((0, 2), (2, 1), (0, 2)): (1, 0),
    ((0, 2), (2, 1), (2, 0)): (1, 1),
    ((0, 2), (2, 1), (0, 1)): (1, 2),
    ((0, 2), (2, 1), (1, 0)): (1, 3),
    ((0, 2), (2, 1), (2, 1)): (1, 4),
    ((0, 2), (2, 1), (1, 2)): (1, 5),
    ((0, 2), (1, 0), (0, 2)): (2, 0),
    ((0, 2), (1, 0), (2, 0)): (2, 1),
    ((0, 2), (1, 0), (0, 1)): (2, 2),
    ((0, 2), (1, 0), (1, 0)): (2, 3),
    ((0, 2), (1, 0), (2, 1)): (2, 4),
    ((0, 2), (1, 0), (1, 2)): (2, 5),
    ((0, 2), (0, 1), (0, 2)): (3, 0),
    ((0, 2), (0, 1), (2, 0)): (3, 1),
    ((0, 2), (0, 1), (0, 1)): (3, 2),
    ((0, 2), (0, 1), (1, 0)): (3, 3),
    ((0, 2), (0, 1), (2, 1)): (3, 4),
    ((0, 2), (0, 1), (1, 2)): (3, 5),
    ((0, 2), (2, 0), (0, 2)): (4, 0),
    ((0, 2), (2, 0), (2, 0)): (4, 1),
    ((0, 2), (2, 0), (0, 1)): (4, 2),
    ((0, 2), (2, 0), (1, 0)): (4, 3),
    ((0, 2), (2, 0), (2, 1)): (4, 4),
    ((0, 2), (2, 0), (1, 2)): (4, 5),
    ((0, 2), (0, 2), (0, 2)): (5, 0),
    ((0, 2), (0, 2), (2, 0)): (5, 1),
    ((0, 2), (0, 2), (0, 1)): (5, 2),
    ((0, 2), (0, 2), (1, 0)): (5, 3),
    ((0, 2), (0, 2), (2, 1)): (5, 4),
    ((0, 2), (0, 2), (1, 2)): (5, 5),
}

class MotifsExpression(nn.Module):
    def __init__(self, n_dim: int = 16):
        super().__init__()
        self.time_seq_embed = nn.Embedding(3, n_dim, _freeze=True)  # hardcoded
        n_roles = 2  # hardcoded
        self.role_embed_base = nn.Embedding(n_roles, n_roles, _weight=torch.stack([
            torch.tensor([0 if j != i else 1 for j in range(n_roles)], dtype=torch.float32)
            for i in range(n_roles)
        ], dim=0), _freeze=True)
        self.role_embed_encoder = nn.Sequential(
            nn.Linear(n_roles, n_dim),
            nn.ReLU(inplace=True),
        )
        self.learnable_role_embed = nn.Parameter(torch.randn(3 * 36, n_dim), requires_grad=True)  # r  # hardcoded
        self.learnable_motifs_embed = nn.Parameter(torch.randn(36, n_dim), requires_grad=True)  # m  # hardcoded
        self.n_dim = n_dim
    
    def __get_current_prototype_pos_emb(self):
        # t_1 + from, t1 + to, t_2 + from, t_2 + to, ...
        role_embed = self.role_embed_encoder(self.role_embed_base.weight)
        return tuple([
            self.time_seq_embed.weight[i] + role_embed[j]
            for i in range(3) for j in range(2)  # hardcoded
        ])

    def __get_motifs_pos_emb(self):
        proto_pos_emb = self.__get_current_prototype_pos_emb()
        motifs_pos_emb = [0] * (len(DIR_MAP) * 3)
        for node_dir, motifs_id in DIR_MAP.items():
            motifs_id = motifs_id[0] * 6 + motifs_id[1]
            for idx_time, time_edge in enumerate(node_dir):
                motifs_pos_emb[motifs_id * 3 + time_edge[0]] += proto_pos_emb[idx_time * 2]
                motifs_pos_emb[motifs_id * 3 + time_edge[1]] += proto_pos_emb[idx_time * 2 + 1]
        motifs_pos_emb = torch.stack([i if isinstance(i, torch.Tensor) else torch.zeros(self.n_dim, dtype=torch.float32).to(proto_pos_emb[0].device) for i in motifs_pos_emb], dim=0)  # p
        return motifs_pos_emb
        
    def forward(self, motifs_count: torch.Tensor | np.ndarray):
        if isinstance(motifs_count, np.ndarray):
            motifs_count = torch.from_numpy(motifs_count)
        if motifs_count.dtype != torch.float32:
            motifs_count = motifs_count.to(torch.float32)
        motifs_count = motifs_count.to(self.learnable_role_embed.device)
        # N x (36 * 3)
        motifs_pos_emb = self.__get_motifs_pos_emb()  # p
        motif_embed = torch.stack([self.learnable_motifs_embed] * 3, dim=0).transpose(0, 1).reshape(-1, self.learnable_motifs_embed.shape[1])  # m
        expression_embed = motifs_pos_emb + self.learnable_role_embed + motif_embed
        final_embed = motifs_count @ expression_embed
        return final_embed


class MotifsSimpleExpression(nn.Module):
    def __init__(self, idim: int = 16):
        super().__init__()
        self.linear = nn.Linear(36 * 3, idim)

    def forward(self, motifs_count: torch.Tensor):
        if motifs_count.dtype != torch.float32:
            motifs_count = motifs_count.to(torch.float32)
        motifs_count = motifs_count.to(self.linear.weight.device)
        return self.linear(motifs_count)


class MultiLayerAggregration(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self, g: dgl.DGLGraph, motifs_embed: torch.Tensor):  # TODO TopK neighbor aggregation by att
        with g.local_scope():
            g.ndata['motifs_embed'] = motifs_embed
            g.update_all(
                dgl.function.copy_u('motifs_embed', 'm'),
                dgl.function.mean('m', 'motifs_embed')
            )
            return g.ndata['motifs_embed']


class FeatureSimpleFusion(nn.Module):
    def __init__(self, *args, i_dim: int = 16, h_dim: int = 16, o_dim: int = 16, n_layers: int = 2, **kwargs):
        super().__init__(*args, **kwargs)
        self.before_motif = nn.Linear(i_dim, h_dim)
        self.before_node = nn.Linear(i_dim, h_dim)
        fusion_layers = []
        if n_layers < 1:
            raise ValueError('n_layers must be greater than 0')
        elif n_layers == 1:
            fusion_layers.append(nn.Linear(i_dim, o_dim))
        else:
            for _ in range(n_layers - 1):
                fusion_layers.append(nn.Linear(h_dim, h_dim))
            fusion_layers.append(nn.Linear(h_dim, o_dim))
        self.fusion_layer = nn.ModuleList(fusion_layers)
        self.motifs_expression = MotifsExpression(i_dim)
    
    def forward(self, node_embed: torch.Tensor, motifs_count: torch.Tensor):
        motifs_embed = self.motifs_expression(motifs_count)
        x = self.before_motif(motifs_embed) + self.before_node(node_embed)
        for layer in self.fusion_layer:
            x = layer(x)
            x = F.relu(x)
        return x
    
    def get_motifs_embedding(self, motifs_count: torch.Tensor):
        return self.motifs_expression(motifs_count)

def get_model_device(model: nn.Module):
    return next(model.parameters()).device

def wrap_class_instance_for_feature_fusion(cls_):  # TODO do too much trick here
    raise NotImplementedError("NEVER USE THIS FUNCTION")
    if not isinstance(cls_, nn.Module):
        return
    og_forward = cls_.forward
    cls_.feature_fusion = FeatureSimpleFusion()
    def new_forward(*args, **kwargs):
        possible_graph = None
        for arg in [*args, *kwargs.values()]:
            if isinstance(arg, dgl.DGLGraph):
                possible_graph = arg
                break
        if possible_graph is None:
            raise ValueError('No graph found in args')
        if get_model_device(cls_) != possible_graph.ndata['feature'].device:
            cls_.to(possible_graph.ndata['feature'].device)
        if 'feature' in possible_graph.ndata and 'motif' in possible_graph.ndata:
            if 'og_feature' not in possible_graph.ndata:
                print(f"load motifs for network {cls_.__class__.__name__} with graph {possible_graph}")
                possible_graph.ndata['og_feature'] = possible_graph.ndata['feature'].detach()
            feat = possible_graph.ndata['og_feature']
            motif = possible_graph.ndata['motif']
            feat = cls_.feature_fusion(feat, motif)
            possible_graph.ndata['feature'] = feat
        return og_forward(*args, **kwargs)
    cls_.forward = new_forward
    return cls_

def wrap_class_for_feature_fusion(cls_):
    og_init = cls_.__init__
    def inject_init(self, *args, **kwargs):
        og_init(self, *args, **kwargs)
        wrap_class_instance_for_feature_fusion(self)
    cls_.__init__ = inject_init
    return cls_
