import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import numpy as np
import torch_geometric.utils
import torch_sparse
import torch_geometric
from typing import (Literal, List)

from tta.base_tta import TTABaseClass
from tta.utils import (hack_model_for_embedding, make_dgl_graph)

def linear_to_triu_idx(n: int, lin_idx: torch.Tensor) -> torch.Tensor:
    row_idx = (
        n
        - 2
        - torch.floor(torch.sqrt(-8 * lin_idx.double() + 4 * n * (n - 1) - 7) / 2.0 - 0.5)
    ).long()
    col_idx = (
        lin_idx
        + row_idx
        + 1 - n * (n - 1) // 2
        + (n - row_idx) * ((n - row_idx) - 1) // 2
    )
    return torch.stack((row_idx, col_idx))

def to_symmetric(edge_index, edge_weight, n, op='mean'):
    symmetric_edge_index = torch.cat(
        (edge_index, edge_index.flip(0)), dim=-1
    )

    symmetric_edge_weight = edge_weight.repeat(2)

    symmetric_edge_index, symmetric_edge_weight = torch_sparse.coalesce(
        symmetric_edge_index,
        symmetric_edge_weight,
        m=n,
        n=n,
        op=op
    )
    return symmetric_edge_index, symmetric_edge_weight

def inner(t1, t2):
    t1 = t1 / (t1.norm(dim=1).view(-1,1) + 1e-15)
    t2 = t2 / (t2.norm(dim=1).view(-1,1) + 1e-15)
    return (1-(t1 * t2).sum(1)).mean()

def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from **logits**."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

def grad_with_checkpoint(outputs, inputs):
    inputs = (inputs,) if isinstance(inputs, torch.Tensor) else tuple(inputs)
    for input in inputs:
        if not input.is_leaf:
            input.retain_grad()
    torch.autograd.backward(outputs)

    grad_outputs = []
    for input in inputs:
        grad_outputs.append(input.grad.clone())
        input.grad.zero_()
    return grad_outputs

def bisection(edge_weights, a, b, n_perturbations, epsilon=1e-5, iter_max=1e5):
    def func(x):
        return torch.clamp(edge_weights - x, 0, 1).sum() - n_perturbations

    miu = a
    for i in range(int(iter_max)):
        miu = (a + b) / 2
        # Check if middle point is root
        if (func(miu) == 0.0):
            break
        # Decide the side to repeat the steps
        if (func(miu) * func(a) < 0):
            b = miu
        else:
            a = miu
        if ((b - a) <= epsilon):
            break
    return miu

def project(n_perturbations, values, eps, inplace=False):
    if not inplace:
        values = values.clone()

    if torch.clamp(values, 0, 1).sum() > n_perturbations:
        left = (values - 1).min()
        right = values.max()
        miu = bisection(values, left, right, n_perturbations)
        values.data.copy_(torch.clamp(
            values - miu, min=eps, max=1 - eps
        ))
    else:
        values.data.copy_(torch.clamp(
            values, min=eps, max=1 - eps
        ))
    return values

class GTrans(TTABaseClass):
    _copy_required = False
    def __init__(
            self, 
            *args,
            lr_feat: float = 1e-4,
            ratio: float = 0.1,
            lr_adj: float = 0.1,
            loop_feat: int = 4,
            loop_adj: int = 0,
            eps: float = 1e-7,
            cuda_synchronize: bool = True,
            loss_type: List[str] | None = None,
            aug_strategy: Literal['shuffle', 'dropedge', 'dropnode', 'rwsample', 'dropmix', 'dropfeat', 'featnoise'] = 'dropedge',
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.lr_feat = lr_feat
        self.ratio = ratio
        self.lr_adj = lr_adj
        self.loop_feat = loop_feat
        self.loop_adj = loop_adj
        self.eps = eps
        self.cuda_synchronize = cuda_synchronize
        self.loss_type = loss_type or ['LC']
        self.aug_strategy = aug_strategy

        # results
        self.final_graph = None

    @staticmethod
    def __sample_random_block(n_pertubations: int, edge_index: torch.Tensor, nnodes: int, eps: float = 1e-7):
        edge_index = edge_index.clone()
        edge_index = edge_index[:, edge_index[0] < edge_index[1]]
        row, col = edge_index
        edge_index_id = ((2 * nnodes - row - 1) * row // 2 + col - row - 1).long()
        modified_edge_index = linear_to_triu_idx(nnodes, edge_index_id)
        perturbed_edge_weight = torch.full_like(
            edge_index_id, eps, dtype=torch.float32, requires_grad=True
        )
        return edge_index_id, modified_edge_index, perturbed_edge_weight
    
    @staticmethod
    def __get_modified_adj(modified_edge_index, perturbed_edge_weight, nnodes, edge_index, edge_weight):
        modified_edge_index, modified_edge_weight = to_symmetric(modified_edge_index, perturbed_edge_weight, nnodes)
        edge_index = torch.cat((edge_index, modified_edge_index), dim=-1)
        edge_weight = torch.cat((edge_weight, modified_edge_weight), dim=-1)

        edge_index, edge_weight = torch_sparse.coalesce(edge_index, edge_weight, nnodes, nnodes, op='sum')
        edge_weight[edge_weight > 1] = 2 - edge_weight[edge_weight > 1]
        return edge_index, edge_weight
    
    @staticmethod
    def __augment(
        strategy: Literal['shuffle', 'dropedge', 'dropnode', 'rwsample', 'dropmix', 'dropfeat', 'featnoise'],
        model: nn.Module, feat: torch.Tensor, delta_feat: torch.Tensor, edge_index: torch.Tensor | None, edge_weight: torch.Tensor | None,
        p: float = 0.5, nnodes: int | None = None
    ) -> torch.Tensor:
        if strategy == 'shuffle':
            new_idx = np.random.permutation(feat.shape[0])
            feat = (feat + delta_feat)[new_idx, :]
            _, output = hack_model_for_embedding(model, args=(make_dgl_graph(feat, edge_index, num_nodes=nnodes), ))
        elif strategy == 'dropedge':
            edge_index, edge_weight = torch_geometric.utils.dropout_adj(edge_index, edge_weight, p=p)
            _, output = hack_model_for_embedding(model, args=(make_dgl_graph(feat, edge_index, edge_weight, self_loop=False, num_nodes=nnodes), ), kwargs={'edge_weight': edge_weight} if edge_weight else {})
        elif strategy == 'dropnode':
            feat += delta_feat
            mask = torch.bernoulli(torch.full_like(feat, 1 - p)).bool()  # og torch.cuda.FloatTensor().uniform_()
            feat = feat * mask.unsqueeze(-1)
            _, output = hack_model_for_embedding(model, args=(make_dgl_graph(feat, edge_index, num_nodes=nnodes), ))
        elif strategy == 'rwsample':
            raise NotImplementedError
        elif strategy == 'dropmix':
            feat += delta_feat
            mask = torch.bernoulli(torch.full_like(feat, 1 - p)).bool()  # og torch.cuda.FloatTensor().uniform_()
            feat = feat * mask.unsqueeze(-1)
            edge_index, edge_weight = torch_geometric.utils.dropout_adj(edge_index, edge_weight, p=p)
            _, output = hack_model_for_embedding(model, args=(make_dgl_graph(feat, edge_index, edge_weight, self_loop=False, num_nodes=nnodes), ), kwargs={'edge_weight': edge_weight} if edge_weight else {})
        elif strategy == 'dropfeat':
            feat = F.dropout(feat, p=p) + delta_feat
            _, output = hack_model_for_embedding(model, args=(make_dgl_graph(feat, edge_index, num_nodes=nnodes), ))
        elif strategy == 'featnoise':
            noise = torch.randn_like(feat) * p
            feat += noise.to(feat.device) + delta_feat
            _, output = hack_model_for_embedding(model, args=(make_dgl_graph(feat, edge_index, num_nodes=nnodes), ))
        else:
            raise NotImplementedError
        assert output is not None, "Unable to hack model for embedding, check the model"
        return output

    @staticmethod
    def __test_time_loss(
        model: nn.Module, 
        feat: torch.Tensor, 
        delta_feat: torch.Tensor,
        edge_index: torch.Tensor, 
        edge_weight: torch.Tensor | None, 
        loss_type: List[str],
        aug_strategy: Literal['shuffle', 'dropedge', 'dropnode', 'rwsample', 'dropmix', 'dropfeat', 'featnoise'],
        nnodes: int | None = None
    ) -> torch.Tensor:
        losses = []
        if 'LC' in loss_type:
            p = 0.5 if aug_strategy == 'rwsample' else 0.05
            output1 = __class__.__augment(aug_strategy, model, feat, delta_feat, edge_index, edge_weight, p, nnodes)
            output2 = __class__.__augment('dropedge', model, feat, delta_feat, edge_index, edge_weight, p=0.0, nnodes=nnodes)
            output3 = __class__.__augment('shuffle', model, feat, delta_feat, edge_index, edge_weight, nnodes=nnodes)
            losses.append(inner(output1, output2) - inner(output2, output3))
        if 'recon' in loss_type:
            output = hack_model_for_embedding(model, args=(make_dgl_graph(feat + delta_feat, edge_index, edge_weight, num_nodes=nnodes)))
            losses.append(inner(output[edge_index[0]], output[edge_index[1]]))
        if 'entropy' in loss_type:
            output = model(make_dgl_graph(feat + delta_feat, edge_index, edge_weight, num_nodes=nnodes))
            sampled = np.random.permutation(np.arange(len(output))[:1000])
            losses.append(softmax_entropy(output[sampled]).mean(0))
        
        return sum(losses)
    
    @staticmethod
    def __sample_final_edges(
        n_perturbations: int, 
        og_feat: torch.Tensor, 
        delta_feat: torch.Tensor,
        perturbed_edge_weight: torch.Tensor, 
        eps: float, 
        max_final_samples: int, 
        modified_edge_index: torch.Tensor, 
        nnodes: int, 
        edge_index: torch.Tensor, 
        edge_weight: torch.Tensor,
        model: nn.Module,
        loss_type: List[str],
        aug_strategy: Literal['shuffle', 'dropedge', 'dropnode', 'rwsample', 'dropmix', 'dropfeat', 'featnoise'],

    ):
        best_loss = float('inf')
        i_perturbed_edge_weight = perturbed_edge_weight
        perturbed_edge_weight = perturbed_edge_weight.detach()
        perturbed_edge_weight[perturbed_edge_weight <= eps] = 0
        for i in range(max_final_samples):
            if i == 0:
                sampled_edges = torch.zeros_like(perturbed_edge_weight)
                sampled_edges[torch.topk(perturbed_edge_weight, min(n_perturbations, perturbed_edge_weight.shape[0])).indices] = 1
            else:
                sampled_edges = torch.bernoulli(perturbed_edge_weight).float()
            
            if sampled_edges.sum() > n_perturbations:
                continue
            i_perturbed_edge_weight = sampled_edges

            t_edge_index, t_edge_weight = __class__.__get_modified_adj(
                modified_edge_index, i_perturbed_edge_weight, nnodes, edge_index, edge_weight
            )
            with torch.no_grad():
                loss = __class__.__test_time_loss(model, og_feat, delta_feat, t_edge_index, t_edge_weight, loss_type, aug_strategy, nnodes)
            if best_loss > loss:
                best_loss = loss
                best_edges = i_perturbed_edge_weight.clone().detach()
        
        i_perturbed_edge_weight = best_edges
        edge_index, edge_weight = __class__.__get_modified_adj(
            modified_edge_index, i_perturbed_edge_weight, nnodes, edge_index, edge_weight
        )
        edge_mask = (edge_weight == 1)
        allowed_perturbations = 2 * n_perturbations
        edges_after_attack  = edge_mask.sum()
        clean_edges = edge_index.shape[1]
        assert (edges_after_attack >= clean_edges - allowed_perturbations
            and edges_after_attack <= clean_edges + allowed_perturbations), \
            f'{edges_after_attack} out of range with {clean_edges} clean edges and {n_perturbations} pertutbations'
        return edge_index[:, edge_mask], edge_weight[edge_mask]

    def adapt(self):
        model = self.model
        for param in model.parameters():
            param.requires_grad = False
        model = model.eval()
        graph = self.test_time_graph

        nnodes = graph.num_nodes()
        d = graph.ndata['feature'].shape[1]

        delta_feat = nn.Parameter(torch.FloatTensor(nnodes, d).to(self.device))
        delta_feat.data.fill_(1e-7)
        feat_optim = torch.optim.Adam([delta_feat], lr=self.lr_feat)

        feat, label = graph.ndata['feature'], graph.ndata['label']
        edge_index = graph.adj().coalesce().indices()  # TODO may vary for different dgl/torch versions
        edge_weight = torch.ones(edge_index.shape[1]).to(self.device)
        og_feat, og_label, og_edge_index, og_edge_weight = feat, label, edge_index, edge_weight

        n_perturbations = int(self.ratio * graph.num_edges() // 2)
        edge_index_id, modified_edge_index, perturbed_edge_weight = self.__sample_random_block(n_perturbations, edge_index, nnodes)
        adj_optim = torch.optim.Adam([perturbed_edge_weight], lr=self.lr_adj)
        edge_index, edge_weight = edge_index, None

        for it in range(self.epoch // (self.loop_feat + self.loop_adj)):
            for _ in range(self.loop_feat):
                feat_optim.zero_grad()
                loss = self.__test_time_loss(model, feat, delta_feat, edge_index, edge_weight, self.loss_type, self.aug_strategy, nnodes)
                loss.backward()
                feat_optim.step()
            
            new_feat = (feat + delta_feat).detach()
            for _ in range(self.loop_adj):
                perturbed_edge_weight.requires_grad = True
                edge_index, edge_weight = self.__get_modified_adj(modified_edge_index, perturbed_edge_weight, nnodes, og_edge_index, og_edge_weight)
                # if self.cuda_synchronize and self.device.type == 'cuda':
                #     torch.cuda.empty_cache()
                #     torch.cuda.synchronize()

                loss = self.__test_time_loss(model, new_feat, delta_feat, edge_index, edge_weight, self.loss_type, self.aug_strategy, nnodes)
                gradient = grad_with_checkpoint(loss, perturbed_edge_weight)[0]  # backward
                
                with torch.no_grad():
                    # ## update_edge_weights
                    adj_optim.zero_grad()
                    perturbed_edge_weight.grad = gradient
                    adj_optim.step()
                    perturbed_edge_weight.data[perturbed_edge_weight < self.eps] = self.eps
                    # ## update_edge_weights
                    perturbed_edge_weight = project(n_perturbations, perturbed_edge_weight, self.eps)
                    del edge_index, edge_weight

                perturbed_edge_weight.requires_grad = True
                adj_optim = torch.optim.Adam([perturbed_edge_weight], lr=self.lr_adj)
            
            if self.loop_adj:
                edge_index, edge_weight = self.__get_modified_adj(modified_edge_index, perturbed_edge_weight, nnodes, og_edge_index, og_edge_weight)
                edge_weight = edge_weight.detach()

            print(f"GTrans epoch {it} loss {loss.item()} acc {self.eval()}")
        
        if self.loop_adj:
            edge_index, edge_weight = self.__sample_final_edges(
                n_perturbations, og_feat, delta_feat, perturbed_edge_weight, self.eps, 20, 
                modified_edge_index, nnodes, og_edge_index, og_edge_weight, model, 
                self.loss_type, self.aug_strategy
            )

        with torch.no_grad():
            final_loss = self.__test_time_loss(model, feat,  delta_feat, edge_index, edge_weight, self.loss_type, self.aug_strategy, nnodes)
        print(f"GTrans final loss {final_loss.item()}")

        self.final_graph = make_dgl_graph((feat + delta_feat).detach(), edge_index, edge_weight, num_nodes=nnodes)
    
    def model_pred(self):
        if self.final_graph is None:
            return super().model_pred()
        return self.model(self.final_graph)
