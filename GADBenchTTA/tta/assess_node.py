import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from torch_scatter import scatter_sum
import dgl

from .base_tta import TTABaseClass
from .utils import get_classifier_layer, hack_model_for_embedding


class DiscriminatorMI(nn.Module):  # similarity discriminator?
    def __init__(self, embed_dim):
        super(DiscriminatorMI, self).__init__()
        self.fc_1 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim)
        )
        self.fc_2 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim)
        )
    
    def forward(self, embeddings_1, embeddings_2):
        emb_1 = self.fc_1(embeddings_1) + embeddings_1
        emb_2 = self.fc_2(embeddings_2) + embeddings_2
        score = torch.sum(emb_1 * emb_2, dim=1)
        return score
    

class ASSESSNode(TTABaseClass):
    _copy_required = False
    def __init__(
            self, 
            *args, 
            shared_thr: float = 0.8, 
            selection_interval: int = 5,
            batch_size: int = 128, 
            omega: float = 0.1,
            lr: float = 1e-2,
            sinkhorn_eps: float = 0.1,
            sinkhorn_weight: float = 1.0,
            prior_weight: float = 0.1,
            mi_loss_weight: float = 0.001,
            beta: float = 0.2,
            sdfa_eval_interval: int = 2,
            **kwargs
        ):
        super().__init__(*args, **kwargs)
        self.train_model = copy.deepcopy(self.model)
        self.model_for_tta = self.model.to(self.sec_device)
        self.shared_thr = shared_thr
        self.selection_interval = selection_interval
        self.batch_size = batch_size
        self.omega = omega
        self.lr = lr
        self.sinkhorn_eps = sinkhorn_eps
        self.sinkhorn_weight = sinkhorn_weight
        self.prior_weight = prior_weight
        self.mi_loss_weight = mi_loss_weight
        self.beta = beta
        self.sdfa_eval_interval = sdfa_eval_interval

    @staticmethod
    def __get_prototype_layer(model: nn.Module) -> nn.Linear:
        return get_classifier_layer(model)
    
    @staticmethod
    def __sinkhorn(scores, eps=0.05, n_iters=10):  # B x num_prototypes
            Q = torch.exp(scores / eps).T  # Kx B
            Q = Q / torch.sum(Q)
            K, B = Q.shape
            r = torch.ones(K, device=scores.device) / K
            c = torch.ones(B, device=scores.device) / B
            for _ in range(n_iters):
                Q *= (r / Q.sum(dim=1)).unsqueeze(1)
                Q *= (c / Q.sum(dim=0)).unsqueeze(0)
            return (Q / Q.sum(dim=0, keepdims=True)).T  

    @staticmethod
    def __drop_node(data: dgl.DGLGraph, p=0.2):
        mask = torch.rand(size=(data.num_nodes(), ), device=data.device)
        mask = (mask > p)
        while torch.sum(mask.int()) == 0:
            mask = torch.rand(size=(data.num_nodes(), ), device=data.device)
            mask = (mask > p)
        subgraph = dgl.node_subgraph(data, mask)
        return subgraph, mask

    @staticmethod
    def __softplus(x):
        return torch.log(1 + torch.exp(x))
    
    def adapt(self):
        og_prototype = self.__get_prototype_layer(self.train_model).weight.detach().clone().T
        discriminator_mi = DiscriminatorMI(embed_dim=og_prototype.shape[1]).to(self.device)
        mi_optimizer = torch.optim.Adam(discriminator_mi.parameters(), lr=self.lr)
        shared_thr = self.shared_thr
        surrogate_loss_memory = torch.zeros(self.test_time_graph.num_nodes(), device=self.device)
        train_params = self.__get_prototype_layer(self.train_model)
        optimizer = torch.optim.Adam(train_params.parameters(), lr=self.lr)
        best_acc = 0
        conf_mask = torch.ones(self.test_time_graph.num_nodes(), device=self.device)
        for epoch in range(self.epoch):
            if (epoch + 1) % self.selection_interval == 1:
                self.train_model.eval()
                with torch.no_grad():
                    embeddings = self.model_for_tta(self.test_time_graph.to(self.sec_device))
                    conf = torch.softmax(embeddings, dim=1).max(dim=1).values
                corrected_conf = conf.to(self.device) - surrogate_loss_memory * self.omega
                thr = torch.kthvalue(corrected_conf, max(1, int(len(corrected_conf) * (1 - shared_thr)))).values
                conf_mask = (corrected_conf > thr).float()
                shared_thr -= 0.02
                print(f"epoch: {epoch}, shared_thr: {shared_thr:.4f}, conf_mask ratio: {(conf_mask.sum() / conf_mask.shape[0]).item() * 100:.2f}%")
            
            self.train_model.train()
            self.__get_prototype_layer(self.train_model).weight.requires_grad = True
            data: dgl.DGLGraph = self.test_time_graph.to(self.device)
            x_emb = self.train_model(data)
            prototypes = self.__get_prototype_layer(self.train_model).weight.T
            with torch.no_grad():
                x_emb_tta = self.model_for_tta(data.to(self.sec_device)).detach().to(self.device)  # emb from og_model
                sim_mat = F.normalize(x_emb_tta, dim=1) @ F.normalize(prototypes, dim=1).T
                sh_prob = self.__sinkhorn(sim_mat.detach(), eps=self.sinkhorn_eps, n_iters=5)
            dist = torch.cdist(x_emb, prototypes, p=2)
            loss_sink = torch.mean(torch.sum(sh_prob * dist.pow(2), dim=1) * conf_mask)
            loss_proto = torch.mean((self.__get_prototype_layer(self.train_model).weight.T - og_prototype).pow(2))

            data_dropped, mask = self.__drop_node(data.clone())
            x_emb_dropped = self.train_model(data_dropped)
            discriminator_pos_score = torch.clamp(discriminator_mi(x_emb[mask], x_emb_dropped), min=-10, max=10)
            rand_perm = torch.randperm(x_emb_dropped.shape[0], device=self.device)
            x_emd_neg = x_emb_dropped[rand_perm]
            discriminator_neg_score = torch.clamp(discriminator_mi(x_emb[mask], x_emd_neg), min=-10, max=10)

            loss_mi_graph = torch.mean(self.__softplus(discriminator_pos_score) + self.__softplus(-discriminator_neg_score))
            loss = loss_sink * self.sinkhorn_weight + loss_proto * self.prior_weight + loss_mi_graph * self.mi_loss_weight
            optimizer.zero_grad()
            mi_optimizer.zero_grad()
            loss.backward()
            grad = self.__get_prototype_layer(self.train_model).weight.grad.mean().item()
            print(f"grad: {grad:.4f}")
            optimizer.step()
            mi_optimizer.step()

            print(f"epoch: {epoch}, loss: {loss.item():.4f}, loss_sink: {loss_sink.item():.4f}, loss_proto: {loss_proto.item():.4f}, loss_mi_graph: {loss_mi_graph.item():.4f}")
            print(f"eval: {self.eval()}")

            surrogate_loss_memory = (1 - self.beta) * surrogate_loss_memory + self.beta * loss_mi_graph

    def get_trained_model(self):
        return self.train_model
    
    def eval(self):
        with torch.no_grad():
            self.train_model.eval()
            pred = self.train_model(self.test_time_graph).softmax(1)[:, 1]
            pred = pred.cpu()
            labels = self.test_time_graph.ndata['label'].cpu()
            result = self._eval(pred, labels)
        return result
