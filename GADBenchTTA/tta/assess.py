import torch
import torch.nn as nn
import dgl
import torch.utils
import torch.utils.data
import torch.nn.functional as F
from torch_scatter import scatter_sum
import copy

from tta.base_tta import TTABaseClass

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

class ASSESS(TTABaseClass):
    _copy_required = False
    def __init__(
            self, 
            *args, 
            shared_thr: float = 0.8, 
            selection_interval: int = 5,
            batch_size: int = 128, 
            omega: float = 0.1,
            lr: float = 1e-4,
            sinkhorn_eps: float = 0.1,
            sinkhorn_weight: float = 1.0,
            prior_weight: float = 0.1,
            mi_loss_weight: float = 0.001,
            beta: float = 0.2,
            sdfa_eval_interval: int = 2,
            **kwargs
        ):
        super().__init__(*args, **kwargs)
        self.model_for_tta = copy.deepcopy(self.model)
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
    def __get_prototype_layer(model: nn.Module, expect_num_classes: int = 2) -> nn.Linear | None :
        module_list = list(model.modules())

        for layer in module_list[::-1]:
            if not isinstance(layer, nn.Linear):
                continue
            if expect_num_classes <= 0:
                return layer  # ignore num_classes restriction
            if layer.out_features == expect_num_classes:
                return layer
        return None
    
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
    def __split_subgraphs(source_graph: dgl.DGLGraph):
        subgraphs = []
        sampled_nodes = set()
        for nid in source_graph.nodes():
            # if nid in sampled_nodes:
            #     continue
            nbr = dgl.sampling.sample_neighbors(source_graph, nid, -1)
            # sampled_nodes.update(nbr.nodes().tolist())
            if nbr.num_nodes() <= 1:
                continue
            nbr.remove_nodes(torch.where((nbr.in_degrees() + nbr.out_degrees()) == 0)[0])
            subgraphs.append(dgl.add_self_loop(nbr))
        return subgraphs

    @staticmethod
    def __drop_node(data, p=0.2):
        mask = torch.rand(size=(data.num_nodes, ), device=data.x.device)
        mask = (mask > p)
        if data.batch is not None:
            batch = data.batch
            while (scatter_sum(mask.int(), batch) == 0).sum() > 0:
                mask = torch.rand(size=(data.num_nodes,), device=data.x.device)
                mask = (mask > p)
        else:
            while torch.sum(mask.int()) == 0:
                mask = torch.rand(size=(data.num_nodes,), device=data.x.device)
                mask = (mask > p)
        subgraph = data.subgraph(mask)
        return subgraph

    @staticmethod
    def __softplus(x):
        return torch.log(1 + torch.exp(x))

    def adapt(self):
        og_prototype = self.__get_prototype_layer(self.model).weight.detach().clone()
        discriminator_mi = DiscriminatorMI(embed_dim=og_prototype.shape[1]).to(self.device)
        mi_optimizer = torch.optim.Adam(discriminator_mi.parameters(), lr=1e-3)
        shared_thr = self.shared_thr
        surrogate_loss_memory = torch.zeros(self.test_time_graph.num_nodes(), device=self.device)
        train_params = self.__get_prototype_layer(self.model)
        optimizer = torch.optim.Adam(train_params.parameters(), lr=1e-3)
        best_acc = 0
        conf_mask = torch.ones(self.test_time_graph.num_nodes(), device=self.device)
        for epoch in range(self.epoch):
            if (epoch + 1) % self.selection_interval == 1:
                self.model.eval()
                with torch.no_grad():
                    data = self.test_time_graph.to(self.device)
                    embeddings = self.model(data)
                    confidences = torch.softmax(embeddings, dim=1).max(dim=1).values
                corrected_confidences = confidences - surrogate_loss_memory * self.omega
                thr = torch.kthvalue(corrected_confidences, max(1, int(len(corrected_confidences) * (1 - shared_thr)))).values
                conf_mask = (corrected_confidences >= thr).float()
                shared_thr -= 0.02

            loss_sh = 0
            loss_pp = 0
            loss_mi = 0
            loss_all = 0
            self.model.train()
            surrogate_losses = []

            # START og data iter
            data = self.test_time_graph.to(self.device)
            conf_mask = conf_mask
            optimizer.zero_grad()
            loss = 0
            x_emb = self.model(data)
            prototypes = self.__get_prototype_layer(self.model).weight
            x_emb_tta = self.model_for_tta(data)
            sim_mat = F.normalize(x_emb_tta, dim=1) @ F.normalize(prototypes, dim=1).T
            sh_prob = self.__sinkhorn(sim_mat.detach(), eps=self.sinkhorn_eps, n_iters=5)
            dist = torch.cdist(x_emb, prototypes)
            loss_sinkhorn = torch.mean(torch.sum(sh_prob * dist.pow(2), dim=1) * conf_mask)
            loss_prototype_prior = torch.mean((self.__get_prototype_layer(self.model).weight - og_prototype).pow(2))

            data_dropped = self.__drop_node(copy.deepcopy(data))
            x_emb_dropped = self.model(data_dropped)
            discriminator_pos_score = torch.clamp(discriminator_mi(x_emb, x_emb_dropped), -10, 10)
            rand_perm = torch.randperm(x_emb.shape[0], device=self.device)
            x_emb_neg = x_emb_dropped[rand_perm]
            discriminator_neg_score = torch.clamp(discriminator_mi(x_emb, x_emb_neg), -10, 10)

            loss_mi_graph = (self.__softplus(-discriminator_pos_score) + self.__softplus(discriminator_neg_score)).mean()
            loss = loss_sinkhorn * self.sinkhorn_weight + loss_prototype_prior * self.prior_weight + loss_mi_graph * self.mi_loss_weight
            loss.backward()
            optimizer.step()
            mi_optimizer.step()
            loss_sh += loss_sinkhorn.item()
            loss_pp += loss_prototype_prior.item()
            loss_mi += loss_mi_graph.item()
            loss_all += loss.item()
            surrogate_losses.append(loss_mi_graph.detach())
            # END og data iter

            surrogate_losses = torch.stack(surrogate_losses, dim=0)
            surrogate_loss_memory = (1 - self.beta) * surrogate_loss_memory + self.beta * surrogate_losses

            if (epoch + 1) % self.sdfa_eval_interval == 0:
                self.model.eval()
                val_acc = self.__assess_eval(self.test_time_graph, self.model)
                print(f"Epoch {epoch + 1}, Validation Accuracy: {val_acc:.4f}")
                if val_acc > best_acc:
                    best_acc = val_acc
                    self.save_model(name=f"best_{self.__class__.__name__}")
    
    @torch.no_grad()
    def __assess_eval(self, data, model, eval_mode=True):
        if eval_mode:
            model.eval()
        else:
            model.train()

        total_correct = 0
        data = data.to(self.device)
        pred = model(data)
        labels = data.ndata['label']
        total_correct += int((pred == labels).sum())
        return total_correct / len(labels)

# test codes
# def test():
#     g = dgl.load_graphs("./datasets/reddit")[0][0]
#     res = ASSESS._ASSESS__split_subgraphs(g)
#     print(res)

# test()