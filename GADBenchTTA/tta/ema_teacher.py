from .base_tta import TTABaseClass
import torch
import torch.nn.functional as F
import copy
from typing import Literal
from .utils import hack_model_for_embedding, get_classifier_layer
import dgl

def batch_info_nce(s_pred, t_pred, trustable_mask, s_m_emb=None, t_m_emb=None, temp=0.1, batch_size=32, tau=0.5):
    # s_pred: [B, D]
    # t_pred: [B, D]
    B = s_pred.shape[0]
    loss_infonce = 0.0
    for i in range(0, B, batch_size):
        s_pred_batch = s_pred[i:i+batch_size]
        t_pred_batch = t_pred[i:i+batch_size]
        mask = trustable_mask[i:i+batch_size]
        if s_m_emb is not None and t_m_emb is not None:
            s_m_emb_batch = s_m_emb[i:i+batch_size]
            t_m_emb_batch = t_m_emb[i:i+batch_size]
            neg_mask = (F.cosine_similarity(s_m_emb_batch.unsqueeze(1), t_m_emb_batch.unsqueeze(0), dim=2) > tau).any(0)
        else:
            neg_mask = None
        
        sim = F.cosine_similarity(s_pred_batch.unsqueeze(1), t_pred_batch.unsqueeze(0), dim=2)
        this_loss = (F.cross_entropy(sim / temp, torch.arange(s_pred_batch.shape[0], device=s_pred_batch.device), reduction='none') * mask)
        if neg_mask is not None:
            this_loss = this_loss * neg_mask
        loss_infonce += this_loss.mean()
    
    return loss_infonce


class EMATeacher(TTABaseClass):
    def __init__(self, *args, 
                 alpha: float = 0.9, 
                 lr: float = 1e-3,
                 loss_type: Literal['kl', 'dist'] = 'dist',
                 lower_tau: float = 0.7, 
                 upper_tau: float = 0.95, 
                 beta: float = 0.9,
                 abla: list | None = None,
                 aug_k: float = 0.1,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.student = copy.deepcopy(self.model)
        self.model = self.model.to(self.sec_device)
        self.alpha = alpha
        self.lr = lr
        self.loss_type = loss_type
        self.lower_tau = lower_tau
        self.upper_tau = upper_tau
        self.beta = beta
        self.aug_k = aug_k
        
        abla = set(abla) if abla is not None else set()
        self.disable_teacher = 'teacher' in abla
        self.disable_infonce = 'infonce' in abla
        self.disable_trustable = 'trustable' in abla

        self.tolerance = 3  # rep_sim

    def __augment(self, graph: dgl.DGLGraph):
        graph = graph.clone()
        num_edges = graph.num_edges()
        random_mask = torch.rand((num_edges, ), device=self.device) < self.aug_k
        graph.remove_edges(random_mask.nonzero().squeeze(1))
        return graph
    
    def adapt(self):
        self.model.eval()

        if self.disable_trustable:
            conf_mask = torch.ones((self.test_time_graph.num_nodes(), ), device=self.device)
        else:
            with torch.no_grad():
                pred = self.model(self.test_time_graph.to(self.sec_device)).to(self.device)
                conf = F.softmax(pred, dim=1).max(dim=1).values
                conf_mask = ((self.lower_tau <= conf) & (conf <= self.upper_tau)).float().to(self.device)
            print(f"Confidence mask ratio: {(conf_mask.sum() / conf_mask.shape[0]).item() * 100:.2f}%")

        self.student.train()
        train_params = []
        classifier_params = [i.shape for i in get_classifier_layer(self.student).parameters()]
        for params in self.student.parameters():
            params.requires_grad = True
            lr = self.lr
            if params.shape in classifier_params:
                print(f"Classifier params: {params.shape}")
                lr = self.lr * 1e-2
            train_params.append((params, lr))
        student_optim = torch.optim.Adam([{'params': p, 'lr': l} for p, l in train_params], weight_decay=1e-5)
        sum_loss = 0.0
        cur_tolerance = 0
        max_rep_sim = 0.0
        for epoch in range(self.epoch):
            with torch.no_grad():
                # t_pred = self.model(self.test_time_graph.to(self.sec_device)).to(self.device)
                t_pred, t_emb = hack_model_for_embedding(self.model, args=(self.test_time_graph.to(self.sec_device), ))
                t_pred = t_pred.to(self.device)
                t_emb = t_emb.to(self.device)
                if "feature_fusion" in dir(self.model) and "motif" in self.test_time_graph.ndata:
                    t_m_emb = self.model.feature_fusion.get_motifs_embedding(self.test_time_graph.ndata['motif'].to(self.sec_device)).to(self.device)
                else:
                    t_m_emb = None
            # s_pred = self.student(self.__augment(self.test_time_graph.to(self.device)))
            s_pred, s_emb = hack_model_for_embedding(self.student, args=(self.__augment(self.test_time_graph.to(self.device)), ))
            if "feature_fusion" in dir(self.student) and "motif" in self.test_time_graph.ndata:
                s_m_emb = self.student.feature_fusion.get_motifs_embedding(self.test_time_graph.ndata['motif'])
            else:
                s_m_emb = None

            rep_sim = F.cosine_similarity(s_emb, t_emb, dim=1).mean()
            if rep_sim > max_rep_sim:
                max_rep_sim = rep_sim
                cur_tolerance = 0
            else:
                cur_tolerance += 1
            if cur_tolerance >= self.tolerance:
                break

            if self.loss_type == 'kl':
                loss_m_1, loss_m_2 = F.log_softmax(s_emb, dim=1), F.softmax(t_emb, dim=1)
                loss_f = F.kl_div(loss_m_1, loss_m_2 , reduction='none') * conf_mask.unsqueeze(1)
                loss_ema = loss_f.mean()
            elif self.loss_type == 'dist':
                loss_f = F.mse_loss(s_emb, t_emb, reduction='none') * conf_mask.unsqueeze(1)
                loss_ema = loss_f.mean()
            # InfoNCE
            loss_infonce = batch_info_nce(s_emb, t_emb, conf_mask, s_m_emb, t_m_emb)
            if not self.disable_infonce and not self.disable_teacher:
                loss = self.beta * loss_ema + (1 - self.beta) * loss_infonce
            elif not self.disable_teacher:
                loss = loss_ema
            elif not self.disable_infonce:
                loss = loss_infonce
            else:
                raise RuntimeError("At least one of teacher and infonce should be enabled")

            student_optim.zero_grad()
            loss.backward()
            grad = max([p.grad.norm().item() for p in self.student.parameters() if p.grad is not None])
            student_optim.step()
            # Update student model parameters using EMA
            for student_param, teacher_param in zip(self.student.parameters(), self.model.parameters()):
                # \tau_t = \alpha \tar_t + (1 - \alpha) \tar_s
                teacher_param.data = teacher_param.data.to(self.sec_device)
                alpha = self.alpha
                if teacher_param.shape in classifier_params:
                    alpha = 1 - (1 - alpha) * 0.1
                teacher_param.data = teacher_param.data * alpha + student_param.data.to(self.sec_device) * (1 - alpha)
            sum_loss += loss.item()
            print(f"epoch: {epoch} , rep_sim: {rep_sim.item():.4f}, loss: {loss.item():.4f} (ema: {loss_ema.item():.4f}, infonce: {loss_infonce.item():.4f}), grad: {grad:.4f}, eval: {self.eval()}")
        print(f"EMA Teacher adapt loss: {sum_loss / self.epoch:.4f}")
    
    def eval(self):
        with torch.no_grad():
            self.model.eval()
            pred = self.model(self.test_time_graph.to(self.sec_device)).softmax(1)[:, 1]
            pred = pred.cpu()
            labels = self.test_time_graph.ndata['label'].cpu()
            result = self._eval(pred, labels)
        return result
