import torch.nn as nn
import torch
from dgl import DGLGraph
import copy
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score
from utils import MotifsDataset

def check_gpu_idle(gpus: list | None = None):  # not functional, pytorch wont release context memory until exit
    if gpus is None:
        gpus = list(range(torch.cuda.device_count()))
    res = []
    for gpu in gpus:
        if torch.cuda.list_gpu_processes(gpu).endswith("no processes are running"):
            res.append(True)
        else:
            res.append(False)
    return res

class TTABaseClass:
    _copy_required = True
    _auto_fit_data = False

    def __init__(self, 
            model: nn.Module, 
            data: MotifsDataset, 
            source_dim: int,
            *_, 
            epoch: int = 10, 
            device: torch.device | str | None = None,
            en_sec_device: bool = False,
            model_save_dir: str = "./tta_model/",
            **__
    ):
        self.epoch = epoch
        self.device = device if device is not None else (
            torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
        )
        if en_sec_device:
            self.sec_device = (None if self.device.type == 'cpu' else (
                torch.device("cuda:1") if torch.cuda.device_count() > 1 else torch.device("cpu")
            )) or torch.device("cpu")
        else:
            self.sec_device = self.device
        print(f"[DEBUG] sec_device: {self.sec_device}")
        self.model = (copy.deepcopy(model) if self._copy_required else model).to(self.device)
        # self.data = data
        # self.test_time_graph = data.graph.to(self.device)  # TODO
        self.test_time_graph : dgl.DGLGraph = data.to(self.device)
        self.save_dir = Path(model_save_dir)
        self.source_dim = source_dim
        if self._auto_fit_data:
            self.fit_feature_to_graph()

    def __fit_feature(self, source_feature: torch.Tensor):
        # svd to self.source_dim
        U, S, V = torch.svd(source_feature)
        return U[:, :self.source_dim] @ torch.diag(S[:self.source_dim])
    
    def fit_feature_to_graph(self, key: str = 'feature'):
        if key not in self.test_time_graph.ndata:
            raise ValueError(f"Key {key} not found in test_time_graph.ndata")
        if self.test_time_graph.ndata[key].shape[1] == self.source_dim:
            return
        self.test_time_graph.ndata[key] = self.__fit_feature(self.test_time_graph.ndata[key])
    
    def adapt(self):
        raise NotImplementedError
    
    def model_pred(self):
        return self.get_trained_model()(self.test_time_graph)
    
    def eval(self):
        result = {}
        with torch.no_grad():
            self.get_trained_model().eval()
            pred = self.model_pred().softmax(1)[:, 1]
            pred = pred.cpu()
            labels = self.test_time_graph.ndata['label'].cpu()
            result = self._eval(pred, labels)
        return result
    
    @staticmethod
    def _eval(pred, labels):
        result = {}
        result['AUROC'] = roc_auc_score(labels, pred)
        result['AUPRC'] = average_precision_score(labels, pred)
        result['F1'] = f1_score(labels, pred > 0.5)
        result['ACC'] = accuracy_score(labels, pred > 0.5)
        result['RecK'] = torch.sum(labels[torch.argsort(pred, descending=True)[:torch.sum(labels).item()]]).item() / torch.sum(labels).item()
        return result

    def save_model(self, name=None, model=None):
        if model is None:
            model = self.model
        if name is None:
            name = self.__class__.__name__
        folder = self.save_dir / str(self.__class__.__name__)  # method specified folder
        folder.mkdir(parents=True, exist_ok=True)
        model._save_to_state_dict(folder / f"{name}.pth")
    
    def get_trained_model(self):
        return self.model
