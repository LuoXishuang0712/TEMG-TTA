from models.gnn import *
from models.attention import *
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score
from sklearn.cluster import KMeans
from dgl.nn.pytorch.factory import KNNGraph
import dgl
import numpy as np
import pandas as pd
import itertools
import psutil, os
from catboost import Pool, CatBoostClassifier, CatBoostRegressor, sum_models
from torch.utils.data import DataLoader
from typing import Iterable
import traceback


class BaseDetector(object):
    def __init__(self, train_config, model_config, data):
        self.model_config = model_config
        self.train_config = train_config
        self.data = data
        model_config['in_feats'] = self.data.graph.ndata['feature'].shape[1]
        graph = self.data.graph.to(self.train_config['device'])
        self.labels = graph.ndata['label']
        self.train_mask = graph.ndata['train_mask'].bool()
        self.val_mask = graph.ndata['val_mask'].bool()
        self.test_mask = graph.ndata['test_mask'].bool()
        self.weight = (1 - self.labels[self.train_mask]).sum().item() / self.labels[self.train_mask].sum().item()
        self.source_graph = graph
        print(train_config['inductive'])
        if train_config['inductive'] == False:
            self.train_graph = graph
            self.val_graph = graph
        else:
            self.train_graph = graph.subgraph(self.train_mask)
            self.val_graph = graph.subgraph(self.train_mask+self.val_mask)
        self.best_score = -1
        self.patience_knt = 0
        
    def train(self):
        pass

    def eval(self, labels, probs):
        score = {}
        with torch.no_grad():
            if torch.is_tensor(labels):
                labels = labels.cpu().numpy()
            if torch.is_tensor(probs):
                probs = probs.cpu().numpy()
            score['AUROC'] = roc_auc_score(labels, probs)
            score['AUPRC'] = average_precision_score(labels, probs)
            score['F1'] = f1_score(labels, probs > 0.5)
            score['ACC'] = accuracy_score(labels, probs > 0.5)
            labels = np.array(labels)
            k = labels.sum()
        score['RecK'] = sum(labels[probs.argsort()[-k:]]) / sum(labels)
        return score

    def plain_eval(self, data):
        graph = data.graph.to(self.train_config['device'])
        mask = graph.ndata['test_mask'].bool()
        label = graph.ndata['label']
        logits = self.model(graph)
        probs = logits.softmax(1)[:, 1]
        score = self.eval(label[mask], probs[mask])
        return score

class BaseGNNDetector(BaseDetector):
    def __init__(self, train_config, model_config, data):
        super().__init__(train_config, model_config, data)
        gnn = globals()[model_config['model']]
        model_config['in_feats'] = self.data.graph.ndata['feature'].shape[1]
        self.model = gnn(**model_config).to(train_config['device'])

    def train(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.model_config['lr'])
        train_labels, val_labels, test_labels = self.labels[self.train_mask], self.labels[self.val_mask], self.labels[self.test_mask]
        for e in range(self.train_config['epochs']):
            self.model.train()
            logits = self.model(self.train_graph)
            loss = F.cross_entropy(logits[self.train_graph.ndata['train_mask']], train_labels,
                                   weight=torch.tensor([1., self.weight], device=self.labels.device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # The following code is used to record the memory usage
            # py_process = psutil.Process(os.getpid())
            # print(f"CPU Memory Usage: {py_process.memory_info().rss / (1024 ** 3)} GB")
            # print(f"GPU Memory Usage: {torch.cuda.memory_reserved() / (1024 ** 3)} GB")
            if self.model_config['drop_rate'] > 0 or self.train_config['inductive']:
                self.model.eval()
                logits = self.model(self.val_graph)
            probs = logits.softmax(1)[:, 1]
            val_score = self.eval(val_labels, probs[self.val_graph.ndata['val_mask']])
            if val_score[self.train_config['metric']] > self.best_score:
                if self.train_config['inductive']:
                    logits = self.model(self.source_graph)
                    probs = logits.softmax(1)[:, 1]
                self.patience_knt = 0
                self.best_score = val_score[self.train_config['metric']]
                test_score = self.eval(test_labels, probs[self.test_mask])
                print('Epoch {}, Loss {:.4f}, Val AUC {:.4f}, PRC {:.4f}, RecK {:.4f}, F1 {:.4f}, test AUC {:.4f}, PRC {:.4f}, RecK {:.4f}, F1 {:.4f}'.format(
                    e, loss, val_score['AUROC'], val_score['AUPRC'], val_score['RecK'], val_score['F1'],
                    test_score['AUROC'], test_score['AUPRC'], test_score['RecK'], test_score['F1']))
            else:
                self.patience_knt += 1
                if self.patience_knt > self.train_config['patience']:
                    break
        return test_score

# RGCN, HGT
class HeteroGNNDetector(BaseDetector):
    def __init__(self, train_config, model_config, data):
        super().__init__(train_config, model_config, data)
        hgnn = globals()[model_config['model']]
        model_config['in_feats'] = self.data.graph.ndata['feature'].shape[1]
        model_config['etypes'] = self.source_graph.canonical_etypes
        self.model = hgnn(**model_config).to(train_config['device'])
        
    def train(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.model_config['lr'])
        train_labels, val_labels, test_labels = self.labels[self.train_mask], self.labels[self.val_mask], self.labels[self.test_mask]
        for e in range(self.train_config['epochs']):
            self.model.train()
            logits = self.model(self.train_graph)
            loss = F.cross_entropy(logits[self.train_graph.ndata['train_mask']], train_labels,
                                   weight=torch.tensor([1., self.weight], device=self.labels.device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if self.model_config['drop_rate'] > 0 or self.train_config['inductive']:
                self.model.eval()
                logits = self.model(self.val_graph)
            probs = logits.softmax(1)[:, 1]
            val_score = self.eval(val_labels, probs[self.val_graph.ndata['val_mask']])
            if val_score[self.train_config['metric']] > self.best_score:
                if self.train_config['inductive']:
                    logits = self.model(self.source_graph)
                    probs = logits.softmax(1)[:, 1]
                self.patience_knt = 0
                self.best_score = val_score[self.train_config['metric']]
                test_score = self.eval(test_labels, probs[self.test_mask])
                print('Epoch {}, Loss {:.4f}, Val AUC {:.4f}, PRC {:.4f}, RecK {:.4f}, F1 {:.4f}, test AUC {:.4f}, PRC {:.4f}, RecK {:.4f}, F1 {:.4f}'.format(
                    e, loss, val_score['AUROC'], val_score['AUPRC'], val_score['RecK'], val_score['F1'],
                    test_score['AUROC'], test_score['AUPRC'], test_score['RecK'], test_score['F1']))
            else:
                self.patience_knt += 1
                if self.patience_knt > self.train_config['patience']:
                    break
        return test_score


class CAREGNNDetector(BaseDetector):
    def __init__(self, train_config, model_config, data):
        super().__init__(train_config, model_config, data)
        model_config['in_feats'] = self.data.graph.ndata['feature'].shape[1]
        self.model = CAREGNN(**model_config).to(train_config['device'])

    def train(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.model_config['lr'])
        train_labels, val_labels, test_labels = self.labels[self.train_mask], \
                                                self.labels[self.val_mask], self.labels[self.test_mask]
        rl_idx = torch.nonzero(self.train_mask & self.labels, as_tuple=False).squeeze(1)
        for e in range(self.train_config['epochs']):
            self.model.train()
            logits = self.model(self.train_graph, e)
            loss = F.cross_entropy(logits[self.train_graph.ndata['train_mask']], train_labels,
                                   weight=torch.tensor([1., self.weight], device=self.labels.device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            self.model.RLModule(self.train_graph, e, rl_idx)
            if self.model_config['drop_rate'] > 0 or self.train_config['inductive']:
                self.model.eval()
                logits = self.model(self.val_graph)
            probs = logits.softmax(1)[:, 1]
            val_score = self.eval(val_labels, probs[self.val_graph.ndata['val_mask']])
            if val_score[self.train_config['metric']] > self.best_score:
                if self.train_config['inductive']:
                    logits = self.model(self.source_graph)
                    probs = logits.softmax(1)[:, 1]
                self.patience_knt = 0
                self.best_score = val_score[self.train_config['metric']]
                test_score = self.eval(test_labels, probs[self.test_mask])
                print('Epoch {}, Loss {:.4f}, Val AUC {:.4f}, PRC {:.4f}, RecK {:.4f}, test AUC {:.4f}, PRC {:.4f}, RecK {:.4f}'.format(
                    e, loss, val_score['AUROC'], val_score['AUPRC'], val_score['RecK'],
                    test_score['AUROC'], test_score['AUPRC'], test_score['RecK']))
            else:
                self.patience_knt += 1
                if self.patience_knt > self.train_config['patience']:
                    break
        return test_score


class NAGNNDetector(BaseDetector):
    def __init__(self, train_config, model_config, data):
        super().__init__(train_config, model_config, data)
        model_config['in_feats'] = self.data.graph.ndata['feature'].shape[1]
        self.model = BWGNN(**model_config).to(train_config['device'])
        self.aggregate = dglnn.GINConv(None, activation=None, init_eps=0,
                                 aggregator_type='mean').to(self.train_config['device'])

    def train(self):
        k = 5 if 'k' not in self.model_config else self.model_config['k']
        dist = 'cosine' if 'dist' not in self.model_config else self.model_config['dist']
        feat = self.data.graph.ndata['feature'].to(self.train_config['device'])
        if k > 0:
            knn_graph = KNNGraph(k)
            knn_g = knn_graph(feat, algorithm="bruteforce-sharemem", dist=dist)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.model_config['lr'])
        train_labels, val_labels, test_labels = self.labels[self.train_mask], \
                                                self.labels[self.val_mask], self.labels[self.test_mask]
        for e in range(self.train_config['epochs']):
            self.model.train()
            logits = self.model(self.source_graph)
            loss = F.cross_entropy(logits[self.train_mask], train_labels,
                                   weight=torch.tensor([1., self.weight], device=self.labels.device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if self.model_config['drop_rate'] > 0:
                self.model.eval()
                logits = self.model(self.source_graph)
            probs = logits.softmax(1)[:, 1]
            if k > 0:
                # neighbor smoothing
                probs = self.aggregate(knn_g, probs)
            
            val_score = self.eval(val_labels, probs[self.val_mask])
            if val_score[self.train_config['metric']] > self.best_score:
                self.patience_knt = 0
                self.best_score = val_score[self.train_config['metric']]
                test_score = self.eval(test_labels, probs[self.test_mask])
                print('Loss {:.4f}, Val AUC {:.4f}, PRC {:.4f}, RecK {:.4f}, test AUC {:.4f}, PRC {:.4f}, RecK {:.4f}'.format(
                    loss, val_score['AUROC'], val_score['AUPRC'], val_score['RecK'],
                    test_score['AUROC'], test_score['AUPRC'], test_score['RecK']))
            else:
                self.patience_knt += 1
                if self.patience_knt > self.train_config['patience']:
                    break
        return test_score


class SVMDetector(BaseDetector):
    def __init__(self, train_config, model_config, data):
        super().__init__(train_config, model_config, data)
        penalty = 'l2' if 'penalty' not in self.model_config else self.model_config['penalty']
        loss = 'squared_hinge' if 'loss' not in self.model_config else self.model_config['loss']
        C = 1 if 'C' not in self.model_config else self.model_config['C']
        self.model = svm.LinearSVC(penalty=penalty, loss=loss, C=C)

    def train(self):
        train_X = self.source_graph.ndata['feature'][self.train_mask].cpu().numpy()
        train_y = self.source_graph.ndata['label'][self.train_mask].cpu().numpy()
        val_X = self.source_graph.ndata['feature'][self.val_mask].cpu().numpy()
        val_y = self.source_graph.ndata['label'][self.val_mask].cpu().numpy()
        test_X = self.source_graph.ndata['feature'][self.test_mask].cpu().numpy()
        test_y = self.source_graph.ndata['label'][self.test_mask].cpu().numpy()
        self.model.fit(train_X, train_y)
        pred_val_y = self.model.decision_function(val_X)
        pred_y = self.model.decision_function(test_X)
        val_score = self.eval(val_y, pred_val_y)
        self.best_score = val_score[self.train_config['metric']]
        test_score = self.eval(test_y, pred_y)
        return test_score


class KNNDetector(BaseDetector):
    def __init__(self, train_config, model_config, data):
        super().__init__(train_config, model_config, data)
        k = 5 if 'k' not in self.model_config else self.model_config['k']
        weights = 'uniform' if 'weights' not in self.model_config else self.model_config['weights']
        p = 2 if 'p' not in self.model_config else self.model_config['p']
        self.model = KNeighborsClassifier(n_neighbors=k, weights=weights, p=p, n_jobs=32)

    def train(self):
        train_X = self.source_graph.ndata['feature'][self.train_mask].cpu().numpy()
        train_y = self.source_graph.ndata['label'][self.train_mask].cpu().numpy()
        val_X = self.source_graph.ndata['feature'][self.val_mask].cpu().numpy()
        val_y = self.source_graph.ndata['label'][self.val_mask].cpu().numpy()
        test_X = self.source_graph.ndata['feature'][self.test_mask].cpu().numpy()
        test_y = self.source_graph.ndata['label'][self.test_mask].cpu().numpy()
        self.model.fit(train_X, train_y)
        pred_val_y = self.model.predict_proba(val_X)[:, 1]
        pred_y = self.model.predict_proba(test_X)[:, 1]
        val_score = self.eval(val_y, pred_val_y)
        self.best_score = val_score[self.train_config['metric']]
        test_score = self.eval(test_y, pred_y)
        return test_score


class XGBODDetector(BaseDetector):
    def __init__(self, train_config, model_config, data):
        from pyod.models.xgbod import XGBOD
        super().__init__(train_config, model_config, data)
        self.model = XGBOD(n_jobs=32, **model_config)

    def train(self):
        train_X = self.source_graph.ndata['feature'][self.train_mask].cpu().numpy()
        train_y = self.source_graph.ndata['label'][self.train_mask].cpu().numpy()
        if self.train_mask.sum() > 100000: # avoid out of time
            train_X = train_X[:2000]
            train_y = train_y[:2000]
        print(train_X.shape, train_y.shape)
        val_X = self.source_graph.ndata['feature'][self.val_mask].cpu().numpy()
        val_y = self.source_graph.ndata['label'][self.val_mask].cpu().numpy()
        test_X = self.source_graph.ndata['feature'][self.test_mask].cpu().numpy()
        test_y = self.source_graph.ndata['label'][self.test_mask].cpu().numpy()
        self.model.fit(train_X, train_y)
        pred_val_y = self.model.decision_function(val_X)
        pred_y = self.model.decision_function(test_X)
        val_score = self.eval(val_y, pred_val_y)
        self.best_score = val_score[self.train_config['metric']]
        test_score = self.eval(test_y, pred_y)
        return test_score


class XGBoostDetector(BaseDetector):
    def __init__(self, train_config, model_config, data):
        super().__init__(train_config, model_config, data)
        import xgboost as xgb
        eval_metric = roc_auc_score if train_config['metric'] == "AUROC" else average_precision_score
        self.model = xgb.XGBClassifier(tree_method='gpu_hist', eval_metric=eval_metric, **model_config)

    def train(self):
        train_X = self.source_graph.ndata['feature'][self.train_mask].cpu().numpy()
        train_y = self.source_graph.ndata['label'][self.train_mask].cpu().numpy()
        val_X = self.source_graph.ndata['feature'][self.val_mask].cpu().numpy()
        val_y = self.source_graph.ndata['label'][self.val_mask].cpu().numpy()
        test_X = self.source_graph.ndata['feature'][self.test_mask].cpu().numpy()
        test_y = self.source_graph.ndata['label'][self.test_mask].cpu().numpy()
        weights = np.where(train_y == 0, 1, self.weight)

        self.model.fit(train_X, train_y, sample_weight=weights, eval_set=[(val_X, val_y)], verbose=False)
        pred_val_y = self.model.predict_proba(val_X)[:, 1]
        pred_y = self.model.predict_proba(test_X)[:, 1]
        val_score = self.eval(val_y, pred_val_y)
        self.best_score = val_score[self.train_config['metric']]
        test_score = self.eval(test_y, pred_y)
        return test_score


class XGBNADetector(BaseDetector):
    def __init__(self, train_config, model_config, data):
        super().__init__(train_config, model_config, data)
        import xgboost as xgb
        eval_metric = roc_auc_score if train_config['metric'] == "AUROC" else average_precision_score
        self.model = xgb.XGBClassifier(tree_method='gpu_hist', eval_metric=eval_metric, **model_config)
        self.aggregate = dglnn.GINConv(None, activation=None, init_eps=0,
                                 aggregator_type='mean').to(self.train_config['device'])

    def train(self):
        k = 5 if 'k' not in self.model_config else self.model_config['k']
        dist = 'cosine' if 'dist' not in self.model_config else self.model_config['dist']
        feat = self.data.graph.ndata['feature'].to(self.train_config['device'])
        if k > 0:
            knn_graph = KNNGraph(k)
            knn_g = knn_graph(feat, algorithm="bruteforce-sharemem", dist=dist)

        train_X = self.source_graph.ndata['feature'][self.train_mask].cpu().numpy()
        train_y = self.source_graph.ndata['label'][self.train_mask].cpu().numpy()
        val_X = self.source_graph.ndata['feature'][self.val_mask].cpu().numpy()
        val_y = self.source_graph.ndata['label'][self.val_mask].cpu().numpy()
        test_y = self.source_graph.ndata['label'][self.test_mask].cpu().numpy()
        weights = np.where(train_y == 0, 1, self.weight)

        self.model.fit(train_X, train_y, sample_weight=weights, eval_set=[(val_X, val_y)], verbose=False)
        X = self.source_graph.ndata['feature'].cpu().numpy()
        probs = torch.tensor(self.model.predict_proba(X)[:, 1]).cuda()
        if k > 0:
            probs = self.aggregate(knn_g, probs)
        pred_val_y = probs[self.val_mask]
        pred_y = probs[self.test_mask]
        val_score = self.eval(val_y, pred_val_y)
        self.best_score = val_score[self.train_config['metric']]
        test_score = self.eval(test_y, pred_y)
        return test_score


class XGBGraphDetector(BaseDetector):
    def __init__(self, train_config, model_config, data):
        super().__init__(train_config, model_config, data)
        import xgboost as xgb
        eval_metric = roc_auc_score if train_config['metric'] == "AUROC" else average_precision_score
        self.model = xgb.XGBClassifier(tree_method='gpu_hist', eval_metric=eval_metric, verbose=2, **model_config)
        gnn = GIN_noparam(**model_config).to(self.source_graph.device)
        new_feat = gnn(self.source_graph)
        if self.train_config['inductive'] == True:
            new_feat[self.train_mask] = gnn(self.source_graph.subgraph(self.train_mask))
            val_graph = self.source_graph.subgraph(self.train_mask+self.val_mask)
            new_feat[self.val_mask] = gnn(val_graph)[val_graph.ndata['val_mask']]
        self.source_graph.ndata['feature'] = new_feat

    def train(self):
        train_X = self.source_graph.ndata['feature'][self.train_mask].cpu().numpy()
        train_y = self.source_graph.ndata['label'][self.train_mask].cpu().numpy()
        val_X = self.source_graph.ndata['feature'][self.val_mask].cpu().numpy()
        val_y = self.source_graph.ndata['label'][self.val_mask].cpu().numpy()
        test_X = self.source_graph.ndata['feature'][self.test_mask].cpu().numpy()
        test_y = self.source_graph.ndata['label'][self.test_mask].cpu().numpy()
        weights = np.where(train_y == 0, 1, self.weight)

        self.model.fit(train_X, train_y, sample_weight=weights, eval_set=[(val_X, val_y)])  # early_stopping_rounds =20
        pred_val_y = self.model.predict_proba(val_X)[:, 1]
        pred_y = self.model.predict_proba(test_X)[:, 1]
        val_score = self.eval(val_y, pred_val_y)
        self.best_score = val_score[self.train_config['metric']]
        test_score = self.eval(test_y, pred_y)
        return test_score


class RFDetector(BaseDetector):
    def __init__(self, train_config, model_config, data):
        super().__init__(train_config, model_config, data)
        n_estimators = 100 if 'n_estimators' not in model_config else model_config['n_estimators']
        criterion = 'gini' if 'criterion' not in model_config else model_config['criterion']
        max_samples = None if 'max_samples' not in model_config else model_config['max_samples']
        max_features = 'sqrt' if 'max_features' not in model_config else model_config['max_features']
        self.model = RandomForestClassifier(n_jobs=32, n_estimators=n_estimators, criterion=criterion,
                                            max_samples=max_samples, max_features=max_features)

    def train(self):
        train_X = self.source_graph.ndata['feature'][self.train_mask].cpu().numpy()
        train_y = self.source_graph.ndata['label'][self.train_mask].cpu().numpy()
        val_X = self.source_graph.ndata['feature'][self.val_mask].cpu().numpy()
        val_y = self.source_graph.ndata['label'][self.val_mask].cpu().numpy()
        test_X = self.source_graph.ndata['feature'][self.test_mask].cpu().numpy()
        test_y = self.source_graph.ndata['label'][self.test_mask].cpu().numpy()
        weights = np.where(train_y == 0, 1, self.weight)
        self.model.fit(train_X, train_y, sample_weight=weights)
        pred_val_y = self.model.predict_proba(val_X)[:, 1]
        pred_y = self.model.predict_proba(test_X)[:, 1]
        val_score = self.eval(val_y, pred_val_y)
        self.best_score = val_score[self.train_config['metric']]
        test_score = self.eval(test_y, pred_y)
        return test_score


class RFGraphDetector(BaseDetector):
    def __init__(self, train_config, model_config, data):
        super().__init__(train_config, model_config, data)
        n_estimators = 100 if 'n_estimators' not in model_config else model_config['n_estimators']
        criterion = 'gini' if 'criterion' not in model_config else model_config['criterion']
        max_samples = None if 'max_samples' not in model_config else model_config['max_samples']
        max_features = 'sqrt' if 'max_features' not in model_config else model_config['max_features']
        self.model = RandomForestClassifier(n_jobs=32, n_estimators=n_estimators, criterion=criterion,
                                            max_samples=max_samples, max_features=max_features)
        gnn = GIN_noparam(**model_config).to(self.source_graph.device)
        new_feat = gnn(self.source_graph)
        if self.train_config['inductive'] == True:
            new_feat[self.train_mask] = gnn(self.source_graph.subgraph(self.train_mask))
            val_graph = self.source_graph.subgraph(self.train_mask+self.val_mask)
            new_feat[self.val_mask] = gnn(val_graph)[val_graph.ndata['val_mask']]
        self.source_graph.ndata['feature'] = new_feat

    def train(self):
        train_X = self.source_graph.ndata['feature'][self.train_mask].cpu().numpy()
        train_y = self.source_graph.ndata['label'][self.train_mask].cpu().numpy()
        val_X = self.source_graph.ndata['feature'][self.val_mask].cpu().numpy()
        val_y = self.source_graph.ndata['label'][self.val_mask].cpu().numpy()
        test_X = self.source_graph.ndata['feature'][self.test_mask].cpu().numpy()
        test_y = self.source_graph.ndata['label'][self.test_mask].cpu().numpy()
        weights = np.where(train_y == 0, 1, self.weight)
        self.model.fit(train_X, train_y, sample_weight=weights)
        pred_val_y = self.model.predict_proba(val_X)[:, 1]
        pred_y = self.model.predict_proba(test_X)[:, 1]
        val_score = self.eval(val_y, pred_val_y)
        self.best_score = val_score[self.train_config['metric']]
        test_score = self.eval(test_y, pred_y)
        return test_score


class GASDetector(BaseDetector):
    def __init__(self, train_config, model_config, data):
        super().__init__(train_config, model_config, data)
        model_config['in_feats'] = self.data.graph.ndata['feature'].shape[1]
        model_config['mlp_layers'] = 0
        self.model = GCN(**model_config).to(train_config['device'])
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.model.h_feats * 2, self.model.h_feats),
            torch.nn.ReLU(),
            torch.nn.Linear(self.model.h_feats, 2)).to(train_config['device'])

    def train(self):
        k = 5 if 'k' not in self.model_config else self.model_config['k']
        dist = 'cosine' if 'dist' not in self.model_config else self.model_config['dist']
        feat = self.data.graph.ndata['feature'].to(self.train_config['device'])
        knn_graph = KNNGraph(k)
        knn_g = knn_graph(feat, algorithm="bruteforce-sharemem", dist=dist)
        knn_g.ndata["feature"] = feat

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.model_config['lr'])
        train_labels, val_labels, test_labels = self.labels[self.train_mask], \
                                                self.labels[self.val_mask], self.labels[self.test_mask]

        for e in range(self.train_config['epochs']):
            self.model.train()
            h_origin = self.model(self.source_graph)
            h_knn = self.model(knn_g)
            h_all = torch.cat([h_origin, h_knn], -1)
            logits = self.mlp(h_all)
            loss = F.cross_entropy(logits[self.train_mask], train_labels,
                                   weight=torch.tensor([1., self.weight], device=self.labels.device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if self.model_config['drop_rate'] > 0:
                self.model.eval()
                logits = self.model(self.source_graph)
            probs = logits.softmax(1)[:, 1]
            val_score = self.eval(val_labels, probs[self.val_mask])
            if val_score[self.train_config['metric']] > self.best_score:
                self.patience_knt = 0
                self.best_score = val_score[self.train_config['metric']]
                test_score = self.eval(test_labels, probs[self.test_mask])
            else:
                self.patience_knt += 1
                if self.patience_knt > self.train_config['patience']:
                    break
        return test_score


class KNNGCNDetector(BaseDetector):
    def __init__(self, train_config, model_config, data):
        super().__init__(train_config, model_config, data)
        model_config['in_feats'] = self.data.graph.ndata['feature'].shape[1]
        self.model = GCN(**model_config).to(train_config['device'])

    def train(self):
        k = 5 if 'k' not in self.model_config else self.model_config['k']
        dist = 'cosine' if 'dist' not in self.model_config else self.model_config['dist']
        feat = self.data.graph.ndata['feature'].to(self.train_config['device'])
        knn_graph = KNNGraph(k)
        knn_g = knn_graph(feat, algorithm="bruteforce-sharemem", dist=dist)
        new_g = dgl.merge([knn_g, self.source_graph])
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.model_config['lr'])
        train_labels, val_labels, test_labels = self.labels[self.train_mask], \
                                                self.labels[self.val_mask], self.labels[self.test_mask]

        for e in range(self.train_config['epochs']):
            self.model.train()
            logits = self.model(new_g)
            loss = F.cross_entropy(logits[self.train_mask], train_labels,
                                   weight=torch.tensor([1., self.weight], device=self.labels.device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if self.model_config['drop_rate'] > 0:
                self.model.eval()
                logits = self.model(self.source_graph)
            if self.model_config['drop_rate'] > 0:
                self.model.eval()
                logits = self.model(self.source_graph)
            probs = logits.softmax(1)[:, 1]
            val_score = self.eval(val_labels, probs[self.val_mask])
            if val_score[self.train_config['metric']] > self.best_score:
                self.patience_knt = 0
                self.best_score = val_score[self.train_config['metric']]
                test_score = self.eval(test_labels, probs[self.test_mask])
            else:
                self.patience_knt += 1
                if self.patience_knt > self.train_config['patience']:
                    break
        return test_score


class GHRNDetector(BaseDetector):
    def __init__(self, train_config, model_config, data):
        super().__init__(train_config, model_config, data)
        model_config['in_feats'] = self.data.graph.ndata['feature'].shape[1]
        self.model = BWGNN(**model_config).to(train_config['device'])

    def random_walk_update(self, delete_ratio):
        graph = self.source_graph
        edge_weight = torch.ones(graph.num_edges()).to(self.train_config['device'])
        norm = dgl.nn.pytorch.conv.EdgeWeightNorm(norm='both')
        graph.edata['w'] = norm(graph, edge_weight)
        aggregate_fn = fn.u_mul_e('h', 'w', 'm')
        reduce_fn = fn.sum(msg='m', out='ay')

        graph.ndata['h'] = graph.ndata['feature']
        graph.update_all(aggregate_fn, reduce_fn)
        graph.ndata['ly'] = graph.ndata['feature'] - graph.ndata['ay']
        graph.apply_edges(self.inner_product_black)
        black = graph.edata['inner_black']
        threshold = int(delete_ratio * graph.num_edges())
        edge_to_move = set(black.sort()[1][:threshold].tolist())
        graph_new = dgl.remove_edges(graph, list(edge_to_move))
        return graph_new

    def inner_product_black(self, edges):
        inner_black = (edges.src['ly'] * edges.dst['ly']).sum(axis=1)
        return {'inner_black': inner_black}

    def train(self):
        del_ratio = 0.015 if 'del_ratio' not in self.model_config else self.model_config['del_ratio']
        if del_ratio != 0.:
            graph = self.random_walk_update(del_ratio)
            graph = dgl.add_self_loop(dgl.remove_self_loop(graph))

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.model_config['lr'])
        train_labels, val_labels, test_labels = self.labels[self.train_mask], \
                                                self.labels[self.val_mask], self.labels[self.test_mask]
        for e in range(self.train_config['epochs']):
            self.model.train()
            logits = self.model(graph)
            loss = F.cross_entropy(logits[self.train_mask], train_labels,
                                   weight=torch.tensor([1., self.weight], device=self.labels.device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            probs = logits.softmax(1)[:, 1]
            val_score = self.eval(val_labels, probs[self.val_mask])
            if val_score[self.train_config['metric']] > self.best_score:
                self.patience_knt = 0
                self.best_score = val_score[self.train_config['metric']]
                test_score = self.eval(test_labels, probs[self.test_mask])
                print('Loss {:.4f}, Val AUC {:.4f}, PRC {:.4f}, RecK {:.4f}, test AUC {:.4f}, PRC {:.4f}, RecK {:.4f}'.format(
                    loss, val_score['AUROC'], val_score['AUPRC'], val_score['RecK'],
                    test_score['AUROC'], test_score['AUPRC'], test_score['RecK']))
            else:
                self.patience_knt += 1
                if self.patience_knt > self.train_config['patience']:
                    break
        return test_score


class PCGNNDetector(BaseDetector):
    def __init__(self, train_config, model_config, data):
        super().__init__(train_config, model_config, data)
        model_config['in_feats'] = self.data.graph.ndata['feature'].shape[1]
        self.model = ChebNet(**model_config).to(train_config['device'])

    def process(self, del_ratio=0.7, add_ratio=0.3, k_max=5, dist='cosine', **kwargs):
        graph = self.source_graph.long()
        features = graph.ndata['feature']
        edges = graph.adj().coalesce()
        edges_num = edges.indices().shape[1]
        dd = torch.zeros([edges_num], device=features.device)
        step = 5000000
        idx = 0
        while idx < edges_num:  # avoid OOM on large datasets
            st = idx
            idx += step
            ed = idx if idx < edges_num else edges_num
            f1 = features[edges.indices()[0, st:ed]]
            f2 = features[edges.indices()[1, st:ed]]
            dd[st:ed] = (f1 - f2).norm(1, dim=1).detach().clone()

        # the choose step: remove edges
        selected_edges = (dd).topk(int(edges_num * del_ratio)).indices.long()
        graph = dgl.remove_edges(graph, selected_edges)
        selected_nodes = (graph.ndata['label'] == 1) & (graph.ndata['train_mask'] == 1)

        # the choose step: add edges
        g_id = selected_nodes.nonzero().squeeze(-1)
        ave_degree = graph.in_degrees(g_id).float().mean() * add_ratio
        k = min(int(ave_degree), k_max) + 1

        knn_g = dgl.knn_graph(graph.ndata['feature'][selected_nodes], algorithm="bruteforce-sharemem",
                              k=k, dist=dist)
        u, v = g_id[knn_g.edges()[0]], g_id[knn_g.edges()[1]]
        graph = dgl.add_edges(graph, u.long(), v.long())
        return graph

    def train(self):
        graph = self.process(**self.model_config)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.model_config['lr'])
        train_labels, val_labels, test_labels = self.labels[self.train_mask], \
                                                self.labels[self.val_mask], self.labels[self.test_mask]
        for e in range(self.train_config['epochs']):
            self.model.train()
            logits = self.model(graph)
            loss = F.cross_entropy(logits[self.train_mask], train_labels,
                                   weight=torch.tensor([1., self.weight], device=self.labels.device), reduce=False)
            degree_weight = graph.in_degrees()
            loss = (loss * degree_weight[self.train_mask]).mean() / degree_weight.max()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if self.model_config['drop_rate'] > 0:
                self.model.eval()
                logits = self.model(self.source_graph)
            probs = logits.softmax(1)[:, 1]
            val_score = self.eval(val_labels, probs[self.val_mask])
            if val_score[self.train_config['metric']] > self.best_score:
                self.patience_knt = 0
                self.best_score = val_score[self.train_config['metric']]
                test_score = self.eval(test_labels, probs[self.test_mask])
            else:
                self.patience_knt += 1
                if self.patience_knt > self.train_config['patience']:
                    break
        return test_score


class DCIDetector(BaseDetector):
    def __init__(self, train_config, model_config, data):
        super().__init__(train_config, model_config, data)
        gnn = globals()[model_config['model']]
        print(gnn)
        model_config['in_feats'] = self.data.graph.ndata['feature'].shape[1]
        self.model = gnn(**model_config).to(train_config['device'])
        self.num_cluster = 2 if 'num_cluster' not in model_config else model_config['num_cluster']
        self.pretrain_epochs = 100 if 'pretrain_epochs' not in model_config else model_config['pretrain_epochs']

        self.kmeans = KMeans(n_clusters=self.num_cluster, random_state=0).fit(self.data.graph.ndata['feature'])
        self.ss_label = self.kmeans.labels_
        self.cluster_info = [list(np.where(self.ss_label == i)[0]) for i in range(self.num_cluster)]

    def train(self):
        train_labels, val_labels, test_labels = self.labels[self.train_mask], \
                                                self.labels[self.val_mask], self.labels[self.test_mask]

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        for e in range(1, self.pretrain_epochs):
            self.model.train()
            loss = self.model(self.source_graph, self.cluster_info, self.num_cluster)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                print(loss)
                optimizer.step()
            # re-clustering
            if e % 20 == 0:
                self.model.eval()
                emb = self.model.get_emb(self.source_graph)
                kmeans = KMeans(n_clusters=self.num_cluster, random_state=0).fit(emb.detach().cpu().numpy())
                ss_label = kmeans.labels_
                self.cluster_info = [list(np.where(ss_label == i)[0]) for i in range(self.num_cluster)]

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.model_config['lr'])
        for e in range(self.train_config['epochs']):
            self.model.train()
            logits = self.model.encoder(self.source_graph, use_mlp=True)
            loss = F.cross_entropy(logits[self.train_mask], train_labels,
                                   weight=torch.tensor([1., self.weight], device=self.labels.device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if self.model_config['drop_rate'] > 0:
                self.model.eval()
                logits = self.model.encoder(self.source_graph)
            probs = logits.softmax(1)[:, 1]
            val_score = self.eval(val_labels, probs[self.val_mask])

            if val_score[self.train_config['metric']] > self.best_score:
                self.patience_knt = 0
                self.best_score = val_score[self.train_config['metric']]
                test_score = self.eval(test_labels, probs[self.test_mask])
                print(
                    'Loss {:.4f}, Val AUC {:.4f}, PRC {:.4f}, RecK {:.4f}, test AUC {:.4f}, PRC {:.4f}, RecK {:.4f}'.format(
                        loss, val_score['AUROC'], val_score['AUPRC'], val_score['RecK'],
                        test_score['AUROC'], test_score['AUPRC'], test_score['RecK']))
            else:
                self.patience_knt += 1
                if self.patience_knt > self.train_config['patience']:
                    break
        return test_score
    

class BGNNDetector(BaseDetector):
    def __init__(self, train_config, model_config, data):
        super().__init__(train_config, model_config, data)
        # gnn = globals()[model_config['model']]
        self.depth = 6 if 'depth' not in model_config else model_config['depth']
        self.iter_per_epoch = 10 if 'iter_per_epoch' not in model_config else model_config['iter_per_epoch']
        self.gbdt_alpha = 1 if 'gbdt_alpha' not in model_config else model_config['gbdt_alpha']
        self.gbdt_lr = 0.1 if 'gbdt_lr' not in model_config else model_config['gbdt_lr']
        self.train_non_gbdt = False if 'train_non_gbdt' not in model_config else model_config['train_non_gbdt']
        self.only_gbdt = False if 'only_gdbt' not in model_config else model_config['only_gdbt']
        self.normalize_features = False if 'nomarlize_features' not in model_config else model_config['normalize_features']

        if not self.only_gbdt:
            model_config['in_feats'] = self.source_graph.ndata['feature'].shape[1] + self.labels.unique().shape[0]
        else:
            model_config['in_feats'] = self.labels.unique().size(0)

        self.model = GCN(**model_config).to(train_config['device'])
        self.gbdt_model = None
    
    def preprocess(self):
        gbdt_X_train = pd.DataFrame(self.source_graph.ndata['feature'][self.train_mask].cpu().numpy())
        gbdt_y_train = pd.DataFrame(self.labels[self.train_mask].cpu().numpy()).astype(float)

        raw_X = pd.DataFrame(self.source_graph.ndata['feature'].clone().cpu().numpy())
        encoded_X = self.source_graph.ndata['feature'].clone()
        if not self.only_gbdt and self.normalize_features:
            min_vals, _ = torch.min(encoded_X[self.train_mask], dim=0, keepdim=True)
            max_vals, _ = torch.max(encoded_X[self.train_mask], dim=0, keepdim=True)
            encoded_X[self.train_mask] = (encoded_X[self.train_mask] - min_vals) / (max_vals - min_vals)
            encoded_X[self.val_mask | self.test_mask] = (encoded_X[self.val_mask | self.test_mask] - min_vals) / (max_vals - min_vals)
            if encoded_X.isnan().any():
                row, col = torch.where(encoded_X.isnan())
                encoded_X[row, col] = self.source_graph.ndata['feature'][row, col]
            if encoded_X.isinf().any():
                row, col = torch.where(encoded_X.isinf())
                encoded_X[row, col] = self.source_graph.ndata['feature'][row, col]

        node_features = torch.empty(encoded_X.shape[0], self.model_config['in_feats'], requires_grad=True, device=self.labels.device)
        if not self.only_gbdt:
            node_features.data[:, :-2] = self.source_graph.ndata['feature'].clone()
        self.source_graph.ndata['feature'] = node_features
        return gbdt_X_train, gbdt_y_train, raw_X, encoded_X

    def train_gbdt(self, gbdt_X_train, gbdt_y_train, epoch):
        pool = Pool(gbdt_X_train, gbdt_y_train)
        if epoch == 0:
            catboost_model_obj = CatBoostClassifier
            catboost_loss_fn = 'MultiClass'
        else:
            catboost_model_obj = CatBoostRegressor
            catboost_loss_fn = 'MultiRMSE'
        
        epoch_gbdt_model = catboost_model_obj(iterations=self.iter_per_epoch,
                                              depth=self.depth,
                                              learning_rate=self.gbdt_lr,
                                              loss_function=catboost_loss_fn,
                                              random_seed=0,
                                              nan_mode='Min')
        epoch_gbdt_model.fit(pool, verbose=False)
        
        if epoch == 0:
            self.base_gbdt = epoch_gbdt_model
        else:
            if self.gbdt_model is None:
                self.gbdt_model = epoch_gbdt_model
            else:
                self.gbdt_model = sum_models([self.gbdt_model, epoch_gbdt_model], weights=[1, self.gbdt_alpha])
                # self.gbdt_model = self.append_gbdt_model(epoch_gbdt_model, weights=[1, self.gbdt_alpha])

    def update_node_features(self, X, encoded_X):
        predictions = self.base_gbdt.predict_proba(X)
        # predictions = self.base_gbdt.predict(X, prediction_type='RawFormulaVal')
        if self.gbdt_model is not None:
            predictions_after_one = self.gbdt_model.predict(X)
            predictions += predictions_after_one

        predictions = torch.tensor(predictions, device=self.labels.device)
        node_features = self.source_graph.ndata['feature']
        if not self.only_gbdt:
            if self.train_non_gbdt:
                predictions = torch.concat((node_features.detach().data[:, :-2], predictions), dim=1)
            else:
                predictions = torch.concat((encoded_X, predictions), dim=1)
        node_features.data = predictions.float().data

    def train(self):
        gbdt_X_train, gbdt_y_train, raw_X, encoded_X = self.preprocess()
        optimizer = torch.optim.Adam(
            itertools.chain(*[self.model.parameters(), [self.source_graph.ndata['feature']]]), lr=self.model_config['lr']
        )
        train_labels, val_labels, test_labels = self.labels[self.train_mask], \
                                                self.labels[self.val_mask], self.labels[self.test_mask]

        for e in range(self.train_config['epochs']):
            self.train_gbdt(gbdt_X_train, gbdt_y_train, e)
            self.update_node_features(raw_X, encoded_X)
            node_features_before = self.source_graph.ndata['feature'].clone()
            
            self.model.train()
            for _ in range(self.iter_per_epoch):
                logits = self.model(self.source_graph)
                loss = F.cross_entropy(logits[self.train_mask], train_labels,
                                   weight=torch.tensor([1., self.weight], device=self.labels.device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            self.model.eval()
            logits = self.model(self.source_graph)
            probs = logits.softmax(1)[:, 1]
            val_score = self.eval(val_labels, probs[self.val_mask])
            if val_score[self.train_config['metric']] > self.best_score:
                self.patience_knt = 0
                self.best_score = val_score[self.train_config['metric']]
                test_score = self.eval(test_labels, probs[self.test_mask])
                print('Loss {:.4f}, Val AUC {:.4f}, PRC {:.4f}, RecK {:.4f}, test AUC {:.4f}, PRC {:.4f}, RecK {:.4f}'.format(
                    loss, val_score['AUROC'], val_score['AUPRC'], val_score['RecK'],
                    test_score['AUROC'], test_score['AUPRC'], test_score['RecK']))
            else:
                self.patience_knt += 1
                if self.patience_knt > self.train_config['patience']:
                    break
            
            # Update GBDT target
            gbdt_y_train = (self.source_graph.ndata['feature'] - node_features_before)[self.train_mask, -2:].detach().cpu().numpy()
            
            # Check if update is frozen
            if np.isclose(gbdt_y_train.sum(), 0.):
                print('Nodes do not change anymore. Stopping...')
                break
        return test_score


class H2FDetector(BaseDetector):
    def __init__(self, train_config, model_config, data):
        super().__init__(train_config, model_config, data)
        model_config['in_feats'] = self.data.graph.ndata['feature'].shape[1]

        g = self.source_graph
        canon = g.canonical_etypes
        new_g = dgl.heterograph({
            canon[0]: g[canon[0]].edges(),
            canon[1]: g[canon[1]].edges(),
            canon[2]: g[canon[2]].edges(),
            (canon[0][0], 'homo', canon[0][0]): dgl.to_homogeneous(g).edges()
            })
        homo_edges = new_g.edges(etype='homo')
        for feat in g.ndata:
            new_g.ndata[feat] = g.ndata[feat].clone()
        
        homo_labels, homo_train_mask = self.generate_edges_labels(homo_edges, new_g.ndata['label'].cpu().tolist(), new_g.ndata['train_mask'].nonzero().squeeze(1).tolist())

        new_g.edges['homo'].data['label'] = homo_labels.cuda()
        new_g.edges['homo'].data['train_mask'] = homo_train_mask.cuda()
        for ntype in g.ntypes:
            for key in g.ndata.keys():
                new_g.nodes[ntype].data[key] = g.nodes[ntype].data[key].clone()
        # dgl.save_graphs('new_amazon', [new_g])
        # new_g = dgl.load_graphs('new_amazon')[0][0].to(train_config['device'])
        self.source_graph = new_g
        model_config['graph'] = self.source_graph
        self.model = H2FD(**model_config).to(train_config['device'])

    def generate_edges_labels(self, edges, labels, train_idx):
        row, col = edges[0].cpu(), edges[1].cpu()
        edge_labels = []
        edge_train_mask = []
        for i, j in zip(row, col):
            i = i.item()
            j = j.item()
            if labels[i] == labels[j]:
                edge_labels.append(1)
            else:
                edge_labels.append(-1)
            if i in train_idx and j in train_idx:
                edge_train_mask.append(1)
            else:
                edge_train_mask.append(0)
        edge_labels = torch.Tensor(edge_labels).long()
        edge_train_mask = torch.Tensor(edge_train_mask).bool()
        return edge_labels, edge_train_mask

    def train(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.model_config['lr'])
        train_labels, val_labels, test_labels = self.labels[self.train_mask], \
                                                self.labels[self.val_mask], self.labels[self.test_mask]
        for e in range(self.train_config['epochs']):
            self.model.train()
            loss, logits = self.model(self.source_graph)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            probs = logits.softmax(1)[:, 1]
            val_score = self.eval(val_labels, probs[self.val_graph.ndata['val_mask']])
            if val_score[self.train_config['metric']] > self.best_score:
                if self.train_config['inductive']:
                    loss, logits = self.model(self.source_graph)
                    probs = logits.softmax(1)[:, 1]
                self.patience_knt = 0
                self.best_score = val_score[self.train_config['metric']]
                test_score = self.eval(test_labels, probs[self.test_mask])
                print('Epoch {}, Loss {:.4f}, Val AUC {:.4f}, PRC {:.4f}, RecK {:.4f}, test AUC {:.4f}, PRC {:.4f}, RecK {:.4f}'.format(
                    e, loss, val_score['AUROC'], val_score['AUPRC'], val_score['RecK'],
                    test_score['AUROC'], test_score['AUPRC'], test_score['RecK']))
            else:
                self.patience_knt += 1
                if self.patience_knt > self.train_config['patience']:
                    break
        return test_score


def infinite_iter(data_list: Iterable):
    it = iter(data_list)
    while True:
        try:
            yield next(it)
        except StopIteration:
            it = iter(data_list)


class GAGADetector(BaseDetector):
    def __init__(self, train_config, model_config, data, cache=True, cache_dir="./gaga_%s.npz"):
        super().__init__(train_config, model_config, data)
        gnn = GAGA
        model_config['in_feats'] = self.data.graph.ndata['feature'].shape[1]
        model_config['num_edge_types'] = len(data.graph.etypes)
        device = train_config['device']
        self.model = gnn(**model_config).to(device)

        dataset_name = data.name
        etypes = data.graph.etypes
        cache_name = cache_dir % (dataset_name.replace("/", "_"))

        if not cache or not os.path.exists(cache_name):  # FIXME random mask may cause recall=0
            self.masked_train_label = torch.clone(self.labels)  # label 0 = benign, 1 = fraud, 2 = masked
            self.masked_train_label[~self.train_mask.bool()] = 2
            while True:  # Loop until each type of label will be sampled
                mask_rate = 0.8
                random_mask = torch.rand(self.train_mask.sum()) < mask_rate
                label_count = (torch.sum(self.masked_train_label[self.train_mask][~random_mask] == 0).item(), torch.sum(self.masked_train_label[self.train_mask][~random_mask] == 1).item())
                label_balance = 1 - (abs(label_count[0] - label_count[1]) / (label_count[0] + label_count[1] + 1e-8))
                dataset_count = (torch.sum(self.masked_train_label[self.train_mask] == 0).item(), torch.sum(self.masked_train_label[self.train_mask] == 1).item())
                dataset_balance = 1 - (abs(dataset_count[0] - dataset_count[1]) / (dataset_count[0] + dataset_count[1] + 1e-8))
                if label_count[0] != 0 and label_count[1] != 0 and label_balance > dataset_balance:  # Ensure both labels are present and balanced
                    break
            print("Label balance: ", label_balance, "Dataset balance:", dataset_balance, "Label count: ", label_count)
            self.masked_train_label[self.train_mask][random_mask] = 2

            self.sampled_train_feature = self.model.pre_feature_sample(self.train_graph, self.masked_train_label)
            self.sampled_val_feature = self.model.pre_feature_sample(self.val_graph, self.masked_train_label)
            self.sampled_test_feature = self.model.pre_feature_sample(self.source_graph, self.masked_train_label)
        else:  # cache == True and cache file exists
            data = np.load(cache_name)
            self.masked_train_label = torch.from_numpy(data['train_label']).to(device)

            train_feature = torch.from_numpy(data['train_feature']).to(device)
            val_feature = torch.from_numpy(data['val_feature']).to(device)
            test_feature = torch.from_numpy(data['test_feature']).to(device)

            self.sampled_train_feature = {etypes[i]: train_feature[i] for i in range(len(etypes))}
            self.sampled_val_feature = {etypes[i]: val_feature[i] for i in range(len(etypes))}
            self.sampled_test_feature = {etypes[i]: test_feature[i] for i in range(len(etypes))}

            # if 'train_mask' in data and 'val_mask' in data and 'test_mask' in data:
            #     self.train_mask = torch.from_numpy(data['train_mask'])
            #     self.val_mask = torch.from_numpy(data['val_mask'])
            #     self.test_mask = torch.from_numpy(data['test_mask'])

        if cache and not os.path.exists(cache_name):
            np.savez(cache_name,
                     train_label=self.masked_train_label.detach().cpu().numpy(),
                     train_feature=torch.stack([self.sampled_train_feature[k] for k in etypes], dim=0).detach().cpu().numpy(),
                     val_feature=torch.stack([self.sampled_val_feature[k] for k in etypes], dim=0).detach().cpu().numpy(),
                     test_feature=torch.stack([self.sampled_val_feature[k] for k in etypes], dim=0).detach().cpu().numpy()
                     )

        batch_size = model_config.get('batch_size', 256)
        self.weight_decay = model_config.get('weight_decay', 1e-4)

        data_length = list(self.sampled_train_feature.items())[0][1].shape[0]
        self.train_loader = infinite_iter(DataLoader(torch.arange(data_length)[self.train_mask.cpu()], batch_size = batch_size, shuffle=True, drop_last=False, num_workers=8))
        self.val_loader = infinite_iter(DataLoader(torch.arange(data_length)[self.val_mask.cpu()], batch_size = batch_size, shuffle=True, drop_last=False, num_workers=8))
        self.test_loader = infinite_iter(DataLoader(torch.arange(data_length)[self.test_mask.cpu()], batch_size = batch_size, shuffle=True, drop_last=False, num_workers=8))

    def get_data(self, type='train'):
        assert type in ['train', 'val', 'test']
        loader = {
            'train': self.train_loader,
            'val': self.val_loader,
            'test': self.test_loader,
        }[type]
        features = {
            'train': self.sampled_train_feature,
            'val': self.sampled_val_feature,
            'test': self.sampled_test_feature,
        }[type]

        data_idx = next(loader)
        res = dict()
        for k in features.keys():
            res[k] = features[k][data_idx]
        return res, data_idx

    def train(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.model_config['lr'], weight_decay=self.weight_decay)
        train_labels, val_labels, test_labels = self.labels[self.train_mask], self.labels[self.val_mask], self.labels[self.test_mask]

        for e in range(self.train_config['epochs']):
            self.model.train()
            data, idx = self.get_data('train')
            logits = self.model(self.train_graph, data)
            loss = F.cross_entropy(logits, self.masked_train_label[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if self.model_config['drop_rate'] > 0 or self.train_config['inductive']:
                self.model.eval()
                data, idx = self.get_data('val')
                logits = self.model(self.val_graph, data)
            probs = logits[:, 0]
            try:
                val_score = self.eval(self.labels[idx], probs)
            except ValueError:
                print("ValueError: ", traceback.format_exc())
            if val_score[self.train_config['metric']] > self.best_score:
                if self.train_config['inductive']:
                    data, idx = self.get_data('test')
                    logits = self.model(self.source_graph, data)
                    probs = logits.softmax(1)[:, 1]
                self.patience_knt = 0
                self.best_score = val_score[self.train_config['metric']]
                test_score = self.eval(self.labels[idx], probs)
                print('Epoch {}, Loss {:.4f}, Val AUC {:.4f}, PRC {:.4f}, RecK {:.4f}, test AUC {:.4f}, PRC {:.4f}, RecK {:.4f}'.format(
                    e, loss, val_score['AUROC'], val_score['AUPRC'], val_score['RecK'],
                    test_score['AUROC'], test_score['AUPRC'], test_score['RecK']))
            else:
                self.patience_knt += 1
                if self.patience_knt > self.train_config['patience']:
                    break
        return test_score

class ConsisGADDetector(BaseDetector):
    def __init__(self, train_config, model_config, data):
        super().__init__(train_config, model_config, data)
        model_config['in_feats'] = self.data.graph.ndata['feature'].shape[1]

        model_args = {
            'to-homo': False,
            'shuffle-train': True,
            'hidden-dim': 64,
            'num-layers': 1,
            'weight-decay': 0.00001,
            'training-ratio': 1,
            'train-procedure': 'CT',
            'mlp-drop': 0.4,
            'input-drop': 0.0,
            'hidden-drop': 0.0,
            'mlp12-dim': 128,
            'mlp3-dim': 128,
            'bn-type': 2,
            'optim': 'adam',
            'store-model': True,
            'trainable-consis-weight': 1.5,
            'trainable-temp': 0.0001,
            'trainable-eps': 0.000000000001,
            'trainable-drop-rate': 0.2,
            'trainable-warm-up': -1,
            'trainable-model': 'mlp',
            'trainable-optim': 'adam',
            'trainable-lr': 0.005,
            'trainable-weight-decay': 0.00001,
            'topk-mode': 4,
            'diversity-type': 'cos',
            'unlabel-ratio': 4,
            'normal-th': 7,
            'fraud-th': 88,
            'trainable-detach-y': True,
            'trainable-div-eps': True,
            'trainable-detach-mask': False,
            'batch-size': 128,
            'train-iterations': 10
        }

        model_args['device'] = 'cuda:0'
        model_args.update(model_config)

        print("[INFO]", "current model args =", model_args)

        self.train_nids = torch.nonzero(self.data.graph.ndata['train_mask']).squeeze()
        self.valid_nids = torch.nonzero(self.data.graph.ndata['val_mask']).squeeze()
        self.test_nids = torch.nonzero(self.data.graph.ndata['test_mask']).squeeze()

        self.labeled_nids = self.train_nids
        self.unlabeled_nids = torch.concatenate([self.train_nids, self.valid_nids, self.test_nids])

        power = 10 if data.name == 'tfinance' else 16
        self.valid_loader = DataLoader(self.valid_nids, batch_size = 2 ** power, shuffle=False, drop_last=False, num_workers=4)
        self.test_loader = DataLoader(self.test_nids, batch_size = 2 ** power, shuffle=False, drop_last=False, num_workers=4)
        self.labeled_loader = DataLoader(self.labeled_nids, batch_size = model_args['batch-size'], shuffle=model_args['shuffle-train'], drop_last=True, num_workers=0)
        self.unlabeled_loader = DataLoader(self.labeled_nids, batch_size = model_args['batch-size'] * model_args['unlabel-ratio'], shuffle=model_args['shuffle-train'], drop_last=True, num_workers=0)

        self.model = ConsisGADGNN(
            model_config['in_feats'],
            64,
            2,
            self.source_graph.etypes,
            128,
            128,
            model_args['input-drop'],
            model_args['hidden-drop'],
            model_args['mlp-drop'],
            model_args['num-layers']
        ).to(train_config['device'])

        if model_args['optim'] == 'rmpprop':
            self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=model_args['lr'])
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=model_args['lr'], weight_decay=0.0)
        self.sampler = dgl.dataloading.MultiLayerFullNeighborSampler(model_args['num-layers'])
        self.argumentator = LearnableDataArugmentation(
            self.model,
            drop_rate=model_args['trainable-drop-rate'],
            lr=model_args['trainable-lr'],
            eps=model_args['trainable-eps'],
            temp=model_args['trainable-temp'],
            weight_decay=model_args['trainable-weight-decay'],
        )

        self.task_loss = ConsisCombinedLoss()

        self.args = model_args
    
    @staticmethod
    def sample_blocks(graph: dgl.DGLGraph, seed_nodes: torch.Tensor, sampler: dgl.dataloading.MultiLayerFullNeighborSampler):
        seed_nodes = seed_nodes.to(graph.idtype)
        input_nodes, output_nodes, blocks = sampler.sample_blocks(graph, seed_nodes)
        return input_nodes, output_nodes, blocks

    @staticmethod
    def UDA_train_epoch(epoch, model, arugmentator, graph, loader, optimizer, sampler, task_loss, args):
        model.train()
        num_iters = args['train-iterations']
        device = "cuda:0"
        
        label_loader, unlabel_loader = loader
        unlabel_loader_iter = infinite_iter(unlabel_loader)
        label_loader_iter = infinite_iter(label_loader)

        for idx in range(num_iters):
            label_idx = next(label_loader_iter)
            unlabel_idx = next(unlabel_loader_iter)
            assert label_idx is not None and unlabel_idx is not None

            if epoch > args['trainable-warm-up']:
                _, u, u_blocks = ConsisGADDetector.sample_blocks(graph, unlabel_idx.to(device), sampler)
                p_h_u, y_w_u = arugmentator(u_blocks)
            else:
                p_h_u, y_w_u = torch.tensor(1.0, requires_grad=False), torch.tensor(1.0, requires_grad=False)

            _, _, s_blocks = ConsisGADDetector.sample_blocks(graph, label_idx.to(device), sampler)
            p_v = model(s_blocks)
            y_v = s_blocks[-1].dstdata['label']

            loss = task_loss(p_v, y_v, y_w_u, p_h_u) + args['weight-decay'] * l2_regularization(model)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        return p_v
    
    @staticmethod
    def get_model_pred(model, graph, data_loader, sampler, args):
        model.eval()

        pred_list = []
        target_list = []
        with torch.no_grad():
            for index, node_index in enumerate(data_loader):  # 1/4s pre 100 samples
                _, _, blocks = sampler.sample_blocks(graph, node_index.to(args['device']).to(graph.idtype))
                pred = model(blocks)
                target = blocks[-1].dstdata['label']

                pred.detach()
                pred[pred.isnan()] = 0

                pred_list.append(pred)
                target_list.append(target.detach())

            pred_list = torch.cat(pred_list, dim=0)
            target_list = torch.cat(target_list, dim=0)
            pred_list = pred_list.exp()[:, 1]
        
        return pred_list, target_list 

    def train(self):
        # train_labels, val_labels, test_labels = self.labels[self.train_mask], \
        #                                         self.labels[self.val_mask], self.labels[self.test_mask]
        for epoch in range(self.train_config['epochs']):
            # print("[INFO]", f"current epoch={epoch}")
            self.UDA_train_epoch(
                epoch, self.model, self.argumentator, self.source_graph,
                (self.labeled_loader, self.unlabeled_loader),
                self.optimizer, self.sampler, self.task_loss, self.args
            )
            
            val_probs, val_labels = self.get_model_pred(self.model, self.source_graph, self.valid_loader, self.sampler, self.args)
            test_probs, test_labels = self.get_model_pred(self.model, self.source_graph, self.test_loader, self.sampler, self.args)

            val_results, test_results = \
                self.eval(val_labels, val_probs), self.eval(test_labels, test_probs)

            # print("[DEBUG]", "current evaluate result =", val_results, test_results)
            
            if val_results[self.train_config['metric']] > self.best_score:
                self.best_score = val_results[self.train_config['metric']]
                test_in_best_val = test_results
                print("[RES]", test_results, "@", epoch)

                if self.args['store-model']:
                    ...  # TODO

        return test_in_best_val
    
class DGAGNNDetector(BaseGNNDetector):
    def __init__(self, train_config, model_config, data):
        graph: dgl.DGLGraph = data.graph
        num_nodes = graph.num_nodes()
        num_etypes = len(graph.etypes)
        model_config['n_nodes'] = num_nodes
        model_config['n_etypes'] = num_etypes
        super().__init__(train_config, model_config, data)

class SECGFDDetector(BaseGNNDetector):
    def __init__(self, train_config, model_config, data):
        graph: dgl.DGLGraph = data.graph
        model_config['graph'] = graph.to(train_config['device'])
        super().__init__(train_config, model_config, data)
