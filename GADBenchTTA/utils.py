import random
from models.detector import *
from dgl.data.utils import load_graphs
import os
import json
import torch
import pandas
from typing import Callable
from collections import defaultdict, Counter


class Dataset:
    def __init__(self, name='tfinance', prefix='datasets/', norm_feat=True):
        graph = load_graphs(prefix + name)[0][0]
        self.name = name

        if graph.ndata['label'].dtype != torch.int64:
            graph.ndata['label'] = graph.ndata['label'].to(torch.int64)
        # if graph.ndata['feature'].dtype != torch.float64:
        #     graph.ndata['feature'] = graph.ndata['feature'].to(torch.float64)

        if norm_feat:
            graph.ndata['feature'] = (graph.ndata['feature'] - graph.ndata['feature'].mean(0)) / graph.ndata['feature'].std(0)

        self.graph: dgl.DGLGraph = graph
        self.avail_groups = [i for i, _ in Counter([i.item() for i in self.graph.ndata['group']]).most_common() if i != 0] if 'group' in self.graph.ndata else []

    def iter_group(self, topk=5):
        for group in (self.avail_groups if topk is None else self.avail_groups[:topk]):
            # if group == 0:  # no label group
            #     continue
            self.set_group(group)
            yield group

    def reset_group(self):
        self.set_group(None)

    def set_group(self, group: int | None = None):
        if group is not None and group not in self.avail_groups:
            raise ValueError(f"Group {group} not in available groups {self.avail_groups}")
        if group is None:
            self.graph = self.graph_og
            return
        if 'graph_og' not in self.__dict__:
            self.graph_og: dgl.DGLGraph = self.graph
        black_mask = self.graph_og.ndata['group'] == group
        adj_csc = self.graph_og.adj().csc()  # (col_ptr, row_ind, val_ind)
        adj_csr = self.graph_og.adj().csr()  # (row_ptr, col_ind, val_ind)
        
        neighbors = torch.unique(torch.cat([
            adj_csc[1][adj_csc[0][torch.where(black_mask)[0]]], 
            adj_csr[1][adj_csr[0][torch.where(black_mask)[0]]],
            torch.where(black_mask)[0]
        ], dim=0))
        self.graph = self.graph_og.subgraph(neighbors)

    def split(self, semi_supervised=True, trial_id=0):
        if semi_supervised:
            trial_id += 10
        self.graph.ndata['train_mask'] = self.graph.ndata['train_masks'][:,trial_id]
        self.graph.ndata['val_mask'] = self.graph.ndata['val_masks'][:,trial_id]
        self.graph.ndata['test_mask'] = self.graph.ndata['test_masks'][:,trial_id]
        print(self.graph.ndata['train_mask'].sum(), self.graph.ndata['val_mask'].sum(), self.graph.ndata['test_mask'].sum())


class MotifsDataset(Dataset):
    def __init__(self, name='tfinance', prefix='datasets/', motifs_file_rule: Callable | None = None):
        super().__init__(name, prefix, norm_feat=True)
        if motifs_file_rule is None:
            motifs_file_rule = lambda x: ('.'.join(x.split('.')[:-1]) if '.' in x else x) + '.rust_motifs.npy'
        self.motifs_file_name = motifs_file_rule(name)
        self.motifs = torch.from_numpy(np.load(prefix + self.motifs_file_name).astype(np.int32))
        self.graph.ndata['motif'] = self.motifs


model_detector_dict = {
    # Classic Methods
    'MLP': BaseGNNDetector,
    'KNN': KNNDetector,
    'SVM': SVMDetector,
    'RF': RFDetector,
    'XGBoost': XGBoostDetector,
    'XGBOD': XGBODDetector,
    'NA': XGBNADetector,

    # Standard GNNs
    'GCN': BaseGNNDetector,
    'SGC': BaseGNNDetector,
    'GIN': BaseGNNDetector,
    'GraphSAGE': BaseGNNDetector,
    'GAT': BaseGNNDetector,
    'GT': BaseGNNDetector,
    'PNA': BaseGNNDetector,
    'BGNN': BGNNDetector,

    # Specialized GNNs
    'GAS': GASDetector,
    'BernNet': BaseGNNDetector,
    'AMNet': BaseGNNDetector,
    'BWGNN': BaseGNNDetector,
    'GHRN': GHRNDetector,
    'GATSep': BaseGNNDetector,
    'PCGNN': PCGNNDetector,
    'DCI': DCIDetector,

    # Heterogeneous GNNs
    'RGCN': HeteroGNNDetector,
    'HGT': HeteroGNNDetector,
    'CAREGNN': CAREGNNDetector,
    'H2FD': H2FDetector, 

    # Tree Ensembles with Neighbor Aggregation
    'RFGraph': RFGraphDetector,
    'XGBGraph': XGBGraphDetector,
    
    # Custom Methods
    'MySAGE': BaseGNNDetector,
    'GAGA': GAGADetector,
    'ConsisGAD': ConsisGADDetector,
    'KYCGCN': BaseGNNDetector,
    'SpaceGNN': BaseGNNDetector,
    'DGAGNN': DGAGNNDetector,
    'SECGFD': SECGFDDetector,
    'ARC': BaseGNNDetector,

    # Motifs Enhenced
    'GCN_M': BaseGNNDetector,
    'GAT_M': BaseGNNDetector,
    'GraphSAGE_M': BaseGNNDetector,
    # 'SpaceGNN_M': BaseGNNDetector,
    # 'ARC_M': BaseGNNDetector,
    'DGAGNN_M': DGAGNNDetector,
    
    # Extened SAGEs
    'GraphSAGEMean': BaseGNNDetector,
    'GraphSAGEPool': BaseGNNDetector,
    'GraphSAGELSTM': BaseGNNDetector,
    'GraphSAGEGCN': BaseGNNDetector,
    'MySAGEMean': BaseGNNDetector,
    'MySAGEPool': BaseGNNDetector,
    'MySAGELSTM': BaseGNNDetector,
    'MySAGEGCN': BaseGNNDetector,
}

def get_model_feature(model: torch.nn.Module):
    features = {}
    for name, params in model.named_parameters():
        features[name] = params.mean().item(), params.std().item()
    return features

def compare_model_feature(model1_feature: dict, model2_feature: dict | torch.nn.Module, eps=1e-6, verbose=False):
    if isinstance(model2_feature, torch.nn.Module):
        model2_feature = get_model_feature(model2_feature)
    any_fail = False
    for name in model1_feature.keys():
        if name not in model2_feature.keys():
            any_fail = True
            if verbose:
                print(f'Warning: {name} not found in model2_feature')
                break
            continue
        mean1, std1 = model1_feature[name]
        mean2, std2 = model2_feature[name]
        if abs(mean1 - mean2) > eps or abs(std1 - std2) > eps:
            any_fail = True
            if verbose:
                print(f'Warning: {name} mean or std differs by more than {eps}: mean1={mean1:.4f}, std1={std1:.4f}, mean2={mean2:.4f}, std2={std2:.4f}')
                break
    return not any_fail

def save_results(results, file_id):
    if not os.path.exists('results/'):
        os.mkdir('results/')
    if file_id is None:
        file_id = 0
        while os.path.exists('results/{}.xlsx'.format(file_id)):
            file_id += 1
    results.transpose().to_excel('results/{}.xlsx'.format(file_id))
    print('save to file ID: {}'.format(file_id))
    return file_id

def better_save_results(results, file_id):
    if not os.path.exists('results/'):
        os.mkdir('results/')
    if file_id is None:
        file_id = 0
        while os.path.exists('results/{}.xlsx'.format(file_id)):
            file_id += 1
    
    # Create Excel writer with multiple sheets
    with pandas.ExcelWriter(f'results/{file_id}.xlsx', engine='openpyxl') as writer:
        # Sheet 0 - All results (same as original function)
        results.transpose().to_excel(writer, sheet_name='All Results')
        
        # Extract model names and dataset names
        models = results['name'].tolist()
        datasets = []
        for col in results.columns:
            if '-AUROC mean' in col:
                datasets.append(col.split('-AUROC mean')[0])
        
        # Create separate sheets for each metric with datasets as rows and methods as columns
        for metric in ['AUROC', 'AUPRC', 'RecK', 'F1', 'ACC']:
            metric_df = pandas.DataFrame(index=datasets, columns=models)
            for dataset in datasets:
                for i, model in enumerate(models):
                    metric_df.loc[dataset, model] = results.iloc[i][f'{dataset}-{metric} mean']
            metric_df.to_excel(writer, sheet_name=f'{metric} Mean')
    
    # Create markdown table
    md_content = "# Results\n\n"
    md_content += "| Dataset | Metric |"
    for model in models:
        md_content += f" {model} |"
    md_content += "\n"
    
    # Header separator
    md_content += "| --- | --- |"
    for _ in models:
        md_content += " --- |"
    md_content += "\n"
    
    # Content rows
    for dataset in datasets:
        metrics = [
            ("AUROC", "AUROC"),
            ("AUPRC", "AUPRC"),
            ("RecK", "RecK"),
            ("F1", "F1-score"), 
            ("ACC", "ACC"), 
        ]
        
        for idx, (metric_key, metric_display) in enumerate(metrics):
            # First column: dataset only on first row
            if idx == 0:
                md_content += f"| {dataset} | {metric_display} |"
            else:
                md_content += f"| | {metric_display} |"
        
            for i, model in enumerate(models):
                mean_col = f'{dataset}-{metric_key} mean'
                std_col = f'{dataset}-{metric_key} std'
        
                try:
                    mean = results.iloc[i][mean_col]
                    std = results.iloc[i][std_col]
                    if not pandas.isna(mean) and not pandas.isna(std):
                        md_content += f" {mean:.4f} ±{std:.4f} |"
                    else:
                        md_content += " N/A |"
                except (KeyError, IndexError):
                    md_content += " N/A |"
        
            md_content += "\n"
    
    # Save markdown to file
    with open(f'results/{file_id}.md', 'w') as f:
        f.write(md_content)
        
    print(f'Results saved to file ID: {file_id} (Excel and Markdown)')
    return file_id

def save_tta_results(results: pandas.DataFrame, file_id):
    if not os.path.exists(f'results/{file_id}.xlsx'):
        raise FileNotFoundError(f'File ID {file_id} not found.')
    tta_methods = defaultdict(set)  # method: tta-ed datasets
    trained_datasets = set()
    for key in results.columns:
        if not key.endswith('mean'):
            continue
        if not key.startswith('TTA-'):
            source_dataset, *_ = key.split('-')
            trained_datasets.add(source_dataset)
            continue
        _, method, source_dataset, target_dataset, *_ = key.split('-')
        tta_methods[method].add((source_dataset, target_dataset))
    
    with pandas.ExcelWriter(f'results/{file_id}_tta.xlsx', engine='openpyxl') as writer:
        for model in results['name'].to_list():
            for metric in ['AUROC', 'AUPRC', 'RecK']:
                for method, tta_datasets in tta_methods.items():
                    target_datasets = {target_dataset for source_dataset, target_dataset in tta_datasets if source_dataset in trained_datasets}
                    tta_df = pandas.DataFrame(index=sorted(list(target_datasets)), columns=sorted(list(trained_datasets)))
                    for dataset in target_datasets:
                        for source_dataset in trained_datasets:
                            if source_dataset == dataset:
                                key = f'{dataset}-{metric}'
                            else:
                                key = f'TTA-{method}-{source_dataset}-{dataset}-{metric}'
                            try:
                                result_mean = results.loc[results['name']==model, f'{key} mean']
                                # print(f"[DEBUG] result_mean {result_mean} {type(result_mean)}")
                                if result_mean is None or (isinstance(result_mean, pandas.Series) and pd.isna(result_mean).any()) or (isinstance(result_mean, np.ndarray) and np.isnan(result_mean).any()):
                                    result_mean = 0.0
                                else:
                                    result_mean = result_mean.item()
                            except KeyError:
                                print(f"key {key} not found")
                                result_mean = 0.0
                            try:
                                result_std = results.loc[results['name']==model, f'{key} std']
                                # print(f"[DEBUG] result_std {result_std} {type(result_std)}")
                                if result_std is None or (isinstance(result_std, pandas.Series) and pd.isna(result_std).any()) or (isinstance(result_std, np.ndarray) and np.isnan(result_std).any()):
                                    result_std = 0.0
                                else:
                                    result_std = result_std.item()
                            except KeyError:
                                result_std = 0.0
                            # tta_df.loc[dataset, source_dataset] = f"{result_mean:.4f} ±{result_std:.4f}"
                            tta_df.loc[dataset, source_dataset] = result_mean
                    tta_df.to_excel(writer, sheet_name=f'{model}|{method}|{metric}')

def sample_param(model, dataset, t=0):
    model_config = {'model': model, 'lr': 0.01, 'drop_rate': 0}
    if t == 0:
        return model_config
    for k, v in param_space[model].items():
        model_config[k] = random.choice(v)
    # Avoid OOM in Random Search
    if model in ['GAT', 'GATSep', 'GT'] and dataset in ['tfinance', 'dgraphfin', 'tsocial']:
        model_config['h_feats'] = 16
        model_config['num_heads'] = 2
    if dataset == 'tsocial':
        model_config['h_feats'] = 16
    if dataset in ['dgraphfin', 'tsocial']:
        if 'k' in model_config:
            model_config['k'] = min(5, model_config['k'])
        if 'num_cluster' in model_config:
            model_config['num_cluster'] = 2
        # if 'num_layers' in model_config:
        #     model_config['num_layers'] = min(2, model_config['num_layers'])
    return model_config


def sample_trial_param(trial, model, dataset, t=0):
    model_config = {'model': model, 'lr': 0.01, 'drop_rate': 0}
    if t == 0:
        return model_config
    for k, v in trial_space[model].items():
        choice_method = v[0]
        choice_args = v[1] if len(v) >= 2 else tuple()
        choice_kwargs = v[2] if len(v) >= 3 else dict()
        model_config[k] = eval(f"trial.suggest_{choice_method}(k, *choice_args, **choice_kwargs)")
    # Avoid OOM in Random Search
    if model in ['GAT', 'GATSep', 'GT'] and dataset in ['tfinance', 'dgraphfin', 'tsocial']:
        model_config['h_feats'] = 16
        model_config['num_heads'] = 2
    if dataset == 'tsocial':
        model_config['h_feats'] = 16
    if dataset in ['dgraphfin', 'tsocial']:
        # if 'k' in model_config:
        #     model_config['k'] = min(5, model_config['k'])
        if 'num_cluster' in model_config:
            model_config['num_cluster'] = 2
    return model_config


param_space = {}
trial_space = {}

param_space['MLP'] = {
    'h_feats': [16, 32, 64],
    'num_layers': [1, 2, 3, 4],
    'drop_rate': [0, 0.1, 0.2, 0.3],
    'lr': 10 ** np.linspace(-3, -1, 1000),
    'activation': ['ReLU', 'LeakyReLU', 'Tanh']
}

param_space['GCN'] = {
    'h_feats': [16, 32, 64],
    'num_layers': [1, 2, 3],
    'drop_rate': [0, 0.1, 0.2, 0.3],
    'lr': 10 ** np.linspace(-3, -1, 1000),
    'activation': ['ReLU', 'LeakyReLU', 'Tanh']
}

param_space['SGC'] = {
    'h_feats': [16, 32, 64],
    'k': [1, 2, 3, 4],
    'drop_rate': [0, 0.1, 0.2, 0.3],
    'mlp_layers': [1, 2],
    'lr': 10 ** np.linspace(-3, -1, 1000),
}

param_space['GIN'] = {
    'h_feats': [16, 32, 64],
    'num_layers': [1, 2, 3],
    'agg': ['sum', 'max', 'mean'],
    'drop_rate': [0, 0.1, 0.2, 0.3],
    'lr': 10 ** np.linspace(-3, -1, 1000),
    'activation': ['ReLU', 'LeakyReLU', 'Tanh']
}

param_space['GraphSAGE'] = {
    'h_feats': [16, 32, 64],
    'num_layers': [1, 2, 3],
    'agg': ['mean', 'gcn', 'pool'],
    'drop_rate': [0, 0.1, 0.2, 0.3],
    'lr': 10 ** np.linspace(-3, -1, 1000),
    'activation': ['ReLU', 'LeakyReLU', 'Tanh']
}
trial_space['GraphSAGE'] = {
    'h_feats': ('categorical', ([16, 32, 64], )),
    'num_layers': ('int', (1, 3)),
    'agg': ('categorical', (['mean', 'gcn', 'pool'], )),
    'drop_rate': ('categorical', ([0, 0.1, 0.2, 0.3], )),
    'lr': ("float", (10e-3, 10e-1)),
    'activation': ('categorical', (['ReLU', 'LeakyReLU', 'Tanh'], ))
}

param_space['MySAGE'] = {
    'h_feats': [16, 32, 64],
    'num_layers': [1, 2, 3],
    'agg': ['mean', 'gcn', 'pool'],
    'dropout': [0, 0.1, 0.2, 0.3],
    'lr': 10 ** np.linspace(-3, -1, 1000),
    'activation': ['ReLU', 'LeakyReLU', 'Tanh']
}
trial_space['MySAGE'] = {
    'h_feats': ('categorical', ([16, 32, 64], )),
    'num_layers': ('int', (1, 3)),
    'agg': ('categorical', (['mean', 'gcn', 'pool'], )),
    'dropout': ('categorical', ([0, 0.1, 0.2, 0.3], )),
    'lr': ("float", (10e-3, 10e-1)),
    'activation': ('categorical', (['ReLU', 'LeakyReLU', 'Tanh'], ))
}

param_space['ChebNet'] = {
    'h_feats': [16, 32, 64],
    'num_layers': [1, 2, 3, 4],
    'drop_rate': [0, 0.1, 0.2, 0.3],
    'mlp_layers': [1, 2],
    'lr': 10 ** np.linspace(-3, -1, 1000),
    'activation': ['ReLU', 'LeakyReLU', 'Tanh']
}

param_space['BernNet'] = {
    'h_feats': [16, 32, 64],
    'drop_rate': [0, 0.1, 0.2, 0.3],
    'mlp_layers': [1, 2],
    'lr': 10 ** np.linspace(-3, -1, 1000),
    'orders': [2, 3, 4, 5],
    'activation': ['ReLU', 'LeakyReLU', 'Tanh']
}

param_space['AMNet'] = {
    'h_feats': [16, 32, 64],
    'drop_rate': [0, 0.1, 0.2, 0.3],
    'num_layers': [1, 2, 3, 4],
    'lr': 10 ** np.linspace(-3, -1, 1000),
    'orders': [2, 3],
    'activation': ['ReLU', 'LeakyReLU', 'Tanh']
}

param_space['BWGNN'] = {
    'h_feats': [16, 32, 64],
    'drop_rate': [0, 0.1, 0.2, 0.3],
    'num_layers': [1, 2, 3, 4],
    'lr': 10 ** np.linspace(-3, -1, 1000),
    'mlp_layers': [1, 2],
    'activation': ['ReLU', 'LeakyReLU', 'Tanh'],
}

param_space['GAS'] = {
    'h_feats': [16, 32, 64],
    'num_layers': [1, 2, 3, 4],
    'drop_rate': [0, 0.1, 0.2, 0.3],
    'k': range(3, 51),
    'dist': ['euclidean', 'cosine'],
    'lr': 10 ** np.linspace(-3, -1, 1000),
    'activation': ['ReLU', 'LeakyReLU', 'Tanh']
}

param_space['GHRN'] = {
    'h_feats': [16, 32, 64],
    'del_ratio': 10 ** np.linspace(-2, -1, 1000),
    'drop_rate': [0, 0.1, 0.2, 0.3],
    'num_layers': [1, 2, 3, 4],
    'lr': 10 ** np.linspace(-3, -1, 1000),
    'mlp_layers': [1, 2],
}

param_space['KNNGCN'] = {
    'h_feats': [16, 32, 64],
    'num_layers': [1, 2, 3, 4],
    'drop_rate': [0, 0.1, 0.2, 0.3],
    'k': list(range(3, 51)),
    'dist': ['euclidean', 'cosine'],
    'lr': 10 ** np.linspace(-3, -1, 1000),
}

param_space['XGBoost'] = {
    'n_estimators': list(range(10, 201)),
    'eta': 0.5 * 10 ** np.linspace(-1, 0, 1000),
    'lambda': [0, 1, 10],
    'subsample': [0.5, 0.75, 1]
}
trial_space['XGBoost'] = {
    'n_estimators': ("int", (10, 200)),
    'eta': ("float", (0.05, 0.5)),
    'lambda': ("categorical", ([0, 1, 10], )),
    'subsample': ("categorical", ([0.5, 0.75, 1], ))
}

param_space['XGBGraph'] = {
    'n_estimators': list(range(10, 201)),
    'eta': 0.5 * 10 ** np.linspace(-1, 0, 1000),
    'lambda': [0, 1, 10],
    # 'alpha': [0, 0.5, 1],
    'subsample': [0.5, 0.75, 1],
    'num_layers': [1, 2, 3, 4],
    'agg': ['sum', 'max', 'mean'],
    'booster': ['gbtree', 'dart']
}
trial_space['XGBGraph'] = {
    'n_estimators': ("int", (10, 200)),
    'eta': ("float", (0.05, 0.5)),
    'lambda': ("categorical", ([0, 1, 10], )),
    # 'alpha': [0, 0.5, 1],
    'subsample': ("categorical", ([0.5, 0.75, 1], )),
    'num_layers': ("categorical", ([1, 2, 3, 4], )),
    'agg': ("categorical", (['sum', 'max', 'mean'], )),
    'booster': ("categorical", (['gbtree', 'dart'], ))
}

param_space['RF'] = {
    'n_estimators': list(range(10, 201)),
    'criterion': ['gini', 'entropy'],
    'max_samples': list(np.linspace(0.1, 1, 1000)),
}

param_space['RFGraph'] = {
    'n_estimators': list(range(10, 201)),
    'criterion': ['gini', 'entropy'],
    'max_samples': [0.5, 0.75, 1],
    'max_features': ['sqrt', 'log2', None],
    'num_layers': [1, 2, 3, 4],
    'agg': ['sum', 'max', 'mean'],
}

param_space['SVM'] = {
    'weights': ['uniform', 'distance'],
    'C': list(10 ** np.linspace(-1, 1, 1000))
}

param_space['KNN'] = {
    'k': list(range(3, 51)),
    'weights': ['uniform', 'distance'],
    'p': [1, 2]
}

param_space['XGBOD'] = {
    'n_estimators': list(range(10, 201)),
    'learning_rate': 0.5 * 10 ** np.linspace(-1, 0, 1000),  # [0.05, 0.1, 0.2, 0.3, 0.5],
    'lambda': [0, 1, 10],
    'subsample': [0.5, 0.75, 1],
    'booster': ['gbtree', 'dart']
}

param_space['GAT'] = {
    'h_feats': [16, 32],
    'num_heads': [1, 2, 4, 8],
    'drop_rate': [0, 0.1, 0.2, 0.3],
    'num_layers': [1, 2, 3],
    'lr': 10 ** np.linspace(-3, -1, 1000),
}

param_space['GATSep'] = {
    'h_feats': [16, 32],
    'num_heads': [1, 2, 4, 8],
    'drop_rate': [0, 0.1, 0.2, 0.3],
    'num_layers': [1, 2, 3],
    'lr': 10 ** np.linspace(-3, -1, 1000),
}

param_space['GT'] = {
    'h_feats': [16, 32],
    'num_heads': [1, 2, 4, 8],
    'drop_rate': [0, 0.1, 0.2, 0.3],
    'num_layers': [1, 2, 3],
    'lr': 10 ** np.linspace(-3, -1, 1000),
}

param_space['PCGNN'] = {
    'h_feats': [16, 32, 64],
    'del_ratio': np.linspace(0.01, 0.8, 1000),
    'add_ratio': np.linspace(0.01, 0.8, 1000),
    'dist': ['euclidean', 'cosine'],
    # 'k': list(range(3, 10)),
    'drop_rate': [0, 0.1, 0.2, 0.3],
    'num_layers': [1, 2, 3],
    'lr': 10 ** np.linspace(-3, -1, 1000),
}

param_space['DCI'] = {
    'h_feats': [16, 32, 64],
    'pretrain_epochs': [20, 50, 100],
    'num_cluster': list(range(2,31)),
    'drop_rate': [0, 0.1, 0.2, 0.3],
    'num_layers': [1, 2, 3],
    'lr': 10 ** np.linspace(-3, -1, 1000),
}

param_space['BGNN'] = {
    'depth': [4,5,6,7],
    'iter_per_epoch': [2,5,10,20],
    'gbdt_lr': 10 ** np.linspace(-2, -0.5, 1000),
    'normalize_features': [True, False],
    'h_feats': [16, 32, 64],
    'num_layers': [1, 2, 3, 4],
    'drop_rate': [0, 0.1, 0.2, 0.3],
    'lr': 10 ** np.linspace(-3, -1, 1000),
    'activation': ['ReLU', 'LeakyReLU', 'Tanh']
}

param_space['NA'] = {
    'n_estimators': list(range(10, 201)),
    'eta': 0.5 * 10 ** np.linspace(-1, 0, 1000),
    'lambda': [0, 1, 10],
    'subsample': [0.5, 0.75, 1],
    'k': list(range(0, 51)),
}

param_space['PNA'] = {
    'h_feats': [16, 32, 64],
    'drop_rate': [0, 0.1, 0.2, 0.3],
    'num_layers': [1, 2, 3, 4],
    'lr': 10 ** np.linspace(-3, -1, 1000),
    'activation': ['ReLU', 'LeakyReLU', 'Tanh'],
}

class AllZeroDict(dict):
    def __getitem__(self, key):
        return 0