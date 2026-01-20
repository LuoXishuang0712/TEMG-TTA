import argparse
import time
from utils import *
import pandas
import os
import warnings
import traceback
warnings.filterwarnings("ignore")
seed_list = list(range(3407, 10000, 10))

def set_seed(seed=3407):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


parser = argparse.ArgumentParser()
parser.add_argument('--trials', type=int, default=10)
parser.add_argument('--semi_supervised', type=int, default=0)
parser.add_argument('--inductive', type=int, default=0)
parser.add_argument('--models', type=str, default=None)
parser.add_argument('--datasets', type=str, default=None)
parser.add_argument('--better_output', choices=['True', 'False'], default='True')
args = parser.parse_args()

better_result = args.better_output == 'True'

columns = ['name']
new_row = {}
datasets = ['reddit', 'weibo', 'amazon', 'yelp', 'tolokers',                        # 0-4
            'questions', 'tfinance', 'elliptic', 'dgraphfin', 'tsocial',            # 5-9
            # 'hetero/amazon', 'hetero/yelp'
            # 'alpha_homora', 'cryptopia_hacker', 'plus_token_ponzi', 'upbit_hack',   # 10-13
            'eth_AlphaHomora.dglgraph', 'eth_CryptopiaHacker.dglgraph', 'eth_PlusTokenPonzi.dglgraph', 'eth_UpbitHack.dglgraph',  # 10-13
            ]
models = model_detector_dict.keys()

if args.datasets is not None:
    if '-' in args.datasets:
        st, ed = args.datasets.split('-')
        datasets = datasets[int(st):int(ed)+1]
    else:
        datasets = [datasets[int(t)] for t in args.datasets.split(',')]
print('Evaluated Datasets: ', datasets)

if args.models is not None:
    models = args.models.split('-')
    print('Evaluated Baselines: ', models)

for dataset in datasets:
    for metric in ['AUROC mean', 'AUROC std', 'AUPRC mean', 'AUPRC std',
                   'RecK mean', 'RecK std', 'Time']:
        columns.append(dataset+'-'+metric)

results = pandas.DataFrame(columns=columns)
file_id = None
for model in models:
    model_result = {'name': model}
    for dataset_name in datasets:
        if model in ['CAREGNN', 'H2FD'] and 'hetero' not in dataset_name:
            continue
        time_cost = 0
        train_config = {
            'device': 'cuda',
            'epochs': 200,
            'patience': 50,
            'metric': 'AUPRC',
            'inductive': bool(args.inductive)
        }
        data = Dataset(dataset_name)
        model_config = {'model': model, 'lr': 0.01, 'drop_rate': 0}
        if dataset_name == 'tsocial':
            model_config['h_feats'] = 16
            # if model in ['GHRN', 'KNNGCN', 'AMNet', 'GT', 'GAT', 'GATv2', 'GATSep', 'PNA']:   # require more than 24G GPU memory
                # continue

        auc_list, pre_list, rec_list, f1_list = [], [], [], []
        for t in range(args.trials):
            torch.cuda.empty_cache()
            print("Dataset {}, Model {}, Trial {}".format(dataset_name, model, t))
            data.split(args.semi_supervised, t)
            seed = seed_list[t]
            set_seed(seed)
            train_config['seed'] = seed
            try:
                detector = model_detector_dict[model](train_config, model_config, data)
                st = time.time()
                print(detector.model)
                test_score = detector.train()  # if no F1-score printed to stdout, check detector! the eval() in super class has been modified to return F1-score so in results.
            except torch.cuda.OutOfMemoryError:
                test_score = {'AUROC': 0, 'AUPRC': 0, 'RecK': 0, 'F1': 0}
                print(f"Out of memory error for {model} on {dataset_name} at trial {t}. OG traceback: \n{traceback.format_exc()}")
            auc_list.append(test_score['AUROC']), pre_list.append(test_score['AUPRC']), rec_list.append(test_score['RecK']), f1_list.append(test_score['F1'])
            ed = time.time()
            time_cost += ed - st
        del detector, data

        for metric in ['AUROC', 'AUPRC', 'RecK', 'F1', 'ACC']:
            model_result[dataset_name+'-'+metric+' mean'] = np.mean(test_score[metric], where=np.array(test_score[metric]) > 0)
            model_result[dataset_name+'-'+metric+' std'] = np.std(test_score[metric], where=np.array(test_score[metric]) > 0)
        model_result[dataset_name+'-Time'] = time_cost/args.trials
    model_result = pandas.DataFrame(model_result, index=[0])
    results = pandas.concat([results, model_result])
    if better_result:
        file_id = better_save_results(results, file_id)
    else:
        file_id = save_results(results, file_id)
    print(results)
