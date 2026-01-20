import argparse
import time
from utils import *
import pandas
import os
import warnings
import traceback
from collections import defaultdict
from tta import tta_dict
import datetime
import logging
import sys
import io
import inspect
import matplotlib.pyplot as plt
import pickle
import gc
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

def run_gc():
    gc.collect()
    torch.cuda.empty_cache()

parser = argparse.ArgumentParser()
parser.add_argument('--trials', type=int, default=10)
parser.add_argument('--semi_supervised', type=int, default=0)
parser.add_argument('--inductive', type=int, default=0)
parser.add_argument('--models', type=str, default=None)
parser.add_argument('--datasets', type=str, default=None)
parser.add_argument('--better_output', choices=['True', 'False'], default='True')
# parser.add_argument('--tta_methods', type=str, default='')
parser.add_argument('--tta_datasets', type=str, default=None)
parser.add_argument('--tta_trials', type=int, default=10)
parser.add_argument('--motifs', action='store_true')
parser.add_argument('--motifs_file_prefix', type=str, default=None)
parser.add_argument('--log_prefix', type=str, default=None)
parser.add_argument('--groups', action='store_true')
args = parser.parse_args()

log_prefix = args.log_prefix or 'ema_ablation'
# logging.basicConfig(filename=f'logs/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}_{log_prefix}.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = False
# stream_handler = logging.StreamHandler(sys.stdout)
# stream_handler.setLevel(logging.INFO)
# stream_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
# logger.addHandler(stream_handler)
file_handler = logging.FileHandler(f'logs/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}_{log_prefix}.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
logger.addHandler(file_handler)
# hack sys.stdout to redirect to logger
class PrintCapture(io.TextIOBase):
    def __init__(self, original_stdout, logger):
        self.original_stdout = original_stdout
        self.logger = logger

    def write(self, message):
        if not message.strip():
            return
        try:
            frame = inspect.currentframe()
            if frame is not None:
                frame = frame.f_back.f_back
            if frame is not None:
                filename = frame.f_code.co_filename
                lineno = frame.f_lineno
                funcname = frame.f_code.co_name
            else:
                filename, lineno, funcname = "<unknown>", 0, "<module>"

            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_message = f"[{timestamp}] {filename}:{lineno} ({funcname}) | {message}"
        except Exception:
            log_message = message

        # always print to real stdout
        self.original_stdout.write(log_message + '\n')

        # log to logger
        try:
            self.logger.info(log_message)
        except Exception:
            pass

    def flush(self):
        self.original_stdout.flush()
sys.stdout = PrintCapture(sys.stdout, logger)

logger.info(f"[INFO] Running with parameters: {args.__dict__}")

better_result = args.better_output == 'True'

columns = ['name']
new_row = {}
datasets = ['reddit', 'weibo', 'amazon', 'yelp', 'tolokers',                        # 0-4
            'questions', 'tfinance', 'elliptic', 'dgraphfin', 'tsocial',            # 5-9
            # 'hetero/amazon', 'hetero/yelp'
            # 'alpha_homora', 'cryptopia_hacker', 'plus_token_ponzi', 'upbit_hack',   # 10-13
            'eth_AlphaHomora.dglgraph', 'eth_CryptopiaHacker.dglgraph', 'eth_PlusTokenPonzi.dglgraph', 'eth_UpbitHack.dglgraph',  # 10-13
            'tron_gl.dglgraph', # 14
            ]
models = model_detector_dict.keys()
motifs_datasets = datasets[10:14]  # eth_datasets

# tta_classes = [tta_dict[i] for i in args.tta_methods.split(',') if i in tta_dict]
tta_class = tta_dict['ema_teacher']
# print("Evaluated TTA Methods: ", [i.__name__ for i in tta_classes])

# search_params = {
#     'alpha': [0.9, 0.99, 0.999],
#     'loss_type': ['kl', 'dist'],
#     'lower_tau': [0.5, 0.7, 0.9],
#     'upper_tau': [0.8, 0.9, 0.95],
# }

# def iter_params():
#     iters = [(k, iter(v)) for k, v in search_params.items()]
#     params = {k: next(v) for k, v in iters}
#     exit_flag = False
#     while not exit_flag:
#         if params['lower_tau'] < params['upper_tau']:
#             yield '_'.join([f'{k}_{v}' for k, v in params.items()]), params
#         for i in range(len(iters) - 1, -1, -1):
#             try:
#                 params[iters[i][0]] = next(iters[i][1])
#                 break
#             except StopIteration:
#                 if i == 0:
#                     exit_flag = True
#                     break
#                 iters[i] = (iters[i][0], iter(search_params[iters[i][0]]))
#                 params[iters[i][0]] = next(iters[i][1])
#                 continue

def iter_params():
    return [
        ('teacher', {'abla': ['teacher']}),
        ('infonce', {'abla': ['infonce']}),
        ('trustable', {'abla': ['trustable']}),
        # ('teacher_trustable', {'abla': ['teacher', 'trustable']}),
        # ('infonce_trustable', {'abla': ['infonce', 'trustable']}),
    ]

og_datasets = datasets
if args.datasets is not None:
    if '-' in args.datasets:
        st, ed = args.datasets.split('-')
        datasets = datasets[int(st):int(ed)+1]
    else:
        datasets = [datasets[int(t)] for t in args.datasets.split(',')]
print('Evaluated Datasets: ', datasets)

def look_motif_rep(model, dataset_name):
    try:
        rep_role = model._feature_fusion.motifs_expression.learnable_role_embed.detach()
        rep_motifs = model._feature_fusion.motifs_expression.learnable_motifs_embed.detach()
    except:
        print('Motif Representation Not Found')
        return
    
    # calc dis matrix
    role_dis = (rep_role @ rep_role.t()).cpu().numpy()
    motif_dis = (rep_motifs @ rep_motifs.t()).cpu().numpy()

    # plot heatmap for each tensor
    plt.figure(figsize=(10, 10))
    plt.imshow(role_dis, cmap='Blues')
    plt.title(f'{dataset_name} Role Dis Matrix')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(f'plots/{dataset_name}_{model.__class__.__name__}_role_rep.png')
    plt.close()

    plt.figure(figsize=(10, 10))
    plt.imshow(motif_dis, cmap='Blues')
    plt.title(f'{dataset_name} Motif Dis Matrix')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(f'plots/{dataset_name}_{model.__class__.__name__}_motif_rep.png')
    plt.close()

tta_datasets = []
if args.tta_datasets is not None:
    if '-' in args.tta_datasets:
        st, ed = args.tta_datasets.split('-')
        tta_datasets = og_datasets[int(st):int(ed)+1]
    else:
        tta_datasets = [og_datasets[int(t)] for t in args.tta_datasets.split(',')]
print('TTA Datasets: ', tta_datasets)

motifs_enabled = args.motifs
motifs_file_prefix = args.motifs_file_prefix or 'rust_motifs'
def get_dataset(dataset: str):
    if motifs_enabled and dataset in motifs_datasets:
        print(f"Enable motifs for dataset {dataset}")
        return MotifsDataset(dataset, motifs_file_rule=lambda x: 'motifs/' + x.split('.')[0] + '.' + motifs_file_prefix + '.npy')
    return Dataset(dataset)

groups_enabled = args.groups

if args.models is not None:
    models = args.models.split('-')
    print('Evaluated Baselines: ', models)

for dataset in datasets:
    for metric in ['AUROC mean', 'AUROC std', 'AUPRC mean', 'AUPRC std',
                   'RecK mean', 'RecK std', 'Time']:
        columns.append(dataset+'-'+metric)

results = pandas.DataFrame(columns=columns)
file_id = None
try:
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
            data = get_dataset(dataset_name)
            model_config = {'model': model, 'lr': 0.01, 'drop_rate': 0}
            if dataset_name == 'tsocial':
                model_config['h_feats'] = 16
                # if model in ['GHRN', 'KNNGCN', 'AMNet', 'GT', 'GAT', 'GATv2', 'GATSep', 'PNA']:   # require more than 24G GPU memory
                    # continue

            eval_results = defaultdict(list)
            best_model = None
            best_score = 0.0
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
                    if test_score[train_config['metric']] > best_score:
                        best_score = test_score[train_config['metric']]
                        best_model = copy.deepcopy(detector.model)
                except torch.cuda.OutOfMemoryError:
                    test_score = AllZeroDict()
                    print(f"Out of memory error for {model} on {dataset_name} at trial {t}. OG traceback: \n{traceback.format_exc()}")
                for metric in ['AUROC', 'AUPRC', 'RecK', 'F1', 'ACC']:
                    eval_results[metric].append(test_score[metric])
                ed = time.time()
                time_cost += ed - st
            og_feature_dim = data.graph.ndata['feature'].shape[1]
            del detector, data
            run_gc()

            look_motif_rep(best_model, dataset_name)

            for metric in ['AUROC', 'AUPRC', 'RecK', 'F1', 'ACC']:
                model_result[dataset_name+'-'+metric+' mean'] = np.mean(eval_results[metric], where=np.array(eval_results[metric]) > 0)
                model_result[dataset_name+'-'+metric+' std'] = np.std(eval_results[metric], where=np.array(eval_results[metric]) > 0)
            model_result[dataset_name+'-Time'] = time_cost/args.trials

            ## TTA
            if best_model is None:
                print(f"Best model is None for {model} on {dataset_name}")
                continue
            og_model_feature = get_model_feature(best_model)
            for tta_dataset_name in tta_datasets:
                # if tta_dataset_name == dataset_name:
                #     continue
                history_model_feature = {}
                for param_name, params in iter_params():
                    tta_dataset = get_dataset(tta_dataset_name)
                    if not compare_model_feature(og_model_feature, best_model):
                        print(f"[WARN] og model has been modified")
                        raise ValueError("og model has been modified")
                    tta_method = tta_class(best_model, tta_dataset.graph, og_feature_dim, en_sec_device=(model in ['SpaceGNN', 'ARC']), **params)
                    print(f"tta: {tta_class.__name__} params: {params}")
                    tta_name = f"{tta_class.__name__}_{param_name}"
                    try:
                        tta_method.adapt()
                        tta_test_score = tta_method.eval()
                        current_model_feature = get_model_feature(tta_method.get_trained_model())
                        print(f"Test score for {tta_method} on {tta_dataset_name}: ", tta_test_score)
                        print(f"Compare {model} and {tta_name} on {tta_dataset_name}: ", compare_model_feature(og_model_feature, current_model_feature))
                        for history_tta_class, history_model in history_model_feature.items():
                            print(f"Compare {tta_name} and {history_tta_class} on {tta_dataset_name}: ", compare_model_feature(current_model_feature, history_model))
                        history_model_feature[tta_name] = current_model_feature
                    except torch.cuda.OutOfMemoryError:
                        tta_test_score = AllZeroDict()
                        print(f"Out of memory error for {tta_method} and {model} on {tta_dataset_name} at trial {t}. OG traceback: \n{traceback.format_exc()}")
                    for metric in ['AUROC', 'AUPRC', 'RecK', 'F1', 'ACC']:
                        model_result['TTA-'+tta_name+'-'+dataset_name+'-'+tta_dataset_name+'-'+metric+' mean'] = tta_test_score[metric]
                        model_result['TTA-'+tta_name+'-'+dataset_name+'-'+tta_dataset_name+'-'+metric+' std'] = 0.0
                    del tta_method
                    run_gc()
                    if not groups_enabled:
                        continue
                    for group in tta_dataset.iter_group():
                        if not compare_model_feature(og_model_feature, best_model):
                            print(f"[WARN] og model has been modified")
                            raise ValueError("og model has been modified")
                        tta_method = tta_class(best_model, tta_dataset.graph, og_feature_dim)
                        try:
                            tta_method.adapt()
                            tta_test_score = tta_method.eval()
                            print(f"Compare {model} and {tta_method} on {tta_dataset_name} group {group}: ", compare_model_feature(og_model_feature, tta_method.get_trained_model()))
                        except torch.cuda.OutOfMemoryError:
                            tta_test_score = AllZeroDict()
                            print(f"Out of memory error for {tta_method} and {model} on {tta_dataset_name}@{group} at trial {t}. OG traceback: \n{traceback.format_exc()}")
                        for metric in ['AUROC', 'AUPRC', 'RecK', 'F1', 'ACC']:
                            model_result['TTA-'+tta_name+'-'+dataset_name+'-'+tta_dataset_name+'@'+str(group)+'-'+metric+' mean'] = tta_test_score[metric]
                            model_result['TTA-'+tta_name+'-'+dataset_name+'-'+tta_dataset_name+'@'+str(group)+'-'+metric+' std'] = 0.0
                        del tta_method
                        run_gc()
                del tta_dataset

        model_result = pandas.DataFrame(model_result, index=[0])
        results = pandas.concat([results, model_result])
        with open(f"debug/result_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.pkl", 'wb') as fp:
            pickle.dump(results, fp)
        if better_result:
            file_id = better_save_results(results, file_id)
            save_tta_results(results, file_id)
        else:
            file_id = save_results(results, file_id)
        print(results)
except Exception as e:
    print(e)
    print(traceback.format_exc())
    raise e
