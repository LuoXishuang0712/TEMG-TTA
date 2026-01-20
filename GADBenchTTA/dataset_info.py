import dgl
import argparse
import json

datasets = ['reddit', 'weibo', 'amazon', 'yelp', 'tolokers',                        # 0-4
            'questions', 'tfinance', 'elliptic', 'dgraphfin', 'tsocial',            # 5-9
            # 'hetero/amazon', 'hetero/yelp'
            # 'alpha_homora', 'cryptopia_hacker', 'plus_token_ponzi', 'upbit_hack',   # 10-13
            'eth_AlphaHomora.dglgraph', 'eth_CryptopiaHacker.dglgraph', 'eth_PlusTokenPonzi.dglgraph', 'eth_UpbitHack.dglgraph',  # 10-13
            'tron_gl.dglgraph', # 14
            ]

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default=None)
args = parser.parse_args()

datasets = [datasets[int(i)] for i in args.dataset.split(',')]

final_res = {}
for dataset in datasets:
    g = dgl.load_graphs(f'./datasets/{dataset}')[0][0]
    res = {}
    res['num_nodes'] = g.num_nodes()
    res['num_edges'] = g.num_edges()
    res['num_feats'] = g.ndata['feature'].shape[1]
    res['fraud_ratio'] = "%.2f%%" % ((g.ndata['label'] == 1).float().mean().item() * 100)
    final_res[dataset] = res
print(json.dumps(final_res, indent=4))
