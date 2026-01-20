#!/bin/bash
set -e

# models="GCN_M-GAT_M-GraphSAGE_M-SpaceGNN-ARC-DGAGNN_M"
models="GCN_M-GraphSAGE_M-SpaceGNN-DGAGNN_M"

PYTHON_PATH=/opt/conda/envs/prog_dgl_xgb/bin/python
# COMMANDS=(
#     "benchmark_tta.py --trials 5 --models $models --datasets 10,11,12,13 --tta_methods assess,gtrans,ema_teacher,no_tta --tta_datasets 10,11,12,13 --log_prefix no_motifs"
#     # "benchmark_tta.py --trials 5 --models $models --datasets 10,11,12,13 --tta_methods assess,gtrans,ema_teacher,no_tta --tta_datasets 10,11,12,13 --log_prefix motifs_w_100 --motifs --motifs_file_prefix w_100_ts"
#     "benchmark_tta.py --trials 5 --models $models --datasets 10,11,12,13 --tta_methods assess,gtrans,ema_teacher,no_tta --tta_datasets 10,11,12,13 --log_prefix motifs_w_100_agg_3600 --motifs --motifs_file_prefix w_100_agg_3600_ts"
# )
COMMANDS=(
    "benchmark_tta_emb.py --trials 5 --models $models --datasets 10,11,12,13 --tta_methods assess,gtrans,ema_teacher,no_tta --tta_datasets 10,11,12,13 --log_prefix no_motifs"
    # "benchmark_tta.py --trials 5 --models $models --datasets 10,11,12,13 --tta_methods assess,gtrans,ema_teacher,no_tta --tta_datasets 10,11,12,13 --log_prefix motifs_w_100 --motifs --motifs_file_prefix w_100_ts"
    "benchmark_tta_emb.py --trials 5 --models $models --datasets 10,11,12,13 --tta_methods assess,gtrans,ema_teacher,no_tta --tta_datasets 10,11,12,13 --log_prefix motifs_w_100_agg_3600 --motifs --motifs_file_prefix w_100_agg_3600_ts"
)
# COMMANDS=(
#     "benchmark_tta.py --trials 5 --models $models --datasets 14 --tta_datasets 10,11,12,13,14 --tta_methods assess,gtrans,ema_teacher,no_tta --log_prefix motifs_test --motifs --motifs_file_prefix w_100_agg_3600_ts"
#     "benchmark_tta.py --trials 5 --models $models --datasets 10,11,12,13,14 --tta_datasets 14 --tta_methods assess,gtrans,ema_teacher,no_tta --log_prefix motifs_test --motifs --motifs_file_prefix w_100_agg_3600_ts"
# )
# COMMANDS=(
#     # "benchmark_tta_ema_ablation.py --trials 5 --models GCN_M --datasets 10 --tta_datasets 10,11,12,13,14 --log_prefix ema_ablation_no_motifs"
#     # "benchmark_tta_ema_ablation.py --trials 5 --models $models --datasets 10,11,12,13 --tta_datasets 10,11,12,13 --log_prefix ema_ablation_motifs_w_100 --motifs --motifs_file_prefix w_100_ts"
#     "benchmark_tta_ema_ablation.py --trials 5 --models GCN_M --datasets 10,11 --tta_datasets 10,11,12,13,14 --log_prefix ema_ablation_motifs_w_100_agg_3600 --motifs --motifs_file_prefix w_100_agg_3600_ts"
#     "benchmark_tta_ema_ablation.py --trials 5 --models GCN_M --datasets 10,11 --tta_datasets 10,11,12,13,14 --log_prefix ema_ablation_motifs_w_100_agg_3600_1 --motifs --motifs_file_prefix w_100_agg_3600_ts"
# )
# COMMANDS=(
#     "benchmark_tta_ema_params.py --trials 5 --models $models --datasets 10,11,12,13 --tta_datasets 10,11,12,13 --log_prefix ema_params_no_motifs"
#     # "benchmark_tta_ema_params.py --trials 5 --models $models --datasets 10,11,12,13 --tta_datasets 10,11,12,13 --log_prefix ema_ablation_motifs_w_100 --motifs --motifs_file_prefix w_100_ts"
#     "benchmark_tta_ema_params.py --trials 5 --models $models --datasets 10,11,12,13 --tta_datasets 10,11,12,13 --log_prefix ema_params_motifs_w_100_agg_3600 --motifs --motifs_file_prefix w_100_agg_3600_ts"
# )
# COMMANDS=(
#     "benchmark_motifs_params.py --trials 5 --models $models --datasets 10,11,12,13 --motifs --motifs_file_prefix 1800 --log_prefix motifs_param_analysis_1800"
#     "benchmark_motifs_params.py --trials 5 --models $models --datasets 10,11,12,13 --motifs --motifs_file_prefix 3600 --log_prefix motifs_param_analysis_3600"
#     "benchmark_motifs_params.py --trials 5 --models $models --datasets 10,11,12,13 --motifs --motifs_file_prefix 7200 --log_prefix motifs_param_analysis_7200"
# )

# get_cuda_device() {
#     echo $((($1*2)%4)),$(((($1*2+1)%4)))
# }
# get_cuda_device() {
#     echo $((($1*2+1)%4)),$(((($1*2+2)%4)))
# }
get_cuda_device() {
    echo $(($1%4))
}

cnt=0
for cmd in "${COMMANDS[@]}"
do
    echo "Running command $cnt: $cmd"
    CUDA_VISIBLE_DEVICES=$(get_cuda_device $cnt) screen -dmS "gadbench$cnt" $PYTHON_PATH $cmd
    # CUDA_VISIBLE_DEVICES=2,3 $PYTHON_PATH $cmd
    cnt=$((cnt+1))
done