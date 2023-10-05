#!/bin/bash
#SBATCH --job-name=sft_gpt_neox6B_v1 # create a short name for your job
#SBATCH --nodes=1 # node count
#SBATCH --ntasks-per-node=1 # total number of tasks across all nodes
#SBATCH --cpus-per-task=32 # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=8G # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:hgx:0 # number of gpus per node
#SBATCH -p pol-preempted # number of gpus per node

CODE_PATH=/cognitive_comp/yangping/nlp/gpt-neox

python $CODE_PATH/tools/preprocess_data_finetune.py \
    --input /cognitive_comp/yangping/data/instructdata/merge_data/merge_sft_data/align_20230330 \
    --output /cognitive_comp/yangping/data/instructdata/merge_data/merge_sft_data/align_20230330/tokenid/ \
    --vocab /cognitive_comp/yangping/pretrained_model/llama-7b-hf/ \
    --tokenizer-type SPMTokenizer \
    --epoch 3 \
    --seq_length 2048
