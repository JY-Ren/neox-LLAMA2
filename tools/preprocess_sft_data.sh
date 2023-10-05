#!/bin/bash
#SBATCH --job-name=preprocess-sft-data
#SBATCH --nodes=1 # node count
#SBATCH --ntasks=1 # total number of tasks across all nodes
#SBATCH --reservation=acagpt
#SBATCH --cpus-per-task 40
#SBATCH -o /cognitive_comp/ganruyi/gpt-neox/tools/%x-%j.log
#SBATCH -e /cognitive_comp/ganruyi/gpt-neox/tools/%x-%j.err

DATA_PATH='/cognitive_comp/yangping/data/instructdata/merge_data/dataset_20230318/train_shuffle.json'
VOCAB_FILE='/cognitive_comp/common_data/BPETokenizer-Mix-NEO-pre2'
export PARAMS="
    --input $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --output-prefix /cognitive_comp/ganruyi/gpt-neox/workspace/gpt_neox6B_v1_sample_exp/dataset_20230318 \
    --tokenizer-type HFGPTNeoXTokenizerFast \
    --jsonl-keys text \
    --dataset-impl mmap \
    --workers 20 \
    --append-eod \
    --sft-data \
    --ftfy \
    --log-data \
    "
SCRIPT_PATH='/cognitive_comp/ganruyi/gpt-neox/tools/preprocess_data.py'
export RUN_CMD="python $SCRIPT_PATH $PARAMS"

srun bash -c '$RUN_CMD'
