#!/bin/bash
#SBATCH --job-name=train_llama65b # create a short name for your job
#SBATCH -p batch
#SBATCH -x dgx047
#SBATCH --reservation=acagpt
#SBATCH -N 1 # node count
#SBATCH --ntasks-per-node 1 # number of tasks to run per node
#SBATCH --cpus-per-task 70 # cpu-cores per task (>1 if multi-threaded tasks),--cpus-per-task
#SBATCH --gpus-per-node 0 # total cpus for job


CODE_PATH=/home/yangping/nlp/gpt-neox

python $CODE_PATH/tools/preprocess_data.py \
    --input /cognitive_comp/common_data/big_corpus/wudaoData2.0/wd100.jsonl \
    --jsonl-keys content \
    --output-prefix /cognitive_comp/yangping/data/pretrain_corpus/llama_continue_pretraining/wudao/data \
    --vocab /cognitive_comp/common_data/LLamaTokenizer_7B/ \
    --dataset-impl mmap \
    --tokenizer-type HFLlamaTokenizer \
    --workers 64 \
    --append-eod \
    --log-data \
    --ftfy
