#!/bin/bash
#SBATCH --job-name=preprocess-code-data
#SBATCH --nodes=1 # node count
#SBATCH --ntasks=1 # total number of tasks across all nodes
#SBATCH --reservation=acagpt
#SBATCH --cpus-per-task 64
#SBATCH -o /cognitive_comp/ganruyi/gpt-neox/workspace/llama/%x-%j.log
#SBATCH -e /cognitive_comp/ganruyi/gpt-neox/workspace/llama/%x-%j.err

DATA_DIR='/cognitive_comp/common_data/big_corpus/code_t/'

files=`ls ${DATA_DIR}`
opt=""
for i in $files;
do
    opt=$opt"${DATA_DIR}$i,"
done

opt=${opt%,*}
echo $opt

VOCAB_FILE='/cognitive_comp/common_data/Huggingface-Models/llama_7b/tokenizer.model'
export PARAMS="
    --input $opt \
    --vocab-file $VOCAB_FILE \
    --output-prefix /cognitive_comp/common_data/chatGPTdata_v2.1/code_spm/code_data \
    --tokenizer-type LlamaTokenizer \
    --jsonl-keys concat_content \
    --dataset-impl mmap \
    --workers 32 \
    --append-eod \
    --append-bos \
    --ftfy \
    --log-data \
    "
SCRIPT_PATH='/cognitive_comp/ganruyi/gpt-neox/tools/preprocess_data.py'
export RUN_CMD="python $SCRIPT_PATH $PARAMS"

srun bash -c '$RUN_CMD'
