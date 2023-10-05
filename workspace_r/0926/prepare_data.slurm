#!/bin/bash
#SBATCH --job-name=prepare_data # create a short name for your job
#SBATCH -p batch
#SBATCH -x dgx047
#SBATCH --reservation=ccp
#SBATCH --qos=ccp
#SBATCH -N 1 # node count
#SBATCH --ntasks-per-node 1 # number of tasks to run per node
#SBATCH --cpus-per-task 96 # cpu-cores per task (>1 if multi-threaded tasks),--cpus-per-task
#SBATCH --gpus-per-node 0 # total cpus for job
#SBATCH -o logs/%x-%j.log # output and error log file names (%x for job id)
#SBATCH -e logs/%x-%j.err # output and error log file names (%x for job id)


code_dir=`git rev-parse --show-toplevel`
input_dir="/shared_space/agpt/pretrain/raw_academic/filtered_thesis"
# input_dir="/platform_tech/jyren/A_data/Pretrain/pubmed"
output_dir="/shared_space/agpt/experiments/0926/data"

file_names=( 
    "thesis_filter_merge.jsonl"
    # "pubmed_text.jsonl" 
    # "wiki_en.jsonl" 
    # "s2.jsonl" 
    # "cc_en.jsonl" 
    # "authors.jsonl" 
    # "latex.jsonl" 

    # "pdf_cs.jsonl" 
    # "pdf_math.jsonl" 
    # "pdf_envir.jsonl" 
    # "pdf_mater.jsonl" 
    # "pdf_physics.jsonl" 
    # "pdf_med.jsonl" 
    # "pdf_bio.jsonl" 
    # "pdf_geo.jsonl" 

    # "prof_luo.jsonl" 
    # "prof_li.jsonl"  
    )

for file in "${file_names[@]}"; do
    echo "processing $file ..."
    python $code_dir/tools/preprocess_data_academic.py \
        --input ${input_dir}/${file} \
        --output-prefix ${output_dir}/${file:0:-6} \
        --vocab-file /shared_space/agpt/models/llama2/hf/70B \
        --dataset-impl mmap \
        --chunk_size 4096 \
        --slide_len 32 \
        --tokenizer-type HFLlamaTokenizer \
        --workers 192 \
        --append-eod \
        --ftfy
done