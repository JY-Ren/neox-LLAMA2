#!/bin/bash
#SBATCH --job-name=convert_13b # create a short name for your job
#SBATCH -p batch
#SBATCH -N 1 # node count
#SBATCH --ntasks-per-node 1 # number of tasks to run per node
#SBATCH --cpus-per-task 2 # cpu-cores per task (>1 if multi-threaded tasks),--cpus-per-task
#SBATCH --gpus-per-node 1 # total cpus for job
#SBATCH -o ./log/convert/%x-%j.log # output and error log file names (%x for job id)
#SBATCH -e ./log/convert/%x-%j.err # output and error log file names (%x for job id)
#SBATCH -p pos # number of gpus per node

# python tools/convert_llama_to_hf.py \
#     --input_dir /cognitive_comp/liuhan/checkpoints/llama-neox-sft/13B-mp4/global_step2000 \
#     --config_file workspace/llama/13b_sft.yml \
#     --output_dir /cognitive_comp/zhangwenjun/checkpoints/llama-neox-sft/13B-mp4/global_step2000-hf

python tools/convert_llama_to_hf.py \
    --input_dir /cognitive_comp/liuhan/checkpoints/llama-neox-sft/13B-0405/global_step25000 \
    --config_file workspace/llama/13b_sft.yml \
    --output_dir /cognitive_comp/zhangwenjun/checkpoints/llama-neox-sft/13B-0405/global_step25000-hf
