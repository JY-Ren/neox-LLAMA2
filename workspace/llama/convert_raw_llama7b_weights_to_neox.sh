#!/bin/bash
#SBATCH --job-name=covert_llama7b_neox # create a short name for your job
#SBATCH -p batch
#SBATCH -x dgx047
#SBATCH --reservation=acagpt
#SBATCH -N 1 # node count
#SBATCH --ntasks-per-node 1 # number of tasks to run per node
#SBATCH --cpus-per-task 100 # cpu-cores per task (>1 if multi-threaded tasks),--cpus-per-task
#SBATCH --gpus-per-node 0 # total cpus for job
#SBATCH -o workspace/llama/%x-%j.log # output and error log file names (%x for job id)
#SBATCH -e workspace/llama/%x-%j.err # output and error log file names (%x for job id)

SCRIPT='/cognitive_comp/ganruyi/gpt-neox/tools/convert_raw_llama_weights_to_neox.py'
raw_model_path="/cognitive_comp/common_data/wanng_warehouse/LLaMA/"
neox_ckpt_path="/cognitive_comp/ganruyi/gpt-neox/workspace/llama/checkpoints_7b"
python "$SCRIPT" --input_dir "$raw_model_path" --model_size 7B --output_dir "$neox_ckpt_path" --num_output_shards 1 --pipeline
