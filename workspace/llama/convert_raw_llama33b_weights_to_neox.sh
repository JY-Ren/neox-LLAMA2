#!/bin/bash
#SBATCH --job-name=covert_llama65b_neox # create a short name for your job
#SBATCH -p batch
#SBATCH -x dgx047
#SBATCH --reservation=acagpt
#SBATCH -N 1 # node count
#SBATCH --ntasks-per-node 1 # number of tasks to run per node
#SBATCH --cpus-per-task 100 # cpu-cores per task (>1 if multi-threaded tasks),--cpus-per-task
#SBATCH --gpus-per-node 0 # total cpus for job
#SBATCH -o workspace/llama/%x-%j.log # output and error log file names (%x for job id)
#SBATCH -e workspace/llama/%x-%j.err # output and error log file names (%x for job id)

# SCRIPT='/cognitive_comp/ganruyi/gpt-neox/tools/convert_raw_llama_weights_to_neox.py'
SCRIPT='/cognitive_comp/yangqi/project/gpt-neox/tools/convert_hf_llama_weights_to_neox.py'

raw_model_path="/cognitive_comp/yangqi/model/models--decapoda-research--llama-30b-hf/snapshots/f991780f9362b2fcaefad066cd235058844562b7"
neox_ckpt_path="/cognitive_comp/yangqi/checkpoints/llama/llama_65B_continue_pretrained/ckpt_33b_ori_pp8mp4"

python "$SCRIPT" --input_dir "$raw_model_path" --model_size 30B --output_dir "$neox_ckpt_path" --num_output_shards 4 --num_output_shards_pipeline 8 --pipeline_parallel 
