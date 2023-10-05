#!/bin/bash
#SBATCH --job-name=convert_llama65b # create a short name for your job
#SBATCH -p batch
#SBATCH -x dgx047
##SBATCH --qos=preemptive
#SBATCH --reservation=acagpt
#SBATCH -N 1 # node count
#SBATCH --ntasks-per-node 1 # number of tasks to run per node
#SBATCH --cpus-per-task 20 # cpu-cores per task (>1 if multi-threaded tasks),--cpus-per-task
#SBATCH --gpus-per-node 0 # total cpus for job
#SBATCH -o slurm_log/%x-%j.log # output and error log file names (%x for job id)
#SBATCH -e slurm_log/%x-%j.err # output and error log file names (%x for job id)

SCRIPT="/cognitive_comp/ganruyi/transformers/src/transformers/models/llama/convert_llama_weights_to_hf.py"
raw_model_path="/cognitive_comp/common_data/wanng_warehouse/LLaMA/"
hf_model_path="/cognitive_comp/common_data/Huggingface-Models/llama_65b"
python "$SCRIPT" --input_dir "$raw_model_path"  --model_size 65B --output_dir "$hf_model_path"
