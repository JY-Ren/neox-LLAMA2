#!/bin/bash
#SBATCH --job-name=convert_neox # create a short name for your job
#SBATCH -p batch
#SBATCH -N 1 # node count
#SBATCH --ntasks-per-node 1 # number of tasks to run per node
#SBATCH --cpus-per-task 20 # cpu-cores per task (>1 if multi-threaded tasks),--cpus-per-task
#SBATCH --gpus-per-node 0 # total cpus for job
#SBATCH -o ./log/convert/%x-%j.log # output and error log file names (%x for job id)
#SBATCH -e ./log/convert/%x-%j.err # output and error log file names (%x for job id)
#SBATCH -p pos # number of gpus per node

SCRIPT="/cognitive_comp/zhangwenjun/idea/gpt-neox/tools/convert_hf_stablelm_weights_to_neox.py"
input_dir="/cognitive_comp/zhangwenjun/pretrained/stablelm/stablelm-base-alpha-3b"
output_dir="/cognitive_comp/zhangwenjun/pretrained/stablelm-neox/stablelm-base-alpha-3b-neox-mp2"
python "$SCRIPT" --input_dir "$input_dir" --model_size 3B --output_dir "$output_dir" --num_output_shards 2 --pipeline_parallel


SCRIPT="/cognitive_comp/zhangwenjun/idea/gpt-neox/tools/convert_hf_stablelm_weights_to_neox.py"
input_dir="/cognitive_comp/zhangwenjun/pretrained/stablelm/stablelm-base-alpha-3b"
output_dir="/cognitive_comp/zhangwenjun/pretrained/stablelm-neox/stablelm-base-alpha-3b-neox-mp2"
python "$SCRIPT" --input_dir "$input_dir" --model_size 3B --output_dir "$output_dir" --num_output_shards 2
