#!/bin/bash
#SBATCH --job-name=fastapi_server # create a short name for your job
#SBATCH -p batch
#SBATCH --reservation=acagpt
#SBATCH -N 1 # node count
#SBATCH --ntasks-per-node 1 # number of tasks to run per node
#SBATCH --cpus-per-task 32 # cpu-cores per task (>1 if multi-threaded tasks),--cpus-per-task
#SBATCH --gpus-per-node 2 # total cpus for job
#SBATCH -o /cognitive_comp/ganruyi/gpt-neox/workspace/fastapi_api/%x-%j.log # output and error log file names (%x for job id)
#SBATCH -e /cognitive_comp/ganruyi/gpt-neox/workspace/fastapi_api/%x-%j.err # output and error log file names (%x for job id)

# on dgx001 192.168.190.1
# srun python /cognitive_comp/wuziwei/codes/fastapi_api/gpt_neox.py
# on dgx012 192.168.190.12
# srun --jobid 413916 python /cognitive_comp/wuziwei/codes/fastapi_api/gpt_neox.py
python /cognitive_comp/ganruyi/gpt-neox/workspace/fastapi_api/hf_nopipline_model_server.py
