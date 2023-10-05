#!/bin/bash
#SBATCH --job-name=test_fused_kernels # create a short name for your job
#SBATCH -p batch
#SBATCH -x dgx047
#SBATCH --reservation=acagpt
#SBATCH -N 1 # node count
#SBATCH --ntasks-per-node 1 # number of tasks to run per node
#SBATCH --cpus-per-task 4 # cpu-cores per task (>1 if multi-threaded tasks),--cpus-per-task
#SBATCH --gpus-per-node 1 # total cpus for job
#SBATCH -o /cognitive_comp/ganruyi/gpt-neox/megatron/fused_kernels/tests/%x-%j.log # output and error log file names (%x for job id)
#SBATCH -e /cognitive_comp/ganruyi/gpt-neox/megatron/fused_kernels/tests/%x-%j.err # output and error log file names (%x for job id)

CODE_ROOT='/cognitive_comp/ganruyi/gpt-neox/megatron/fused_kernels/tests/test_fused_kernels.py'
python "$CODE_ROOT"
