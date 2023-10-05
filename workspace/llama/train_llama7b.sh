#!/bin/bash
#SBATCH --job-name=train_llama7b # create a short name for your job
#SBATCH -p batch
#SBATCH -x dgx047
#SBATCH --reservation=acagpt
#SBATCH -N 1 # node count
#SBATCH --ntasks-per-node 1 # number of tasks to run per node
#SBATCH --cpus-per-task 100 # cpu-cores per task (>1 if multi-threaded tasks),--cpus-per-task
#SBATCH --gpus-per-node 8 # total cpus for job
#SBATCH -o workspace/llama/%x-%j.log # output and error log file names (%x for job id)
#SBATCH -e workspace/llama/%x-%j.err # output and error log file names (%x for job id)

NNODES=1
GPUS_PER_NODE=8

CODE_ROOT='/cognitive_comp/ganruyi/gpt-neox'
CONFIG_ROOT='/cognitive_comp/ganruyi/gpt-neox/workspace/llama'
MODEL_CONFIG="7b_simple.yml"
# LOCAL_CONFIG="slurm_local.yml"

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=53006

export LAUNCHER="torchrun \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    --max_restarts 0 \
    "

# 因为这里复用的代码是用deepy.py里面的，默认传入了一个script_path，我们这里没用到，所以实际上第二个参数是无效的，可以不用关注。
export RUN_CMD="$CODE_ROOT/train.py $CODE_ROOT/train.py -d $CONFIG_ROOT $MODEL_CONFIG $LOCAL_CONFIG"

srun bash -c '$LAUNCHER --node_rank $SLURM_PROCID $RUN_CMD'
