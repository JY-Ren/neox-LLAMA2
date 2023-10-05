#!/bin/bash
#SBATCH --job-name=test_panda_2.7B # create a short name for your job
#SBATCH -p batch
#SBATCH -x dgx030
#SBATCH --reservation=acagpt
#SBATCH -N 2 # node count
#SBATCH --ntasks-per-node 1 # number of tasks to run per node
#SBATCH --cpus-per-task 40 # cpu-cores per task (>1 if multi-threaded tasks),--cpus-per-task
#SBATCH --gpus-per-node 8 # total cpus for job



NNODES=2
GPUS_PER_NODE=8

CODE_ROOT='/home/yangping/nlp/gpt-neox'
CONFIG_ROOT='/home/yangping/nlp/gpt-neox/workspace/llama/config/'
MODEL_CONFIG="13b_generate.yml"
# LOCAL_CONFIG="slurm_local.yml"

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=$(shuf -n 1 -i 40000-65535)

export LAUNCHER="torchrun \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    --max_restarts 0 \
    "

# 因为这里复用的代码是用deepy.py里面的，默认传入了一个script_path，我们这里没用到，所以实际上第二个参数是无效的，可以不用关注。
export RUN_CMD="$CODE_ROOT/generate.py $CODE_ROOT/generate.py -d $CONFIG_ROOT $MODEL_CONFIG"

# export RUN_CMD=deepy.py generate.py $CONFIG_ROOT -i prompt.txt -o sample_outputs.txt

# ./deepy.py generate.py $CONFIG_ROOT -i prompt.txt -o sample_outputs.txt

srun bash -c '$LAUNCHER --node_rank $SLURM_PROCID $RUN_CMD'
