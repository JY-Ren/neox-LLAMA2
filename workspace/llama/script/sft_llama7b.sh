#!/bin/bash
#SBATCH --job-name=eval_llama-7B # create a short name for your job
#SBATCH --nodes=1 # node count
#SBATCH --ntasks-per-node=1 # total number of tasks across all nodes
#SBATCH --cpus-per-task=64 # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=16G # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:hgx:4 # number of gpus per node
#SBATCH -p pol # number of gpus per node

#SBATCH -o ../log_sft2/%x-%j.log # output and error log file names (%x for job id)
#SBATCH -e ../log_sft2/%x-%j.err # output and error log file names (%x for job id)

NNODES=1
GPUS_PER_NODE=4

CODE_ROOT='/cognitive_comp/yangping/nlp/gpt-neox'
# CODE_ROOT='/cognitive_comp/yangping/nlp/copy/gpt-neox'
# CODE_ROOT='/cognitive_comp/gaoxinyu/gitlab/gpt-neox'
CONFIG_ROOT='/cognitive_comp/yangping/nlp/gpt-neox/workspace/llama_eval/config'
MODEL_CONFIG="7b_sft.yml"
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
export RUN_CMD="$CODE_ROOT/train.py $CODE_ROOT/train.py -d $CONFIG_ROOT $MODEL_CONFIG $LOCAL_CONFIG"

srun bash -c '$LAUNCHER --node_rank $SLURM_PROCID $RUN_CMD'
