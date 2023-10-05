NNODES=1
GPUS_PER_NODE=4

CODE_ROOT='/cognitive_comp/yangping/nlp/gpt-neox'
CONFIG_ROOT='/cognitive_comp/yangping/nlp/gpt-neox/workspace/llama/config'
# CODE_ROOT='/opt/gpt-neox'
# CONFIG_ROOT='/opt/workspace/6B_rlhf'

MODEL_CONFIG="7b_sft.yml"

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=$(shuf -n 1 -i 40000-65535)

# export LAUNCHER="torchrun \
#     --nproc_per_node $GPUS_PER_NODE \
#     --nnodes $NNODES \
#     --master_addr $MASTER_ADDR \
#     --master_port $MASTER_PORT \
#     --max_restarts 0 \
#     "

# 因为这里复用的代码是用deepy.py里面的，默认传入了一个script_path，我们这里没用到，所以实际上第二个参数是无效的，可以不用关注。

export RUN_CMD="$CODE_ROOT/train.py $CODE_ROOT/train.py -d $CONFIG_ROOT $MODEL_CONFIG"

#srun -N $NNODES --gres=gpu:$GPUS_PER_NODE --ntasks-per-node=1 --cpus-per-task=8 -o ${MODEL_NAME}-%j.log bash -c '$LAUNCHER --node_rank $SLURM_PROCID $RUN_CMD'

echo "START"
# $LAUNCHER --node_rank 0 $RUN_CMD
python3 -m torch.distributed.run --nproc_per_node $GPUS_PER_NODE $RUN_CMD
