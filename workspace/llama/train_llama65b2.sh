NNODES=20
GPUS_PER_NODE=8

CODE_ROOT='/cognitive_comp/yangqi/project/gpt-neox'
CONFIG_ROOT='/cognitive_comp/yangqi/project/gpt-neox/workspace/llama'
MODEL_CONFIG="/cognitive_comp/yangqi/project/gpt-neox/workspace/llama/config/65b.yml"

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=$[RANDOM%10000+50000]

export LAUNCHER="torchrun \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    --max_restarts 0 \
    "

# source activate BBT2
# 因为这里复用的代码是用deepy.py里面的，默认传入了一个script_path，我们这里没用到，所以实际上第二个参数是无效的，可以不用关注。
export RUN_CMD="$CODE_ROOT/train.py $CODE_ROOT/train.py -d $CONFIG_ROOT $MODEL_CONFIG $LOCAL_CONFIG"

# 删除旧配置文件、删除旧长度的 indexmap 缓存
rm $CONFIG_ROOT/config_megatron/*.json
rm /cognitive_comp/yangqi/data/neox_test/arxiv/data_text_document*
echo "clear up config megatron"

srun bash -c '$LAUNCHER --node_rank $SLURM_PROCID $RUN_CMD'
