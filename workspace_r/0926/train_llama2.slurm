#!/bin/bash
#SBATCH --job-name=pretrain # create a short name for your job
#SBATCH -p batch
#SBATCH -x dgx[006,059,020] # 排除一些节点
#SBATCH --reservation=ccp
#SBATCH --qos ccp
#SBATCH -N 20 # node count
#SBATCH --ntasks-per-node 1 # number of tasks to run per node
#SBATCH --cpus-per-task 100 # cpu-cores per task (>1 if multi-threaded tasks),--cpus-per-task
#SBATCH --gpus-per-node 8 # total cpus for job
#SBATCH -o logs/%x-%j.log # output and error log file names (%x for job id)
#SBATCH -e logs/%x-%j.err # output and error log file names (%x for job id)

NNODES=20
GPUS_PER_NODE=8

CODE_ROOT='/platform_tech/jyren/pretrainpipeline'
CONFIG_ROOT=${CODE_ROOT}'/ren-workspace/0926'
MODEL_CONFIG=${CODE_ROOT}'/ren-workspace/0926/config/70b.yml'

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=$[RANDOM%10000+50000]
export NUMEXPR_MAX_THREADS=256

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
# rm /platform_tech/xiajun/pretrainpipeline/xj-workspace/0625-debug/data/pajama_cc_1_p0_text_document_*
echo "clear up config megatron"

srun bash -c '$LAUNCHER --node_rank $SLURM_PROCID $RUN_CMD'
