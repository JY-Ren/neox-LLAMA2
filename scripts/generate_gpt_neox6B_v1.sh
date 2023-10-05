#!/bin/bash
#SBATCH --job-name=gen_gpt_neox6B_v1 # create a short name for your job
#SBATCH -p batch
#SBATCH -x dgx047
#SBATCH --qos=preemptive
#SBATCH -N 1 # node count
#SBATCH --ntasks-per-node 1 # number of tasks to run per node
#SBATCH --cpus-per-task 20 # cpu-cores per task (>1 if multi-threaded tasks),--cpus-per-task
#SBATCH --gpus-per-node 1 # total cpus for job
#SBATCH -o slurm_log/%x-%j.log # output and error log file names (%x for job id)
#SBATCH -e slurm_log/%x-%j.err # output and error log file names (%x for job id)

CODE_ROOT="/cognitive_comp/ganruyi/gpt-neox/tools/generate_by_hf.py"
hf_model_path="/cognitive_comp/ganruyi/gpt-neox/workspace/gpt_neox6B_v1/hf_model_8000"
prompt="I have a dream."
# prompt="这是一个装满爆米花的袋子，袋子里没有巧克力。 然而，袋子上的标签上写着“巧克力”而不是“爆米花”。 山姆找到了包。 她以前从未见过这个包。 她看不到袋子里装的是什么。 她读了标签。然后她打开包查看里面的物品，她看到包里都是？"
# prompt="怎么实现财富自由？"
# prompt="北京是中国的"
srun python $CODE_ROOT --hf_model_path $hf_model_path --prompt "$prompt" --max_length 300
