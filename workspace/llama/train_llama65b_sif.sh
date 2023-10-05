#!/bin/bash
#SBATCH --job-name=train_llama65b # create a short name for your job
#SBATCH -p batch
#SBATCH -x dgx[047,037,021]
#SBATCH --reservation=ccp
#SBATCH --qos ccp
#SBATCH -N 20 # node count
#SBATCH --ntasks-per-node 1 # number of tasks to run per node
#SBATCH --cpus-per-task 100 # cpu-cores per task (>1 if multi-threaded tasks),--cpus-per-task
#SBATCH --gpus-per-node 8 # total cpus for job
#SBATCH -o %x-%j.log # output and error log file names (%x for job id)
#SBATCH -e %x-%j.err # output and error log file names (%x for job id)

# README
# singularity 镜像无需安装环境和改动 deepspeed 代码、triton，已经集成 
# 镜像路径：/cognitive_comp/yangqi/images/neox_cu113_0630.sif 
# 挂载目录：-B 挂载代码中涉及到的数据、代码相应目录 -e 环境变量

singularity exec --nv \
	-B /cognitive_comp/yangqi/:/cognitive_comp/yangqi/ \
	-B /home/yangqi/:/home/yangqi/ \
	-B /cognitive_comp/common_data/Huggingface-Models/:/cognitive_comp/common_data/Huggingface-Models/                   \
	/cognitive_comp/yangqi/images/neox_cu113_0630.sif bash /cognitive_comp/yangqi/project/gpt-neox/workspace/neox_test/train_llama65b2.sh
