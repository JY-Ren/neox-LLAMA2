#!/bin/bash
#SBATCH --job-name=eval_llama-7B # create a short name for your job
#SBATCH --nodes=1 # node count
#SBATCH --ntasks-per-node=1 # total number of tasks across all nodes
#SBATCH --cpus-per-task=64 # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=16G # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:hgx:4 # number of gpus per node
#SBATCH -p pol-preempted # number of gpus per node

#SBATCH -o ../log_sft3/%x-%j.log # output and error log file names (%x for job id)
#SBATCH -e ../log_sft3/%x-%j.err # output and error log file names (%x for job id)

singularity exec --nv \
-B /cognitive_comp/yangping/:/cognitive_comp/yangping/ \
-B /home/yangping/:/home/yangping/ \
/cognitive_comp/yangping/containers/neox_cu113.sif bash /cognitive_comp/yangping/nlp/gpt-neox/workspace/llama_eval/script/sft_llama7b1.sh
