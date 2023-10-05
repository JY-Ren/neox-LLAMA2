#!/bin/bash
#SBATCH --job-name=eval_llama-7B # create a short name for your job
#SBATCH --nodes=1 # node count
#SBATCH --ntasks-per-node=1 # total number of tasks across all nodes
#SBATCH --cpus-per-task=16 # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=16G # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:hgx:0 # number of gpus per node
#SBATCH -p pol # number of gpus per node


# SCRIPT='/cognitive_comp/yangping/nlp/gpt-neox/tools/convert_raw_llama_weights_to_neox.py'
SCRIPT='/cognitive_comp/yangping/nlp/gpt-neox/tools/convert_hf_llama_weights_to_neox.py'
# raw_model_path="/cognitive_comp/yangping/checkpoints/llama/llama_ori/"
raw_model_path="/cognitive_comp/sunqianguo/pretrained/sft/sft_model_version/sft_llama_7b_0405_v1/"
# raw_model_path="/cognitive_comp/yangping/checkpoints/llama/sft_7b/checkpoints_d0330-tokenid/llama_7B_sft_79w_step16800"
# raw_model_path="/cognitive_comp/yangping/pretrained_model/llama-7b-hf"
neox_ckpt_path="/cognitive_comp/yangping/checkpoints/llama/ckpt_hf_sft_7b_pp2mp1"
python "$SCRIPT" --input_dir "$raw_model_path" --model_size 7B --output_dir "$neox_ckpt_path" --num_output_shards 1 --pipeline_parallel --num_output_shards_pipeline 2
