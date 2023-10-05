python /platform_tech/jyren/pretrainpipeline/scripts_r/convert_llama_pp_tp_to_hf_batch.py \
	--input_dir /shared_space/agpt/experiments/0922_1B_from_scratch/checkpoints \
	--model_size 1.3B \
	--output_dir /shared_space/agpt/experiments/0922_1B_from_scratch/hf \
	--num_shards 4 \
	--tokenizer_path /shared_space/agpt/models/llama2/hf/70B \
	--steps 13000
