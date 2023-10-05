

SCRIPT="/cognitive_comp/yangqi/project/gpt-neox/tools/convert_llama_to_hf.py"
BATCH_SCRIPT="/cognitive_comp/yangqi/project/gpt-neox/tools/convert_llama_pp_tp_to_hf_batch.py"

python $BATCH_SCRIPT --input_dir "/shared_space/agpt/0627/checkpoints/global_step1000" \
               --model_size 30B \
               --output_dir "/shared_space/agpt/0627/hf-checkpoints/global_step1000"\
               --num_shards 4 \
               --tokenizer_path /shared_space/agpt/models/llama-65b-hf-merged/ 
