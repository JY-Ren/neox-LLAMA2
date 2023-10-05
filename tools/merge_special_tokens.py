from transformers import  LlamaTokenizer

llama_65b_hf = '/platform_tech/xiajun/PLMs/llama-65b-hf/'
merged_llama_65b_hf = '/shared_space/agpt/models/llama-65b-hf-merged/'

special_tokens = [
    '[START_REF]', 
    '[END_REF]', 
    '[START_AUTHOR_ID]', 
    '[END_AUTHOR_ID]',
    '[START_VENUE]', 
    '[END_VENUE]', 
    '[START_CITATIONS]', 
    '[END_CITATIONS]',
    '[START_DATE]', 
    '[END_DATE]'
]

llama_tokenizer = LlamaTokenizer.from_pretrained(llama_65b_hf)
print(f"Old Llama tokenizer: {len(llama_tokenizer)}")
llama_tokenizer.add_special_tokens({"additional_special_tokens":special_tokens})
llama_tokenizer.save_pretrained(save_directory=merged_llama_65b_hf)

llama_tokenizer = LlamaTokenizer.from_pretrained(merged_llama_65b_hf)
print(f"New Llama tokenizer: {len(llama_tokenizer)}")