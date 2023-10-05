import argparse
from transformers import (
    GPTNeoXForCausalLM,
    GPTNeoXTokenizerFast,
    LlamaForCausalLM,
    LlamaTokenizer,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate by HF Model.")
    parser.add_argument(
        "--hf_model_path",
        type=str,
        help="Path to NeoX hf models",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="请把下面这句话翻译成中文。Hello world!",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--num_return_sequences",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--model_type",
        choices=["neox", "llama"],
        type=str,
        default="neox",
    )

    args = parser.parse_args()

    if args.model_type == "neox":
        model = GPTNeoXForCausalLM.from_pretrained(
            args.hf_model_path, device_map="auto"
        )
        tokenizer = GPTNeoXTokenizerFast.from_pretrained(args.hf_model_path)
    else:
        model = LlamaForCausalLM.from_pretrained(args.hf_model_path, device_map="auto")
        tokenizer = LlamaTokenizer.from_pretrained(args.hf_model_path)
    print("prompt is:", args.prompt)
    input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids
    print("input_ids:", input_ids)

    gen_tokens = model.generate(
        input_ids,
        do_sample=True,
        temperature=1.0,
        max_length=args.max_length,
        num_return_sequences=args.num_return_sequences,
    )
    gen_texts = tokenizer.batch_decode(gen_tokens)
    for gen_text in gen_texts:
        print(gen_text)
