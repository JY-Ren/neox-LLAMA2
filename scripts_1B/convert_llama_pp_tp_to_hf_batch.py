# Copyright 2022 EleutherAI and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import gc
import json
import math
import os
import shutil
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
from loguru import logger
import torch
from typing import List
from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer

os.environ['CUDA_VISIBLE_DEVICES']=''

"""
Sample usage:
```
python src/transformers/models/llama/convert_llama_weights_to_hf.py \
    --input_dir /path/to/downloaded/llama/weights --model_size 7B --output_dir /output/path
```
Thereafter, models can be loaded via:
```py
from transformers import LlamaForCausalLM, LlamaTokenizer
model = LlamaForCausalLM.from_pretrained("/output/path")
tokenizer = LlamaTokenizer.from_pretrained("/output/path")
```
Important note: you need to be able to host the whole model in RAM to execute this script (even if the biggest versions
come in several checkpoints they each contain a part of each weight of the model, so we need to load them all in RAM).
"""

INTERMEDIATE_SIZE_MAP = {
    "7B": 11008,
    "13B": 13824,
    "30B": 17920,
    "65B": 22016,
}


NUM_SHARDS = {
    "7B": 1,
    "13B": 1,
    "30B": 1,
    "65B": 1,
}

PARAMS ={
    "1.3B":{"dim": 2048, "multiple_of": 256, "n_heads": 32, "n_layers": 24, "norm_eps": 1e-06, "vocab_size": -1},
    "7B":{"dim": 4096, "multiple_of": 256, "n_heads": 32, "n_layers": 32, "norm_eps": 1e-06, "vocab_size": -1},
    "13B":{"dim": 5120, "multiple_of": 256, "n_heads": 40, "n_layers": 40, "norm_eps": 1e-06, "vocab_size": -1},
    "30B":{"dim": 6656, "multiple_of": 256, "n_heads": 52, "n_layers": 60, "norm_eps": 1e-06, "vocab_size": -1},
    "65B":{"dim": 8192, "multiple_of": 256, "n_heads": 64, "n_layers": 80, "norm_eps": 1e-05, "vocab_size": -1}
}


def compute_intermediate_size(n):
    return int(math.ceil(n * 8 / 3) + 255) // 256 * 256


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def write_json(text, path):
    with open(path, "w") as f:
        json.dump(text, f)

def load_partitions(
    input_checkpoint_path, mp_partitions, layer_idx
) -> List[torch.Tensor]:
    """Returns a list containing all weights in a given layer from a model (across MP partitions)"""

    loaded_tp_ranks = [
        torch.load(
            os.path.join(
                input_checkpoint_path,
                f"layer_{layer_idx:02}-model_{i:02}-model_states.pt",
            ),
            map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
        for i in range(mp_partitions)
    ]

    return loaded_tp_ranks

def merge_layers(loaded_tp_ranks):
    d0_params = ["mlp.w1.weight","mlp.w3.weight","attention.query_key_value.weight"]
    d1_params = ["mlp.w2.weight","attention.dense.weight"]
    dup_params = ["input_layernorm.scale","post_attention_layernorm.scale","attention.rotary_emb.inv_freq"]
    state_dict = {}
    for key in d0_params:
            state_dict[key] = torch.cat([t[key] for t in loaded_tp_ranks], dim=0)
    for key in d1_params:
            state_dict[key] = torch.cat([t[key] for t in loaded_tp_ranks], dim=1)
    for key in dup_params:
            state_dict[key] = loaded_tp_ranks[0][key]
    return state_dict
    
def write_model(model_path, input_base_path, model_size,num_shards,tokenizer_path):
    os.makedirs(model_path, exist_ok=True)
    tmp_model_path = os.path.join(model_path, "tmp")
    os.makedirs(tmp_model_path, exist_ok=True)

    params = PARAMS[model_size]
    n_layers = params["n_layers"]
    n_heads = params["n_heads"]
    n_heads_per_shard = n_heads // num_shards
    dim = params["dim"]
    dims_per_head = dim // n_heads
    base = 10000.0
    hidden_size = params["dim"]
    inv_freq = 1.0 / (base ** (torch.arange(0, dims_per_head, 2).float() / dims_per_head))

    # permute for sliced rotary
    def permute(w):
        return w.view(n_heads, dim // n_heads // 2, 2, dim).transpose(1, 2).reshape(dim, dim)

    def rev_permute_rotary(w):
        return w.view(n_heads, 2, dims_per_head // 2, hidden_size) \
            .transpose(1, 2) \
            .reshape(n_heads, dims_per_head, hidden_size)

    logger.info(f"Fetching all parameters from the checkpoint at {input_base_path}.")
    num_output_shards = 1
    num_heads_per_output_shard = num_heads_per_input_shard = n_heads

    param_count = 0

    if model_size == "1.3B":
        cached = torch.load(input_base_path +"/mp_rank_00_model_states.pt",map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    index_dict = {"weight_map": {}}
    for layer_i in range(n_layers):
        filename = f"pytorch_model-{layer_i + 1}-of-{n_layers + 1}.bin"
        if model_size == "1.3B":
            loaded = dict([(key.replace(f'sequential.{layer_i+2}.','') ,cached['module'][key]) for key in cached['module'].keys() if f'sequential.{layer_i+2}.' in key])
        else:
            loaded = merge_layers(load_partitions(input_base_path,num_shards,layer_i+2))
        sharded_qkv = loaded["attention.query_key_value.weight"]
        sharded_qkv_1 = sharded_qkv.view(
            num_output_shards,
            num_heads_per_output_shard, 3, dims_per_head,
            hidden_size,
        )
        wq = sharded_qkv_1[0, :, 0, :, :]
        wq = rev_permute_rotary(wq).view(num_heads_per_input_shard * dims_per_head, hidden_size)

        wk = sharded_qkv_1[0, :, 1, :, :]
        wk = rev_permute_rotary(wk).view(num_heads_per_input_shard * dims_per_head, hidden_size)

        wv = sharded_qkv_1[0, :, 2, :, :]
        wv = wv.reshape(num_heads_per_input_shard * dims_per_head, hidden_size)

        # Unsharded
        state_dict = {
            f"model.layers.{layer_i}.self_attn.q_proj.weight": permute(
                wq
            ),
            f"model.layers.{layer_i}.self_attn.k_proj.weight": permute(
                wk
            ),
            f"model.layers.{layer_i}.self_attn.v_proj.weight": wv,
            f"model.layers.{layer_i}.self_attn.o_proj.weight": loaded[f"attention.dense.weight"],
            f"model.layers.{layer_i}.mlp.gate_proj.weight": loaded[f"mlp.w1.weight"],
            f"model.layers.{layer_i}.mlp.down_proj.weight": loaded[f"mlp.w2.weight"],
            f"model.layers.{layer_i}.mlp.up_proj.weight": loaded[f"mlp.w3.weight"],
            f"model.layers.{layer_i}.input_layernorm.weight": loaded[f"input_layernorm.scale"],
            f"model.layers.{layer_i}.post_attention_layernorm.weight": loaded[f"post_attention_layernorm.scale"],
        }
        # state_dict[f"model.layers.{layer_i}.self_attn.rotary_emb.inv_freq"] = loaded[f"sequential.{layer_i + 1}.attention.rotary_emb.inv_freq"]
        state_dict[f"model.layers.{layer_i}.self_attn.rotary_emb.inv_freq"] = inv_freq
        for k, v in state_dict.items():
            index_dict["weight_map"][k] = filename
            param_count += v.numel()
        torch.save(state_dict, os.path.join(tmp_model_path, filename))
        logger.info(f"finished process layer {layer_i}")
    filename = f"pytorch_model-{n_layers + 1}-of-{n_layers + 1}.bin"
    logger.info("process embedding")
    word_embeddings = torch.cat([t['word_embeddings.weight'] for t in load_partitions(input_base_path,num_shards,0)],dim=0) if model_size != "1.3B" else cached['module']['sequential.0.word_embeddings.weight']
    logger.info("process norm.scale")
    #loaded =  load_partitions(input_base_path,num_shards,n_layers+3)[0]
    norm_scale = load_partitions(input_base_path,num_shards,n_layers+3)[0]["norm.scale"] if model_size != "1.3B" else cached['module'][f'sequential.{n_layers+3}.norm.scale']
    logger.info("process lm head")
    lm_head = torch.cat([t['final_linear.weight'] for t in load_partitions(input_base_path,num_shards,n_layers+4)],dim=0) if model_size != "1.3B" else cached['module'][f'sequential.{n_layers + 4}.final_linear.weight']
    # Unsharded
    state_dict = {
        "model.embed_tokens.weight": word_embeddings,
        "model.norm.weight": norm_scale,
        "lm_head.weight": lm_head,
    }

    for k, v in state_dict.items():
        index_dict["weight_map"][k] = filename
        param_count += v.numel()
    torch.save(state_dict, os.path.join(tmp_model_path, filename))

    # Write configs
    print("###################Write configs###########################")
    index_dict["metadata"] = {"total_size": param_count * 2}
    write_json(index_dict, os.path.join(tmp_model_path, "pytorch_model.bin.index.json"))

    final_index_dict = {"weight_map": {}}
    
    for k, v in index_dict["weight_map"].items():
        final_index_dict["weight_map"][k] = "pytorch_model.bin"

    write_json(final_index_dict, os.path.join(model_path, "pytorch_model.bin.index.json"))

    config = LlamaConfig(
        hidden_size=dim,
        intermediate_size=compute_intermediate_size(dim),
        num_attention_heads=params["n_heads"],
        num_hidden_layers=params["n_layers"],
        rms_norm_eps=params["norm_eps"],
    )
    config.save_pretrained(tmp_model_path)

    # Make space so we can load the model properly now.
    del state_dict
    del loaded
    gc.collect()

    logger.info("Loading the checkpoint in a Llama model.")
    model = LlamaForCausalLM.from_pretrained(tmp_model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    # Avoid saving this as part of the config.
    del model.config._name_or_path

    logger.info("Saving in the Transformers format.")
    model.save_pretrained(model_path)
    tokenier = LlamaTokenizer.from_pretrained(tokenizer_path)
    tokenier.save_pretrained(model_path)
    shutil.rmtree(tmp_model_path)

def format_model(model_path,save_path):
    model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    model.save_pretrained(save_path)



def write_tokenizer(tokenizer_path, input_tokenizer_path):
    logger.info(f"Fetching the tokenizer from {input_tokenizer_path}.")
    # Initialize the tokenizer based on the `spm` model
    tokenizer = LlamaTokenizer(input_tokenizer_path)
    tokenizer.save_pretrained(tokenizer_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        default="/platform_tech/sf.w/projects/llm_dev/train_scratch/ckpt_new",
        help="Location of LLaMA weights, which contains tokenizer.model and model folders",
    )
    parser.add_argument(
        "--model_size",
        default="1.3B",
        choices=["7B", "13B", "30B", "65B", "1.3B", "tokenizer_only"],
    )
    parser.add_argument(
        "--output_dir",
        default = "/platform_tech/sf.w/projects/llm_dev/train_scratch/ckpt_hf",
    )
    parser.add_argument(
        "--num_shards",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--tokenizer_path",
        default="/shared_space/agpt/models/llama-65b-hf-merged/",
        type=str,
    )
    parser.add_argument("--steps",default="100,5000,10000,14000",type=str)
    args = parser.parse_args()
    logger.info(args)
    if "global_step" in args.input_dir:
        write_model(model_path=args.output_dir,
            input_base_path=args.input_dir,
            model_size=args.model_size,
            num_shards = args.num_shards,
            tokenizer_path = args.tokenizer_path)
    else:
        sub_dirs = os.listdir(args.input_dir)
        if args.steps is not None:
            steps = set(args.steps.split(','))
        else:
            steps = None
        logger.info(f"steps is {steps}")
        for sub_dir in sub_dirs:
            if "global_step" in sub_dir and (steps is None or sub_dir.split('_step')[1] in steps):
                logger.info(f"process {args.input_dir+os.sep+sub_dir}")
                write_model(model_path=args.output_dir+os.sep+"hf_{}".format(sub_dir),
                input_base_path=args.input_dir+os.sep+sub_dir,
                model_size=args.model_size,
                num_shards = args.num_shards,
                tokenizer_path = args.tokenizer_path)
            else:
                logger.info(f"subdir {sub_dir} not process")
                
if __name__ == "__main__":
    main()
