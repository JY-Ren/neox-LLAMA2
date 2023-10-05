# Copyright (c) 2021, EleutherAI
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

import os
import sys
import yaml
import argparse

from tqdm import tqdm
import torch
from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)
from megatron.tokenizer import build_tokenizer


"""
A script for converting saved NeoX Checkpoints to Huggingface (HF) compatible GPT-NeoX type models.

Note that this script does not support all NeoX features.
Please investigate carefully whether your model is compatible with all architectures supported by the GPTNeoXForCausalLM class in HF.

(e.g. position embeddings such as AliBi may not be supported by Huggingface's GPT-NeoX architecture.
"""


def load_partitions(
    input_checkpoint_path, mp_partitions, layer_idx
):
# ) -> list[torch.Tensor]:
    """Returns a list containing all weights in a given layer from a model (across MP partitions)"""

    loaded_tp_ranks = [
        torch.load(
            os.path.join(
                input_checkpoint_path,
                f"layer_{layer_idx:02}-model_{i:02}-model_states.pt",
            ),
            map_location="cuda:0"
        )
        for i in range(mp_partitions)
    ]

    return loaded_tp_ranks


def get_key(loaded_config, key, default=None):
    """
    Search for a given key in a NeoX yaml. normalizes underscores -> hyphens
    """
    key = key.replace("_", "-")
    try:
        return loaded_config[key]
    except KeyError:
        key = key.replace("-", "_")
        try:
            return loaded_config[key]
        except KeyError:
            return default


def create_config(neox_config):
    """take in a loaded yaml from NeoX and assign relevant values to HF config.
    Returns: LlamaConfig() object
    """

    class TokenizerArgs:
        # kinda hacky.
        # this is to get something with the same interface as is used in build_tokenizer()
        # without diving into loading a neox_args object or using argparse etc.
        def __init__(self, neox_config):
            self.make_vocab_size_divisible_by = get_key(
                neox_config, "make-vocab-size-divisible-by", default=128
            )
            self.model_parallel_size = get_key(neox_config, "model-parallel-size")
            self.vocab_file = get_key(neox_config, "vocab-file")
            self.merge_file = get_key(neox_config, "merge-file")
            # self.tokenizer_type = get_key(neox_config, "tokenizer-type")
            self.tokenizer_type = "LlamaTokenizer" # 部分checkpoint的config仍为SPM

            self.rank = 0

    args = TokenizerArgs(neox_config)
    tokenizer = build_tokenizer(args)

    # TODO: change the default value here based on discussion regarding `gpt_j_tied` config parameter's default
    use_tied_lns = get_key(neox_config, "gpt-j-tied", False)

    if use_tied_lns:
        raise NotImplementedError(
            """ERROR: Huggingface Transformers does not yet support a single shared layernorm
                per transformer block for GPT-NeoX models trained  w/ GPT-J parallel residuals.
                See https://github.com/EleutherAI/gpt-neox/pull/481 for further details."""
        )

    # set all config values.
    # reference: megatron/model/transformer.py#154
    hidden_size = get_key(neox_config, "hidden-size")
    multiple_of = get_key(neox_config, "llama_mlp_multiple_of", default=256)
    intermediate_size = multiple_of * ((int(2 * hidden_size * 4 / 3) + multiple_of - 1) // multiple_of)

    hf_config = LlamaConfig(
        vocab_size=args.padded_vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=get_key(neox_config, "num-layers"),
        num_attention_heads=get_key(neox_config, "num-attention-heads"),
        intermediate_size=intermediate_size,
        hidden_act=get_key(neox_config, "activation", default="gelu"),
        rotary_pct=get_key(neox_config, "rotary-pct", default=1.0),
        rotary_emb_base=get_key(neox_config, "rotary-emb-base", default=10000),
        max_position_embeddings=get_key(neox_config, "max-position-embeddings"),
        initializer_range=get_key(neox_config, "init-method-std", 0.02),
        layer_norm_eps=get_key(neox_config, "layernorm-epsilon", 1e-5),
        use_cache=True,
        bos_token_id=tokenizer.bos,
        eos_token_id=tokenizer.eod,
        tie_word_embeddings=(not get_key(neox_config, "no-weight-tying", False)),
        use_parallel_residual=get_key(neox_config, "gpt-j-residual", False),
    )
    return hf_config


def convert(input_checkpoint_path, loaded_config):
    """convert a NeoX checkpoint to a HF model format.
    should perform model-parallel merging correctly
    but only supports features allowed by HF GPT-NeoX implementation (e.g. rotary embeddings)
    """

    """
    参数对应：
    word_embeddings.weight           --> embed_tokens.weight
    attention.query_key_value.weight --> [self_attn.q_proj.weight self_attn.k_proj.weight self_attn.v_proj.weight]
    attention.rotary_emb.inv_freq    --> self_attn.rotary_emb.inv_freq
    input_layernorm.scale            --> input_layernorm.weight
    post_attention_layernorm.scale   --> post_attention_layernorm.weight
    mlp.w1.weight                    --> mlp.gate_proj.weight
    mlp.w3.weight                    --> mlp.up_proj.weight
    mlp.w2.weight                    --> mlp.down_proj.weight
    norm.scale                       --> norm.weight
    final_linear.weight              --> lm_head.weight
    """

    hf_config = create_config(loaded_config)

    hf_model = LlamaForCausalLM(
        hf_config
    ).half()  # nice-to-have: lazy init weights somehow?

    mp_partitions = get_key(loaded_config, "model-parallel-size")

    ### Embedding layer ###
    loaded_tp_ranks = load_partitions(input_checkpoint_path, mp_partitions, 0)
    state_dict = {"weight": torch.cat([t["word_embeddings.weight"] for t in loaded_tp_ranks], dim=0)}
    hf_model.model.embed_tokens.load_state_dict(state_dict)

    assert (
        hf_config.vocab_size == hf_model.model.embed_tokens.weight.shape[0]
    ), f"ERROR: calculated vocab size {hf_config.vocab_size} != embed param size {hf_model.model.embed_tokens.weight.shape[0]}"
    ### End Embedding Layer ###

    ### Transformer Layers ###
    for layer_i in tqdm(range(get_key(loaded_config, "num-layers"))):

        ## get layer from hf model
        hf_layer = hf_model.model.layers[layer_i]

        ## + 2 bc of embed layer and a dummy _pre_transformer_block
        loaded_tp_ranks = load_partitions(
            input_checkpoint_path, mp_partitions, layer_i + 2
        )

        state_dict = {}
        ## attention params
        # qkv split
        """
        TODO: 实测`attention.query_key_value.weight`以128维切分给query、key、value，推测为(hidden_size / num_attention_heads)
        ----------------
        | query_part_1 | 
        |  key_part_1  |
        | value_part_1 |
        | query_part_2 | 
        |  key_part_2  |
        | value_part_2 |
            ......
        | query_part_n | 
        |  key_part_n  |
        | value_part_n |
        ----------------
        """
        query_key_value = torch.cat([t["attention.query_key_value.weight"] for t in loaded_tp_ranks], dim=0)
        split_size = int(get_key(loaded_config, "hidden_size") / get_key(loaded_config, "num_attention_heads"))
        q_list, k_list, v_list = [], [], []
        for i in range(0, query_key_value.shape[0], split_size * 3):
            q, k, v = torch.split(query_key_value[i: i + split_size * 3, :], split_size)
            q_list.append(q)
            k_list.append(k)
            v_list.append(v)

        state_dict["self_attn.q_proj.weight"] = torch.cat(q_list)
        state_dict["self_attn.k_proj.weight"] = torch.cat(k_list)
        state_dict["self_attn.v_proj.weight"] = torch.cat(v_list)
        state_dict["self_attn.o_proj.weight"] = torch.cat([t["attention.dense.weight"] for t in loaded_tp_ranks], dim=1)
        # Just take one
        state_dict["self_attn.rotary_emb.inv_freq"] = loaded_tp_ranks[0]["attention.rotary_emb.inv_freq"]

        ## average layernorm stats over mp ranks
        state_dict["input_layernorm.weight"] = (sum([t["input_layernorm.scale"] for t in loaded_tp_ranks])) / mp_partitions
        state_dict["post_attention_layernorm.weight"] = (sum([t["post_attention_layernorm.scale"] for t in loaded_tp_ranks])) / mp_partitions

        ## mlp params
        state_dict["mlp.gate_proj.weight"] = torch.cat([t["mlp.w1.weight"] for t in loaded_tp_ranks], dim=0)
        state_dict["mlp.up_proj.weight"] = torch.cat([t["mlp.w3.weight"] for t in loaded_tp_ranks], dim=0)
        state_dict["mlp.down_proj.weight"] = torch.cat([t["mlp.w2.weight"] for t in loaded_tp_ranks], dim=1)

        ## load state_dict into layer
        hf_layer.load_state_dict(state_dict)
    ### End Transformer Layers ###

    ### Norm Layer ###
    loaded_tp_ranks = load_partitions(
        input_checkpoint_path, mp_partitions, get_key(loaded_config, "num-layers") + 3
    )
    state_dict = {"weight": (sum([t["norm.scale"] for t in loaded_tp_ranks])) / mp_partitions}
    hf_model.model.norm.load_state_dict(state_dict)
    del loaded_tp_ranks
    ### End Norm Layer ###

    ### LM Head ###
    loaded_tp_ranks = load_partitions(
        input_checkpoint_path, mp_partitions, get_key(loaded_config, "num-layers") + 4
    )
    state_dict = {"weight": torch.cat([t["final_linear.weight"] for t in loaded_tp_ranks], dim=0)}
    hf_model.lm_head.load_state_dict(state_dict)
    del loaded_tp_ranks
    ### End LM Head ###

    return hf_model


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Merge MP partitions and convert to HF Model."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        help="Path to NeoX checkpoint, e.g. /path/to/model/global_step143000",
    )
    parser.add_argument(
        "--config_file",
        type=str,
        help="Path to config file for the input NeoX checkpoint.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output dir, where to save the HF Model, tokenizer, and configs",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Set to true in order to upload to the HF Hub directly.",
    )
    args = parser.parse_args()

    with open(args.config_file) as f:
        loaded_config = yaml.full_load(f)

    hf_model = convert(args.input_dir, loaded_config)

    hf_model.save_pretrained(args.output_dir, max_shard_size="1GB")

    # save tokenizer to directory as well, for easy loading of model as a HF model
    tokenizer = LlamaTokenizer.from_pretrained(os.path.dirname(get_key(loaded_config, "vocab-file")))
    print("loaded tokenizer: ", tokenizer)
    tokenizer.save_pretrained(args.output_dir)
    print("tokenizer saved!")

    if args.upload:
        # before running script:
        # `pip install --upgrade transformers`
        # `huggingface-cli login`
        #
        from huggingface_hub import create_repo, HfApi

        repo_name = input("Provide a repository name for the HF Hub: ")
        create_repo(repo_name, repo_type="model", private=False, use_auth_token=True)

        api = HfApi()
        api.upload_folder(
            folder_path=args.output_dir,
            repo_id=repo_name,
            repo_type="model",
        )
