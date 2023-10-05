# Copyright (c) 2021 EleutherAI
# This file is based on code by the authors denoted below and has been modified from its original version.
#
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

"""Transformer."""

import math
import torch
import torch.nn.functional as F
import torch.nn as nn

from .norms import get_norm
from megatron import mpu, print_rank_0
from megatron.model.fused_softmax import FusedScaleMaskSoftmax
from megatron.model.activations import get_activation
from megatron.model.utils import exists, get_fusion_type
from megatron.model.positional_embeddings import (
    RotaryEmbedding,
    DynamicRotaryEmbedding,
    apply_rotary_pos_emb,
    apply_rotary_pos_emb_torch,
    AliBi,
)
from megatron.model.fused_bias_dropout import (
    get_bias_dropout_add,
    bias_dropout_add_fused_train,
    bias_dropout_add_fused_inference,
)
from megatron.model.utils import configure_sparse_attention
import time
# flags required to enable jit fusion kernels
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)

""" We use the following notation throughout this file:
     h: hidden size
     n: number of attention heads
     p: number of model parallel partitions
     np: n/p
     hp: h/p
     hn: h/n
     b: batch size
     s: sequence length
     l: number of layers
    Transformer takes input of size [s, b, h] and returns a
    tensor of the same size. We use the following arguments:
        hyperparameters: transformer hyperparameters
        attention_mask_func: a function that takes `unmasked-attention-scores`
            with size [b, np, s, s] and an `attention-mask` and will apply
            the masking. The function should return a masked score of the
            same size [b, np, s, s].
               masked-attention-scores = attention_mask_func(
                                     unmasked-attention-scores, attention-mask)
"""


class ParallelMLP(nn.Module):
    """MLP.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension. At the end, dropout is also
    applied.
    """

    def __init__(
        self, neox_args, init_method, output_layer_init_method, parallel_output=False
    ):
        super().__init__()

        self.activation_func = get_activation(neox_args)
        self.activation_type = neox_args.activation
        self.bias_gelu_fusion = neox_args.bias_gelu_fusion
        self.use_bias_in_mlp = neox_args.use_bias_in_mlp

        # auto scale so geglu has equal parameters
        ff_mult = 4 * 2 / 3 if self.activation_type == "geglu" else 4
        ff_dim = (
            int(ff_mult * neox_args.hidden_size) * 2
            if self.activation_type == "geglu"
            else ff_mult * neox_args.hidden_size
        )
        self.dense_h_to_4h = mpu.ColumnParallelLinear(
            neox_args=neox_args,
            input_size=neox_args.hidden_size,
            output_size=ff_dim,
            gather_output=False,
            init_method=init_method,
            skip_bias_add=True,
            bias=neox_args.use_bias_in_mlp,
        )
        ff_dim_in = ff_dim // 2 if self.activation_type == "geglu" else ff_dim
        # Project back to h.
        self.dense_4h_to_h = mpu.RowParallelLinear(
            neox_args=neox_args,
            input_size=ff_dim_in,
            output_size=neox_args.hidden_size,
            input_is_parallel=True,
            init_method=output_layer_init_method,
            skip_bias_add=True,
            parallel_output=parallel_output,
            bias=neox_args.use_bias_in_mlp,
        )

    def forward(self, hidden_states):

        # [s, b, 4hp]
        intermediate_parallel, bias_parallel = self.dense_h_to_4h(hidden_states)
        if not self.use_bias_in_mlp:
            intermediate_parallel = self.activation_func(intermediate_parallel)
        else:
            if (
                self.activation_type == "gelu" and self.bias_gelu_fusion
            ) or self.activation_type == "geglu":
                intermediate_parallel = self.activation_func(
                    intermediate_parallel, bias_parallel
                )
            else:
                intermediate_parallel = self.activation_func(
                    intermediate_parallel + bias_parallel
                )

        # [s, b, h]
        output, output_bias = self.dense_4h_to_h(intermediate_parallel)
        return output, output_bias


class LLaMAParallelMLP(nn.Module):
    """LLaMA's MLP.
    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension. At the end, dropout is also
    applied.
    """

    def __init__(
        self, neox_args, init_method, output_layer_init_method, parallel_output=False
    ):
        super().__init__()

        self.activation_func = get_activation(neox_args)
        self.activation_type = neox_args.activation
        self.multiple_of = neox_args.llama_mlp_multiple_of

        ff_dim = int(2 * neox_args.hidden_size * 4 / 3)
        ff_dim = int(neox_args.ffn_dim_multiplier * ff_dim)
        ff_dim = self.multiple_of * (
            (ff_dim + self.multiple_of - 1) // self.multiple_of
        )
        ff_dim += neox_args.llama_mlp_extend_hidden_size
        self.w1 = mpu.ColumnParallelLinear(
            neox_args=neox_args,
            input_size=neox_args.hidden_size,
            output_size=ff_dim,
            gather_output=False,
            init_method=init_method,
            skip_bias_add=True,
            bias=False,
        )
        self.w3 = mpu.ColumnParallelLinear(
            neox_args=neox_args,
            input_size=neox_args.hidden_size,
            output_size=ff_dim,
            gather_output=False,
            init_method=init_method,
            skip_bias_add=True,
            bias=False,
        )
        self.w2 = mpu.RowParallelLinear(
            neox_args=neox_args,
            input_size=ff_dim,
            output_size=neox_args.hidden_size,
            input_is_parallel=True,
            init_method=output_layer_init_method,
            skip_bias_add=True,
            parallel_output=parallel_output,
            bias=False,
        )

    def forward(self, hidden_states):
        w1_out, _ = self.w1(hidden_states)
        w3_out, _ = self.w3(hidden_states)
        return self.w2(self.activation_func(w1_out) * w3_out)


class ParallelLinear(nn.Module):
    """
    A Parallel Linear Layer transforming the transformer outputs from hidden_size -> vocab_size
    """

    def __init__(
        self,
        neox_args,
        parallel_output=True,
        init_method=nn.init.xavier_normal_,
        is_last_layer=False,
    ):
        super().__init__()
        parallelism = neox_args.output_layer_parallelism
        if parallelism == "column":
            self.final_linear = mpu.ColumnParallelLinear(
                neox_args=neox_args,
                input_size=neox_args.hidden_size,
                output_size=neox_args.padded_vocab_size,
                bias=False,
                init_method=init_method,
                gather_output=not parallel_output,
                skip_bias_add=False,
                mup_rescale_parameters=is_last_layer,  # rescale params only called if neox_args.use_mup = True, despite it not being included here
            )
        else:
            self.final_linear = mpu.RowParallelLinear(
                neox_args=neox_args,
                input_size=neox_args.hidden_size,
                output_size=neox_args.padded_vocab_size,
                bias=False,
                input_is_parallel=False,
                init_method=init_method,
                parallel_output=parallel_output,
                skip_bias_add=False,
                mup_rescale_parameters=is_last_layer,  # only called if neox_args.use_mup = True, despite it not being included here
            )

    def forward(self, hidden_states):
        return self.final_linear(hidden_states)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )

class ParallelSelfAttention(nn.Module):
    """Parallel self-attention layer abstract class.

    Self-attention layer takes input with size [b, s, h]
    and returns output of the same size.
    """

    def __init__(
        self,
        neox_args,
        attention_mask_func,
        init_method,
        output_layer_init_method,
        layer_number,
        rpe=None,
        rotary=False,
        use_cache=False,
        parallel_output=False,
    ):
        super().__init__()

        self.fp16 = neox_args.precision == "fp16"
        self.bf16 = neox_args.precision == "bfloat16"
        self.multi_query = neox_args.multi_query
        self.attention_mask_func = attention_mask_func
        self.apply_query_key_layer_scaling = neox_args.apply_query_key_layer_scaling
        self.use_cache = use_cache
        self.attention_softmax_in_fp32 = neox_args.attention_softmax_in_fp32
        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True
        self.layer_number = layer_number
        # Per attention head and per partition values.
        world_size = mpu.get_model_parallel_world_size()
        self.hidden_size_per_partition = mpu.divide(neox_args.hidden_size, world_size)
        self.hidden_size_per_attention_head = mpu.divide(
            neox_args.hidden_size, neox_args.num_attention_heads
        )
        self.num_attention_heads_per_partition = mpu.divide(
            neox_args.num_attention_heads, world_size
        )
        self.num_attention_heads_kv = neox_args.num_attention_heads_kv
        self.num_attention_heads_kv_per_partition = mpu.divide(
            neox_args.num_attention_heads_kv, world_size
        )
        self.num_attention_heads_kv_groups = int(neox_args.num_attention_heads // neox_args.num_attention_heads_kv)
        
        self.pos_emb = neox_args.pos_emb
        self.use_length_scale_in_att = neox_args.use_length_scale_in_att

        # Strided linear layer.
        if self.multi_query:
            # self.query = mpu.ColumnParallelLinear(
            #     neox_args=neox_args,
            #     input_size=neox_args.hidden_size,
            #     output_size=neox_args.hidden_size,
            #     gather_output=False,
            #     init_method=init_method,
            #     bias=neox_args.use_bias_in_attn_linear,
            # )

            # self.key_value = torch.nn.Linear(
            #     neox_args.hidden_size,
            #     2 * self.hidden_size_per_attention_head * self.num_attention_heads_kv,
            #     bias=neox_args.use_bias_in_attn_linear,
            # )
    
            self.query = mpu.ColumnParallelLinear(
                neox_args=neox_args,
                input_size=neox_args.hidden_size,
                output_size=neox_args.hidden_size,
                gather_output=False,
                init_method=init_method,
                bias=neox_args.use_bias_in_attn_linear,
            )
            
            self.key_value = mpu.ColumnParallelLinear(
                neox_args=neox_args,
                input_size=neox_args.hidden_size,
                output_size=self.hidden_size_per_attention_head * self.num_attention_heads_kv * 2,
                gather_output=False,
                init_method=init_method,
                bias=neox_args.use_bias_in_attn_linear,
            )


        else:
            self.query_key_value = mpu.ColumnParallelLinear(
                neox_args=neox_args,
                input_size=neox_args.hidden_size,
                output_size=3 * neox_args.hidden_size,
                gather_output=False,
                init_method=init_method,
                bias=neox_args.use_bias_in_attn_linear,
            )

        coeff = None
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        if self.apply_query_key_layer_scaling:
            coeff = max(1, self.layer_number)
            self.norm_factor *= coeff

        if neox_args.use_mup:
            self.norm_factor = self.hidden_size_per_attention_head

        self.rpe = rpe

        if self.pos_emb == "alibi":
            self.alibi_embed = AliBi(
                neox_args.num_attention_heads,
                neox_args.model_parallel_size,
                mpu.get_model_parallel_rank(),
            )

        # TODO: this arg shouldn't need to be passed in - get from neox_args
        if rotary:
            if neox_args.rotary_pct == 1:
                self.rotary_ndims = None
            else:
                assert neox_args.rotary_pct < 1
                self.rotary_ndims = int(
                    self.hidden_size_per_attention_head * neox_args.rotary_pct
                )
            dim = (
                self.rotary_ndims
                if self.rotary_ndims is not None
                else self.hidden_size_per_attention_head
            )
            if neox_args.dynamic_rotary:
                print( f"DynamicRotary is turn on, with ntk_alpah: {neox_args.ntk}, ori_position_embeddings:{neox_args.origin_position_embeedings}" )
                self.rotary_emb = DynamicRotaryEmbedding(dim=dim,base=neox_args.rotary_emb_base,
                precision=neox_args.params_dtype, ntk=neox_args.ntk, 
                ori_position_embeddings=neox_args.origin_position_embeedings)
            else:    
                self.rotary_emb = RotaryEmbedding(
                dim, base=neox_args.rotary_emb_base, precision=neox_args.params_dtype
            )
        else:
            self.rotary_emb = None

        self.attention_type = neox_args.attention_config[layer_number]
        self.use_flash_attention = self.attention_type == "flash"
        self.use_xformer_attention = self.attention_type == "xformer"
        self.use_flash_triton_attention = self.attention_type == "flash_triton"
        self.sparse = self.attention_type not in (
            "global",
            "flash",
            "xformer",
            "flash_triton",
        )
        self.sparse = (
            self.attention_type != "global"
            and not self.use_flash_attention
            and not self.use_xformer_attention
            and not self.use_flash_triton_attention
        )
        if self.sparse:
            self.sparse_attn = configure_sparse_attention(
                neox_args,
                self.attention_type,
                self.num_attention_heads_per_partition,
                mpu=mpu,
            )
        else:
            if self.use_flash_attention:
                # from megatron.model.flash_attention import (
                #     flash_attn_unpadded_qkvpacked_func,
                # )

                # self.flash_attention_function = flash_attn_unpadded_qkvpacked_func
                
                from flash_attn import flash_attn_varlen_qkvpacked_func
                self.flash_attention_function = flash_attn_varlen_qkvpacked_func
                
                if self.pos_emb == "alibi":
                    raise ValueError(
                        "Flash attention is currently not compatible with AliBi positional embeddings. Use sinuisoidal, learned, or rotary embeddings instead."
                    )
                
            elif self.use_xformer_attention:
                from xformers.components.attention import ScaledDotProduct
                from megatron.model.xformer_mha import MultiHeadDispatch

                self.sdp = ScaledDotProduct(seq_len=neox_args.seq_length, causal=True)
                self.xformer_mask_sftmax = MultiHeadDispatch(
                    num_heads=self.num_attention_heads_per_partition,
                    dim_qkv_head=self.hidden_size_per_attention_head,
                    attention=self.sdp,
                )
            elif self.use_flash_triton_attention:
                from megatron.model.flash_attn_triton import flash_attn_func

                self.flash_triton_attention_function = flash_attn_func

            else:
                self.scale_mask_softmax = FusedScaleMaskSoftmax(
                    input_in_fp16=self.fp16,
                    input_in_bf16=self.bf16,
                    fusion_type=get_fusion_type(neox_args),
                    mask_func=self.attention_mask_func,
                    softmax_in_fp32=self.attention_softmax_in_fp32,
                    scale=coeff,
                )

            # Dropout. Note that for a single iteration, this layer will generate
            # different outputs on different number of parallel partitions but
            # on average it should not be partition dependent.
            self.dropout_p = neox_args.attention_dropout
            self.attention_dropout = nn.Dropout(self.dropout_p)

        # Output.
        self.dense = mpu.RowParallelLinear(
            neox_args=neox_args,
            input_size=neox_args.hidden_size,
            output_size=neox_args.hidden_size,
            input_is_parallel=True,
            init_method=output_layer_init_method,
            skip_bias_add=True,
            parallel_output=parallel_output,
            bias=neox_args.use_bias_in_attn_linear,
        )

    def attention(
        self, query_layer, key_layer, value_layer, layer_past, attention_mask
    ):
        # ===================================
        # Raw attention scores. [b, np, s, s]
        # ===================================

        # [b, np, sq, sk]
        output_size = (
            query_layer.size(1),
            query_layer.size(2),
            query_layer.size(0),
            key_layer.size(0),
        )

        # [sq, b, np, hn] -> [sq, b * np, hn]
        query_layer = query_layer.view(
            output_size[2], output_size[0] * output_size[1], -1
        )
        key_layer = key_layer.view(output_size[3], output_size[0] * output_size[1], -1)

        # preallocating result tensor: [b * np, sq, sk]
        matmul_result = torch.empty(
            output_size[0] * output_size[1],
            output_size[2],
            output_size[3],
            dtype=query_layer.dtype,
            device=torch.cuda.current_device(),
        )

        # Raw attention scores. [b * np, sq, sk]
        matmul_result = torch.baddbmm(
            matmul_result,
            query_layer.transpose(0, 1),  # [b * np, sq, hn]
            key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
            beta=0.0,
            alpha=(1.0 / self.norm_factor),
        )

        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.view(*output_size)

        # ==================================================
        # Update attention mask for inference. [b, np, sq, sk]
        # ==================================================

        if self.use_cache:
            with torch.no_grad():
                attention_mask = attention_mask[
                    ..., : attention_scores.size(3), : attention_scores.size(3)
                ]

        # ===========================
        # Attention probs and dropout
        # ===========================

        if exists(self.rpe):
            rpe = self.rpe(query_layer.size(0), key_layer.size(0))
            attention_scores += rpe  # [1, np, sq, sk]

        if self.pos_emb == "alibi":
            attention_scores = self.alibi_embed(attention_scores)

        if self.use_length_scale_in_att:
            length_scale = torch.log(
                torch.sum(1 - attention_mask.type_as(query_layer), dim=-1) + 1
            ).unsqueeze(-1)
            attention_scores = torch.multiply(attention_scores, length_scale)

        # attention scores and attention mask [b, np, sq, sk]
        # print_rank_0("attention_scores:", type(attention_scores))
        attention_probs = self.scale_mask_softmax(attention_scores, attention_mask)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        with mpu.get_cuda_rng_tracker().fork():
            attention_probs = self.attention_dropout(attention_probs)

        # =========================
        # Context layer. [sq, b, hp]
        # =========================

        # value_layer -> context layer.
        # [sk, b, np, hn] --> [b, np, sq, hn]

        # context layer shape: [b, np, sq, hn]
        output_size = (
            value_layer.size(1),
            value_layer.size(2),
            query_layer.size(0),
            value_layer.size(3),
        )

        # change view [sk, b * np, hn]
        value_layer = value_layer.view(
            value_layer.size(0), output_size[0] * output_size[1], -1
        )

        # change view [b * np, sq, sk]
        attention_probs = attention_probs.view(
            output_size[0] * output_size[1], output_size[2], -1
        )

        # matmul: [b * np, sq, hn]
        context_layer = torch.bmm(attention_probs, value_layer.transpose(0, 1))

        # change view [b, np, sq, hn]
        context_layer = context_layer.view(*output_size)
        return context_layer

    def multi_query_attention(
        self, query_layer, key_layer, value_layer, layer_past, attention_mask
    ):
        # ===================================
        # Raw attention scores. [b, np, s, s]
        # ===================================

        # [b, np, sq, sk]
        output_size = (
            query_layer.size(1),
            query_layer.size(2),
            query_layer.size(0),
            key_layer.size(0),
        )

        # [sq, b, np, hn] -> [b, np*sq , hn]
        query_layer = (
            query_layer.transpose(0, 1)
            .transpose(1, 2)
            .reshape(output_size[0], output_size[2] * output_size[1], -1)
        )

        key_layer = (
            key_layer.transpose(0, 1)
            .transpose(1, 2)
            .reshape(output_size[0], output_size[3] * 1, -1)
        )

        # preallocating result tensor: [b, np*sq, sk]
        matmul_result = torch.empty(
            output_size[0],
            output_size[1] * output_size[2],
            output_size[3],
            dtype=query_layer.dtype,
            device=torch.cuda.current_device(),
        )

        # Raw attention scores. [b, np*sq, sk]
        matmul_result = torch.baddbmm(
            matmul_result,
            query_layer,  # [b, sq * np, hn]
            key_layer.transpose(1, 2),  # [b, hn, 1* sk]
            beta=0.0,
            alpha=(1.0 / self.norm_factor),
        )

        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.view(*output_size)

        # ==================================================
        # Update attention mask for inference. [b, np, sq, sk]
        # ==================================================

        if self.use_cache:
            with torch.no_grad():
                attention_mask = attention_mask[
                    ..., : attention_scores.size(3), : attention_scores.size(3)
                ]

        # ===========================
        # Attention probs and dropout
        # ===========================

        if exists(self.rpe):
            rpe = self.rpe(query_layer.size(0), key_layer.size(0))
            attention_scores += rpe  # [1, np, sq, sk]

        if self.pos_emb == "alibi":
            attention_scores = self.alibi_embed(attention_scores)

        # attention scores and attention mask [b, np, sq, sk]
        # print_rank_0("attention_scores:", attention_scores.dtype,attention_scores.shape)

        attention_probs = self.scale_mask_softmax(attention_scores, attention_mask)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        with mpu.get_cuda_rng_tracker().fork():
            attention_probs = self.attention_dropout(attention_probs)

        # =========================
        # Context layer. [sq, b, hp]
        # =========================

        # value_layer -> context layer.
        # [sq, b, np, hn] --> [b, np, sq, hn]

        # context layer shape: [b, np, sq, hn]
        output_size = (
            attention_probs.size(0),
            attention_probs.size(1),
            attention_probs.size(2),
            value_layer.size(3),
        )

        # change view [b, 1*sk, hn]
        value_layer = (
            value_layer.transpose(0, 1)
            .transpose(1, 2)
            .view(output_size[0], 1 * value_layer.size(0), -1)
        )

        # change view [b, np*sq, sk]
        attention_probs = attention_probs.view(
            output_size[0], output_size[1] * output_size[2], -1
        )

        # matmul: [b, np*sq, hn]
        # print('attention_probs',attention_probs.shape)
        # print('value_layer',value_layer.shape)
        context_layer = torch.bmm(attention_probs, value_layer)
        # change view [b, np, sq, hn]
        context_layer = context_layer.view(*output_size)
        return context_layer

    def flash_attention(self, query_layer, key_layer, value_layer):
        # [b, np, sq, sk]
        output_size = (
            query_layer.size(1),
            query_layer.size(2),
            query_layer.size(0),
            key_layer.size(0),
        )
        # [s, b, np, hn] -> [b, s, np, hn] -> [b * s, 1, np, hn]

        query_layer = query_layer.transpose(0, 1).reshape(
            output_size[0] * output_size[2], 1, output_size[1], -1
        )
        key_layer = key_layer.transpose(0, 1).reshape(
            output_size[0] * output_size[3], 1, output_size[1], -1
        )
        value_layer = value_layer.transpose(0, 1).reshape(
            output_size[0] * output_size[3], 1, output_size[1], -1
        )

        # Combined q/k/v into [b * s, 3, np, hn].
        qkv = torch.concat([query_layer, key_layer, value_layer], dim=1)

        batch_size = output_size[0]
        seqlen = output_size[2]
        max_s = seqlen

        cu_seqlens = torch.arange(
            0,
            (batch_size + 1) * seqlen,
            step=seqlen,
            dtype=torch.int32,
            device=qkv.device,
        )
        output = self.flash_attention_function(
            qkv,
            cu_seqlens,
            max_s,
            self.dropout_p if self.training else 0.0,
            softmax_scale=None,
            causal=True,
        )
        # [b * sq, np, hn] -> [b, sq, np, hn]
        matmul_result = output.view(
            output_size[0], output_size[2], output.shape[1], output.shape[2]
        )
        # [b, sq, np, hn] -> [b, np, sq, hn]
        matmul_result = matmul_result.transpose(1, 2)

        return matmul_result

    def flash_triton_attention(
        self, query_layer, key_layer, value_layer, attention_mask
    ):

        # [s, b, np, hn] -> [b, s, np, hn]

        query_layer = query_layer.transpose(0, 1)
        key_layer = key_layer.transpose(0, 1)
        value_layer = value_layer.transpose(0, 1)

        if self.multi_query:
            mh_kv = key_layer.size(2)
            key_layer = key_layer.contiguous().view(
                key_layer.size(0),
                key_layer.size(1),
                1,
                key_layer.size(2) * query_layer.size(3),
            )
            value_layer = value_layer.contiguous().view(
                value_layer.size(0),
                value_layer.size(1),
                1,
                value_layer.size(2) * query_layer.size(3),
            )

            key_layer = key_layer.expand(
                *key_layer.shape[:2],
                int(query_layer.shape[2] / mh_kv),
                key_layer.size(-1)
            )
            value_layer = value_layer.expand(
                *value_layer.shape[:2],
                int(query_layer.shape[2] / mh_kv),
                value_layer.size(-1)
            )

            key_layer = key_layer.contiguous().view(
                key_layer.size(0),
                key_layer.size(1),
                query_layer.size(2),
                query_layer.size(3),
            )
            value_layer = value_layer.contiguous().view(
                value_layer.size(0),
                value_layer.size(1),
                query_layer.size(2),
                query_layer.size(3),
            )

        attention_mask = (attention_mask.type_as(query_layer) * 1.0).masked_fill(
            attention_mask, -9900
        )

        matmul_result = self.flash_triton_attention_function(
            query_layer,
            key_layer,
            value_layer,
            attention_mask,
            False,
            None,
        )

        # [b, sq, np, hn] -> [b, np, sq, hn]
        matmul_result = matmul_result.transpose(1, 2)

        return matmul_result

    def sparse_attention(self, query_layer, key_layer, value_layer, attention_mask):
        # TODO: sparse attn dropout?
        # TODO: pad to block size
        # shape of q/k/v is [sq, b, np, hn] and needs to be transposed to [b, np, sq, hn]
        query_layer, key_layer, value_layer = map(
            lambda t: t.permute(1, 2, 0, 3).contiguous(),
            (query_layer, key_layer, value_layer),
        )
        # output shape [b, np(heads), sq, hn]
        attn_mask = attention_mask.to(query_layer.dtype) * -10000
        if exists(self.rpe):
            rpe = self.rpe(query_layer.size(0), key_layer.size(0))
        else:
            rpe = None
        return self.sparse_attn(
            query_layer, key_layer, value_layer, attn_mask=attn_mask, rpe=rpe
        )

    def xformer_attention(self, query_layer, key_layer, value_layer, attention_mask):
        input_query_size = query_layer.shape
        query_layer = query_layer.transpose(0, 1).view(
            input_query_size[1],
            input_query_size[0],
            input_query_size[2] * input_query_size[3],
        )
        key_layer = key_layer.transpose(0, 1).view(
            input_query_size[1],
            input_query_size[0],
            input_query_size[2] * input_query_size[3],
        )
        value_layer = value_layer.transpose(0, 1).reshape(
            input_query_size[1],
            input_query_size[0],
            input_query_size[2] * input_query_size[3],
        )

        attention_mask = attention_mask.squeeze(dim=1)

        output = self.xformer_mask_sftmax(
            query_layer, key_layer, value_layer, attention_mask
        )
        # [b,sq,np*nh] [b, np, sq, hn]
        output = output.view(
            input_query_size[1],
            input_query_size[0],
            input_query_size[2],
            input_query_size[3],
        ).permute(0, 2, 1, 3)
        return output

    def forward(self, hidden_states, attention_mask, layer_past=None):

        # hidden_states: [sq, b, h]

        # =====================
        # Query, Key, and Value
        # =====================

        if self.multi_query:
            kv_input = hidden_states
            # Attention heads [sq, b, h] --> [sq, b, (mh_kv * hn * 2)]
            mixed_kv_layer, _ = self.key_value(kv_input)
            # Reduce the KV gradients in the tensor-parallel direction.
            # This is different from multi-head attention which reduces the KV input,
            # mixed_kv_layer = mpu.copy_to_model_parallel_region(mixed_kv_layer)

            # [sq, b, (mh_kv * hn * 2)] --> [sq, b, mh_kv, hn * 2]
            # new_tensor_shape = mixed_kv_layer.size()[:-1] + (
            #     self.num_attention_heads_kv,
            #     2 * self.hidden_size_per_attention_head,
            # )
            
            new_tensor_shape = mixed_kv_layer.size()[:-1] + (
                self.num_attention_heads_kv_per_partition,
                2 * self.hidden_size_per_attention_head,
            )

            mixed_kv_layer = mixed_kv_layer.view(*new_tensor_shape)
            # [sq, b, np, 2 * hn] --> 2 [sq, b, np, hn]
            (key_layer, value_layer) = mpu.split_tensor_along_last_dim(
                mixed_kv_layer, 2
            )

            key_layer = repeat_kv(key_layer, self.num_attention_heads_kv_groups) 
            value_layer = repeat_kv(value_layer, self.num_attention_heads_kv_groups)

            # Attention head [sq, b, h] --> [sq, b, np * hn]
            query_layer, _ = self.query(hidden_states)
            # [sq, b, np * hn] --> [sq, b, np, hn]
            new_tensor_shape = query_layer.size()[:-1] + (
                self.num_attention_heads_per_partition,
                self.hidden_size_per_attention_head,
            )
            query_layer = query_layer.view(*new_tensor_shape)

        else:
            # Attention heads [sq, b, h] --> [sq, b, (np * 3 * hn)]
            mixed_x_layer, _ = self.query_key_value(hidden_states)

            # [sq, b, (np * 3 * hn)] --> [sq, b, np, 3 * hn]
            new_tensor_shape = mixed_x_layer.size()[:-1] + (
                self.num_attention_heads_per_partition,
                3 * self.hidden_size_per_attention_head,
            )
            mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

            # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn]
            (query_layer, key_layer, value_layer) = mpu.split_tensor_along_last_dim(
                mixed_x_layer, 3
            )

        if exists(self.rotary_emb):
            if exists(self.rotary_ndims):
                # partial rotary
                query_rot, query_pass = (
                    query_layer[..., : self.rotary_ndims],
                    query_layer[..., self.rotary_ndims :],
                )
                key_rot, key_pass = (
                    key_layer[..., : self.rotary_ndims],
                    key_layer[..., self.rotary_ndims :],
                )
            else:
                # full rotary
                query_rot, key_rot = query_layer, key_layer
            apply_rotary_fn = (
                apply_rotary_pos_emb_torch if self.bf16 else apply_rotary_pos_emb
            )

            seq_len = key_layer.shape[0]
            offset = 0
            if exists(layer_past) and layer_past.numel() > 0:
                offset = layer_past[0].shape[0]
                seq_len += offset
            cos, sin = self.rotary_emb(value_layer, seq_len=seq_len)
            query_layer, key_layer = apply_rotary_fn(
                query_rot, key_rot, cos, sin, offset=offset
            )

            if exists(self.rotary_ndims):
                query_layer = torch.cat((query_layer, query_pass), dim=-1)
                key_layer = torch.cat((key_layer, key_pass), dim=-1)

        # ==================================
        # Cache key and value for inference
        # ==================================

        if exists(layer_past) and layer_past.numel() > 0:
            past_key, past_value = layer_past
            key_layer = torch.cat((past_key.type_as(key_layer), key_layer), dim=0)
            value_layer = torch.cat(
                (past_value.type_as(value_layer), value_layer), dim=0
            )

        if self.use_cache:
            present = torch.stack((key_layer, value_layer))

        if self.use_flash_attention:
            context_layer = self.flash_attention(query_layer, key_layer, value_layer)
        elif self.use_flash_triton_attention:
            context_layer = self.flash_triton_attention(
                query_layer, key_layer, value_layer, attention_mask
            )
        elif self.use_xformer_attention:
            context_layer = self.xformer_attention(
                query_layer, key_layer, value_layer, attention_mask
            )
        elif not self.sparse:
            if self.multi_query:
                context_layer = self.multi_query_attention(
                    query_layer, key_layer, value_layer, layer_past, attention_mask
                )
            else:
                context_layer = self.attention(
                    query_layer, key_layer, value_layer, layer_past, attention_mask
                )
        else:
            context_layer = self.sparse_attention(
                query_layer, key_layer, value_layer, attention_mask
            )

        # [b, np, sq, hn] --> [sq, b, np, hn]
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

        # [sq, b, np, hn] --> [sq, b, hp]
        new_context_layer_shape = context_layer.size()[:-2] + (
            self.hidden_size_per_partition,
        )
        context_layer = context_layer.view(*new_context_layer_shape)

        # =================
        # Output. [sq, b, h]
        # =================

        output, bias = self.dense(context_layer)

        if self.use_cache:
            output = [output, present]

        return output, bias


class ParallelTransformerLayer(nn.Module):
    """A single transformer layer.

    Transformer layer takes input with size [b, s, h] and returns an
    output of the same size.
    """

    def __init__(
        self,
        neox_args,
        attention_mask_func,
        init_method,
        output_layer_init_method,
        layer_number,
        rpe=None,
        rotary=False,
        use_cache=False,
    ):

        super().__init__()
        self.layer_number = layer_number

        norm, eps = get_norm(neox_args)

        # Layernorm on the input data.
        self.input_layernorm = norm(neox_args.hidden_size, eps=eps)
        self.use_cache = use_cache

        self.hidden_dropout = neox_args.hidden_dropout
        self.bias_dropout_fusion = neox_args.bias_dropout_fusion
        self.gpt_j_residual = neox_args.gpt_j_residual
        self.gpt_j_tied = neox_args.gpt_j_tied
        self.mlp_type = neox_args.mlp_type
        self.deepnorm = neox_args.deepnorm
        self.residual_norm = neox_args.residual_norm
        self.residual_input_norm_scale = neox_args.residual_input_norm_scale

        if self.gpt_j_residual:
            self.reduce = mpu.mappings.reduce_from_model_parallel_region

        # Self attention.
        self.attention = ParallelSelfAttention(
            neox_args=neox_args,
            attention_mask_func=attention_mask_func,
            init_method=init_method,
            output_layer_init_method=output_layer_init_method,
            layer_number=layer_number,
            rpe=rpe,
            use_cache=self.use_cache,
            rotary=rotary,
            parallel_output=self.gpt_j_residual,
        )

        # Layernorm on the output of the attention layer.
        # If GPT-J residuals are used, this is surpurfulous but leaving it in
        # leads to cleaner code
        if not self.gpt_j_tied:
            self.post_attention_layernorm = norm(neox_args.hidden_size, eps=eps)

        # MLP
        if neox_args.mlp_type == "llama":
            self.mlp = LLaMAParallelMLP(
                neox_args=neox_args,
                init_method=init_method,
                output_layer_init_method=output_layer_init_method,
                parallel_output=self.gpt_j_residual,
            )
        elif neox_args.mlp_type == "regular":
            self.mlp = ParallelMLP(
                neox_args=neox_args,
                init_method=init_method,
                output_layer_init_method=output_layer_init_method,
                parallel_output=self.gpt_j_residual,
            )
        else:
            raise KeyError(neox_args.mlp_type)

        self.layer_past = None  # used to cache k/v pairs in inference

    def _get_bias_dropout(self):
        if self.bias_dropout_fusion:
            fn = (
                bias_dropout_add_fused_train
                if self.training
                else bias_dropout_add_fused_inference
            )
        else:
            fn = get_bias_dropout_add(self.training)
        return fn

    def prenorm_forward(self, x, attention_mask, layer_past=None):
        # print(f'self.layer_number{self.layer_number}',x)
        bias_dropout_fn = self._get_bias_dropout()
        # x: [b, s, h]
        if self.gpt_j_residual:
            # pseudocode:
            # x = x + attn(ln(x)) + mlp(ln(x))
            # this means we can avoid doing the allreduce in the attn / mlp outputs
            # to save communication time (we can do a single allreduce after we add mlp / attn outputs).
            # due to a bug, the two layernorms are not tied in GPT-NeoX-20B. This is non-desirable, but
            # we preserve the functionality for backwards compatibility

            residual = x
            # applies the correct normalization depending on if the norms are tied
            if self.gpt_j_tied:
                x = self.input_layernorm(x)
                x1, x2 = x, x
            else:
                x1, x2 = self.input_layernorm(x), self.post_attention_layernorm(x)

            # attention operator
            attention_output, attention_bias = self.attention(
                x1, attention_mask, layer_past=layer_past
            )
            if self.use_cache:
                attention_output, presents = attention_output
                self.layer_past = presents

            with torch.enable_grad():
                if attention_bias is not None:
                    attention_output = bias_dropout_fn(
                        attention_output,
                        bias=attention_bias.expand_as(attention_output),
                        residual=None,
                        prob=self.hidden_dropout,
                    )
                else:
                    attention_output = torch.nn.functional.dropout(
                        attention_output,
                        p=self.hidden_dropout,
                        training=self.training,
                    )
            # mlp operator
            mlp_output, mlp_bias = self.mlp(x2)
            with torch.enable_grad():
                if mlp_bias is not None:
                    output = bias_dropout_fn(
                        mlp_output,
                        bias=mlp_bias.expand_as(mlp_output),
                        residual=attention_output,
                        prob=self.hidden_dropout,
                    )
                else:
                    output = torch.nn.functional.dropout(
                        mlp_output + attention_output,
                        p=self.hidden_dropout,
                        training=self.training,
                    )

            # output = (x + attn(ln(x)) + mlp(ln(x))
            output = residual + self.reduce(output)
        else:
            # pseudocode:
            # x = x + attn(ln1(x))
            # x = x + mlp(ln2(x))

            residual = x

            # x = x + attn(ln1(x))
            t0=time.time()
            attention_output, attention_bias = self.attention(
                self.input_layernorm(x), attention_mask, layer_past=layer_past
            )
            # print_rank_0("self.attention=====", time.time()-t0)
            if self.use_cache:
                attention_output, presents = attention_output
                self.layer_past = presents
            with torch.enable_grad():
                if attention_bias is not None:
                    attention_output = bias_dropout_fn(
                        attention_output,
                        bias=attention_bias.expand_as(residual),
                        residual=residual,
                        prob=self.hidden_dropout,
                    )
                else:
                    attention_output = (
                        torch.nn.functional.dropout(
                            attention_output,
                            p=self.hidden_dropout,
                            training=self.training,
                        )
                        + residual
                    )

            # output = x + mlp(ln2(x))
            t0=time.time()
            mlp_output, mlp_bias = self.mlp(
                self.post_attention_layernorm(attention_output)
            )
            # print_rank_0("self.mlp=====", time.time()-t0)
            with torch.enable_grad():
                if self.mlp_type == "llama":
                    assert mlp_bias is None
                    output = mlp_output + attention_output
                else:
                    output = bias_dropout_fn(
                        mlp_output,
                        bias=mlp_bias.expand_as(attention_output),
                        residual=attention_output,
                        prob=self.hidden_dropout,
                    )
        return output

    def deepnorm_forward(self, x, attention_mask, layer_past=None):
        bias_dropout_fn = self._get_bias_dropout()

        x = self.input_layernorm(x)
        residual = x

        # x = ln1(x + attn(x))
        attention_output, attention_bias = self.attention(
            x, attention_mask, layer_past=layer_past
        )
        alpha = (2 * self.layer_number) ** 0.5
        if self.use_cache:
            attention_output, presents = attention_output
            self.layer_past = presents
        with torch.enable_grad():
            if attention_bias is not None:
                attention_output = bias_dropout_fn(
                    attention_output,
                    bias=attention_bias.expand_as(residual * alpha),
                    residual=residual * alpha,
                    prob=self.hidden_dropout,
                )
            else:
                attention_output = (
                    torch.nn.functional.dropout(
                        attention_output, p=self.hidden_dropout, training=self.training
                    )
                    + residual * alpha
                )

        attention_output = self.post_attention_layernorm(attention_output)
        mlp_output, mlp_bias = self.mlp(attention_output)

        with torch.enable_grad():
            if self.mlp_type == "llama":
                assert mlp_bias is None
                output = mlp_output + attention_output * alpha
            else:
                output = bias_dropout_fn(
                    mlp_output,
                    bias=mlp_bias.expand_as(attention_output * alpha),
                    residual=attention_output * alpha,
                    prob=self.hidden_dropout,
                )

        return output

    def residualnorm_forward(self, data, attention_mask, layer_past=None):
        bias_dropout_fn = self._get_bias_dropout()
        x, res = data
        x = self.input_layernorm(x)
        residual = x

        # x = ln1(x + attn(x))
        # print_rank_0('layer_number:',self.layer_number)
        attention_output, attention_bias = self.attention(
            x, attention_mask, layer_past=layer_past
        )
        if self.use_cache:
            attention_output, presents = attention_output
            self.layer_past = presents

        res = (
            res + attention_output * self.residual_input_norm_scale
        )  # self.res_input_norms[0](x)
        with torch.enable_grad():
            if attention_bias is not None:
                attention_output = bias_dropout_fn(
                    attention_output,
                    bias=attention_bias.expand_as(residual),
                    residual=residual,
                    prob=self.hidden_dropout,
                )
            else:
                attention_output = (
                    torch.nn.functional.dropout(
                        attention_output, p=self.hidden_dropout, training=self.training
                    )
                    + residual
                )

        attention_output = self.post_attention_layernorm(attention_output)

        mlp_output, mlp_bias = self.mlp(attention_output)

        res = res + mlp_output * self.residual_input_norm_scale

        with torch.enable_grad():
            if self.mlp_type == "llama":
                assert mlp_bias is None
                output = mlp_output + attention_output
            else:
                output = bias_dropout_fn(
                    mlp_output,
                    bias=mlp_bias.expand_as(attention_output),
                    residual=attention_output,
                    prob=self.hidden_dropout,
                )
        return (output, res)

    def forward(self, x, attention_mask, layer_past=None):
        layer_past = layer_past if layer_past is not None else self.layer_past

        if self.deepnorm:
            output = self.deepnorm_forward(x, attention_mask, layer_past)
        elif self.residual_norm:
            output = self.residualnorm_forward(x, attention_mask, layer_past)
        else:
            output = self.prenorm_forward(x, attention_mask, layer_past)
        return output


class ParallelTransformerLayerPipe(ParallelTransformerLayer):
    """Extends ParallelTransformerLayer to forward attention_mask through the pipeline."""

    def forward(self, args):
        assert (
            len(args) == 2
        ), "ParallelTransformerLayerPipe expects 2 arguments - hidden_states and attention_mask"
        hidden_states, attention_mask = args
        # we are returning just [hidden_states, mask]
        return super().forward(hidden_states, attention_mask), attention_mask


class ParallelLinearPipe(ParallelLinear):
    """Another helper class to pass presents through to the output when doing inference with a Pipe Parallel model"""

    def forward(self, args):
        assert isinstance(
            args, torch.Tensor
        ), "ParallelLinearPipe expects a single argument - hidden_states"
        logits, bias = super().forward(args)
        return logits


class NormPipe(nn.Module):
    """Just a helper class to pass presents through to the output when doing inference with a Pipe Parallel model"""

    def __init__(self, neox_args, norm_class, hidden_size, eps):
        super().__init__()
        self.residual_norm = neox_args.residual_norm
        self.residual_alpha = neox_args.residual_alpha
        self.norm = norm_class(hidden_size, eps=eps)
        if self.residual_norm:
            self.norm_residual = norm_class(hidden_size, eps=eps)

    def forward(self, args):
        # assert not isinstance(
        #     args, tuple
        # ), "NormPipe should only receive a single tensor as input"

        if self.residual_norm:
            return self.norm(args[0]) * self.residual_alpha + self.norm_residual(
                args[1]
            )
        else:
            return self.norm(args)


def parallel_lm_logits(input_, word_embeddings_weight, parallel_output, bias=None):
    """LM logits using word embedding weights."""
    # Parallel logits.
    input_parallel = mpu.copy_to_model_parallel_region(input_)

    # Matrix multiply.
    if bias is None:
        logits_parallel = F.linear(input_parallel, word_embeddings_weight)
    else:
        logits_parallel = F.linear(input_parallel, word_embeddings_weight, bias)

    # Gather if needed.
    if parallel_output:
        return logits_parallel

    return mpu.gather_from_model_parallel_region(logits_parallel)
