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

import torch
from torch.nn import LayerNorm as LayerNorm
import importlib
import apex 

def get_norm(neox_args):
    if neox_args.norm == "rmsnorm":
        norm = RMSNorm
        eps = neox_args.rms_norm_epsilon
    if neox_args.norm == "llamarmsnorm":
        norm = LlamaRMSNorm
        eps = neox_args.rms_norm_epsilon
    elif neox_args.norm == "layernorm":
        eps = neox_args.layernorm_epsilon
        norm = LayerNorm
    elif neox_args.norm == "scalenorm":
        eps = neox_args.scalenorm_epsilon
        norm = ScaleNorm
    elif neox_args.norm == "apexrmsnorm":
        try:
            from apex.normalization import FusedRMSNorm
            from apex._autocast_utils import _cast_if_autocast_enabled
            print("for debug: use apex RMSNorm")
            norm = ApexRMSNorm
        except:
            print("warning: cannot import apex so use LLaMARMSNorm")
            norm = LlamaRMSNorm
        eps = neox_args.rms_norm_epsilon
    else:
        raise ValueError(f"norm {neox_args.norm} not recognized")
    return norm, eps


class RMSNorm(torch.nn.Module):
    def __init__(self, dim, p=-1.0, eps=1e-8, bias=False):
        """
            Root Mean Square Layer Normalization
        :param dim: model size
        :param p: partial RMSNorm, valid value [0, 1], default -1.0 (disabled)
        :param eps:  epsilon value, default 1e-8
        :param bias: whether use bias term for RMSNorm, disabled by
            default because RMSNorm doesn't enforce re-centering invariance.
        """
        super(RMSNorm, self).__init__()

        self.eps = eps
        self.d = dim
        self.p = p
        self.bias = bias

        self.scale = torch.nn.Parameter(torch.ones(dim))
        self.register_parameter("scale", self.scale)

        if self.bias:
            self.offset = torch.nn.Parameter(torch.zeros(dim))
            self.register_parameter("offset", self.offset)

    def forward(self, x):
        if self.p < 0.0 or self.p > 1.0:
            norm_x = x.norm(2, dim=-1, keepdim=True)
            d_x = self.d
        else:
            partial_size = int(self.d * self.p)
            partial_x, _ = torch.split(x, [partial_size, self.d - partial_size], dim=-1)

            norm_x = partial_x.norm(2, dim=-1, keepdim=True)
            d_x = partial_size

        rms_x = norm_x * d_x ** (-1.0 / 2)
        x_normed = x / (rms_x + self.eps)

        if self.bias:
            return self.scale * x_normed + self.offset

        return self.scale * x_normed


class LlamaRMSNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.register_parameter("scale", self.scale)

    def forward(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.scale.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.scale.dtype)

        return self.scale * hidden_states


class ScaleNorm(torch.nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.g = torch.nn.Parameter(torch.ones(1))
        self.eps = eps

    def forward(self, x):
        n = torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps)
        return x / n * self.g

class FusedRMSNormAffineFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, normalized_shape, eps):
        global fused_layer_norm_cuda
        if fused_layer_norm_cuda is None:
            fused_layer_norm_cuda = importlib.import_module("fused_layer_norm_cuda")
        ctx.normalized_shape = normalized_shape
        ctx.eps = eps
        input_ = input.contiguous()
        weight_ = weight.contiguous()
        output, invvar = fused_layer_norm_cuda.rms_forward_affine(
            input_, ctx.normalized_shape, weight_, ctx.eps)
        ctx.save_for_backward(input_, weight_, invvar)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_, weight_, invvar = ctx.saved_tensors
        grad_input = grad_weight = None
        grad_input, grad_weight = fused_layer_norm_cuda.rms_backward_affine(
           grad_output.contiguous(), invvar, input_, ctx.normalized_shape, weight_, ctx.eps
        )
        return grad_input, grad_weight, None, None

class ApexRMSNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-5, elementwise_affine=True):
        super().__init__()

        global fused_layer_norm_cuda
        fused_layer_norm_cuda = importlib.import_module("fused_layer_norm_cuda")
        

        self.normalized_shape = torch.Size((hidden_size,))
        self.scale = torch.nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.register_parameter("scale", self.scale)

    def fused_rms_norm_affine(self, input, weight, normalized_shape, eps=1e-6):
        from apex._autocast_utils import _cast_if_autocast_enabled
        args = _cast_if_autocast_enabled(input, weight, normalized_shape, eps)
        with torch.cuda.amp.autocast(enabled=False):
            return FusedRMSNormAffineFunction.apply(*args)
        
    def forward(self, input):
        return self.fused_rms_norm_affine(input, self.scale, self.normalized_shape, self.variance_epsilon)
        
    
class ApexRMSNorm2(torch.nn.Module):
    """封装FusedLayerNorm, 
    TODO 继续训练会对不上参数名字 module.weight -> scale，改一下
    """
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        from apex.normalization import FusedRMSNorm
        self.module = FusedRMSNorm(
            normalized_shape=hidden_size,
            eps=eps,
            elementwise_affine=True
        )

    def forward(self, hidden_states):
        return self.module.forward(hidden_states)
