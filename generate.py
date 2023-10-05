#!/usr/bin/env python
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

# from megatron.utils import print_rank_0, setup_for_inference_or_eval

from megatron.text_generation_utils import (
    generate_samples_input_from_file,
    generate_samples_from_prompt,
    generate_samples_unconditional,
    generate_samples_interactive,
)

from megatron import print_rank_0
from megatron.initialize import initialize_megatron
from megatron.training import setup_model_and_optimizer
from megatron.neox_arguments import NeoXArgs
import json
from datetime import datetime
from pprint import pprint
from eval_tasks import run_eval_harness
from megatron.logging import tb_wandb_log
from megatron.utils import setup_for_inference_or_eval, init_wandb
from megatron.training import forward_step
import os
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)


def main():
    """
    Generate text/sample model
    """
    neox_args = NeoXArgs.consume_deepy_args()
    deepspeed_main_args = neox_args.get_deepspeed_main_args()
    args = deepspeed_main_args[2:]

    # 传入我们生成的参数列表
    neox_args = NeoXArgs.consume_neox_args(args=args)
    neox_args.configure_distributed_args()
    # tokenizer needs to be build in training in order to set the padding vocab
    neox_args.build_tokenizer()
    # is initialized if tensorboard directory is defined
    neox_args.initialize_tensorboard_writer()

    initialize_megatron(neox_args=neox_args)

    use_cache = False
    model, _, _ = setup_model_and_optimizer(
        neox_args=neox_args,
        use_cache=use_cache,
        iteration=neox_args.iteration,
    )  # we use setup_model_and_optimizer instead of get_model in order to initialize deepspeed
    print_rank_0("Finished loading model")

    model.module.inference_mode(use_cache=use_cache)

    # model, neox_args = setup_for_inference_or_eval(use_cache=True)

    if neox_args.recompute:
        model.module.inference_mode(
            use_cache=False
        )  # don't use kv cache if recomputing
    if neox_args.text_gen_type == "unconditional":
        print_rank_0(
            f"Generating samples unconditionally and saving results to {neox_args.sample_output_file}"
        )
        generate_samples_unconditional(
            neox_args=neox_args,
            model=model,
            number_of_samples=neox_args.num_samples,
            output_file=neox_args.sample_output_file,
            maximum_tokens=neox_args.maximum_tokens,
            recompute=neox_args.recompute,
            temperature=neox_args.temperature,
            top_k=neox_args.top_k,
            top_p=neox_args.top_p,
        )

    elif neox_args.text_gen_type == "input-file":
        print_rank_0(
            f"Generating samples from input file {neox_args.sample_input_file}"
        )
        assert neox_args.sample_input_file is not None
        generate_samples_input_from_file(
            neox_args=neox_args,
            model=model,
            input_file=neox_args.sample_input_file,
            output_file=neox_args.sample_output_file,
            maximum_tokens=neox_args.maximum_tokens,
            prompt_end=neox_args.prompt_end,
            recompute=neox_args.recompute,
            temperature=neox_args.temperature,
            top_k=neox_args.top_k,
            top_p=neox_args.top_p,
        )

    elif neox_args.text_gen_type == "interactive":
        generate_samples_interactive(
            neox_args=neox_args,
            model=model,
            recompute=neox_args.recompute,
            temperature=neox_args.temperature,
            maximum_tokens=neox_args.maximum_tokens,
            prompt_end=neox_args.prompt_end,
            top_k=neox_args.top_k,
            top_p=neox_args.top_p,
        )

    else:
        raise ValueError(
            f"`text-gen-type` either not specified or not recognised: {neox_args.text_gen_type}"
        )


if __name__ == "__main__":
    main()
