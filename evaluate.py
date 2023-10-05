# Copyright (c) 2021, EleutherAI
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

"""Evaluation tasks - modified from https://github.com/EleutherAI/lm-evaluation-harness"""
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

    # model, neox_args = setup_for_inference_or_eval(use_cache=False)

    results = run_eval_harness(
        model,
        forward_step,
        neox_args,
        eval_tasks=neox_args.eval_tasks,
        bootstrap_iters=10000,
    )
    if neox_args.rank == 0:
        init_wandb(neox_args=neox_args)
        # log to wandb
        for k, v in results["results"].items():
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    k3 = "_".join([k, k2])
                    tb_wandb_log(
                        f"eval/{k3}",
                        v2,
                        neox_args.iteration,
                        use_wandb=neox_args.use_wandb,
                    )
            else:
                tb_wandb_log(
                    f"eval/{k}",
                    v,
                    neox_args.iteration,
                    use_wandb=neox_args.use_wandb,
                )

        pprint(results)
        results_path = (
            f'eval_results_{datetime.now().strftime("%m-%d-%Y-%H-%M-%S")}.json'
        )
        if neox_args.eval_results_prefix:
            results_path = f"{neox_args.eval_results_prefix}_{results_path}"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
