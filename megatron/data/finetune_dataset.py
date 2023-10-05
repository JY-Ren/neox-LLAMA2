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

import math
import torch
import numpy as np
from typing import List, Tuple
from itertools import zip_longest
from functools import partial

from megatron import mpu, print_rank_0
from megatron.data.indexed_dataset import make_dataset as make_indexed_dataset
from megatron.data.blendable_dataset import BlendableDataset
from megatron.data.samplers import DistributedBatchSampler
from megatron.tokenizer.tokenizer import build_tokenizer
from datasets import load_dataset
from dataclasses import dataclass
from torch.utils.data._utils.collate import default_collate
from sklearn.utils import shuffle


@dataclass
class SFTCollator:
    neox_args = None
    tokenizer = None

    def __call__(self, samples):
        sample_list = []
        for item in samples:
            position_ids = np.arange(self.neox_args.seq_length)
            attention_mask = np.zeros(
                (self.neox_args.seq_length, self.neox_args.seq_length)
            )
            for sidx in item["sample_idx"]:
                attention_mask[sidx[0] : sidx[1], sidx[0] : sidx[1]] = np.tril(
                    np.ones((sidx[1] - sidx[0], sidx[1] - sidx[0]))
                )[: self.neox_args.seq_length, : self.neox_args.seq_length]
                position_ids[sidx[0] : sidx[1]] = np.arange(sidx[1] - sidx[0])[
                    : self.neox_args.seq_length
                ]

            attention_mask.dtype = "int64"
            position_ids.dtype = "int64"
            sample = {
                "text": np.array(
                    item["text"][: self.neox_args.seq_length], dtype=np.int64
                ),
                "loss_mask": np.array(
                    item["loss_mask"][: self.neox_args.seq_length - 1], dtype=np.int64
                ),
                "attention_mask": attention_mask[
                    : self.neox_args.seq_length - 1, : self.neox_args.seq_length - 1
                ],
                "position_ids": position_ids[: self.neox_args.seq_length - 1],
            }
            sample_list.append(sample)
        return default_collate(sample_list)


def make_data_loader(dataset, neox_args):
    """Build dataloader given an input dataset."""
    if dataset is None:
        return None
    # Data parallel arguments.
    world_size = mpu.get_data_parallel_world_size()
    rank = mpu.get_data_parallel_rank()
    global_batch_size = neox_args.batch_size * world_size
    num_workers = neox_args.num_workers

    # Use a simple sampler with distributed batch sampler.
    sampler = torch.utils.data.SequentialSampler(dataset)
    batch_sampler = DistributedBatchSampler(
        sampler=sampler,
        batch_size=global_batch_size,
        drop_last=True,
        rank=rank,
        world_size=world_size,
    )
    # Torch dataloader.
    collator = SFTCollator()
    collator.neox_args = neox_args
    collator.tokenizer = neox_args.tokenizer

    return torch.utils.data.DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collator,
    )


def build_train_valid_test_datasets(data_prefix, seq_length, neox_args):
    """Build train, valid, and test datasets."""
    train_dataset, validation_dataset, test_dataset = None, None, None
    # Indexed dataset.
    dataset = load_dataset(data_prefix)

    batch_size = (
        neox_args.global_num_gpus
        * neox_args.train_micro_batch_size_per_gpu
        * neox_args.gradient_accumulation_steps
    )

    # batch_size =neox_args.batch_size

    eval_iters = (
        neox_args.train_iters // neox_args.eval_interval + 1
    ) * neox_args.eval_iters
    test_iters = neox_args.eval_iters

    if "train" in dataset.keys():
        epoch = (neox_args.train_iters * batch_size) // len(dataset["train"]) + 1
        nd = [d for d in dataset["train"]]
        train_dataset = []
        for _ in range(epoch):
            train_dataset.extend(nd)

    if "validation" in dataset.keys():
        epoch = (eval_iters * batch_size) // len(dataset["validation"]) + 1
        nd = [d for d in dataset["validation"]]
        valid_dataset = []
        for _ in range(epoch):
            valid_dataset.extend(nd)

    if "dev" in dataset.keys():
        epoch = (eval_iters * batch_size) // len(dataset["dev"]) + 1
        nd = [d for d in dataset["dev"]]
        valid_dataset = []
        for _ in range(epoch):
            valid_dataset.extend(nd)

    if "test" in dataset.keys():
        epoch = (test_iters * batch_size) // len(dataset["test"]) + 1
        nd = [d for d in dataset["test"]]
        test_dataset = []
        for _ in range(epoch):
            test_dataset.extend(nd)

    return train_dataset, valid_dataset, test_dataset


def build_finetune_train_valid_test_data_iterators(neox_args):
    """XXX"""

    (train_dataloader, valid_dataloader, test_dataloader) = (None, None, None)

    print_rank_0("> building train, validation, and test datasets ...")

    # Ensure only the first/last pipeline stages have data loaders
    if neox_args.is_pipe_parallel:
        is_first_stage = mpu.get_pipe_parallel_rank() == 0
        is_last_stage = (
            mpu.get_pipe_parallel_rank() == mpu.get_pipe_parallel_world_size() - 1
        )
        pipe_load = is_first_stage or is_last_stage
    else:
        pipe_load = True

    # Data loader only on rank 0 of each model parallel group.
    if mpu.get_model_parallel_rank() == 0 and pipe_load:
        # Number of train/valid/test samples.
        train_iters = neox_args.train_iters
        eval_iters = (train_iters // neox_args.eval_interval + 1) * neox_args.eval_iters
        test_iters = neox_args.eval_iters
        train_val_test_num_samples = [
            train_iters * neox_args.train_batch_size,
            eval_iters * neox_args.train_batch_size,
            test_iters * neox_args.train_batch_size,
        ]

        train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
            data_prefix=neox_args.data_path,
            seq_length=neox_args.seq_length,
            neox_args=neox_args,
        )

        # Build dataloders.
        train_dataloader = make_data_loader(train_ds, neox_args=neox_args)
        valid_dataloader = make_data_loader(valid_ds, neox_args=neox_args)
        test_dataloader = make_data_loader(test_ds, neox_args=neox_args)

        # Flags to know if we need to do training/validation/testing.
        do_train = train_dataloader is not None and neox_args.train_iters > 0
        do_valid = valid_dataloader is not None and neox_args.eval_iters > 0
        do_test = test_dataloader is not None and neox_args.eval_iters > 0
        # Need to broadcast num_tokens and num_type_tokens.
        flags = torch.cuda.LongTensor([int(do_train), int(do_valid), int(do_test)])
    else:
        flags = torch.cuda.LongTensor([0, 0, 0])

    # Broadcast num tokens.
    if neox_args.is_pipe_parallel:
        # Only first/last pipeline stages have data loaders, so pipeline parallelism should
        # broadcast globally instead of just the model parallel group.
        torch.distributed.broadcast(flags, src=0)
    else:
        torch.distributed.broadcast(
            flags,
            mpu.get_model_parallel_src_rank(),
            group=mpu.get_model_parallel_group(),
        )
    neox_args.do_train = flags[0].item()
    neox_args.do_valid = flags[1].item()
    neox_args.do_test = flags[2].item()

    # Shift the start iterations.
    if train_dataloader is not None:
        train_dataloader.batch_sampler.start_iter = (
            neox_args.iteration * neox_args.gradient_accumulation_steps
        ) % len(train_dataloader)
        print_rank_0(
            "setting training data start iteration to {}".format(
                train_dataloader.batch_sampler.start_iter
            )
        )
    if valid_dataloader is not None:
        start_iter_val = (
            (neox_args.iteration * neox_args.gradient_accumulation_steps)
            // neox_args.eval_interval
        ) * neox_args.eval_iters
        valid_dataloader.batch_sampler.start_iter = start_iter_val % len(
            valid_dataloader
        )
        print_rank_0(
            "setting validation data start iteration to {}".format(
                valid_dataloader.batch_sampler.start_iter
            )
        )

    # Build iterators.
    if train_dataloader is not None:
        train_data_iterator = iter(train_dataloader)
    else:
        train_data_iterator = None

    if valid_dataloader is not None:
        valid_data_iterator = iter(valid_dataloader)
    else:
        valid_data_iterator = None

    if test_dataloader is not None:
        test_data_iterator = iter(test_dataloader)
    else:
        test_data_iterator = None

    return train_data_iterator, valid_data_iterator, test_data_iterator


def compile_helper():
    """Compile helper function at runtime. Make sure this
    is invoked on a single process."""
    import os
    import subprocess

    path = os.path.abspath(os.path.dirname(__file__))
    ret = subprocess.run(["make", "-C", path])
    if ret.returncode != 0:
        print("Making C++ dataset helpers module failed, exiting.")
        import sys

        sys.exit(1)
