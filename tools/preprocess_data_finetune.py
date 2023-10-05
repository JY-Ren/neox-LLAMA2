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

"""Processing data for pretraining."""

import argparse
import multiprocessing
import os
import sys
import json

import lm_dataformat as lmd
import numpy as np

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)
import time
from tqdm import tqdm
import torch
import ftfy

from megatron.tokenizer import build_tokenizer
from megatron.data import indexed_dataset
from threading import Semaphore
from datasets import load_dataset
from dataclasses import dataclass
from torch.utils.data._utils.collate import default_collate
from sklearn.utils import shuffle

LOG_DATA_NUM = 10


class SampleConcat:
    def __init__(self, neox_args, tokenizer):
        self.neox_args = neox_args
        self.tokenizer = tokenizer

    def encode(self, item):
        query_response = []
        if len(item["prompt"]) <= len(item["output"]):
            for t in range(len(item["prompt"])):
                query_response.append(item["prompt"][t])
                query_response.append(item["output"][t])
        else:
            for t in range(len(item["output"])):
                query_response.append(item["prompt"][t])
                query_response.append(item["output"][t])
            query_response.append(item["prompt"][len(item["output"])])
        query_s = ""
        for t in range(len(query_response) - 1):
            role = "Human:" if t % 2 == 0 else "Assistant:"
            query_s = query_s + role + query_response[t] + "\n"
        role = "Assistant:" if role == "Human:" else "Human:"
        query = query_s + role
        response = query_response[-1]
        query = self.tokenizer.tokenize(query)
        response = self.tokenizer.tokenize(response)[1:]
        tokens = query + response + [self.tokenizer.eod_id]
        loss_mask = [0] * (len(query) - 1) + [1] * (len(response) + 1) + [0]
        return {"tokens": tokens, "loss_mask": loss_mask}

    def concat(self, samples, split, epoch):
        sample_list = []
        e = 0
        print("epoch", epoch)
        for e in tqdm(range(epoch)):
            idx = 0
            samples = shuffle(samples)

            while idx < len(samples):
                tokens = []
                loss_mask = []
                sample_idx = []
                cur = 0
                while len(tokens) < self.neox_args.seq_length and idx < len(samples):
                    encode_result = self.encode(samples[idx])
                    if (
                        len(tokens) + len(encode_result["tokens"])
                        > self.neox_args.seq_length
                        and tokens != []
                    ):
                        break
                    tokens.extend(encode_result["tokens"])
                    loss_mask.extend(encode_result["loss_mask"])
                    sample_idx.append([cur, cur + len(encode_result["tokens"])])
                    idx += 1
                    cur += len(encode_result["tokens"])
                tokens = tokens + [self.tokenizer.eod_id] * self.neox_args.seq_length
                loss_mask = loss_mask + [0] * self.neox_args.seq_length
                sample = {
                    "text": tokens[: self.neox_args.seq_length],
                    "loss_mask": loss_mask[: self.neox_args.seq_length - 1],
                    "sample_idx": sample_idx,
                }
                sample_list.append(sample)

            print(
                "split: {}, epoch: {}, samples: {}, samples_concat: {}".format(
                    split, e + 1, len(samples), len(sample_list)
                )
            )
        return sample_list


def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title="input data")
    group.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input jsonl files or lmd archive(s) - if using multiple archives, put them in a comma separated "
        "list",
    )

    group = parser.add_argument_group(title="tokenizer")
    group.add_argument(
        "--tokenizer-type",
        type=str,
        required=True,
        choices=[
            "HFGPT2Tokenizer",
            "HFTokenizer",
            "GPT2BPETokenizer",
            "CharLevelTokenizer",
            "TiktokenTokenizer",
            "HFGPTNeoXTokenizerFast",
            "SPMTokenizer",
        ],
        help="What type of tokenizer to use.",
    )
    group.add_argument(
        "--vocab-file", type=str, default=None, help="Path to the vocab file"
    )

    group.add_argument("--ftfy", action="store_true", help="Use ftfy to clean text")
    group = parser.add_argument_group(title="output data")
    group.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to binary output file without suffix",
    )

    group = parser.add_argument_group(title="runtime")
    group.add_argument(
        "--workers", type=int, default=1, help="Number of worker processes to launch"
    )
    group.add_argument(
        "--log-interval",
        type=int,
        default=100,
        help="Interval between progress updates",
    )
    group.add_argument(
        "--epoch",
        type=int,
        default=1,
        help="Interval between progress updates",
    )
    group.add_argument(
        "--seq_length",
        type=int,
        default=512,
        help="Interval between progress updates",
    )
    group.add_argument(
        "--log-data",
        action="store_true",
        default=False,
        help="whether log data",
    )
    args = parser.parse_args()
    args.keep_empty = False

    # some default/dummy values for the tokenizer
    args.rank = 0
    args.make_vocab_size_divisible_by = 128
    args.model_parallel_size = 1

    return args


def save_data(data, file_path):
    sv_dir = "/".join(file_path.split("/")[:-1]) + "/"
    if not os.path.exists(sv_dir):
        os.makedirs(sv_dir)
    with open(file_path, "w", encoding="utf8") as f:
        for line in data:
            json_data = json.dumps(line, ensure_ascii=False)
            f.write(json_data + "\n")


def main():
    args = get_args()
    tokenizer = build_tokenizer(args)
    dataset = load_dataset(args.input)
    concat_model = SampleConcat(args, tokenizer)

    if "train" in dataset.keys():
        train_dataset = concat_model.concat(
            [d for d in dataset["train"]], "train", args.epoch
        )
        save_data(train_dataset, os.path.join(args.output, "train.json"))
        print("train done")

    if "validation" in dataset.keys():
        valid_dataset = concat_model.concat(
            [d for d in dataset["validation"]], "valid", args.epoch
        )
        save_data(valid_dataset, os.path.join(args.output, "validation.json"))
        print("validation done")

    if "dev" in dataset.keys():
        valid_dataset = concat_model.concat(
            [d for d in dataset["dev"]], "valid", args.epoch
        )
        save_data(valid_dataset, os.path.join(args.output, "validation.json"))
        print("dev done")

    if "test" in dataset.keys():
        test_dataset = concat_model.concat(
            [d for d in dataset["test"]], "test", args.epoch
        )
        save_data(test_dataset, os.path.join(args.output, "test.json"))
        print("test done")


if __name__ == "__main__":
    main()
