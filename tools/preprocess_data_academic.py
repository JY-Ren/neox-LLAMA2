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
import tqdm
import torch
import ftfy

from megatron.tokenizer import build_tokenizer
from megatron.data import indexed_dataset
from threading import Semaphore

LOG_DATA_NUM = 10

def chunk_list(input_list, chunk_size):
    return [input_list[i:i+chunk_size] for i in range(0, len(input_list), chunk_size)]

def chunk_list_with_slide(input_list, chunk_size, slide_len):
    assert slide_len < chunk_size, 'Invalid: slide_len > chunk_size'
    
    if len(input_list) <= chunk_size:
        return [input_list]
    
    chunks = [input_list[:chunk_size]]
    begin = chunk_size
    while begin-slide_len < len(input_list):
        chunks.append(input_list[begin-slide_len: begin+chunk_size-slide_len])
        begin += chunk_size - slide_len
        if len(chunks[-1]) < chunk_size:
            break
    
    return chunks


class Encoder(object):
    def __init__(self, args):
        self.args = args

    def initializer(self):
        # Use Encoder class as a container for global data
        Encoder.tokenizer = build_tokenizer(self.args)

    def encode(self, text):
        if self.args.ftfy:
            # fix_character_width=False 能够保持中文符号，否则会统一变成半角符号
            text = ftfy.fix_text(text, fix_character_width=False)

        ids = {}
        for key in self.args.jsonl_keys:
            if "tail" not in key:
                doc_ids = []
                text_ids = Encoder.tokenizer.tokenize(text)
                if self.args.append_bos:
                    text_ids = [Encoder.tokenizer.bos] + text_ids
                
                # splited_text_ids = chunk_list(text_ids, self.args.chunk_size)
                splited_text_ids = chunk_list_with_slide(text_ids, self.args.chunk_size, self.args.slide_len)
                
                for local_text_ids in splited_text_ids:
                    if len(local_text_ids) > 0:
                        doc_ids.append(local_text_ids)
                if self.args.append_eod:
                    doc_ids[-1].append(Encoder.tokenizer.eod)
                chunked_doc_ids = [ele for ele in doc_ids if len(ele) == self.args.chunk_size]
                un_chunked_doc_ids = [ele for ele in doc_ids if len(ele) != self.args.chunk_size]
                if len(chunked_doc_ids):
                    ids[key] = chunked_doc_ids
                if len(un_chunked_doc_ids):
                    ids[f"{key}_tail"] = un_chunked_doc_ids
                global LOG_DATA_NUM
                if self.args.log_data and LOG_DATA_NUM > 0:
                    LOG_DATA_NUM -= 1
                    print(
                        "text before process: \n",
                        text,
                        "\nid after processed: \n",
                        ids[f"{key}_tail"],
                    )
        return ids, len(text)


"""
Input: prompt, output
Output: <human>：xxx\n<bot>: xxx
"""


class SFTEncoder(object):
    def __init__(self, args):
        self.args = args

    def initializer(self):
        # Use Encoder class as a container for global data
        Encoder.tokenizer = build_tokenizer(self.args)

    def encode(self, text):
        json_text = json.loads(text)
        prompts = json_text["prompt"]
        outputs = json_text["output"]
        prompt_text = ""
        assert len(prompts) == len(outputs)
        for prompt, output in zip(prompts, outputs):
            prompt_text += "<human>：{}<bot>：{}".format(prompt, output)

        if self.args.ftfy:
            prompt_text = ftfy.fix_text(prompt_text, fix_character_width=False)

        ids = {}
        key = "text"
        doc_ids = []
        text_ids = Encoder.tokenizer.tokenize(prompt_text)
        if self.args.append_bos:
            text_ids = [Encoder.tokenizer.bos] + text_ids
        if len(text_ids) > 0:
            doc_ids.append(text_ids)
        if self.args.append_eod:
            doc_ids[-1].append(Encoder.tokenizer.eod)
        ids[key] = doc_ids
        global LOG_DATA_NUM
        if self.args.log_data and LOG_DATA_NUM > 0:
            LOG_DATA_NUM -= 1
            print(
                "text before process: \n",
                prompt_text,
                "\nid after processed: \n",
                ids[key],
            )
        return ids, len(prompt_text)


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
    group.add_argument(
        "--jsonl-keys",
        nargs="+",
        default=["text","text_tail"],
        help="space separate listed of keys to extract from jsonl. Defa",
    )
    group.add_argument(
        "--num-docs",
        default=None,
        help="Optional: Number of documents in the input data (if known) for an accurate progress bar.",
        type=int,
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
            "LlamaTokenizer",
            "HFLlamaTokenizer",
        ],
        help="What type of tokenizer to use.",
    )
    group.add_argument(
        "--vocab-file", type=str, default=None, help="Path to the vocab file"
    )
    group.add_argument(
        "--chunk_size", type=int, default=8192, help=""
    )
    group.add_argument(
        "--slide_len", type=int, default=0, help=""
    )
    group.add_argument(
        "--merge-file",
        type=str,
        default=None,
        help="Path to the BPE merge file (if necessary).",
    )
    group.add_argument(
        "--append-eod",
        action="store_true",
        help="Append an <eod> token to the end of a document.",
    )
    group.add_argument(
        "--append-bos",
        action="store_true",
        help="Append an <bos> token to the start of a document.",
    )
    group.add_argument(
        "--sft-data",
        action="store_true",
        help="Use special logic of sft-data.",
    )
    group.add_argument("--ftfy", action="store_true", help="Use ftfy to clean text")
    group = parser.add_argument_group(title="output data")
    group.add_argument(
        "--output-prefix",
        type=str,
        required=True,
        help="Path to binary output file without suffix",
    )
    group.add_argument(
        "--file-include",
        type=str,
        default='*'
    )
    group.add_argument(
        "--dataset-impl",
        type=str,
        default="mmap",
        choices=["lazy", "cached", "mmap"],
        help="Dataset implementation to use. Default: mmap",
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


def yield_from_files(fnames: list, semaphore, key="text"):
    """
    Iterator over input documents using lm_dataformat. Should be able to handle jsons / texts /
    other compressed formats. Also filters out empty documents.

    :param fnames: list of filenames
    """

    def yielder(fname, semaphore):
        for f in filter(lambda x: x, lmd.Reader(fname).stream_data(key=key)):
            semaphore.acquire()
            yield f

    for fname in fnames:
        semaphore.acquire()
        yield from yielder(fname, semaphore)


def main():
    args = get_args()
    if not args.sft_data:
        encoder = Encoder(args)
    else:
        encoder = SFTEncoder(args)
    tokenizer = build_tokenizer(args)
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Output prefix: {args.output_prefix}")

    if not args.input.endswith(".jsonl"):
        import fnmatch
        input_files = [os.path.join(args.input,f) for f in os.listdir(args.input) if fnmatch.fnmatch(f, args.file_include)]
    else:
        input_files = args.input.split(",")

    # use multiprocessing to iterate over input documents
    if not args.sft_data:
        # build a semaphore object to stop `yield_from_files` from getting ahead of encoder.encode and
        # hence building up memory
        semaphore = Semaphore(10000 + args.workers)
        fin = yield_from_files(input_files, semaphore, key=args.jsonl_keys[0])
    else:
        """
        only support a single input file for sft-data
        """
        fin = open(args.input, "r")

    if args.workers > 1:
        pool = multiprocessing.Pool(args.workers, initializer=encoder.initializer)
        encoded_docs = pool.imap(encoder.encode, fin, chunksize=25)
    else:
        encoder.initializer()
        encoded_docs = (encoder.encode(doc) for doc in fin)

    # make a dataset builder for each key in args.jsonl_keys
    # each key will output to a different file beginning with args.output_prefix
    output_bin_files = {}
    output_idx_files = {}
    builders = {}
    for key in args.jsonl_keys:
        output_bin_files[key] = "{}_{}_{}.bin".format(
            args.output_prefix, key, "document"
        )
        output_idx_files[key] = "{}_{}_{}.idx".format(
            args.output_prefix, key, "document"
        )
        builders[key] = indexed_dataset.make_builder(
            output_bin_files[key],
            impl=args.dataset_impl,
            vocab_size=tokenizer.vocab_size,
        )

    # actually do tokenization
    proc_start = time.time()
    total_bytes_processed = 0
    total_tokens_processed = 0
    pbar = tqdm.tqdm()
    for i, (doc, bytes_processed) in enumerate(encoded_docs, start=1):
        total_bytes_processed += bytes_processed

        # release semaphore so `yield_from_files` can add another file to the buffer
        if not args.sft_data:
            semaphore.release()

        # add each tokenized document / sentence
        for key, sentences in doc.items():
            for sentence in sentences:
                total_tokens_processed += len(sentence)
                builders[key].add_item(np.array(sentence, dtype=builders[key].dtype))
            # separate with eos token
            builders[key].end_document()

        # log progress
        if i % args.log_interval == 0:
            current = time.time()
            elapsed = current - proc_start
            mbs = total_bytes_processed / elapsed / 1024 / 1024
            pbar.set_description(
                f"Processed {i}{'' if args.num_docs is None else '/' + str(args.num_docs)} documents {total_tokens_processed} tokens ({i / elapsed} docs/s, {mbs} MB/s)."
            )
            if i != 0:
                pbar.update(args.log_interval)

    # save output file
    for key in args.jsonl_keys:
        builders[key].finalize(output_idx_files[key])

    with open(f"{args.output_prefix}-token_counts.txt", "w", encoding="utf-8") as w:
        w.writelines(f"{total_tokens_processed} tokens \n{i} docs")


if __name__ == "__main__":
    main()
