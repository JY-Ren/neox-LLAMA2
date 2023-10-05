import os
import argparse
from transformers import  LlamaTokenizer
from sentencepiece import sentencepiece_model_pb2 as model
from tokenizers import AddedToken

parser = argparse.ArgumentParser()
parser.add_argument('--llama_model', default='/cognitive_comp/yangqi/model/LLamaTokenizer_7B_origin', type=str)
parser.add_argument('--merged_model', default='/cognitive_comp/yangqi/model/LLamaTokenizer_7B_repro2', type=str)
parser.add_argument('--bert_vocab', default='/cognitive_comp/yangqi/model/BertTokenizer_Zhuiyi/vocab.txt', type=str)
args = parser.parse_args()

llama_model = args.llama_model
merged_model = args.merged_model
bert_vocab = args.bert_vocab

_ZH_RANGES = (
    ("\u3400", "\u4db5"),  # CJK Unified Ideographs Extension A, release 3.0
    ("\u4e00", "\u9fa5"),  # CJK Unified Ideographs, release 1.1
    ("\u9fa6", "\u9fbb"),  # CJK Unified Ideographs, release 4.1
    ("\uf900", "\ufa2d"),  # CJK Compatibility Ideographs, release 1.1
    ("\ufa30", "\ufa6a"),  # CJK Compatibility Ideographs, release 3.2
    ("\ufa70", "\ufad9"),  # CJK Compatibility Ideographs, release 4.1
    ("\u20000", "\u2a6d6"),  # (UTF16) CJK Unified Ideographs Extension B, release 3.1
    ("\u2f800", "\u2fa1d"),  # (UTF16) CJK Compatibility Supplement, release 3.1
    ("\uff00", "\uffef"),  # Full width ASCII, full width of English punctuation,
                                                # half width Katakana, half wide half width kana, Korean alphabet
    ("\u2e80", "\u2eff"),  # CJK Radicals Supplement
    ("\u3000", "\u303f"),  # CJK punctuation mark
    ("\u31c0", "\u31ef"),  # CJK stroke
    ("\u2f00", "\u2fdf"),  # Kangxi Radicals
    ("\u2ff0", "\u2fff"),  # Chinese character structure
    ("\u3100", "\u312f"),  # Phonetic symbols
    ("\u31a0", "\u31bf"),  # Phonetic symbols (Taiwanese and Hakka expansion)
    ("\ufe10", "\ufe1f"),
    ("\ufe30", "\ufe4f"),
    ("\u2600", "\u26ff"),
    ("\u2700", "\u27bf"),
    ("\u3200", "\u32ff"),
    ("\u3300", "\u33ff"),
)

def is_char(uchar, _UCODE_RANGES):
    for start, end in _UCODE_RANGES:
        if start <= uchar <= end:
            return True
    return False

def get_bert_char(bert_vocab):
    f = open(bert_vocab,"r")
    special_tokens = [l.strip() for l in f.readlines() if is_char(l.strip(),_ZH_RANGES)]
    
    fout = open("chinese_token.txt","w")
    fout.write("\n".join(special_tokens))
        
    return special_tokens

def merge_sp_model(llama_model, merge_model, special_tokens):
    m = model.ModelProto()
    m.ParseFromString(open(llama_model, "rb").read())

    # deduplicate chinese token in llama-origin
    llama_spm_tokens_set=set(p.piece for p in m.pieces)
    print(f"Before merge chinese token:{len(llama_spm_tokens_set)}")
    
    for token in special_tokens:
        if token not in llama_spm_tokens_set:
            new_p = model.ModelProto().SentencePiece()
            new_p.piece = token
            new_p.score = 0
            m.pieces.append(new_p)
    print(f"After merge chinese token: {len(m.pieces)}")

    with open(merge_model, 'wb') as f:
        f.write(m.SerializeToString())
        
# backup original llama
llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model)
llama_tokenizer.save_pretrained(save_directory=merged_model)
print(f"Backup Llama Tokenizer and Save to {merged_model}")

# load from bert vocab
special_tokens = get_bert_char(bert_vocab)
print(f"Add Chinese token {len(special_tokens)}")

# load from saved txt
# special_tokens = open("chinese_token.txt", "r").read().split("\n")

# inplace-merge to avoid special token conflict
llama_sp_model = os.path.join(merged_model,"tokenizer.model")
merged_sp_model = os.path.join(merged_model,"tokenizer.model")
merge_sp_model(llama_sp_model,merged_sp_model,special_tokens)
human = AddedToken("<human>",lstrip=False,rstrip=False,single_word=False,normalized=True)
bot = AddedToken("<bot>",lstrip=False,rstrip=False,single_word=False,normalized=True)

llama_tokenizer = LlamaTokenizer.from_pretrained(merged_model)
llama_tokenizer.add_special_tokens({"additional_special_tokens":[human, bot]})
llama_tokenizer.save_pretrained(save_directory=merged_model)

llama_tokenizer = LlamaTokenizer.from_pretrained(merged_model)
print(f"New Llama tokenizer: {len(llama_tokenizer)}")
