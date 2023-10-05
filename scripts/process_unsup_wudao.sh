python /cognitive_comp/wuziwei/codes/gpt-neox/tools/preprocess_data.py \
            --input /cognitive_comp/common_data/big_corpus/wudaoData2.0/wudao180.jsonl \
            --jsonl-keys text \
            --output-prefix /cognitive_comp/wuziwei/task/academicGPT/ftfy_wudao/wudao180 \
            --vocab /cognitive_comp/common_data/BPETokenizer-Mix-NEO-pre \
            --dataset-impl mmap \
            --tokenizer-type HFGPTNeoXTokenizerFast \
            --workers 32 \
            --append-eod \
            --log-data \
            --ftfy