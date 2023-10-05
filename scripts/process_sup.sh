python /cognitive_comp/wuziwei/codes/gpt-neox/tools/preprocess_data.py \
            --input /cognitive_comp/common_data/instructdata/pretrained_data/neox_v1_sup/train_shuffle.jsonl \
            --jsonl-keys text \
            --output-prefix /cognitive_comp/wuziwei/task/academicGPT/ftfy_zh_sup/zh_sup \
            --vocab /cognitive_comp/common_data/BPETokenizer-Mix-NEO-pre \
            --dataset-impl mmap \
            --tokenizer-type HFGPTNeoXTokenizerFast \
            --workers 32 \
            --append-eod \
            --log-data \
            --ftfy