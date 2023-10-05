echo $1
echo $2

python /cognitive_comp/wuziwei/codes/gpt-neox/tools/preprocess_data.py \
            --input $1 \
            --jsonl-keys text \
            --output-prefix $2 \
            --vocab /cognitive_comp/common_data/BPETokenizer-Mix-NEO-pre \
            --dataset-impl mmap \
            --tokenizer-type HFGPTNeoXTokenizerFast \
            --workers 32 \
            --append-eod \
            --log-data \
            --ftfy