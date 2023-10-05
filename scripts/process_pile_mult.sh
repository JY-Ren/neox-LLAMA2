TAR_DATA_PATH=$1/

files=`ls ${TAR_DATA_PATH}`
opt="" 
for i in $files;
do 
    opt=$opt"${TAR_DATA_PATH}$i,"
done 

opt=${opt%,*}
echo $opt

python /cognitive_comp/wuziwei/codes/gpt-neox/tools/preprocess_data.py \
            --input $opt \
            --jsonl-keys text \
            --output-prefix $2 \
            --vocab /cognitive_comp/common_data/BPETokenizer-Mix-NEO-pre2 \
            --dataset-impl mmap \
            --tokenizer-type HFGPTNeoXTokenizerFast \
            --workers 64 \
            --append-eod \
            --log-data \
            --ftfy