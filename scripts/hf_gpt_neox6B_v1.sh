#!/bin/bash
#usage: sh xxx.sh xxx/checkpoint 20000 academic_step2
# $1：checkpoint 路径
# $2：step数
# $3：模型名字

CODE_ROOT='/cognitive_comp/wuziwei/codes/gpt-neox'
MODEL_CONFIG="/cognitive_comp/wuziwei/codes/gpt-neox/workspace/academic_6B/academic_6B.yml"
input_dir=$1
step=$2
output_name=$3
input_dir="${input_dir}/global_step${step}"
output_dir="/cognitive_comp/common_checkpoint/${output_name}/hf_model_${step}"

if [ -d ${output_dir} ];then 
echo '目标checkpoint已经存在请确认后重新执行'
echo "目标位置：${output_dir}"
exit
fi

echo "converting Megatron model checkpoint $step to huggingface mode......"

export RUN_CMD="python $CODE_ROOT/tools/convert_to_hf.py --input_dir $input_dir --config_file $CONFIG_ROOT/$MODEL_CONFIG --output_dir $output_dir"
bash -c "$RUN_CMD"

echo "model saved at:${output_dir}"
