#!/bin/bash

# 监控目录
ckpt_path="/shared_space/agpt/0627/checkpoints/"

SCRIPT="/cognitive_comp/yangqi/project/gpt-neox/tools/convert_llama_to_hf.py"
BATCH_SCRIPT="/cognitive_comp/yangqi/project/gpt-neox/tools/convert_llama_pp_tp_to_hf_batch.py"

YAML_SCRIPT="/cognitive_comp/yangqi/project/gpt-neox/monitor/eval_yaml.py"
EVAL_SCRIPT="/cognitive_comp/yangqi/project/evaluate/run_eval.sh"

# 初始化最新的文件夹名称为空
latest_dir=""
latest_but_last_dir=""

# 循环检查目录下的文件夹，并找到最新的文件夹
while true
do
    for file in $ckpt_path*/
    do
        if [[ -d "$file" ]]; then
            # 解析 step 数 
            step=$(echo $file | sed -n 's/.*global_step\([0-9]\+\).*/\1/p')
            last_step=$(echo $latest_dir | sed -n 's/.*global_step\([0-9]\+\).*/\1/p')
            step=$(expr $step + 0)
            last_step=$(expr $last_step + 0)
            
            if [[ "$step" -gt "$last_step" ]]; then
                latest_but_last_dir="$latest_dir"
                latest_dir="$file"
            fi
        fi
    done
    
    # 最新文件夹执行
    if [[ "$latest_dir" != "" ]]; then
        echo "$(date) lastest folder $latest_dir" >> /shared_space/agpt/0627/results/ckpt_monitor.log
        echo "$(date) lastest but last folder $latest_but_last_dir" >> /shared_space/agpt/0627/results/ckpt_monitor.log

        lastest_but_last_step=$(echo $latest_but_last_dir | sed -n 's/.*global_step\([0-9]\+\).*/\1/p')

        # 转 ckpt
        python $BATCH_SCRIPT --input_dir $latest_but_last_dir \
               --model_size 30B \
               --output_dir "/shared_space/agpt/0627/hf-checkpoints/global_step$lastest_but_last_step" \
               --num_shards 4 \
               --tokenizer_path /shared_space/agpt/models/llama-65b-hf-merged/
        echo "$(date) turning neox ckpt to huggingface ckpt" >> /shared_space/agpt/0627/results/ckpt_monitor.log
        echo "$(date) hf ckpt path: /shared_space/agpt/0627/hf-checkpoints/global_step$lastest_but_last_step" >> /shared_space/agpt/0627/results/ckpt_monitor.log

        # 生成 yaml
        python $YAML_SCRIPT \
        --yaml_dir "/shared_space/agpt/0627/results/config" \
        --result_dir "/shared_space/agpt/0627/results" \
        --model_name_or_path "/shared_space/agpt/0627/hf-checkpoints/global_step$lastest_but_last_step" \
        --models_alias "llama-30b-global-step$lastest_but_last_step" \
        --models_group llama \
        --ceval \
        --mmlu \
        --paperqa
        echo "$(date) generating-auto yaml for ceval/mmlu/paperqa" >> /shared_space/agpt/0627/results/ckpt_monitor.log

        # 评估
        # python $EVAL_SCRIPT /shared_space/agpt/0627/results/config/eval.yaml
        sbatch $EVAL_SCRIPT
        echo "$(date) submit sbatch tasks of eval" >> /shared_space/agpt/0627/results/ckpt_monitor.log

        break
    fi
    
    # 如果没有找到最新的文件夹，等 1h 再扫描
    sleep 36000
done
# rm /shared_space/agpt/0627/results/config/eval.yaml
# echo "$(date) clear config /shared_space/agpt/0627/results/config/eval.yaml" >> /shared_space/agpt/0627/results/ckpt_monitor.log
# echo "$(date) eval finished" >> /shared_space/agpt/0627/results/ckpt_monitor.log
