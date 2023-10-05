#!/bin/bash
#SBATCH --job-name=eval # create a short name for your job
#SBATCH -p batch
##SBATCH -x dgx[047,037,021] # 排除一些节点
#SBATCH --reservation=ccp
#SBATCH --qos ccp
#SBATCH -N 1 # node count
#SBATCH --ntasks-per-node 1 # number of tasks to run per node
#SBATCH --cpus-per-task 100 # cpu-cores per task (>1 if multi-threaded tasks),--cpus-per-task
#SBATCH --gpus-per-node 8 # total cpus for job
#SBATCH -o logs/%x-%j.log # output and error log file names (%x for job id)
#SBATCH -e logs/%x-%j.err # output and error log file names (%x for job id)

CKPT_DIR="/shared_space/agpt/experiments/0808"
TOKENIER="/shared_space/agpt/models/llama2/hf/70B"
CONVERT_CODE="/platform_tech/xiajun/pretrainpipeline/tools/convert_llama2_pp_tp_to_hf_batch.py"
EVAL_CODE="/platform_tech/xiajun/evaluate/"
MODEL_ALIAS="llama2-70b-pretrain"

# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

function send_notify {  

    test_url="https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=1a7b78d4-d352-44a5-9358-94beedbbfe0d"
    # p100_url="https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=abc6a3f2-4f39-49c9-8822-bb956d052bcd"
    # gpt_url="https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=8d66a8c1-1c7f-41b3-a651-c9d1c0c44948"

    for url in $p100_url $gpt_url
    do
        curl "$url" \
            -H 'Content-Type: application/json' \
            -d '
            {
                "msgtype": "text",
                "text": {
                    "content": "'$1'"
                },
                "mentioned_list":["xiajun"]
            }'
    done
}

send_notify "启动${MODEL_ALIAS}预训练任务评估监控。"
echo "启动${MODEL_ALIAS}预训练任务评估监控。"

# 循环检查目录下的文件夹，并找到最新的文件夹
latest_dir=""
latest_but_last_dir=""
while true
do
    for file in $CKPT_DIR/checkpoints/*
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

    STEP=$(echo $latest_but_last_dir | sed -n 's/.*global_step\([0-9]\+\).*/\1/p')

    if [ -z "$STEP" ]; then
        echo "no ckpt found, skipped"
        sleep 3600
        continue
    fi

    if [ ! -d "./results" ]; then
        mkdir "./results"
    fi

    # 判断当前STEP是否已经测试过
    count=$(ls ./results|grep "STEP${STEP}." -c)
    echo "当前step $STEP 是否评测过： $count"

    if [ $count != 6 ]; then
        # neox -> hf
        echo "converting ckpt ${STEP} to hf ..."
        if [ ! -d ${CKPT_DIR}/hf/hf_global_step${STEP} ]; 
        then
            python ${CONVERT_CODE} \
                --input_dir ${CKPT_DIR}/checkpoints \
                --model_size 70B \
                --output_dir ${CKPT_DIR}/hf \
                --num_input_shards 4 \
                --tokenizer_path ${TOKENIER} \
                --steps ${STEP}
            
            if [ $? -ne 0 ]; then
                send_notify "【评估监控】ckpt转hf格式出错..."
                exit 1
            fi
        else
            echo "existed ${STEP} hf ckpt, skipped converting "
        fi 

        
        # 生成 yaml. 默认长度应该4096，误设为8192
        yaml_path=./results/${STEP}.yml
        python ${EVAL_CODE}/auto_yaml.py \
            --yaml_path ${yaml_path} \
            --result_dir ./results \
            --model_name_or_path ${CKPT_DIR}/hf/hf_global_step${STEP} \
            --models_alias ${MODEL_ALIAS}-STEP${STEP} \
            --max_seq_len 8192 \
            --models_group llama \
            --ceval \
            --mmlu \
            --paperqa

        if [ $? -ne 0 ]; then
            send_notify "【评估监控】生成yml文件出错..."
            exit 1
        fi

        # evaluate
        echo "evaluating ..."
        python ${EVAL_CODE}/run_eval.py ${yaml_path}

        if [ $? -ne 0 ]; then
            send_notify "【评估监控】评估异常..."
            exit 1
        fi

        base="./results/llama.${MODEL_ALIAS}-STEP${STEP}"

        mmlu=$(cat ${base}.mmlu.5-shot.txt)
        ceval=$(cat ${base}.ceval.5-shot.txt)
        paperqa=$(cat ${base}.paper_qa.3-shot.txt)
        
        send_notify "${MODEL_ALIAS}-STEP${STEP}:\nmmlu=$mmlu\nceval=$ceval\npaper-qa=$paperqa"
    
        # python ${EVAL_CODE}/bpb.py \
        #     --model_names \
        #         "hf_global_step${STEP}" \
        #     --model_paths  \
        #         "${CKPT_DIR}/hf/hf_global_step${STEP}"  \
        #     --result_dir "./results"
        
        # bpb=$(cat ./results/hf_global_step${STEP})
        # send_notify "${MODEL_ALIAS}-STEP${STEP}:\n${bpb}"

    else
        echo "waiting for next ckpt..."
        sleep 600
    fi

echo ""评估完成。""

done
