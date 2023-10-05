#! /bin/bash

# 监控日志路径
target_dir="/platform_tech/xiajun/pretrainpipeline/xj-workspace/0808/logs"
# log更新时间差阈值（10分钟，单位秒）
threshold=$((10 * 60))
# slurm中预训练任务名
task="pretrain"

function send_notify {

  test_url="https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=1a7b78d4-d352-44a5-9358-94beedbbfe0d"
  p100_url="https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=abc6a3f2-4f39-49c9-8822-bb956d052bcd"
  gpt_url="https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=8d66a8c1-1c7f-41b3-a651-c9d1c0c44948"

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
              "mentioned_list":["yangqi","weishufa","xiajun"]
          }'
  done

}

send_notify "启动【llama2-70b-pretrain】预训练监控..."

counter=0
while true; do
    count=$(squeue -u xiajun | grep $task -c)
    pid=$(squeue -u xiajun | grep $task | grep -v grep | awk '{print $1}')
    status=$(squeue -u xiajun | grep $task | grep -v grep | awk '{print $5}')
    runtime=$(squeue -u xiajun | grep $task | grep -v grep | awk '{print $6}')

    if [ count == 0 ]; then
      send_notify "【监控警告】预训练任务已结束。请确认是否正常，任务id：$pid，运行时间：$runtime"
      break
    fi

    if [ "$status" != 'R' ]; then
      send_notify "【监控警告】预训练异常，请确认是否正常，任务id：$pid，运行时间：$runtime"
    fi

    # 检查latest文件是否存在
    if [ -f "$target_dir/$task-$pid.log" ]; then
        # 获取loss的更新时间（秒）
        last_update=$(stat -c %Y "$target_dir/$task-$pid.log")
        # 获取当前时间（秒）
        current_time=$(date +%s)
        # 获取更新时间差
        time_diff=$((current_time - last_update))

        if [ "$time_diff" -gt "$threshold" ]; 
        then
          send_notify "【监控警告】预训练任务异常：loss更新超时。任务id：$pid，运行时间：$runtime。请前往查看$target_dir/$task-$pid.log"
          break
        else
          echo "【监控进度】预训练任务状态：正常。任务id：$pid，运行时间：$runtime"
          
          # if [ $(($counter % 360)) -eq 0 ]; then
            # send_notify "【监控进度】预训练任务状态：正常。任务id：$pid，运行时间：$runtime"
          # fi

        fi
    else
        echo "$(date): WARNING - The latest file does not exist in $target_dir."
    fi

    sleep 60
    ((counter++))
done
