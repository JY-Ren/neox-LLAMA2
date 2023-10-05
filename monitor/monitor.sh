#! /bin/bash

. /cognitive_comp/yangqi/tools/monitor/notify.sh

function output_notify {
  text=$1
  result=$(send_notify $text)
  echo "$text" >> /shared_space/agpt/0627/logs/pretrain_notify.log
}

count=$(squeue | grep pretrain -c)
pname=$(squeue | grep pretrain | grep -v grep | awk '{print $3}')
pid=$(squeue | grep pretrain | grep -v grep | awk '{print $1}')
nodes=$(squeue | grep pretrain | grep -v grep | awk '{print $7}')
runtime=$(squeue | grep pretrain | grep -v grep | awk '{print $6}')
DATE=$(date)

# lossnan=$(cat /shared_space/agpt/0627/logs/$pname-$pid.log | grep "loss: nan" -c)
# lossscale=$(cat /shared_space/agpt/0627/logs/$pname-$pid.log | grep "Attempted loss scale:" -c)
lossnan=$(cat /shared_space/agpt/0627/logs/logs_33b/dgx006.scc.idea_stdout.txt | grep "loss: nan" -c)
# lossscale=$(cat /shared_space/agpt/0627/logs/logs_33b/dgx006.scc.idea_stdout.txt | grep "Attempted loss scale:" -c)

if [$count == 0]
then
    $(output_notify "【监控警告】预训练任务已中断，请前往查看，任务 id $pid，运行时间 $runtime，loss nan $lossnan 次")
else
    echo "$(date) : 进程运行正常" >> /shared_space/agpt/0627/logs/pretrain_notify.log
fi
