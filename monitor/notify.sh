#! /bin/bash

function send_notify {
  keyword = "【预训练监控通知】"
  
  curl 'https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=c3a4863b-7ec6-4bb1-a54a-3229be4870fe' \
        -H 'Content-Type: application/json' \
        -d '
        {
            "msgtype": "text",
            "text": {
	    "content": "'$1'"
            },
            "mentioned_list":["yangqi","weishufa","xiajun"]
        }'
}
