import os
import sys
import json
import torch
import uvicorn
import numpy as np
from time import time
from fastapi import FastAPI
from transformers import AutoModelForCausalLM, AutoTokenizer


sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)
# os.environ["MASTER_ADDR"] = '127.0.0.1'
# os.environ["MASTER_PORT"] = '6000'
model_dict = {
    "wenzhong-v1": "",
    "wenzhong-v2": "/cognitive_comp/common_data/Huggingface-Models/IDEA-CCNL/Wenzhong2.0-GPT2-3.5B-chinese",
    "wenzhong-v2-110M": "/cognitive_comp/common_data/Huggingface-Models/IDEA-CCNL/Wenzhong2.0-GPT2-110M",
    "yuyuan": "/cognitive_comp/common_data/Huggingface-Models/IDEA-CCNL/Yuyuan-GPT2-3.5B",
    "yuyuanQA": "/cognitive_comp/common_data/Huggingface-Models/IDEA-CCNL/YuyuanQA-GPT2-3.5B",
    "yuyuanSciFi": "/cognitive_comp/common_data/Huggingface-Models/IDEA-CCNL/Yuyuan-SciFi-GPT2-110M-Chinese",
    "neox_path": "/cognitive_comp/wuziwei/pretrained_model_hf/gpt-neox-20b",
    "opt_path": "/cognitive_comp/wuziwei/pretrained_model_hf/opt-iml-max-30b",
    "6B-1800": "/cognitive_comp/ganruyi/gpt-neox/workspace/gpt_neox6B_v1/hf_model",
    "6B-8192-2000": "/cognitive_comp/ganruyi/gpt-neox/workspace/gpt_neox6B_v1/hf_model_2000",
    "6B-8192-5000": "/cognitive_comp/ganruyi/gpt-neox/workspace/gpt_neox6B_v1/hf_model_5000",
    "6B-8192-8000": "/cognitive_comp/ganruyi/gpt-neox/workspace/gpt_neox6B_v1/hf_model_8000",
    "6B-2048-6000": "/cognitive_comp/wuziwei/codes/gpt-neox/workspace/6B/hf_model",
}

model_list = ["6B-8192-8000"]

assert all([i in model_dict.keys() for i in model_list]), "UNKNOWN MODEL"
# app = Flask(__name__)
app = FastAPI()

models = dict()
tokenizers = dict()
for model in model_list:
    print(f"loading:{model_dict[model]}")
    models[model] = AutoModelForCausalLM.from_pretrained(
        model_dict[model], device_map="auto"
    )
    tokenizers[model] = AutoTokenizer.from_pretrained(model_dict[model])
    tokenizers[model].pad_id = 1
    models[model].eval()

# @app.post('/generate')


@app.get("/generate")
async def generate_neox(
    input_text: str,
    max_new_tokens: int = 128,
    top_p: float = 0.7,
    num_return_sequences: int = 2,
    temperature: float = 1.0,
    repetition_penalty: float = 2.5,
    use_model: str = "neox",
):
    st = time()
    if use_model in models.keys():
        with torch.no_grad():
            input_ids = tokenizers[use_model](
                input_text, return_tensors="pt"
            ).input_ids.to(models[use_model].device)
            r = models[use_model].generate(
                input_ids,
                do_sample=True,
                max_new_tokens=max_new_tokens,
                top_p=top_p,
                num_return_sequences=num_return_sequences,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                pad_token_id=1,
            )
            s = [{"generated_text": i} for i in tokenizers[use_model].batch_decode(r)]
    else:
        s = [{"generated_text": "未知模型"}]
    s.append({"time cost": f"{(time()-st):.2f}"})
    return s


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=6615, log_level="info")
