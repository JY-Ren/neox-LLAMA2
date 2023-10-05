import os
import sys
import json
import torch
import uvicorn
import numpy as np
from time import time
from fastapi import FastAPI
from transformers import pipeline


sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)

model_path = {
    "6B-1800": "/cognitive_comp/ganruyi/gpt-neox/workspace/gpt_neox6B_v1/hf_model",
    "6B-2000": "/cognitive_comp/ganruyi/gpt-neox/workspace/gpt_neox6B_v1/hf_model_2000",
}

app = FastAPI()
generate_dicts = dict()
for k, v in model_path.items():
    generate_dicts[k] = pipeline("text-generation", model=v, device_map="auto")


@app.get("/generate")
async def generate_neox(
    input_text: str,
    max_new_tokens: int = 128,
    top_p: float = 0.7,
    num_return_sequences: int = 2,
    temperature: float = 1.0,
    use_model: str = "neox",
):
    st = time()
    if use_model in generate_dicts.keys():
        r = generate_dicts[use_model](
            input_text,
            do_sample=True,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            num_return_sequences=num_return_sequences,
            temperature=temperature,
        )
        # r = generator_neox(input_text, do_sample=True, max_length=1500,top_p=0.7,num_return_sequences=2,temperature=1.0)
    else:
        r = [{"generated_text": "未知模型"}]
    r.append({"time cost": f"{(time()-st):.2f}"})
    return r


torch.Tensor
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=6615, log_level="info")
