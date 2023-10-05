import yaml
import argparse
import os

# data = {
#     'result_dir': '/shared_space/agpt/0627/results',
#     'overwrite': True,
#     'gpus': "1,2,3,4,5,6,7,8",
#     'debug': False,
#     'models': {
#         'group': 'llama',
#         'alias': 'llama-30b-global-step1000',
#         'model_name_or_path': '/shared_space/agpt/0627/hf-checkpoints/global_step1000/',
#         'max_seq_len': 8192,
#         'request_params':{
#             'max_new_tokens': 1,
#             'min_new_tokens': 1,
#             'temperature': 0.0,
#             'top_p': 1.0,
#             'use_cache': True
#         }
#     }
# }

import argparse

# 创建一个argparse解析器
parser = argparse.ArgumentParser(description='Description of your program')

# 添加命令行参数
parser.add_argument('--yaml_dir', type=str, default='/shared_space/agpt/0627/results/config', help='Result directory path')
parser.add_argument('--result_dir', type=str, default='/shared_space/agpt/0627/results', help='Result directory path')
parser.add_argument('--overwrite', action='store_true', default=True, help='Overwrite existing result files')
parser.add_argument('--gpus', type=str, default='1,2,3,4,5,6,7,8', help='GPU IDs to use')
parser.add_argument('--debug', action='store_false', default=False, help='Enable debugging mode')
parser.add_argument('--ceval', action='store_true', default=False)
parser.add_argument('--mmlu', action='store_true', default=False)
parser.add_argument('--paperqa', action='store_true', default=False)

# 添加嵌套参数
parser.add_argument('--models_group', type=str, default='llama', help='Model group name')
parser.add_argument('--models_alias', type=str, default='llama-30b-global-step1000', help='Model alias')
parser.add_argument('--model_name_or_path', type=str, default='/shared_space/agpt/0627/hf-checkpoints/global_step1000/', help='Model checkpoint path')
parser.add_argument('--max_seq_len', type=int, default=8192, help='Maximum sequence length')
parser.add_argument('--max_new_tokens', type=int, default=1, help='Maximum number of new tokens to generate')
parser.add_argument('--min_new_tokens', type=int, default=1, help='Minimum number of new tokens to generate')
parser.add_argument('--temperature', type=float, default=0.0, help='Generation temperature')
parser.add_argument('--top_p', type=float, default=1.0, help='Top-p sampling threshold')
parser.add_argument('--use_cache', type=bool, default=True, help='Enable or disable caching')

# 解析命令行参数
args = parser.parse_args()

# 将命令行参数转换为字典
data = {
    'result_dir': args.result_dir,
    'overwrite': args.overwrite,
    'gpus': args.gpus,
    'debug': args.debug,
    'models': [{
        'group': args.models_group,
        'alias': args.models_alias,
        'model_name_or_path': args.model_name_or_path,
        'max_seq_len': args.max_seq_len,
        'request_params':{
            'max_new_tokens': args.max_new_tokens,
            'min_new_tokens': args.min_new_tokens,
            'temperature': args.temperature,
            'top_p': args.top_p,
            'use_cache': args.use_cache
        }
    }],
    'icl_tasks':[]
}

if args.ceval:
    ceval_cfg = {
        'name':'ceval',
        'dataset_uri': '/cognitive_comp/yangqi/project/evaluate/data/ceval_dev_val.jsonl',
        'num_fewshot': 5,
        'max_eval_instances': 300,
        'icl_task_type': "multiple_choice",
        'prompt_string': '以下是一些关于{subject}学科的多项选择题（带答案）',
        'example_delimiter':'\n',
        'continuation_delimiter': '答案',
        'has_categories': True,
        'request_params':{
            'max_new_tokens': 1
        }
    }
    data["icl_tasks"].append(ceval_cfg)

if args.mmlu:
    mmlu_cfg = {
        'name':'mmlu',
        'dataset_uri': '/cognitive_comp/yangqi/project/evaluate/data/mmlu.jsonl',
        'num_fewshot': 5,
        'max_eval_instances': 300,
        'icl_task_type': "multiple_choice",
        'prompt_string': 'The following are multiple choice questions (with answers) about {subject}.',
        'example_delimiter':'\n',
        'continuation_delimiter': 'Answer:',
        'has_categories': True,
        'request_params':{
            'max_new_tokens': 1
        }
    }
    data["icl_tasks"].append(mmlu_cfg)

if args.paperqa:
    paperqa_cfg = {
        'name':'paper_qa',
        'dataset_uri': '/cognitive_comp/yangqi/project/evaluate/data/paper_qa_v1.jsonl',
        'num_fewshot': 3,
        'max_eval_instances': 1000,
        'icl_task_type': "multiple_choice",
        'prompt_string': 'The following are multiple choice questions (with answers) about {subject}.',
        'example_delimiter':'\n',
        'continuation_delimiter': 'Answer:',
        'has_categories': True,
        'request_params':{
            'max_new_tokens': 1
        }   
    }
    data["icl_tasks"].append(paperqa_cfg)

os.makedirs(args.yaml_dir, exist_ok=True)
with open(os.path.join(args.yaml_dir,'eval.yaml'), 'w') as file:
    yaml.dump(data, file)
    print(f"finish yaml {os.path.join(args.yaml_dir,'eval.yaml')}")
