import math


def evaluate(num_params, num_tokens, num_gpus, flops=140):
    """
    num_params: B
    num_tokens: B
    flops: teraFLOPS
    """
    eval_time = math.ceil(
        8 * num_tokens * num_params * 1e6 / num_gpus / flops / 3600 / 24
    )
    print(
        "{}B params, {}B tokens, {} gpus, eval {} days".format(
            num_params, num_tokens, num_gpus, eval_time
        )
    )
    return eval_time


num_params = 65
num_tokens_list = [1400]
num_gpus_list = [192, 2048]
if __name__ == "__main__":
    for num_tokens in num_tokens_list:
        for num_gpus in num_gpus_list:
            evaluate(num_params, num_tokens, num_gpus, flops=120)
