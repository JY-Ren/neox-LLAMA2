import math
from copy import deepcopy

def calculate_iterations_target(
        dataset_tokens,
        PP=2,
        TP=4,
        num_gpus=192,
        seq_length=8192,
        gradient_accumulation=4,
        mult_dataset_weight=None,
        weight_index=4
):
    """
    dataset_tokens:目标数据集的token数
    mult_dataset_weight：权重列表
    weight_index：目标权重index
    
    """
    # batch size
    bs = num_gpus / (PP*TP) * gradient_accumulation
    # 实际上这个数据集的sample数
    ex_sample_target = math.floor(dataset_tokens / seq_length)
    if mult_dataset_weight:
        # 通过比例，计算每个数据集期望应该有的sample数
        ex_samples = [math.floor(i*ex_sample_target/mult_dataset_weight[weight_index] ) for i in mult_dataset_weight]
        ex_total_sample = sum(ex_samples)
    print('期望的样本数：',ex_samples)
    # 由于neox代码里面有*1.005的逻辑，会放大所过的sample数，容易导致epoch+1，这里除掉它，这里就是正确的训练step数
    # 新代码中没有 1.005 已删除
    num_iter = math.floor(ex_total_sample / bs ) 
    # 通过实际的step数*batch size 反推，具体漏过了多少token
    ac_samples = [math.ceil(num_iter * bs * i/sum(mult_dataset_weight)) for i in mult_dataset_weight]
    print('实际的样本数：',ac_samples)
    lose_target_tokens = (ex_samples[weight_index] - ac_samples[weight_index]) * seq_length
    total_tokens = sum(ac_samples) * seq_length
    lose_total_tokens = (sum(ex_samples) - sum(ac_samples)) * seq_length
    batch_tokens = math.floor(bs*seq_length)
    print(f'设置训练step数为：{num_iter}')
    print(f'损失目标token：{lose_target_tokens}')
    print(f'训练共损失token数：{lose_total_tokens}')
    print(f'一个batch token数：{batch_tokens}')
    print(f'训练完成共学习token数：{total_tokens}')
    return ac_samples


def calculate_iterations(
        tokens_list,
        PP=2,
        TP=1,
        num_gpus=192,
        seq_length=8192,
        gradient_accumulation=4,
):
    """
    dataset_tokens:目标数据集的token数
    mult_dataset_weight：权重列表
    weight_index：目标权重index
    
    """
    # batch size
    bs = num_gpus / (PP*TP) * gradient_accumulation
    # 实际上这个数据集的sample数
    if not isinstance(tokens_list,list):
        print(f'{tokens_list} 是一个各个数据集的token数量的列表')
    samples = [math.floor(i / seq_length) for i in tokens_list]
    total_sample = sum(samples) / 1.005
    iters = total_sample / bs 
    print('训练轮数：',iters)


def count_learning_tokens(
        num_iter,
        tokens=None,
        seq_length=8192,
        bs=3072,
        mult_dataset_weight=None,
        weight_index=-1
):
    """
    查看微调step数，导致数据的epoch数和漏掉的token数变化
    """
    ac_sample = [math.ceil(num_iter * bs * i/sum(mult_dataset_weight) * 1.005) for i in mult_dataset_weight]
    print('实际的样本数：',ac_sample)
    total_tokens = sum(ac_sample) * seq_length
    epoch = math.ceil(ac_sample[weight_index]*seq_length / tokens)
    lose_target_tokens = tokens - ac_sample[weight_index]*seq_length 
    return f"设置训练步数为：{num_iter}\n目标数据epoch数：{epoch}\n损失目标token：{lose_target_tokens}\n练完成共学习token数：{total_tokens}"

def calculate_training_weights(
    mult_dataset_tokens,       # 其他数据集 bin 文件最大 token 数
    mult_dataset_weights,      # 其他数据集占自身占比
    arxiv_data_tokens=10000,   # arxiv token 数
    arxiv_dataset_weights=0.5, # arxiv 占总数据，1：1 就是 50%
    num_gpus =160,
    PP=8,
    TP=4,
    grad_accu=16,
    seq_len=8192
):
    all_tokens = sum(arxiv_data_tokens) / arxiv_dataset_weights
    # 其他数据集比例和 samples 数
    res_tokens = all_tokens - sum(arxiv_data_tokens)
    
    # normalize weight
    weight_sum = sum(mult_dataset_weights) 
    weights = [weight / weight_sum for weight in mult_dataset_weight]
    res_tokens_list = [int(math.floor(res_tokens*weight)) for weight in weights]
    res_tokens_list.extend(arxiv_data_tokens)
    mult_dataset_tokens.extend(arxiv_data_tokens)
    
    # 重新计算比例
    weights_with_arxiv = [num_samples / all_tokens for num_samples in res_tokens_list]
    print("综合权重（包含arxiv在内）",weights_with_arxiv)
    print("总数据集 token B 数",all_tokens / 10**9)
    print("分数据集 token 数",res_tokens_list)
    
    #  反向验证是否正确
    # from megatron.data.data_utils import get_normalized_weights_and_num_samples
    def get_normalized_weights_and_num_samples(
    weights, num_samples
    ):
        # Normalize weights
        weight_sum = sum(weights)
        assert weight_sum > 0.0
        weights = [weight / weight_sum for weight in weights]
        # Add 0.5% (the 1.005 factor) so in case the blending dataset does
        # not uniformly distribute the number of samples, we still have
        # samples left to feed to the network.
        weighted_num_samples = []
        for weight in weights:
            # weighted_num_samples.append(int(math.ceil(num_samples * weight * 1.005)))
            weighted_num_samples.append(int(math.ceil(num_samples * weight)))
        return weights, weighted_num_samples

    train_batch_size = num_gpus / (PP*TP) * grad_accu
    train_iters =  math.floor( all_tokens / seq_len / train_batch_size ) 
    num_samples = train_iters * train_batch_size
    weights, weighted_num_samples = get_normalized_weights_and_num_samples(res_tokens_list, num_samples)
    print("框架下验证")
    print(f"参数：bs {train_batch_size} seq_len {seq_len}")
    print(f"参数：TPxPP {TP} {PP} num_gpus {num_gpus}")
    print(f"训练代数：{train_iters}")
    print("框架下 输出 weight", weights)
    print("框架下输出 weighted_num_samples", weighted_num_samples)
    
    return weights_with_arxiv, res_tokens_list
    
    
    

if __name__ == '__main__':
    #                    book,      c4,        cc,        ,github,   stack,     wiki
    mult_dataset_tokens=[1345378446,4234915249,21278228280,289448619,1579013472,1480596526]
    mult_dataset_weight=[0.045, 0.15, 0.67, 0.045, 0.02, 0.045]
    arxiv_dataset_tokens = [5602566144, 1320550988]
    arxiv_dataset_weights = 0.5
    
    calculate_training_weights(mult_dataset_tokens,mult_dataset_weight,arxiv_dataset_tokens,arxiv_dataset_weights)

    # print('*'*50)
    # print(count_learning_tokens(20003,tokens=50592969575,bs=384.0,mult_dataset_weight=mult_dataset_weight))
