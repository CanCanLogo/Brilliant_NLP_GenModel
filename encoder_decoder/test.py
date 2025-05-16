from transformers import GPT2Tokenizer
import torch
import numpy as np
import torch.nn.functional as F

tokenizer = GPT2Tokenizer.from_pretrained("../gpt2", max_vocab_size = 10000)

vocab_size = tokenizer.vocab_size

# print("Vocabulary Size:", vocab_size)


tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.add_tokens(["[SOS]", "[EOS]", "[UNK]"])
# 设置特殊标记
tokenizer.unk_token = "[UNK]"
tokenizer.sos_token = "[SOS]"
tokenizer.eos_token = "[EOS]"

test_string = 'David noticed he had put on a lot of weight recently.'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('input:')
print(test_string)

# 前后加上开始和结束标志
# test_string = '<SOS> ' + test_string
# 把 test_string 转化为token
test_string_token = tokenizer(test_string, padding=True, truncation=True, return_tensors="pt",
                                          max_length=32)
# 截取后10个字
raw_long = len(test_string_token['input_ids'].squeeze())

test_string_token = test_string_token['input_ids'].squeeze().tolist()
# 转化为 numpy，补齐
# print(test_string_token)
test_string_mat = np.array([test_string_token])

# test_string_mat 转化为 tensor
test_string_tensor = torch.tensor(test_string_mat).to(device)
# print(len(test_string_tensor))

# model = torch.load("model_one_1e-7.ckpt")

model = torch.load("model_de_1e-7.ckpt")

'''
.: 13
SOS:50258
'''

# def log_probs_from_logits(logits, labels):
#     logp = F.log_softmax(logits, dim=-1)
#     logp_label = torch.gather(logp, 2, labels.unsqueeze(2)).squeeze(-1)
#     return logp_label
# def sequence_logprob(logits, labels, input_len=0):
#     with torch.no_grad():
#         log_probs = log_probs_from_logits(
#             logits[:, :-1, :], labels[:, 1:])
#         seq_log_prob = torch.sum(log_probs[:, input_len:])
#     return seq_log_prob.cpu().numpy()

def decode(model, src, max_iter, SOS_token, EOS_token, raw_long, temperature = 0.1):
    # model = model.eval()
    src = src
    tgt = torch.LongTensor([[SOS_token]]).to(device)
    # 一个一个词预测，直到预测为<eos>，或者达到句子最大长度
    for i in range(max_iter):
        out = model(src, tgt)
        predict = model.predictor(out[:, -1])
        # 对 logits 进行 softmax 操作，并应用温度参数
        probabilities = F.softmax(predict / temperature, dim=-1)
        # 从概率分布中采样一个 token
        sampled_token = torch.multinomial(probabilities[ -1, :], 1)
        # 和之前的预测结果拼接到一起

        tgt = torch.concat([tgt, sampled_token.unsqueeze(0)], dim=1)

        # # predict = model.predictor(predict[:, -1])
        # # print(out.shape)
        # # 找出最大值的index
        # y = torch.argmax(predict, dim=1)
        # # 和之前的预测结果拼接到一起
        # tgt = torch.concat([tgt, y.unsqueeze(0)], dim=1)
        raw_long += 1
        # 如果为<eos>，说明预测结束，跳出循环
        if sampled_token == EOS_token:
            break
    return tgt

SOS_token = 50258
EOS_token = 13

decoded = decode(model,
       src=test_string_tensor,
       max_iter=20,
       SOS_token=SOS_token,
       EOS_token=EOS_token,
       raw_long=raw_long)

# logp = sequence_logprob(test_string_tensor, test_string_tensor, input_len=len(test_string_tensor[0]))

decoded_array = decoded.cpu().numpy()

# 将每个样本的序列转换为文本
decoded_texts = [tokenizer.decode(sample) for sample in decoded_array]
print('output:')
print(['[SOS]....................'])
