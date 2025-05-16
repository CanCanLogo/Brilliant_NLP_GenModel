import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import pickle
# import nltk
# from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

from dateset_64 import TrainCorpos
from TransformerModel import GenerateModel

from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2", max_vocab_size = 10000)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.add_tokens(["[SOS]", "[EOS]", "[UNK]"])
# 设置特殊标记
tokenizer.unk_token = "[UNK]"
tokenizer.sos_token = "[SOS]"
tokenizer.eos_token = "[EOS]"


path_train = '../raw_dataset/ROCStories_train.csv'
path_val =  '../raw_dataset/ROCStories_val.csv'
corpos_train = TrainCorpos(path_train)
corpos_valid = TrainCorpos(path_val)

# with open('dataset/train_x.pkl', 'wb') as file:
#     pickle.dump(corpos_train.x_seq_ls, file)
# with open('dataset/train_y.pkl', 'wb') as file:
#     pickle.dump(corpos_train.y_seq_ls, file)
# with open('dataset/val_x.pkl', 'wb') as file:
#     pickle.dump(corpos_valid.x_seq_ls, file)
# with open('dataset/val_y.pkl', 'wb') as file:
#     pickle.dump(corpos_valid.y_seq_ls, file)

vocab_length = corpos_train.vocab_length

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_x = torch.tensor(corpos_train.x_seq_ls).to(device)
# print(len(corpos_train.y_seq_ls[1]))
# print(len(corpos_train.y_seq_ls))
train_y = torch.tensor(corpos_train.y_seq_ls).to(device)
val_x = torch.tensor(corpos_valid.x_seq_ls).to(device)
val_y = torch.tensor(corpos_valid.y_seq_ls).to(device)


# y_train_onehot = F.one_hot(train_y, num_classes=corpos_train.vocab_length)
# y_val_onehot = F.one_hot(val_y, num_classes=corpos_train.vocab_length)
#
# decoder_output_train_tensor = torch.tensor(y_train_onehot).long()
# decoder_output_test_tensor = torch.tensor(y_val_onehot).long()

# 设置 cuda

model = GenerateModel().to(device)

# criterion = nn.CrossEntropyLoss().to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-7)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
# model = torch.load('model_de_1e-6.ckpt').to(device)

'''
第一次3e-4 batch = 128
'''
def log_probs_from_logits(logits, labels):
    logp = F.log_softmax(logits, dim=-1)
    logp_label = torch.gather(logp, 2, labels)
    print(logp_label.shape)
    return logp_label
def sequence_logprob(logits, labels):
    with torch.no_grad():
        log_probs = log_probs_from_logits(
            logits, labels)
        seq_log_prob = torch.sum(log_probs[:, :])
    return seq_log_prob.cpu().numpy()

batch_size = 32
epochs = 1
d_emb = 128
n_tokens = 128
n_head = 4
num_encoder_layers = 5
num_decoder_layers = 3
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-7)

model = torch.load("model_de_1e-7.ckpt")

for epoch in range(epochs):
    total_loss = 0
    clo = 0
    loss_ls = []
    acc_ls = []
    # tqdm_iterator = tqdm(range(0, len(train_x), batch_size), dynamic_ncols=True, desc=f'Epoch {epoch + 1}/{epochs}')
    # for i in tqdm_iterator:
    #
    #     src = train_x[i:i + batch_size].to(device)
    #     tgt = train_y[i:i + batch_size, :-1].to(device)
    #     tgt_y = train_y[i:i + batch_size, 1:].to(device)
    #
    #     tgt_y_hot = F.one_hot(tgt_y, num_classes=vocab_length).to(device)
    #
    #     # print(tokenizer.decode(src[0].tolist()))
    #     #
    #     # print(tokenizer.decode(tgt[0].tolist()))
    #
    #     # print(src.shape)
    #     # print(tgt.shape)
    #     # 清空梯度
    #     optimizer.zero_grad()
    #     # 进行transformer的计算
    #     # print(src.shape)
    #     # print(tgt.shape)
    #     out = model(src, tgt)
    #     # 将结果送给最后的线性层进行预测
    #     # print(out)
    #     # print(out.shape)
    #     # out = model.predictor(out[:, -1].squeeze())
    #     out = model.predictor(out)
    #     # print(tgt_y[0][0])
    #     # 计算loss
    #
    #     # loss = criterion(out.view(-1, out.size(-1)), tgt_y.squeeze().float())
    #     loss = criterion(out.view(-1, out.size(-1)), tgt_y_hot.view(-1, out.size(-1)).float())
    #     # print(out.shape)
    #     # print(tgt_y_hot.shape)
    #
    #     # log_p = sequence_logprob(out.float(), tgt_y_hot) / batch_size * 31
    #     # print(log_p)
    #
    #     # 计算梯度
    #     loss.backward()
    #     # 更新参数
    #     optimizer.step()
    #     # scheduler.step(loss.item())
    #     total_loss += loss.item()
    #
    #     out = out.argmax(dim=-1)
    #
    #     out_decode = [tokenizer.decode(sample).split() for sample in out]
    #     tgt_decode = [tokenizer.decode(sample).split() for sample in tgt_y]
    #     bleu_score = corpus_bleu(tgt_decode, out_decode, smoothing_function=SmoothingFunction().method7)
    #
    #     acc = (out == tgt_y).float().mean()
    #     acc_ls.append(acc.item())
    #     loss_ls.append(loss.item())
    #
    #     tqdm_iterator.set_postfix(loss=sum(loss_ls) / len(loss_ls),
    #                               acc=sum(acc_ls) / len(acc_ls)
    #                             )
    # print("epoch: {}, loss: {}".format(epoch, total_loss / len(train_x)))

    colon = 0
    accu = 0
    accu_ls = []
    tqdm_iterator_val = tqdm(range(0, len(val_x), batch_size), dynamic_ncols=True, desc=f'Epoch {epoch + 1}/{epochs}')
    for i in tqdm_iterator_val:
        colon += 1
        src = val_x[i:i + batch_size].to(device)
        tgt = val_y[i:i + batch_size, :-1].to(device)
        tgt_y = val_y[i:i + batch_size, 1:].to(device)
        # tgt_y = F.one_hot(tgt_y, num_classes=vocab_length).to(device)
        out = model(src, tgt)

        out = model.predictor(out)
        out = out.argmax(dim=-1)
        # print(out)
        bleu_score1 = 0.0107
        out_decode = [tokenizer.decode(sample).split() for sample in out]
        tgt_decode = [tokenizer.decode(sample).split() for sample in tgt_y]
        bleu_score = corpus_bleu(tgt_decode, out_decode, smoothing_function=SmoothingFunction().method7)
        acc = (out == tgt_y).float().mean()
        # print(acc)
        accu += acc.item()
        # print(acc)
        accu_ls.append(acc.item())

        tqdm_iterator_val.set_postfix(
                                      bleu_score = bleu_score1)
    # print("train acc: {}".format(accu / colon))
    # torch.save(model, "model_de_1e-7.ckpt")


    # # 计算训练集上的准确率


    # with torch.no_grad():
    #     src = val_x
    #     tgt = val_y
    #     tgt_y = decoder_output_test_tensor
    #
    #     out = model(src, tgt)
    #     out = model.predictor(out)
    #     out = out.argmax(dim=-1)
    #     acc = (out == tgt_y).float().mean()
    #     print("test acc: {}".format(acc))


