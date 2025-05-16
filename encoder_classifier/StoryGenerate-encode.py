import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import pickle

from dateset import TrainCorpos
from TransformerModelEncode import GenerateModel


# path_train = 'raw_dataset/ROCStories_train.csv'
# path_val =  'raw_dataset/ROCStories_val.csv'
# corpos_train = TrainCorpos(path_train)
# corpos_valid = TrainCorpos(path_val)

# with open('dataset/train_x.pkl', 'wb') as file:
#     pickle.dump(corpos_train.x_seq_ls, file)
# with open('dataset/train_y.pkl', 'wb') as file:
#     pickle.dump(corpos_train.y_seq_ls, file)
# with open('dataset/val_x.pkl', 'wb') as file:
#     pickle.dump(corpos_valid.x_seq_ls, file)
# with open('dataset/val_y.pkl', 'wb') as file:
#     pickle.dump(corpos_valid.y_seq_ls, file)
# vocab_length = corpos_train.vocab_length

vocab_length = 50260

with open('dataset/train_x.pkl', 'rb') as file:
    corpos_train_x_seq_ls = pickle.load(file)
with open('dataset/train_y.pkl', 'rb') as file:
    corpos_train_y_seq_ls = pickle.load(file)
with open('dataset/val_x.pkl', 'rb') as file:
    corpos_valid_x_seq_ls = pickle.load(file)
with open('dataset/val_y.pkl', 'rb') as file:
    corpos_valid_y_seq_ls = pickle.load(file)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# train_x = torch.tensor(corpos_train.x_seq_ls).to(device)
# train_y = torch.tensor(corpos_train.y_seq_ls).to(device)
# val_x = torch.tensor(corpos_valid.x_seq_ls).to(device)
# val_y = torch.tensor(corpos_valid.y_seq_ls).to(device)

train_x = torch.tensor(corpos_train_x_seq_ls).to(device)
train_y = torch.tensor(corpos_train_y_seq_ls).to(device)
val_x = torch.tensor(corpos_valid_x_seq_ls).to(device)
val_y = torch.tensor(corpos_valid_y_seq_ls).to(device)

# y_train_onehot = F.one_hot(train_y, num_classes=corpos_train.vocab_length)
# y_val_onehot = F.one_hot(val_y, num_classes=corpos_train.vocab_length)
#
# decoder_output_train_tensor = torch.tensor(y_train_onehot).long()
# decoder_output_test_tensor = torch.tensor(y_val_onehot).long()

# 设置 cuda

model = GenerateModel(vocab_size=50260, ntoken=128).to(device)

criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-7)

'''
第一次3e-4 batch = 128
'''

batch_size = 64
epochs = 5

for epoch in range(epochs):
    total_loss = 0
    loss_ls = []
    acc_ls = []
    clo = 0
    tqdm_iterator = tqdm(range(0, len(train_x), batch_size), dynamic_ncols=True, desc=f'Epoch {epoch + 1}/{epochs}')
    for i in tqdm_iterator:
        src = train_x[i:i + batch_size].to(device)
        tgt = train_y[i:i + batch_size].to(device)
        tgt_y = F.one_hot(tgt, num_classes=vocab_length).to(device)
        optimizer.zero_grad()
        out = model(src)
        loss = criterion(out.view(-1, out.size(-1)), tgt_y.squeeze().float())
        loss.backward()
        optimizer.step()
        out = out.argmax(dim=-1)
        acc = (out == tgt).float().mean()

        acc_ls.append(acc.item())
        loss_ls.append(loss.item())

        total_loss += loss.item()
        tqdm_iterator.set_postfix(loss=sum(loss_ls) / len(loss_ls),
                                  acc=sum(acc_ls) / len(acc_ls))
    print("epoch: {}, loss: {}".format(epoch, total_loss / len(train_x)))

    colon = 0
    accu = 0
    for i in tqdm(range(0, len(val_x), batch_size)):
        colon += 1
        src = val_x[i:i + batch_size].to(device)
        tgt = val_y[i:i + batch_size].to(device)
        out = model(src)
        out = model.predictor(out)
        out = out.argmax(dim=-1)
        # print(out.shape)
        # print(tgt_y.shape)
        acc = (out == tgt).float().mean()
        # print(acc)
        accu += acc.item()
        # print(acc)
    print("train acc: {}".format(accu / colon))
    torch.save(model, "model_en_1e-7.ckpt")


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


