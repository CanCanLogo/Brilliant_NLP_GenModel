from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, AdamW, get_linear_schedule_with_warmup
import torch
import os
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pandas as pd

from dateset_64 import TrainCorpos

# from transformers import AdamW

class DataFull(Dataset):

    def __init__(self, path, truncate=False, gpt2_type="../gpt2", max_length=1024):

        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
        self.lyrics = []
        self.output_ls = self.ToList(path)
        for row in self.output_ls:
            self.lyrics.append(torch.tensor(
                self.tokenizer.encode(f"<|beginoftext|>{row[:max_length]}<|endoftext|>")
            ))
        if truncate:
            self.lyrics = self.lyrics[:5000]
        self.lyrics_count = len(self.lyrics)
    def ToList(self, path):
        data = pd.read_csv(path, sep=',', encoding='utf-8')
        # csv的分隔符是,
        '''
        Index(['storyid', 'storytitle', 'sentence1', 'sentence2', 'sentence3', 'sentence4', 'sentence5']
        '''
        self.data_length = len(data)
        input_ls = []
        output_ls = []
        for i in range(self.data_length):
            # 以下得出输出句子列表
            input_ls.append(data['sentence1'][i])
            out = ''
            for _ in range(1, 6, 1):

                category = 'sentence' + str(_)
                # print(data[category][i])
                out = out + data[category][i]
                # join“将字符串abc中的每个成员以字符','分隔开再拼接成一个字符串”
            output_ls.append(out)

        return output_ls
    def __len__(self):
        return self.lyrics_count

    def __getitem__(self, item):
        return self.lyrics[item]

class DataPrompt(Dataset):

    def __init__(self, path, truncate=False, gpt2_type="../gpt2", max_length=128):

        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
        self.prompt = []
        self.labels = []
        self.input_ls, self.output_ls = self.ToList(path)
        self.pad = self.tokenizer.encode(f"<|pad|>")[0]
        # print(self.pad[0])

        for row in self.output_ls:
            self.labels.append(torch.tensor(
                self.tokenizer.encode(f"<|beginoftext|>{row[:max_length]}<|endoftext|>")
            ))
        for row in self.input_ls:
            self.prompt.append(torch.tensor(
                self.tokenizer.encode(f"<|beginoftext|>{row[:max_length]}<|endoftext|>")
            ))


        if truncate:
            self.prompt = self.prompt[:5000]
            self.labels = self.labels[:5000]

        self.lyrics_count = len(self.prompt)
    def ToList(self, path):
        data = pd.read_csv(path, sep=',', encoding='utf-8')
        # csv的分隔符是,
        '''
        Index(['storyid', 'storytitle', 'sentence1', 'sentence2', 'sentence3', 'sentence4', 'sentence5']
        '''
        self.data_length = len(data)
        input_ls = []
        output_ls = []
        for i in range(self.data_length):
            # 以下得出输出句子列表
            input_ls.append(data['sentence1'][i])
            out = ''
            for _ in range(2, 6, 1):

                category = 'sentence' + str(_)
                # print(data[category][i])
                out = out + data[category][i]
                # join“将字符串abc中的每个成员以字符','分隔开再拼接成一个字符串”
            output_ls.append(out)

        return input_ls, output_ls
    def __len__(self):
        return self.lyrics_count

    def __getitem__(self, item):
        input_ids, labels = self.pad_to_max_length(self.prompt[item], self.labels[item])
        return input_ids, labels

    def pad_to_max_length(self, input_ids, labels):
        # 确保 input_ids 和 labels 长度相同
        input_length = input_ids.size(0)
        label_length = labels.size(0)

        if input_length > label_length:
            labels = torch.nn.functional.pad(labels, value=self.pad, pad=(0, input_length - label_length))
        elif label_length > input_length:
            input_ids = torch.nn.functional.pad(input_ids, value=self.pad, pad=(0, label_length - input_length))

        return input_ids, labels



def test(text):
    indexed_tokens = tokenizer.encode(text)
    # 转换为pytorch tensor
    # tensor([[ 8241,   373,  5395,   367, 19069,  5633,  5395,   367, 19069,   373, 257]])
    # shape为 torch.Size([1, 11])
    tokens_tensor = torch.tensor([indexed_tokens]).to(device)
    model.eval()
    # 预测所有token
    with torch.no_grad():
        for i in range(50):
            # 将输入tensor输入，就得到了模型的输出，非常简单
            # outputs是一个元组，所有huggingface/transformers模型的输出都是元组
            # 本初的元组有两个，第一个是预测得分（没经过softmax之前的，也叫作logits），
            # 第二个是past，里面的attention计算的key value值
            # 此时我们需要的是第一个值
            outputs = model(tokens_tensor)
            # predictions shape为 torch.Size([1, 11, 50257])，
            # 也就是11个词每个词的预测得分（没经过softmax之前的）
            # 也叫做logits
            predictions = outputs[0]

            # tokens_tensor = torch.argmax(predictions, dim = 2)
            # tokens_tensor = torch.argmax(predictions[0, -1, :])

            new_element = torch.tensor([[torch.argmax(predictions[0, -1, :]).item()]]).to(device)
            tokens_tensor = torch.cat((tokens_tensor, new_element), dim=1)
            print(tokens_tensor)

    predicted_text = tokenizer.decode(tokens_tensor.squeeze().cpu().tolist())
    # 我们需要预测下一个单词，所以是使用predictions第一个batch，最后一个词的logits去计算
    # predicted_index = 582，通过计算最大得分的索引得到的
    # predicted_index = torch.argmax(predictions[0, -1, :]).item()
    # 反向解码为我们需要的文本
    # predicted_text = tokenizer.decode(indexed_tokens + [predicted_index])
    # 解码后的文本：'Who was Jim Henson? Jim Henson was a man'
    # 成功预测出单词 'man'
    print(predicted_text)

def beamSearch(text, max_length, num_beams):
    # 将文本编码为模型输入
    num_beams = 1
    input_ids = tokenizer.encode(text, return_tensors='pt').to(device)
    # 为输入设置 attention_mask
    attention_mask = torch.ones_like(input_ids).to(device)
    model.eval()
    output_beam = model.generate(input_ids, attention_mask=attention_mask, max_length=max_length,temperature=3,
                                 num_beams=num_beams, do_sample=False)
    print('input:')
    print(text)
    print('output:')
    print(tokenizer.decode(output_beam[0]))





#Accumulated batch size (since GPT2 is so big)
def pack_tensor(new_tensor, packed_tensor, max_seq_len):
    if packed_tensor is None:
        return new_tensor, True, None
    if new_tensor.size()[1] + packed_tensor.size()[1] > max_seq_len:
        return packed_tensor, False, new_tensor
    else:
        packed_tensor = torch.cat([new_tensor, packed_tensor[:, 1:]], dim=1)
        return packed_tensor, True, None

def train(
    dataset, model, tokenizer,
    batch_size=16, epochs=3, lr=5e-6,
    max_seq_len=400, warmup_steps=200,
    gpt2_type="gpt2", output_dir="", output_prefix="gpt2-5e-6-",
    test_mode=False,save_model_on_epoch=True,
):

    acc_steps = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.cuda()
    model.train()

    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=-1
    )
    # print(dataset.shape)

    train_dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    loss=0
    accumulating_batch_count = 0
    input_tensor = None

    for epoch in range(epochs):
        tqdm_iterator = tqdm(enumerate(train_dataloader), dynamic_ncols=True, desc=f'Epoch {epoch + 1}/{epochs}')
        for idx, entry in tqdm_iterator:
            (input_tensor, carry_on, remainder) = pack_tensor(entry, input_tensor, 768)
            if carry_on and idx != len(train_dataloader) - 1:
                continue
            input_tensor = input_tensor.to(device)
            outputs = model(input_tensor, labels=input_tensor)
            loss = outputs[0]
            loss.backward()
            '''
            return CausalLMOutputWithCrossAttentions(
                loss=loss,
                logits=lm_logits,
                past_key_values=transformer_outputs.past_key_values,
                hidden_states=transformer_outputs.hidden_states,
                attentions=transformer_outputs.attentions,
                cross_attentions=transformer_outputs.cross_attentions
            )
            '''


            tqdm_iterator.set_postfix(loss=loss.item()
                                      )

            if (accumulating_batch_count % batch_size) == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                model.zero_grad()

            accumulating_batch_count += 1
            input_tensor = None
        if save_model_on_epoch:
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, f"{output_prefix}-{epoch}.pt"),
            )
    return model

def promptTrain(
    dataset, model, tokenizer,
    batch_size=16, epochs=3, lr=2e-5,
    max_seq_len=400, warmup_steps=200,
    gpt2_type="gpt2", output_dir="", output_prefix="prompt",
    test_mode=False,save_model_on_epoch=True,
):
    acc_steps = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.cuda()
    model.train()

    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=-1
    )
    # print(dataset.shape)

    train_dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    loss=0
    accumulating_batch_count = 0
    # input_tensor = None

    for epoch in range(epochs):
        tqdm_iterator = tqdm(enumerate(train_dataloader), dynamic_ncols=True, desc=f'Epoch {epoch + 1}/{epochs}')
        for idx, data in tqdm_iterator:
            batch_x, batch_y = data[0].to(device), data[1].to(device)
            input_tensor = batch_x.to(device)
            labels = batch_y.to(device)
            outputs = model(input_tensor, labels=labels)

            loss = outputs[0]
            '''
            return CausalLMOutputWithCrossAttentions(
                loss=loss,
                logits=lm_logits,
                past_key_values=transformer_outputs.past_key_values,
                hidden_states=transformer_outputs.hidden_states,
                attentions=transformer_outputs.attentions,
                cross_attentions=transformer_outputs.cross_attentions
            )
            '''
            loss.backward()

            tqdm_iterator.set_postfix(loss=loss.item()
                                      )

            if (accumulating_batch_count % batch_size) == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                model.zero_grad()

            accumulating_batch_count += 1
            input_tensor = None
        if save_model_on_epoch:
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, f"{output_prefix}-{epoch}.pt"),
            )
    return model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = GPT2Config.from_pretrained('../gpt2')

tokenizer = GPT2Tokenizer.from_pretrained('../gpt2')
model = GPT2LMHeadModel.from_pretrained('../gpt2', config=config).to(device)
model.load_state_dict(torch.load('gpt2-2.pt'))
test_string = 'Every weekend the Jefferson\'s go to the park.'
# test(test_string)
beamSearch(test_string, 50, 5)

# for i, param in enumerate(model.parameters()):
#     print(i)
#     if i < 20:  # 前面一些参数冻结
#         param.requires_grad = False

# model.load_state_dict(torch.load('gpt2-2.pt'))

# path_train = '../raw_dataset/ROCStories_train.csv'
# # corpos_train = TrainCorpos(path_train)
# # train_y = torch.tensor(corpos_train.y_seq_ls)
#
# dataset = DataFull(path_train, truncate=True, gpt2_type="gpt2")
#
# # dataset = DataPrompt(path_train)
# # model = promptTrain(dataset, model, tokenizer)
# model = train(dataset, model, tokenizer)



