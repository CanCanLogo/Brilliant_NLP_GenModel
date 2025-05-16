import pandas as pd
from transformers import GPT2Tokenizer
import random
from tqdm import tqdm


class TrainCorpos():
    def __init__(self, path, max_vocab_size = 10000, max_length_in = 32, max_length_out = 128):
        self.data_length = None
        self.vocab_length = 50260
        self.max_item_length = 107
        self.token_counts = None
        self.max_length_out = max_length_out

        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2", max_vocab_size = max_vocab_size)
        self.output_ls = self.ToList(path)
        self.tokenized_corpus = self.ToToken(ls = self.output_ls, max_length=max_length_out, max_vocab_size=max_vocab_size)
        self.x_seq_ls, self.y_seq_ls = self.split()



    def ToList(self, path):
        data = pd.read_csv(path, sep=',', encoding='utf-8')
        # csv的分隔符是,
        '''
        Index(['storyid', 'storytitle', 'sentence1', 'sentence2', 'sentence3', 'sentence4', 'sentence5']
        '''
        self.data_length = len(data)
        output_ls = []
        for i in range(self.data_length):
            # 以下得出输出句子列表
            out = ''
            for _ in range(1, 6, 1):
                category = 'sentence' + str(_)
                # print(data[category][i])
                out = out + data[category][i]
                # join“将字符串abc中的每个成员以字符','分隔开再拼接成一个字符串”
            output_ls.append(out)

        return output_ls

    def ToToken(self, ls, max_length=128, max_vocab_size = 10000):
        # 添加填充标记
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.tokenizer.add_tokens(["[SOS]", "[EOS]", "[UNK]"])
        # 设置特殊标记
        self.tokenizer.unk_token = "[UNK]"
        self.tokenizer.sos_token = "[SOS]"
        self.tokenizer.eos_token = "[EOS]"

        # 分词和预处理
        # tokenized_corpus = self.tokenizer(ls, padding=True, truncation=True, return_tensors="pt",
        #                                     max_length=max_length, max_vocab_size=max_vocab_size)
        tokenized_corpus = self.tokenizer(ls, padding=True, truncation=True, return_tensors="pt",
                                          max_length=max_length)

        # 获取 attention mask
        attention_mask = tokenized_corpus["attention_mask"]

        # 计算每个句子在 padding 之前的 token 数
        self.token_counts = attention_mask.sum(dim=1).tolist()
        max_tokens = max(self.token_counts)
        # 打印每个句子在 padding 之前的 token 数
        # max_le = 0
        for i, count in enumerate(self.token_counts):
            # max_le = max(max_le, count)
            print(f"Sentence {i + 1} had {count} tokens before padding.")
        print(f"The maximum number of tokens in a sentence is {max_tokens}.")

        return tokenized_corpus
    def split(self):

        tokenized_list = self.tokenized_corpus["input_ids"].tolist()

        x_seq_ls = []
        y_seq_ls = []
        for i in tqdm(range(self.data_length)):
            for _ in range(10):
                # 随机选一个子串
                end = random.randint(10, self.token_counts[i]-1)
                # 保存到 x_seq_ls 和 y_seq_ls 中
                '''
                补充到128的长度
                '''
                x_seq = tokenized_list[i][0:end] + [50257] * (self.max_length_out - end)

                x_seq_ls.append(x_seq)
                y_seq_ls.append([tokenized_list[i][end]])
        return x_seq_ls, y_seq_ls

    def testCorpos(self, switch_code = True):
        # 查看出现频率最高的前 10 个 token

        # 这里的true和false是用来选择查看哪一个tokenize
        # if switch_code == True:
        #     tokenized_corpus = self.tokenized_corpus_input
        # else:
        #     tokenized_corpus = self.tokenized_corpus_output
        tokenized_corpus = self.tokenized_corpus
        vocab = self.tokenizer.get_vocab()
        # print(vocab)

        # 统计 token 出现的频次
        token_counts = {}
        for sentence_tokens in tokenized_corpus["input_ids"]:
            for token_id in sentence_tokens.tolist():
                token = self.tokenizer.convert_ids_to_tokens(token_id)
                token_counts[token] = token_counts.get(token, 0) + 1

        # 对 token 按频次进行排序
        sorted_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)

        # 打印出现频率最高的前 10 个 token
        top_10_tokens = sorted_tokens[:10]
        for token, count in top_10_tokens:
            print(f"{token}: {count}")
        return vocab

if __name__ == "__main__":
    path = 'raw_dataset/ROCStories_train.csv'
    save_path = 'dataset/train.txt'
    corpos = TrainCorpos(path)
    vocab = corpos.testCorpos()
    print(vocab)
    print(len(corpos.x_seq_ls[0]))
    # vocab length: 50260