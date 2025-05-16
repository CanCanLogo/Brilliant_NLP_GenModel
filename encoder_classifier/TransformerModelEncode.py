import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 初始化Shape为(max_len, d_model)的PE (positional encoding)
        pe = torch.zeros(max_len, d_model)
        # 初始化一个tensor [[0, 1, 2, 3, ...]]
        position = torch.arange(0, max_len).unsqueeze(1)
        # 这里就是sin和cos括号中的内容，通过e和ln进行了变换
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        # 计算PE(pos, 2i)
        pe[:, 0::2] = torch.sin(position * div_term)
        # 计算PE(pos, 2i+1)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 为了方便计算，在最外面在unsqueeze出一个batch
        pe = pe.unsqueeze(0)
        # 如果一个参数不参与梯度下降，但又希望保存model的时候将其保存下来
        # 这个时候就可以用register_buffer
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x 为embedding后的inputs，例如(1,7, 128)，batch size为1,7个单词，单词维度为128
        """
        # 将x和positional encoding相加。
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)

class GenerateModel(nn.Module):

    def __init__(self, vocab_size, ntoken, d_emb=128, d_hid=2048, nhead=4, nlayers=6, dropout=0.2, num_classes=15,
                 embedding_weight=None):
        super(GenerateModel, self).__init__()
        # 将"预训练的词向量"整理成 token->embedding 的二维映射矩阵 emdedding_weight 的形式，初始化 _weight
        # 当 emdedding_weight == None 时，表示随机初始化

        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_emb, _weight=embedding_weight)
        self.pos_encoder = PositionalEncoding(d_model=d_emb, max_len=ntoken, dropout = 0.1)
        self.encode_layer = nn.TransformerEncoderLayer(d_model=d_emb, nhead=nhead, dim_feedforward=d_hid)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=self.encode_layer, num_layers=nlayers)
        self.predictor = nn.Linear(d_emb, vocab_size)

        # ------------------------------------------------------end------------------------------------------------------#

    def forward(self, x):
        x = self.embed(x)
        x = x.permute(1, 0, 2)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)
        x = torch.mean(x, dim=1)
        x = self.predictor(x)

        return x





