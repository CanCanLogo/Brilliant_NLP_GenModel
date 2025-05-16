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

    def __init__(self, d_emb = 128, ntokens = 128, PAD_token = 50257, num_embeddings=50260):
        super(GenerateModel, self).__init__()
        self.PAD_token = PAD_token

        # 词典数为10000
        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=128)
        self.transformer = nn.Transformer(d_model=d_emb, nhead = 4 ,num_encoder_layers=5, num_decoder_layers=3, dim_feedforward=128, batch_first=True)
        self.positional_encoding = PositionalEncoding(d_emb, dropout=0)
        self.predictor = nn.Linear(d_emb, num_embeddings)

    def forward(self, src, tgt):
        # 生成mask
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size()[-1]).to(device)
        src_key_padding_mask = GenerateModel.get_key_padding_mask(src)
        tgt_key_padding_mask = GenerateModel.get_key_padding_mask(tgt)

        src = self.embedding(src)
        tgt = self.embedding(tgt)
        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)

        # 将准备好的数据送给transformer
        out = self.transformer(src, tgt,
                               tgt_mask=tgt_mask,
                               src_key_padding_mask=src_key_padding_mask,
                               tgt_key_padding_mask=tgt_key_padding_mask)
        """
        这里直接返回transformer的结果。因为训练和推理时的行为不一样，
        所以在该模型外再进行线性层的预测。
        """
        return out

    # def get_key_padding_mask(self, tokens):
    #     """
    #     用于key_padding_mask
    #     """
    #
    #     key_padding_mask = torch.zeros(tokens.size())
    #     key_padding_mask[tokens == self.PAD_token] = -torch.inf
    #     return key_padding_mask
    # @staticmethod
    def get_key_padding_mask(tokens):
        """
        用于key_padding_mask
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        PAD_token = 50257
        key_padding_mask = torch.zeros(tokens.size())
        # print(key_padding_mask.shape)
        key_padding_mask[tokens == PAD_token] = -torch.inf
        return key_padding_mask.bool().to(device)



