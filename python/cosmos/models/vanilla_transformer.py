import math

import torch
from config import ModelArgs
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        # 位置エンコーディングの計算
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, seq_len, embedding_dim].
        x = x + self.pe[:, : x.size(1), :]
        return x


class VanillaTransformerModel(nn.Module):
    def __init__(self, params: ModelArgs) -> None:
        super().__init__()
        self.d_model = params.dim

        # トークン埋め込み層
        self.embedding = nn.Embedding(params.vocab_size, params.dim)
        self.pos_encoder = PositionalEncoding(params.dim)

        # Transformerエンコーダー
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=params.dim,
            nhead=params.n_heads,
            dim_feedforward=params.dim * 4,
            dropout=0.0,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=params.n_layers)

        # 予測ヘッド
        self.fc_out = nn.Linear(params.dim, params.vocab_size)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc_out.bias.data.zero_()
        self.fc_out.weight.data.uniform_(-initrange, initrange)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, seq_len].
        seq_len = x.size(1)
        device = x.device
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(
            x,
            mask=nn.Transformer.generate_square_subsequent_mask(seq_len).to(device),
        )
        x = self.fc_out(x)
        return x
