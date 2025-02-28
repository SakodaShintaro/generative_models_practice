"""Wayve101トークンデータセットを使ったTransformerモデルの学習"""

import argparse
import math
from pathlib import Path

import torch
from dataset import Wayve101TokensDataset
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

VOCAB_SIZE = 64_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Transformer model on Wayve101 tokens")
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--seq_len", type=int, default=8, help="Sequence length for the dataset")
    parser.add_argument("--d_model", type=int, default=512, help="Model dimension")
    parser.add_argument("--save_dir", type=Path, default=Path("./checkpoints"))

    return parser.parse_args()


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


class TransformerModel(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        # トークン埋め込み層
        self.embedding = nn.Embedding(VOCAB_SIZE, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        # Transformerエンコーダー
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)

        # 予測ヘッド
        self.fc_out = nn.Linear(d_model, VOCAB_SIZE)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc_out.bias.data.zero_()
        self.fc_out.weight.data.uniform_(-initrange, initrange)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # src: [batch_size, seq_len].
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)

        # エンコーダー通過
        memory = self.transformer_encoder(
            src,
            mask=src_mask,
            src_key_padding_mask=src_key_padding_mask,
        )

        # 予測
        output = self.fc_out(memory)
        return output


def create_masks(seq_len: int, device: torch.device) -> torch.Tensor:
    # 自己回帰マスク (future tokens are masked)
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device) * float("-inf"), diagonal=1)
    return mask


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0

    for batch_idx, data in enumerate(tqdm(dataloader, desc="Training")):
        # トークンシーケンス [batch_size, seq_len, token_dim].
        data = data.to(device)
        batch_size, seq_len, token_dim = data.shape

        # トークンをフラット化 [batch_size, seq_len, token_dim] -> [batch_size, seq_len * token_dim]
        flattened_data = data.reshape(batch_size, seq_len * token_dim)

        # 入力は最後のトークンを除いたシーケンス
        src = flattened_data[:, :-1].long()
        # ターゲットは最初のトークンを除いたシーケンス（次のトークンを予測）
        tgt = flattened_data[:, 1:].long()

        # マスクを作成
        src_mask = create_masks(src.size(1), device)

        # モデル予測
        optimizer.zero_grad()
        output = model(src, src_mask=src_mask)

        # ロス計算
        loss = criterion(output.reshape(-1, output.size(-1)), tgt.reshape(-1))
        loss.backward()

        # 勾配クリッピング
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")

    return total_loss / len(dataloader)


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for data in tqdm(dataloader, desc="Validation"):
            # トークンシーケンス [batch_size, seq_len, token_dim].
            data = data.to(device)
            batch_size, seq_len, token_dim = data.shape

            # トークンをフラット化
            flattened_data = data.reshape(batch_size, seq_len * token_dim)

            # 入力は最後のトークンを除いたシーケンス
            src = flattened_data[:, :-1].long()
            # ターゲットは最初のトークンを除いたシーケンス（次のトークンを予測）
            tgt = flattened_data[:, 1:].long()

            # マスクを作成
            src_mask = create_masks(src.size(1), device)

            # モデル予測
            output = model(src, src_mask=src_mask)

            # ロス計算
            loss = criterion(output.reshape(-1, output.size(-1)), tgt.reshape(-1))
            total_loss += loss.item()

    return total_loss / len(dataloader)


def generate_sequence(
    model: nn.Module,
    start_tokens: torch.Tensor,
    max_len: int,
    device: torch.device,
) -> torch.Tensor:
    model.eval()

    with torch.no_grad():
        current_seq = start_tokens.clone().to(device)

        for _ in range(max_len):
            # マスクを作成
            mask = create_masks(current_seq.size(1), device)

            # 次のトークンを予測
            output = model(current_seq, src_mask=mask)
            next_token_logits = output[:, -1, :]

            # 次のトークンを取得
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            # シーケンスに追加
            current_seq = torch.cat([current_seq, next_token], dim=1)

    return current_seq


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # データセット読み込み
    train_dataset = Wayve101TokensDataset(
        args.data_dir,
        scene_low=1,
        scene_high=90,
        seq_len=args.seq_len,
    )
    valid_dataset = Wayve101TokensDataset(
        args.data_dir,
        scene_low=91,
        scene_high=101,
        seq_len=args.seq_len,
    )

    # データローダー作成
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size)

    # トークンの次元を取得
    sample = train_dataset[0]
    print(f"{sample.shape=}")
    _, token_dim = sample.shape

    # モデルの初期化
    model = TransformerModel(
        d_model=args.d_model,
        nhead=8,
        num_encoder_layers=3,
    ).to(device)

    # 最適化器とロス関数
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # 保存ディレクトリ作成
    args.save_dir.mkdir(exist_ok=True, parents=True)

    # 学習ループ
    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")

        # 学習
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

        # 検証
        val_loss = validate(model, valid_loader, criterion, device)

        print(f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # モデル保存（検証ロスが改善した場合）
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                },
                args.save_dir / "transformer_best.pt",
            )

        # 定期的に保存
        if (epoch + 1) % 5 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                },
                args.save_dir / f"transformer_epoch_{epoch + 1}.pt",
            )
