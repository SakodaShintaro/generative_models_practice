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
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--frame_len", type=int, default=8)
    parser.add_argument("--d_model", type=int, default=256)
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
        d_model: int,
        nhead: int = 8,
        num_encoder_layers: int = 6,
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
            dim_feedforward=d_model * 4,
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


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
) -> float:
    model.train()
    device = next(model.parameters()).device
    total_loss = 0

    for batch_idx, data in enumerate(tqdm(dataloader, desc="Training")):
        # [batch_size, seq_len].
        data = data.to(device)

        src = data[:, :-1].long()
        tgt = data[:, 1:].long()

        optimizer.zero_grad()
        output = model(src)

        loss = criterion(output.reshape(-1, output.size(-1)), tgt.reshape(-1))
        loss.backward()
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
) -> float:
    model.eval()
    device = next(model.parameters()).device
    total_loss = 0

    with torch.no_grad():
        for data in tqdm(dataloader, desc="Validation"):
            # [batch_size, seq_len].
            data = data.to(device)

            src = data[:, :-1].long()
            tgt = data[:, 1:].long()

            output = model(src)

            loss = criterion(output.reshape(-1, output.size(-1)), tgt.reshape(-1))
            total_loss += loss.item()

    return total_loss / len(dataloader)


def generate_sequence(
    model: nn.Module,
    start_tokens: torch.Tensor,
    generate_len: int,
) -> torch.Tensor:
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        current_seq = start_tokens.clone().to(device)
        for _ in range(generate_len):
            output = model(current_seq)
            next_token_logits = output[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            current_seq = torch.cat([current_seq, next_token], dim=1)
    return current_seq


if __name__ == "__main__":
    args = parse_args()

    # GPUの設定
    assert torch.cuda.is_available(), "GPU is not available"
    use_data_parallel = True
    if use_data_parallel:
        available_gpu_count = torch.cuda.device_count()
        gpu_ids = list(range(available_gpu_count))
        device = torch.device(f"cuda:{gpu_ids[0]}")
        print(f"Using DataParallel on GPUs: {gpu_ids}")
    else:
        device = torch.device("cuda")
        print(f"Using single GPU: {torch.cuda.get_device_name(0)}")

    # データセット読み込み
    train_dataset = Wayve101TokensDataset(
        args.data_dir,
        scene_low=1,
        scene_high=90,
        frame_len=args.frame_len,
    )
    valid_dataset = Wayve101TokensDataset(
        args.data_dir,
        scene_low=91,
        scene_high=101,
        frame_len=args.frame_len,
    )

    # データローダー作成
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size)

    # モデルの初期化
    model = TransformerModel(
        d_model=args.d_model,
        nhead=8,
        num_encoder_layers=3,
    ).to(device)

    # DataParallelの適用
    if use_data_parallel:
        gpu_ids = list(range(torch.cuda.device_count()))
        model = nn.DataParallel(model, device_ids=gpu_ids)
        print(f"Model wrapped with DataParallel on {len(gpu_ids)} GPUs")

        # DataParallelを使用する場合、バッチサイズを調整することで効率が向上
        print(f"Effective batch size: {args.batch_size * len(gpu_ids)}")

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
        train_loss = train_epoch(model, train_loader, optimizer, criterion)

        # 検証
        val_loss = validate(model, valid_loader, criterion)

        print(f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # モデル保存（検証ロスが改善した場合）
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # DataParallelを使用している場合は .module にアクセスしてモデルを保存
            model_to_save = model.module if hasattr(model, "module") else model
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model_to_save.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                },
                args.save_dir / "transformer_best.pt",
            )

        # 定期的に保存
        if (epoch + 1) % 5 == 0:
            # DataParallelを使用している場合は .module にアクセスしてモデルを保存
            model_to_save = model.module if hasattr(model, "module") else model
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model_to_save.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                },
                args.save_dir / f"transformer_epoch_{epoch + 1}.pt",
            )
