import argparse
from pathlib import Path

import torch
import wandb
from dataset import Wayve101TokensDataset
from models.config import ModelArgs
from models.llama_transformer import LlamaTransformer
from models.vanilla_transformer import VanillaTransformerModel
from torch import nn
from torch.utils.data import DataLoader

VOCAB_SIZE = 64_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--frame_len", type=int, default=8)
    parser.add_argument("--save_dir", type=Path, default=Path("./checkpoints"))
    return parser.parse_args()


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
) -> float:
    model.train()
    device = next(model.parameters()).device
    total_loss = 0
    total_num = 0

    for batch_idx, data in enumerate(dataloader):
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
        total_num += len(data)

        progress = 100.0 * batch_idx / len(dataloader)
        print(
            f"{total_num=:06d}/{len(dataloader.dataset)}({progress:5.1f}%), {loss.item()=:.4f}",
            end="\r",
        )
        wandb.log({"train_loss": loss.item()})

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
        for data in dataloader:
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
    wandb.init(project="Wayve101_Sequential_Modeling", config=vars(args))

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
    params = ModelArgs(
        dim=256,
        n_layers=3,
        n_heads=8,
        vocab_size=VOCAB_SIZE,
        max_seq_length=args.frame_len * 128,  # multiply `num_tokens_per_frame`
    )
    model_name = "llama"
    match model_name:
        case "llama":
            model = LlamaTransformer(params).to(device)
        case "vanilla":
            model = VanillaTransformerModel(params).to(device)

    # DataParallelの適用
    if use_data_parallel:
        gpu_ids = list(range(torch.cuda.device_count()))
        model = nn.DataParallel(model, device_ids=gpu_ids)
        print(f"Model wrapped with DataParallel on {len(gpu_ids)} GPUs")

    # 最適化器とロス関数
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # 保存ディレクトリ作成
    args.save_dir.mkdir(exist_ok=True, parents=True)

    # 学習ループ
    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        valid_loss = validate(model, valid_loader, criterion)

        wandb.log({"epoch": epoch, "valid_loss": valid_loss})
        print(f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f}, Val Loss: {valid_loss:.4f}")

        # モデル保存（検証ロスが改善した場合）
        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            # DataParallelを使用している場合は .module にアクセスしてモデルを保存
            model_to_save = model.module if hasattr(model, "module") else model
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model_to_save.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "valid_loss": valid_loss,
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
                    "valid_loss": valid_loss,
                },
                args.save_dir / f"transformer_epoch_{epoch + 1}.pt",
            )
