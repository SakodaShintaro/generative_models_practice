# Video Generation — 系列モデルのテストベッド

過去フレームを固定長の状態に圧縮し、そこから行動系列で条件づけた未来フレームを
拡散モデル (Flow Matching) で予測する world model。**状態への圧縮に使う系列モデルを
差し替えられる**ようにして、「周囲の状況をどこまで記憶できるか」を比較するのが目的。

## パイプライン

```text
context frames (T,3,H,W)
  └─ 凍結 TAESD VAE encode ─▶ latents (T,4,H/8,W/8)
       └─ STEncoder (Spatial=Attention, Temporal=系列モデル) ─▶ state (S,D)  ← 最終タイムステップ
state + future actions (N,A)
  └─ FlowPredictor (Flow Matching DiT) ─▶ future latents (N,4,H/8,W/8)
       └─ 凍結 TAESD VAE decode ─▶ future frames (N,3,H,W)
```

- **Spatial** はフレーム内の Attention、**Temporal** は下記4種を切り替え。
- 状態は最終タイムステップの `S` トークン（`T` に依存しない固定長）。
- VAE (`madebyollin/taesd`) は encode/decode とも凍結。入出力は `[0,1]`。

## 切り替えられる系列モデル (`--temporal`)

| 名前 | 内容 |
|------|------|
| `attention` | Causal multi-head self-attention（全履歴を保持） |
| `gru` | GRU |
| `gated_deltanet` | Gated DeltaNet（delta rule の線形アテンション型 SSM） |
| `ttt` | Test-Time Training。2層 MLP を fast weight とし、RoboTTT (arXiv:2607.15275) の式1–3に準拠 |

TTT は RoboTTT に揃えてある: fast model = 2層MLP、更新則 `W_t = W_{t-1} - η∇‖f_W(K_t)-V_t‖²`
（式1）→ apply `O_t = f_{W_t}(Q_t)`（式2）、**学習可能な内側学習率 η**・**メタ学習される初期値 W₀**・
**tanh ゲート**（式3, α≈0.001 初期化）。内側勾配は閉形式で計算し、外側最適化が
fast-weight 更新を貫いて逆伝播する（勾配の勾配）。

学習は RoboTTT の **sequence forcing**（フレームごとに独立なノイズ準位 τ=0.999·(1−u),
u∼Beta(1.5,1)）を採用。

## データ

Bench2Drive (`/media/sakoda/samsung_4t/bench2drive`) の `camera/rgb_front` と `anno`。
行動は `[throttle, steer, brake, speed/10]`。`frame_stride` で 10Hz を間引いて動きを出す。

## 使い方

`train.py` は **4手法を1エポックずつラウンドロビン**で学習する（同時に4モデルを保持）:

```text
ttt e1 → gated_deltanet e1 → attention e1 → gru e1 → ttt e2 → ...
```

各エポックの終わりに、その手法の固定名チェックポイント（`latest.pt`、毎回上書き）と
可視化 PNG を出力する。

```bash
# 学習（4手法をまとめて進める）
uv run python train.py --results_dir results

# サンプリング（clip=1ショット / rollout=自己回帰でN步超え、記憶ストレステスト）
uv run python sample.py --ckpt results/ttt/checkpoints/latest.pt --mode clip   --temporal ttt --use_ema
uv run python sample.py --ckpt results/ttt/checkpoints/latest.pt --mode rollout --temporal ttt --use_ema
```

`sample.py` はアーキテクチャ引数（`--hidden_size` 等）と `--temporal` を学習時と一致させること。

## 出力

手法ごとに `results/<temporal>/` 以下へ:

- `samples/epoch_XXXX.png` — 各サンプル2行（上=正解 context+future、下=予測 context+future）。
- `metrics.csv` — epoch ごとの loss。
- `checkpoints/latest.pt` — 毎エポック上書き（`model` / `ema` / `args` / `temporal` / `epoch`）。

## ファイル

| ファイル | 役割 |
|----------|------|
| `sequence_models.py` | 4種の temporal mixer と factory |
| `models.py` | STEncoder / FlowPredictor / WorldModel |
| `dataset.py` | Bench2Drive クリップデータセット |
| `common.py` | VAE・モデル構築・Flow Matching サンプラ |
| `train.py` | 学習ループ |
| `sample.py` | 推論・可視化（clip / rollout） |
