# llm-project — 百事通（generalist）PyTorch 重写

把原来纯 NumPy 的通用小模型（“百事通”：什么都懂一点、什么都敢试，
broad-but-shallow 是设计目标，不是失败）改写成标准 PyTorch。
**门控双半脑设计不变**，训练权重不变——旧 `.npz` 检查点 1:1 转成 `.pt` 继续用。

## 门控双半脑（一句话版）

一个共享 stem（1 块）先读输入，取隐状态均值池化过一个 sigmoid 门
`g = sigmoid(w·mean_pool(h_stem) + b)`，每序列一个 g；输入按 g/(1-g)
分流、输出按同一个 g 混合：左支吃经典/代码/对话/常识/推理，
右支吃世界域，BCE 监督（左 t=1，右 t=0，权重 1.0）。
d1024 / 4 块 / 16 头 / 8 KV 头， tied embedding，RoPE，GQA，SwiGLU，
pre-norm，dropout 0.1，label smoothing 0.1，seq 128，batch 8，
LR 1e-3 cosine，配比 v2（经典 .25 / 代码 .20 / 对话 .10 / 常识 .15 /
推理 .15 / 世界 .15），字符级词表 4622。
门在 s600 上分得开（左 1.0000 / 世界 0.2500，分离度 0.75），
但 loop 行为还没跟上（3/12），所以产品仍是 d512 的 s5100。
详见 `checkpoints/README.md` 的 swap rule 与 `evaluation/README.md`。

## NumPy → PyTorch 与检查点转换

- 模型、训练、推理、评估全部用 PyTorch；NumPy 只留在数据准备/分词
  （`tokenizer/`、`data/scripts/`），不进模型。
- 旧权重靠 `scripts/convert_npz_to_pt.py` 活下来：自包含
  （numpy + torch，不 import 项目代码），`p_<name>` → `model[<name>]`，
  `m_/v_` → Adam 状态，`meta_json` → config/step/extra。
  转完 `--verify` 核对参数个数与全量 sha256 校验和，
  `tests/test_parity.py` 再比 PyTorch/NumPy 的 logits。
- 续跑教训已写进新代码：检查点只 `torch.load` 一次（NumPy 时代
  读 3 次曾把内存顶到 5.3 GB 撞上 6 GB 上限 OOM）；
  门 BCE 用 `BCEWithLogitsLoss`（NumPy 的 float32 sigmoid 饱和到
  恰好 1.0 导致 log(0) → NaN 的问题在新实现里不存在）。

## 安装

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

- 默认 CPU 版 torch（`requirements.txt` 里 `torch` 不 pin 版本——
  2026-11-21 写本文时协调员正在往 `.venv` 里装 torch，尚未装好，
  故未 pin 已验证的版本号；装好后请把实际版本 pin 回去）。
- CUDA：`pip install torch --index-url https://download.pytorch.org/whl/cu121`
  （按本机 CUDA 版本换 cu121/cu124 等）。
- 另需 `numpy==1.26.4`（numpy 2.x 未经本代码库验证）、`pyyaml`。

## 用法

```bash
# 训练（默认 generalist-v5 门控，300 步一块）
python scripts/train.py --config configs/train_default.yaml
# 续跑：--resume 接 _last.pt，且省略 --steps（由 trainer 自动算剩余步数）
python scripts/train.py --config configs/train_default.yaml \
    --resume checkpoints/generalist3_d1024_s600_last.pt \
    --d_model 1024 --n_blocks 4 --n_heads 16 --n_kv_heads 8
# 生成
python scripts/generate.py --checkpoint /tmp/generalist3_d1024_s600.pt \
    --prompt "孙悟空来到" --max-new-tokens 64
# 转换旧检查点（大文件放 /tmp，别进仓库）
python scripts/convert_npz_to_pt.py \
    ~/workspace/dual_sft/out/generalist3_d1024_s600.npz \
    /tmp/generalist3_d1024_s600.pt --verify
# 测试
python -m pytest tests/ -x -q
```

## 仓库布局

```
configs/      模型与训练配置（扁平 YAML：model_gated / model_small / train_default）
tokenizer/    字符级分词器（已完成，纯 NumPy，无 torch 依赖）
model/        PyTorch 模型：DualTransformer + checkpoint 读写（Agent A）
training/     数据集 / 损失 / 优化器 / 训练循环 / DDP（Agent B）
inference/    生成 + 极简 http.server（Agent C）
evaluation/   探针与 loop 验收（Agent C）
data/         raw/ 原始语料、processed/ 训练用文件 + MANIFEST.md、scripts/ 语料生成脚本
scripts/      train.py / generate.py / convert_npz_to_pt.py（本 Agent D）
tests/        pytest 套件（见下）
checkpoints/  空目录 + README（检查点不进仓库，见 checkpoints/README.md）
```

## 状态（实话版，2026-11-21）

已验证：

- 数据：14 个文件从旧仓库复制，逐字节比对一致；计数见
  `data/processed/MANIFEST.md`（世界域 4,459 条、434,728 字符，0 词表外字符）。
- 分词器：4622 词表往返测试通过（`tests/test_tokenizer.py`）。
- 配置：3 份 YAML 的值与真实运行（GATED_V5_S300.log、pilot 默认值）一致。
- 转换器：在真实 s600 检查点上验证通过——74 个参数、85,542,235 标量、
  step=300（当块步数）、优化器 t=300，转换前后全量 sha256 校验和一致
  （`06c67809b27a4b87…`）；完整版 `.pt` 在 `~/workspace/convert-scratch/`，
  模型版在 `/tmp`（见 `checkpoints/README.md`）。
- 轻量测试 12 项全过（tokenizer 6 + configs 3 + conversion 3，
  `.venv/bin/python -m pytest tests/test_tokenizer.py tests/test_configs.py tests/test_checkpoint_conversion.py`）。
  torch 已装好：`.venv` 里是 `torch==2.15.0+cpu`（已 pin 进 requirements.txt），
  另装了 pytest 9.2.0 + pyyaml 供测试用。
- 两 CLI `--help` 正常；Agent D 全部源码 `py_compile` 通过。
- model / training / inference / evaluation 各包已由 Agent A/B/C 落地
  （见 `MAPPING.md`）；`scripts/train.py` 已按 Agent B 的真实 `train()`
  签名接线，`scripts/generate.py` 已按 Agent C 的真实 `generate()` 签名接线。

暂缓 / 待集成（不拦着写代码，但别当成已完成）：

- **PAUSE（2026-11-21 04:26，chunk 3 训练优先，训练器已从 step 220 续跑）**：
  `tests/test_dataset.py`、`tests/test_parity.py`、`tests/test_gradcheck.py`、
  `tests/test_smoke_train.py` 的运行全部暂缓，任何检查点转换重跑、torch
  模型加载、pip 安装也都停了。等协调员通知训练结束后再跑。
- 待 Agent A：`model/transformer.py` 补 `gate_logits(tokens) -> (B, S, 1)`
  （PRE-sigmoid 门 logit；`gate_values` 保持 sigmoid 输出）。
  训练器与 gradcheck 测试已写好 hasattr 分支，有则自动走 BCEWithLogits 稳定路径。
- `data/scripts/` 的脚本仍指旧路径，重跑前需改路径（见其 README）。
