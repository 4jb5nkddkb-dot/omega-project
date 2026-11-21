# MAPPING — 新布局审计 trail（2026-11-21）

每个文件的：新路径、来源（旧路径或 "new"）、改了什么。
"Agent D" 是本次任务的作者；其他 Agent 的文件只按其头注释/导出表
简述，未逐行审计（见各节末尾说明）。

## 根目录

| 新路径 | 来源 | 改动 |
| --- | --- | --- |
| `CONTRACT.md` | new（协调员） | 重构合同；最终交付前删除，不进 shipped tree |
| `README.md` | new（Agent D） | 项目总览、门控双半脑一句话、安装/用法、实话版状态 |
| `MAPPING.md` | new（Agent D） | 本文件 |
| `.gitignore` | 旧 staging `.gitignore` 改 | 增加 `*.pt`、`checkpoints/*.pt`、`.venv/`；注明小测试 fixture 可 `git add -f` 例外、data/raw 已跟踪 |
| `requirements.txt` | 旧 staging `requirements.txt` 改 | 增加 `torch==2.15.0+cpu`（2026-11-21 在 `.venv` 实装验证）、`pyyaml`；保留 `numpy==1.26.4` 约束 |

## configs/（重构前已在 staging，Agent D 只写了测试断言其值）

| 新路径 | 来源 | 改动 |
| --- | --- | --- |
| `configs/model_gated.yaml` | staging 已有 | 未动。d1024 门控 v5：值与 GATED_V5_S300.log 一致（test_configs 断言） |
| `configs/model_small.yaml` | staging 已有 | 未动。d512 single v2，13,916,328 参数 |
| `configs/train_default.yaml` | staging 已有 | 未动。NumPy 版训练配置（300 步一块）；`checkpoint_pattern` 仍是 `.npz`，scripts/train.py 运行时把后缀换成 `.pt` |
| `configs/train_pytorch.yaml` | new（非 Agent D，见头注释） | 从 train_default.yaml 派生的 PyTorch 版训练配置，同值 |

## tokenizer/（重构前已在 staging，Agent D 只验证）

| 新路径 | 来源 | 改动 |
| --- | --- | --- |
| `tokenizer/__init__.py` | staging 已有 | 未动。导出 CharTokenizer / ByteTokenizer / tokenizer_from_meta |
| `tokenizer/char_tokenizer.py` | `~/workspace/omega-chat/dual_transformer_numpy.py` 第 75–161 行抽出 | 未动。纯 NumPy/标准库，无 torch 依赖；BOS/EOS/UNK 0/1/2，词表 4622 |
| `tokenizer/build_vocab.py` | `~/workspace/dual_sft/build_generalist_vocab.py` 改编 | 非 Agent D。逻辑一致，改走本仓库 tokenizer |

## model/（Agent A；Agent D 未动，仅在测试/脚本里按 CONTRACT 引用）

| 新路径 | 来源 | 改动 |
| --- | --- | --- |
| `model/__init__.py` | new（Agent A） | 重导出 DualTransformer、RMSNorm、MultiHeadAttention、SwiGLU、GELUFeedForward、FeedForward、TransformerBlock、checkpoint 四函数 |
| `model/transformer.py` | new（Agent A，单体 `dual_transformer_numpy.py` 的 DualTransformerNP 移植） | PyTorch DualTransformer：forward / forward_cached / gate_values / config()；state_dict 键与 NumPy 参数名 1:1。**缺 `gate_logits()`**：Agent B 的 trainer（`training/trainer.py:205-211`）已用 hasattr 分支处理（有则走 BCEWithLogits 稳定路径，无则 clamp 概率反推）；Agent A 补上 `gate_logits(tokens) -> (B, S, 1)`（PRE-sigmoid z）后训练器自动切稳定路径，`gate_values` 保持 sigmoid 输出不变 |
| `model/checkpoint.py` | new（Agent A） | save/load/read_checkpoint_meta/last_checkpoint_path，`.pt` dict 格式与 CONTRACT 一致 |
| `model/norm.py` | new（Agent A） | RMSNorm，对应单体 `_norm_fwd` 的 use_rms 分支 |
| `model/attention.py` | new（Agent A） | MultiHeadAttention：RoPE + GQA + 因果掩码 + KV cache，融合 QKV 投影 |
| `model/block.py` | new（Agent A） | TransformerBlock |
| `model/feedforward.py` | new（Agent A） | SwiGLU / GELUFeedForward / FeedForward，ff_dim = round(8D/3) |

## training/（Agent B；Agent D 未动，仅在脚本/测试里引用其 API）

| 新路径 | 来源 | 改动 |
| --- | --- | --- |
| `training/__init__.py` | new（Agent B） | 重导出 dataset / loss / optimizer / trainer / distributed 的公开 API |
| `training/dataset.py` | staging 已有的 NumPy 参考实现 → Agent B 移植 | 保留 `domain_texts` / `make_batch_fn` / `live_corpus_paths` / `RATIOS` / `GATE_*`，新增 torch `GeneralistDataset` / `StratifiedBatchSampler` / `get_dataloader`（每批返回 `(inputs, targets, gate_targets)` 张量） |
| `training/loss.py` | new（Agent B） | `cross_entropy_loss`（F.cross_entropy，label_smoothing）、`gate_bce_loss`（BCEWithLogits，吃 PRE-sigmoid logit）、`masked_loss`。注意：`gate_bce_loss` 内部把输入转 float32，float64 gradcheck 不能走它（tests/test_gradcheck.py 直接调 F.binary_cross_entropy_with_logits） |
| `training/optimizer.py` | new（Agent B） | `build_optimizer`（AdamW，两组：ndim>=2 wd=0.01，ndim<2 wd=0）+ `lr_at`（与 NumPy 同式的 warmup+cosine，trainer 每步手动设 lr） |
| `training/trainer.py` | new（Agent B） | `train(model, *, batch_fn/dataloader, steps, total_steps, ...)`；断点只 torch.load 一次、resume 省略 steps 自动算剩余、config-mismatch 校验、save `_last`。含自带 `__main__` CLI |
| `training/distributed.py` | new（Agent B） | `maybe_wrap_ddp`，单机先行，DDP 只作包装 |

## inference/（Agent C；Agent D 未动，仅在脚本里引用）

| 新路径 | 来源 | 改动 |
| --- | --- | --- |
| `inference/__init__.py` | new（Agent C） | 重导出 generate / generate_with_gate / DecodeController 等 |
| `inference/generate.py` | new（Agent C，单体 `_filter_probs` / `_pick_token` / `DecodeController` 移植） | `generate(model, tokenizer, prompt, ...)` 返回续写文本（不回显提示）；`generate_ids` 返回完整 id；`generate_with_gate` 返回 `(text, gate)`，gate 为提示的每序列左支占比 |
| `inference/server.py` | new（Agent C） | stdlib http.server 最小服务：`create_app` / `run_server`，无 FastAPI |

## evaluation/（Agent C；world_state.py 为 staging 已有）

| 新路径 | 来源 | 改动 |
| --- | --- | --- |
| `evaluation/__init__.py` | new（Agent C） | 导出 scoring 的 QUESTIONS / score |
| `evaluation/README.md` | new（Agent C） | 各探针测什么、怎么跑、swap rule |
| `evaluation/_common.py` | new（Agent C） | 从 `.pt` 检查点加载模型 + 词表（只 load 一次） |
| `evaluation/world_state.py` | staging 已有 | 未动。冻结的世界状态机副本（探针自带，不 import 产品代码） |
| `evaluation/scoring.py` | `~/workspace/dual_sft/world_self_probe.py` 移植 | Q0..Q3 问题集 + 严格字符串评分，判卷逻辑与原版一致 |
| `evaluation/sidecar.py` | `~/workspace/dual_sft/omega_v010/omega_kernel_bridge.py` 移植 | PyTorch Sidecar：generate / ask_question / narrate_arrival / talk / execute + 在线记忆缓存；未移植 Rust 桥接器的文件协议（observation/action.json 轮询） |
| `evaluation/causal_probe.py` | `~/workspace/dual_sft/causal_probe.py` 移植 | 因果链探针：强制世界移动，测模型报新地点 |
| `evaluation/loop.py` | 旧配置 12 步循环移植 | 冻结 12 步 loop 验收（swap rule 的量尺） |
| `evaluation/gate_probe.py` | `~/workspace/dual_sft/probe_gate_v5_pilot_lean.py` + `probe_gate_loop_prompts.py` 移植 | 门分流探针：每域 mean g + 真实循环提示走哪支 |
| `evaluation/q_probe.py` | `~/workspace/dual_sft/diag_q0_s600.py` 移植 | Q0/Q1/Q2/Q3 探针集 |

## data/（Agent D）

| 新路径 | 来源 | 改动 |
| --- | --- | --- |
| `data/README.md` | new（Agent D） | 布局、来源/许可、重建方法 |
| `data/raw/Xiyouji.txt` | `~/workspace/omega-chat/Xiyouji.txt` 复制 | 未动。677,421 字符；公有领域 |
| `data/raw/classics_rulin.txt` | `~/workspace/dual_sft/generalist_corpus/classics_rulin.txt` 复制 | 未动。241,580 字符；**注意是加工本**（Gutenberg #24032 繁体本经 OpenCC t2s 转简体、按词表过滤），原文见旧仓库 `rulinwaishi_raw.txt` |
| `data/processed/generalist_vocab.json` | `~/workspace/dual_sft/generalist_vocab.json` 复制 | 未动。4619 字 + 3 特殊符 = 4622 |
| `data/processed/dialogue_pilot.txt` | `generalist_corpus/dialogue_pilot.txt` 复制 | 未动。2,918 字符 |
| `data/processed/knowledge_wiki.txt` | `generalist_corpus/knowledge_wiki.txt` 复制 | 未动。73,905 字符；CC BY-SA 4.0 |
| `data/processed/knowledge_youxue.txt` | `generalist_corpus/knowledge_youxue.txt` 复制 | 未动。888 字符；Gutenberg #52269 公版 |
| `data/processed/world_tool_records_v9.jsonl` | `~/workspace/dual_sft/world_tool_records_v9.jsonl` 复制 | 未动。1,763 条 |
| `data/processed/self_report_records_v9.jsonl` | `~/workspace/dual_sft/self_report_records_v9.jsonl` 复制 | 未动。1,200 条 |
| `data/processed/probe_format_records.jsonl` | `~/workspace/dual_sft/probe_format_records.jsonl` 复制 | 未动。722 条（dataset.py docstring 的旧数 120 条已过期，以 MANIFEST 为准） |
| `data/processed/register_format_records.jsonl` | 同上复制 | 未动。220 条 |
| `data/processed/carried_register_format_records.jsonl` | 同上复制 | 未动。181 条 |
| `data/processed/daytime_register_format_records.jsonl` | 同上复制 | 未动。171 条 |
| `data/processed/character_register_format_records.jsonl` | 同上复制 | 未动。150 条 |
| `data/processed/video_making_records.jsonl` | 同上复制 | 未动。52 条 |
| `data/processed/MANIFEST.md` | new（Agent D） | 实测计数：世界域合计 4,459 条 / 434,728 字符 / 434,728 token，0 词表外字符；各文件 sha256（前 16 位） |
| `data/scripts/README.md` | new（Agent D） | 脚本→产出对照 + 路径警告 |
| `data/scripts/build_*.py`（共 24 个） | `~/workspace/dual_sft/build_*.py` 复制 | **逻辑未改，路径未改**：内部仍引用 `~/workspace/omega-chat` / `~/workspace/dual_sft` 与 NumPy 单体分词器；重跑前须把路径指到本布局（见 data/scripts/README.md）。其中 `build_minimal_pairs.py` 产出 4 个最小对/寄存器文件；人物寄存器文件的生成脚本不在本目录 |
| `data/scripts/fetch_wiki_knowledge.py` 等 3 个 | `~/workspace/dual_sft/fetch_*.py` 复制 | 未改。维基导言抓取（首批/续抓/留验证集） |
| `data/scripts/filter_knowledge.py` | `~/workspace/dual_sft/filter_knowledge.py` 复制 | 未改。维基导言按词表过滤成句 → knowledge_wiki.txt |
| `data/scripts/convert_rulin.py` | `~/workspace/dual_sft/convert_rulin.py` 复制 | 未改。Gutenberg 繁体本 → 转简体 → 按词表过滤 |

全部 14 个数据复制文件与源文件逐字节比对一致（`cmp` 全过，2026-11-21）。
`.npz` 检查点与 `.log` 日志**一律未复制**。

## scripts/（Agent D）

| 新路径 | 来源 | 改动 |
| --- | --- | --- |
| `scripts/train.py` | new（Agent D） | 薄 CLI：读扁平 YAML（PyYAML 优先，缺时内置简易读取器）→ 建 DualTransformer → tokenizer + `domain_texts`/`make_batch_fn` → 调 Agent B 的 `train(model, batch_fn=..., steps, checkpoint_path, resume_from, init_from, extra_meta, gated, ...)`。`--resume` 省略 `--steps` 的规则与警告在内；`--out` 缺省用配置的 `checkpoint_pattern`（`.npz` 后缀自动换 `.pt`）。`--help` 已验证可跑 |
| `scripts/generate.py` | new（Agent D） | 薄 CLI：`load_checkpoint` → DualTransformer → Agent C 的 `generate` / `generate_with_gate`（传 tokenizer + 提示字符串）。待 Agent A 的 checkpoint/transformer 落地后可用（写本文时已落地，见下） |
| `scripts/convert_npz_to_pt.py` | new（Agent D） | 自包含转换器（numpy + torch，不 import 项目模块）：`read_npz`（p_/m_/v_/meta_json 映射）+ `to_pt_dict` + `checksum`（全量 sha256）+ `verify` + CLI `--verify` / `--no-optimizer`。含目标磁盘余量预检（/tmp 512 MB tmpfs 装不下完整 1 GB `.pt`，见“验证”节）。已在真实 s600 上验证通过 |

## tests/（Agent D）

| 新路径 | 来源 | 改动 |
| --- | --- | --- |
| `tests/conftest.py` | new（Agent D） | sys.path bootstrap：项目根目录可 import |
| `tests/test_tokenizer.py` | new（Agent D） | 7 项：词表 4622、BOS/EOS/UNK 0/1/2、中文+代码往返、BOS/EOS 标志、UNK、tokenizer_from_meta。**已跑：全过** |
| `tests/test_configs.py` | new（Agent D） | 3 项：3 份 YAML 的值与真实运行一致（d1024 门控 / d512 / 训练默认）。**已跑：全过** |
| `tests/test_checkpoint_conversion.py` | new（Agent D） | 微型 fixture：`read_npz` 映射（无 torch 可测）+ 完整 `.pt` 往返（需 torch）。**已跑：全过**（含 torch 部分） |
| `tests/test_dataset.py` | new（Agent D） | live 语料加载 + 分层采样：batch 形状、门目标 7 左/1 右、targets 右移。**未跑**：PAUSE（2026-11-21 04:26，chunk 3 训练中）——全语料编码虽轻，但按暂停令暂缓，等协调员通知后再跑。torch 版 GeneralistDataset 断言待 Agent B 移植后追加 |
| `tests/test_parity.py` | new（Agent D） | 真实 s600：PyTorch vs NumPy logits，max abs diff 上限 1e-3，打印实测值。**未跑**：PAUSE——torch 模型加载暂缓；前置 `.pt` 已就绪（/tmp 模型版 + ~/workspace/convert-scratch 完整版） |
| `tests/test_gradcheck.py` | new（Agent D） | 微型门控模型 float64 gradcheck：前向 + CE+门 BCE 组合损失（经 `_LossWrapper` + functional_call，使门参数也被覆盖；优先用 `gate_logits`，无则 clamp 后备）。**未跑**：PAUSE——torch 模型加载暂缓 |
| `tests/test_smoke_train.py` | new（Agent D） | 3 步 CPU 冒烟：损失有限、参数变化。**未跑**：PAUSE——torch 模型加载暂缓 |

## checkpoints/

| 新路径 | 来源 | 改动 |
| --- | --- | --- |
| `checkpoints/README.md` | new（Agent D） | 命名、Drive 文件 ID（s5100 / s6000）、swap rule、转换方法 |
| `checkpoints/gated_v5_pilot_s3.pt` | 非 Agent D（2026-11-21 09:24 出现，约 342 MB） | 疑似 Agent A/协调员用转换器转的试点检查点；本文件不在 Agent D 任务内，仅记录其存在；`*.pt` 已 gitignore，不进仓库 |

## 未进本布局的东西（有意排除）

- `.npz` 检查点、`.log` 训练日志：不复制，见 `checkpoints/README.md`（Drive ID + 转换方法）。
- Omega 产品侧（Rust 内核、omega_v010 GUI/桥接器、File Seal 等）：不在模型包内，留在旧仓库；探针只用了其行为的最小移植（`evaluation/sidecar.py`、`world_state.py`）。
- 旧仓库的 DPO 数据与脚本：DPO 轮次已关闭，只复制了生成脚本（`build_dpo_*.py`），数据文件未进 `processed/`。

## 验证状态（2026-11-21，PAUSE 生效中）

- 已验证：转换器在真实 s600 上通过（74 参数 / 85,542,235 标量 / 校验和一致，见 checkpoints/README.md 与最终报告）；轻量测试 12 项全过（tokenizer 6 + configs 3 + conversion 3）；两 CLI `--help` 正常；全部源码 `py_compile` 通过。
- 暂缓（PAUSE 2026-11-21 04:26，chunk 3 训练优先，训练器已从 step 220 续跑）：`test_dataset.py`、`test_parity.py`、`test_gradcheck.py`、`test_smoke_train.py` 的运行；任何检查点转换重跑；torch 模型加载；pip 安装。协调员通知训练结束后再跑。
- 待 Agent A：`model/transformer.py` 补 `gate_logits(tokens) -> (B, S, 1)`（PRE-sigmoid z；`gate_values` 保持 sigmoid 输出）。trainer 与 gradcheck 测试已写好 hasattr 分支，有则自动走稳定路径。
