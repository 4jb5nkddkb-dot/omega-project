# Chunk 5 训练计划：s1200 → s1500（先读后答）

> 方法见 `Q0_TEACHING_DESIGN.md`（先读后答：答案区 loss + 抽取演练 + 支架问答）。
> 本计划只覆盖下一块的教学设计与执行，不启动训练，不改动 s5100 产品版。

---

## 0. 前置：先测 s1200 基线（不测就分不清是第 5 块的功劳还是第 4 块的）

s1200 的 Q0 **尚未测过**。第 4 块（s900→s1200，PyTorch，Kaggle T4，300 步，终 loss 3.8010）没针对 Q0，Q0 在 s1200 上大概率仍是 0/7 附近，但必须以实测为准——否则第 5 块跑完无法归因。

1. 从 Kaggle Notebook Version 3 输出取回 `generalist3_d1024_s1200.pt`（scriptVersionId=372388932，Version URL：https://www.kaggle.com/code/zhanyingjia/omega-project-gated-d1024-t4-benchmark?scriptVersionId=372388932）。确认文件约 1.03 GB；同时取回 `_last.pt`（备用续跑点）。
2. 在本地（或 Kaggle）对 s1200 跑三项探针，记下基线：
   - `python -m evaluation.q_probe --checkpoint <s1200.pt> --log DIAG_Q0_S1200.log`
   - `python -m evaluation.causal_probe --checkpoint <s1200.pt> --dir run_causal_s1200`
   - `python -m evaluation.loop --checkpoint <s1200.pt> --out loop_s1200.jsonl`
3. 基线数字写入本计划 §5 的对照表（s900 一列已填，s1200 一列等实测填）。

---

## 1. 起点与目标

| 项 | 内容 |
|---|---|
| 起点 | s1200（Kaggle Version 3，`generalist3_d1024_s1200.pt`，约 1.03 GB，chunk 4 终 loss 3.8010） |
| 目标 | s1500 |
| 步数 | **300 步**（`--steps 300`，`--init` 起新块，不恢复优化器） |
| 模型 | d1024 gated v5，recipe generalist-v5，85,542,235 参数；batch 8，seq 128，lr 1e-3 cosine，seed 7，save-every 10 |
| 运行地 | Kaggle T4（PyTorch），约 0.43 秒/步，300 步训练约 2–3 分钟（加 setup 约 10 分钟） |
| 产物 | `checkpoints/generalist3_d1024_s1500.pt` + `_last.pt` |

**执行命令（Kaggle 上，仓库已克隆、依赖已装的前提下）：**

```bash
python3 scripts/train.py --config configs/train_default.yaml \
  --init <s1200.pt 路径> \
  --out checkpoints/generalist3_d1024_s1500.pt \
  --d_model 1024 --n_blocks 4 --n_heads 16 --n_kv_heads 8
```

（`--init` 是权重初始化，不恢复优化器/步数；`--steps` 默认取配置的 300，不用另传。续跑才用 `--resume <_last.pt>` 且**必须省略 `--steps`**，由程序按剩余步数自动算——见 `configs/train_default.yaml` 的续跑规则。）

**保存铁律（第 4 块第一次丢 s1200 的教训）：** 跑完先用 Save & Run All 存为 Kaggle 新版本（确认输出里有 s1500.pt 和 `_last.pt`，版本状态 Successful），**再停会话**。工作区是临时的，不存版本就清零。

---

## 2. 数据构成（新教法的数据）

### 2.1 代码改动（先做，训练前验证）

对应设计文档 §3 改动 A（答案区 loss）：

- 改 `training/dataset.py`：世界域记录保留答案区边界（用记录里已有的 `answer_start` 字段），batch 附带答案掩码。
- 改 trainer 的 loss：掩码外位置不计 loss（label -100）。非世界域保持全序列 loss。
- **验证**：改完先跑一个 3 步冒烟，确认 loss 只算答案区、训练能正常跑通，再开整块。若冒烟失败或 loss 异常，**先回退到无掩码跑（只上新数据）**，不要带着坏代码开整块。

### 2.2 新记录（共 365 条）

构建脚本：`build_read_then_answer.py`（仿 `~/workspace/dual_sft/build_minimal_pairs.py` 的写法：同格式 jsonl，含 `text`/`summary`/`answer_start`/`state`/`question`/`answer`；过词表检查，0 OOV；写完同步回 `~/workspace/dual_sft/` 活语料与 `llm-project/data/processed/`）。

| 类别 | 数量 | 构成 | 去向 |
|---|---|---|---|
| 抽取演练（drills） | **155** | Q0 70（10 模板 × 7 地点）、Q3 20（10 × 2 昼夜）、Q2 30（5 × 6 随身组合）、Q1 35（5 × 7 地点，人物表随地点变，南海多观音） | `probe_format_records.jsonl`（追加，另立 `read_drills.jsonl` 备查） |
| 支架问答（scaffolded） | **120** | Q0 70（10 × 7）、Q3 20（10 × 2）、Q2 30（5 × 6，随身非空） | 同上 |
| 直接问答（direct） | **90** | Q0 50、Q3 20、Q2 20（新模板，防止只会带支架答） | 同上 |
| **合计新增** | **365** | | |
| 已有最小对子 | 240（保留，不删） | Q0 70、Q2 30、Q3 20，各配寄存器格式（2026-11-20 已入语料） | 已在语料里 |

- 世界域约 439,186 字 → 约 48 万字（新增约 365 条 × 约 110 字）。
- 配比不变（世界域 15%），其他五域（经典/代码/对话/常识/推理）不动。只改世界域内部构成，不动整条线的配比——一次只改一件事。
- 抽取演练的问法与探针问题不同（“地点是哪里。” vs “你在什么地方。”），练通用技能；支架问答的抽取前缀只含真实字段值，不含他值（判分兼容已在设计文档 §3 核对）。

### 2.2b 数据落点（2026-11-23 实测，建数据时补记）

- 365 条已生成，构建脚本 `build_read_then_answer.py`（llm-project 根目录），校验全过：0 OOV、一致性 0 坏、与现有 probe 记录 0 重复、内部 0 重复。
- **落点（新文件，现有语料未动）**：`data/processed/read_drills.jsonl`（365 条）+ 同内容副本 `~/workspace/dual_sft/read_drills.jsonl`。原计划写的是“追加进 probe_format_records.jsonl”，本次按“只新增、不改现有语料文件”的要求改成了独立新文件。
- **接线（开训前必须做，否则训练器读不到这 365 条）**：把 `"read_drills.jsonl"` 加进 `training/dataset.py` 的 `WORLD_RECORD_FILES`（live 语料路径下同名文件随之生效），或把这 365 条并入 `probe_format_records.jsonl`。二选一，做完再开第 6 块。
- **实际字数**：新记录共 27,252 字（平均约 75 字/条，不是预估的 110 字/条）；世界域 439,186 → **466,802 字（约 46.7 万字）**，不是预估的约 48 万字。条数准确（365），字数以实测为准。

### 2.3 数据检查清单（开训前逐项打勾）

- [ ] 新记录 365 条已写入，0 OOV。
- [ ] 世界域字数已复核（约 48 万字）。
- [ ] 答案区掩码 3 步冒烟通过。
- [ ] s1200 基线三项探针已跑完、数字已填入 §5 对照表。

---

## 3. 验收（只认探针与环，不认 loss）

跑完 s1500 后，用与基线**同一套命令**再跑三项探针（`evaluation/q_probe.py`、`causal_probe.py`、`loop.py`）。验收标准：

### 通过线（学会了 → 继续第 6 块）

三条全过，方法成立：

| 测量 | 通过线 | s900 对照 |
|---|---|---|
| Q0 探针（probe 格式） | **≥ 5/7** | 0/7 |
| Q0 探针（register 格式） | **≥ 5/7** | 0/7 |
| 因果探针：Q0 跟上移动 | **≥ 4/7** | 0/7 |
| 12 步环总分 | **≥ 9/12**，且 Q0 ≥ 2/3、Q3 ≥ 2/3 | 6/12（Q0 0/3、Q3 0/3） |

继续的方向：第 6 块同方法，逐步撤支架（提高直接问答比例）、加抽取演练，巩固迁移；目标是环上 12/12 再换产品（换货规则不变，s5100 仍是产品版）。

### 有进展（继续，但调比例）

- Q0 探针 **2–4/7**：固定默认已破，但还没稳。第 6 块同方法，加抽取演练比例、减直接问答，再测。

### 没学会（换办法，停下 redesign）

- Q0 探针 **0–1/7 且仍是固定默认**（7 个地点答同一个）：方法失败。下一假设：瓶颈不在教法，在右支容量、门路由或世界域配比（15%），需另想办法，**不再加同类数据**。
- 或 **loss 明显下降但 Q0 不动**：loss 与 Q0 脱钩，同样按“没学会”处理，不拿 loss 充数。（s900 已经证明过一次：loss 4.03 的同时 Q2 进步了、Q0 没动。）

### 决策点

**s1500 三项探针跑完之时。** 哪条线过、哪条没过，决定继续还是换办法（见上表）。决策只看这三项探针，不看训练 loss、不看门分离度（门在 s600 已是 1.0000/0.2500、分离 0.7500，门不是瓶颈）。

---

## 4. 风险与回退

- **答案区掩码是代码改动**：冒烟不过就回退到无掩码跑（只上新数据），不要带着坏代码开整块（§2.1）。
- **支架格式与桥接器解析**：探针用的 `evaluation/scoring.py` 已核对放行支架格式；旧配置 12 步环的桥接器自有 grounded_err 算法。若 s1500 探针动了但环不动，先核对桥接器对支架回复的判分，再决定是否调整支架格式——不要直接判定方法失败。
- **s1200 取回**：若 Kaggle Version 3 输出取不到，先复核版本链接与输出文件清单，不要凭印象换起点。
- **s5100 不动**：无论 s1500 验收如何，产品版仍是 s5100，直到某块环上 12/12 超过它才换。

---

## 5. 对照表（基线与验收数字）

| 测量 | s900（已测，2026-11-21） | s1200 基线（待测，第 5 块前） | s1500 验收（待测，第 5 块后） | 通过线 |
|---|---|---|---|---|
| Q0 探针 probe | 0/7（固定默认南海） | | | ≥ 5/7 |
| Q0 探针 register | 0/7 | | | ≥ 5/7 |
| 因果 Q0 跟上移动 | 0/7 | | | ≥ 4/7 |
| 因果到达叙述 | 5/5 | | | ≥ 5/5（不退化） |
| 12 步环总分 | 6/12（avg_err 0.547） | | | ≥ 9/12 |
| 环 Q0 / Q1 / Q2 / Q3 | 0/3 / 3/3 / 3/3 / 0/3 | | | Q0 ≥ 2/3、Q3 ≥ 2/3 |
| 终 loss（仅记录，不验收） | 4.03 | 3.8010 | | 不作验收依据 |

---

## 附：数字出处一览

- s900 三项探针与 loss：`~/memory/2026-11-21.md`（09:41 段落、chunk 3 完成段落）；原始日志 `DIAG_Q0_S900.log`、`CAUSAL_PROBE_S900.log`、`omega_v010/run_g3s900_bridge.log`。
- 最小对子 240 条：`~/memory/2026-11-20.md`；脚本 `~/workspace/dual_sft/build_minimal_pairs.py`。
- s1200：`~/memory/2026-11-21.md`（chunk 4 完成与重跑保存段落）；Kaggle Version 3，scriptVersionId=372388932。
- 训练命令与续跑规则：`scripts/train.py --help`、`configs/train_default.yaml`、`configs/model_gated.yaml`。
- 探针命令：`evaluation/README.md`（`q_probe.py`、`causal_probe.py`、`loop.py`）。
