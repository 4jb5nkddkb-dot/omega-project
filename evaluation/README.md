# evaluation/ — 探针与循环验收（PyTorch 版）

Port 自 `~/workspace/dual_sft/` 的 NumPy 探针，改走 PyTorch 模型
（`inference.generate` + `model.DualTransformer`）。判卷逻辑与原版
一字不差——改判卷等于改测量的东西。

## 共用件

| 文件 | 作用 |
|---|---|
| `world_state.py` | 冻结的世界状态机副本（地点/人物/物品/桩时钟/校验）。探针自带一份冻结副本，不 import 产品代码；旧仓库修了状态机须同步更新此副本。 |
| `scoring.py` | Q0..Q3 问题集 + 严格字符串评分 `score(state, qi, cont)`（port 自 `world_self_probe.py`）。 |
| `sidecar.py` | PyTorch 版 Sidecar：`generate` / `ask_question` / `narrate_arrival` / `talk` / `execute` + 在线记忆缓存 `CognitiveMemoryBuffer`（答对才记，容量 3，同状态才拼）。**没有**移植 Rust 桥接器的文件协议（observation.json / action.json 轮询）——那是产品侧内核选动作的通道，探针不需要。 |
| `_common.py` | 从 `.pt` 检查点加载模型 + 词表（只 load 一次）。 |

## 各探针测什么、怎么跑

所有探针都需要一个已转换的 `.pt` 检查点（`scripts/convert_npz_to_pt.py`
的产物）。`--checkpoint` 不给会报清晰的错并退出，不会去找默认检查点。
`--device` 默认 `cpu`。

### `causal_probe.py` — 因果链探针：移动后 Q0 跟不跟得上

```
python -m evaluation.causal_probe --checkpoint checkpoints/generalist-v5_s600.pt
python -m evaluation.causal_probe --checkpoint <ckpt.pt> --no-memory --register
```

13 步：起点问 Q0，随后 5 次「移动→问 Q0」，最后再问一次。
输出 `Q0 跟上移动：X/7`（7 次 Q0 答对数）与 `到达叙述无矛盾：Y/5`
（5 次到达叙述经 `world_state.validate()` 无矛盾的次数），
每步写进 `<dir>/causal_probe.jsonl`（默认 `run_causal/`）。
`--no-memory` 关在线记忆缓存（隔离记忆示例是否污染移动后的问答）；
`--register` 开寄存器原型（地点句复制到提示最前面）。

### `loop.py` — 冻结 12 步循环（换货的唯一标尺）

```
python -m evaluation.loop --checkpoint checkpoints/generalist-v5_s600.pt
python -m evaluation.loop --checkpoint <ckpt.pt> --out loop_run.jsonl
```

12 步全是 stabilize：轮询 Q0..Q3（Q0 在 1/5/9 步，Q1 在 2/6/10 步，
Q2 在 3/7/11 步，Q3 在 4/8/12 步），世界冻结在 S0（取经路、step 0、
白天，一步不动），分 = `grounded_err` 为 0.00 的步数。打印每题
X/3 与总分 X/12，并对照换货线。

**与 live 循环的区别**：没有 Rust 桥接器与内核选动作（动作表固定为
stabilize×12），没有 observation.json / action.json 文件协议，
直接在进程内驱动 Sidecar。原版 stabilize 循环里世界本来就不动、
时间不走，所以冻结版与原版在“问什么、判什么”上完全一致；
差的是动作选择（内核在 live 循环里 12 步全选 stabilize 是实测结果，
不是设计保证）与记忆累积的时序细节（冻结版同样开着记忆缓存）。

### `gate_probe.py` — 门控分流探针

```
python -m evaluation.gate_probe --checkpoint checkpoints/generalist-v5_s600.pt
python -m evaluation.gate_probe --checkpoint <ckpt.pt> --skip-domains
```

两部分（A：域窗口，B：循环提示），只读不写检查点：
- **A. 域窗口**：每域 4 个 128-token 窗口的 mean g。左域
  （经典/代码/对话/常识/推理）应 →1，世界域应 →0。
  通过线：左均 > 0.6 且世界 < 0.4；趋势线：separation（左均 − 世界）> 0.05。
- **B. 循环提示**：真实循环提示的 g——S0 的 Q0..Q3、六个地点的 Q0、
  两条到达叙述、两条探针格式训练记录、世界/经典语料窗对照。
  回答 2026-11-18 的未解问题：QA 格式提示是不是被门送去了左支。

**测不到的东西**：合同 API 只暴露 `gate_values()`（g），不暴露门输入
`||h_pool||`，所以 v4/v5 诊断里的 `||h_pool||` 与
`||mean_left − mean_world||` 两列在这里没有——门输入的量级问题只能
靠训练日志或模型内部探针看。

### `q_probe.py` — Q0/Q1/Q2/Q3 探针集

```
python -m evaluation.q_probe --checkpoint checkpoints/generalist-v5_s600.pt
```

Q0：7 个地点 × 两种格式（probe：摘要 + 问题；register：地点句放最前面
再跟摘要 + 问题）；Q1/Q2/Q3：只在取经路（step 2，白天）做对照。
每题判 PASS/FAIL，汇总每题与 Q0 两种格式的分数，日志写进
`diag_q0.log`（可用 `--log` 改路径）。生成长度默认 16 个新 token
（原版同款），可用 `--max-new` 改。

## 换货规则（swap rule）

只有冻结循环分**打赢 12/12** 的块才能替换当前产品。当前产品是
**s5100（generalist2_s5100，12/12）**。挑战者至今没到过 12/12
（v5 s600 是 3/12，d512 线最好的是 s5100 自己），所以 s5100 留任。
`loop.py` 跑完会直接打印对照换货线的结果。

## 已知失败模式

- **Q0 固定答案**：模型不读状态摘要，三次都答同一个地点
  （v5 s600：三次都答“你在花果山”，Q0 0/3；d512 旧线 s61800
  也曾 0/3）。`q_probe.py` 的 7 地点 × 2 格式就是抓这个的；
  `causal_probe.py` 的“移动后 Q0”是从另一面抓同一件事。
- **Q2/Q3 跷跷板**：随身（Q2）与昼夜（Q3）从不同时 3/3——s7500
  修好随身 3/3 时昼夜掉到 0/3，s7800 修好昼夜 3/3 时随身掉到 0/3；
  自 s6300 的 12/12 之后 Q2+Q3 再没同时满过。`loop.py` 的分题
  X/3 是看跷跷板的第一现场。
- **门分得开、行为跟不上**：v5 的门在 s600 仍分得干净
  （左 1.0000、世界 0.2500、separation 0.7500），但循环只有 3/12、
  因果 Q0 只有 1/7——门不是瓶颈，右支还没学会回答。`gate_probe.py`
  的 B 部分（循环提示的 g）与 `loop.py` 必须一起读：g 全 0
  说明问题在右支本身，不在门。
- **记忆示例翻答案**：同状态下只隔一步答案就翻了，两步之间唯一变的
  是缓存多了一条示例（s2100 因果复测：记忆开 4/7、关 7/7）。
  怀疑记忆污染时用 `--no-memory` 对照。
- **寄存器 B/B2 已证伪**：B（一次贴三行）循环 9/12 = 基线，B2
  （按问题贴一行）7/12 < 基线且昼夜 3/3→1/3；两者 Q2 都是 0/3。
  寄存器原型只留作探针对照，不进产品。
