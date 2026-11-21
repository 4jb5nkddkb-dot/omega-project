# 数据（data/）

## 布局

```
data/
  raw/        原始语料（只读来源的副本，不做加工）
    Xiyouji.txt            《西游记》全文
    classics_rulin.txt     《儒林外史》加工本（见下“来源说明”）
  processed/  训练直接读的文件（词表 + 各域文本 + 世界域 .jsonl 记录）
    generalist_vocab.json  字符表：4619 字 + BOS/EOS/UNK = 4622 词
    dialogue_pilot.txt     对话域
    knowledge_wiki.txt     常识域：维基百科导言
    knowledge_youxue.txt   常识域：《幼学琼林》
    world_tool_records_v9.jsonl 等 8 个世界域 .jsonl 记录文件
    MANIFEST.md            每个文件的记录数 / 字符数 / token 数（实测，非估计）
  scripts/    生成 processed/ 的脚本（从旧仓库原样复制，逻辑未改）
    README.md              每个脚本产出什么、输入输出路径说明
```

训练代码默认读新布局（`training/dataset.py::default_paths`）；
过渡期可用 `live_corpus_paths()` 指回旧仓库的 live 语料。
**不进仓库的东西**：`.npz` 检查点、`.log` 训练日志（见根目录 `.gitignore`）。

## 来源与许可（provenance / licenses）

| 文件 | 来源 | 许可 |
| --- | --- | --- |
| `raw/Xiyouji.txt` | 《西游记》全文（旧仓库 `omega-chat/Xiyouji.txt` 的副本） | 公有领域（作者逝世逾百年） |
| `raw/classics_rulin.txt` | 《儒林外史》：Project Gutenberg #24032 繁体本，经 OpenCC t2s 转简体、按 4622 词表整句过滤后的加工本。**注意：这里存的是加工后的文件**，不是 Gutenberg 原文；原始繁体本见旧仓库 `dual_sft/generalist_corpus/rulinwaishi_raw.txt`，转简体本见 `rulinwaishi_simplified.txt`，转换脚本是 `data/scripts/convert_rulin.py` | 公有领域（Gutenberg 公版） |
| `processed/knowledge_wiki.txt` | 中文维基百科条目导言（zh.wikipedia.org API，exintro），2700 条中 1008 条过滤后保留，2238 句、71667 字；条目清单见旧仓库 `dual_sft/generalist_corpus/knowledge_provenance.md` | CC BY-SA 4.0（署名 + 相同方式共享） |
| `processed/knowledge_youxue.txt` | 《幼学琼林》：Project Gutenberg eBook #52269，明程登吉撰 | 公有领域（Gutenberg 公版） |
| `processed/*.jsonl` 世界域记录 | 本项目自产：小世界循环的协议记录（工具调用、自述、探针/寄存器格式、最小对、做视频经验），纯文字，无图片/视频 | 本项目自有数据 |
| `processed/generalist_vocab.json` | 由 `data/scripts/build_generalist_vocab.py` 从语料生成 | 本项目自产 |

**明确不含**：grok 包、用户粘贴的手机对话、旧检查点（见 `training/dataset.py` 模块 docstring）。

## 如何从 raw/ 重建 processed/

```bash
# 1. 词表（读 raw/ 两本经典 + 其余域文本，产出 generalist_vocab.json）
python data/scripts/build_generalist_vocab.py

# 2. 《儒林外史》加工（Gutenberg 原文 -> 转简体 -> 按词表整句过滤）
python data/scripts/convert_rulin.py

# 3. 常识域：抓维基导言 -> 按词表过滤成句
python data/scripts/fetch_wiki_knowledge.py   # 首批
python data/scripts/fetch_wiki_knowledge_more.py  # 续抓
python data/scripts/fetch_wiki_val.py          # 留出验证集
python data/scripts/filter_knowledge.py        # 过滤成 knowledge_wiki.txt

# 4. 世界域记录：各 build_*.py 生成 .jsonl（最小对见 build_minimal_pairs.py）
python data/scripts/build_minimal_pairs.py
```

注意：`data/scripts/` 里的脚本是从旧仓库原样复制的，内部仍引用旧路径
（`~/workspace/omega-chat`、`~/workspace/dual_sft`）和 NumPy 单体分词器；
重跑前需按 `data/scripts/README.md` 的说明把输入输出路径指到本布局。
已复制进 `processed/` 的文件是当前训练用的权威版本，重建结果应与
`MANIFEST.md` 的计数一致（允许因上游重抓产生小幅差异）。
