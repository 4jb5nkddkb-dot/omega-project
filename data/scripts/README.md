# data/scripts/ 说明

本目录是从旧仓库 `~/workspace/dual_sft/` 原样复制的语料生成脚本，
**逻辑未改**。它们产出 `data/processed/` 里的文件。
复制时只改了路径的脚本：无——全部脚本内部仍引用旧路径
（`~/workspace/omega-chat`、`~/workspace/dual_sft`）和 NumPy 单体
（`dual_transformer_numpy.py` 里的分词器/词表检查）。
重跑前必须把脚本里的输入输出路径指到本布局（`data/raw/`、`data/processed/`），
并把 `sys.path` 指向的单体分词器换成本仓库的 `tokenizer` 包；
否则脚本会直接读写旧仓库。已复制进 `processed/` 的文件是当前训练用的
权威版本，不需要重跑就能训练。

## 脚本 → 产出

| 脚本 | 产出 | 说明 |
| --- | --- | --- |
| `build_generalist_vocab.py` | `processed/generalist_vocab.json` | 从各域文本建 4622 字符表 |
| `convert_rulin.py` | `raw/classics_rulin.txt` | Gutenberg #24032 繁体本 → OpenCC t2s 转简体 → 按词表整句过滤 |
| `convert_honglou.py` | `raw/classics_honglou.txt` | Gutenberg #24264 繁体本 → OpenCC t2s 转简体 → 按词表整句过滤（与上行同一套规矩，2026-11-23） |
| `fetch_wiki_knowledge.py` | （中间产物，维基导言首批） | zh.wikipedia.org API 随机条目 exintro |
| `fetch_wiki_knowledge_more.py` | （中间产物，维基导言续抓） | 同上，第二批 |
| `fetch_wiki_val.py` | （验证集留出） | 从维基常识域留出验证集 |
| `filter_knowledge.py` | `processed/knowledge_wiki.txt` | 维基导言按词表过滤成句 |
| `build_minimal_pairs.py` | `processed/probe_format_records.jsonl`、`register_format_records.jsonl`、`carried_register_format_records.jsonl`、`daytime_register_format_records.jsonl` | 同一底本只改一个字段（地点/随身/昼夜），答案随之变 |
| `build_character_register_format_records` | `processed/character_register_format_records.jsonl` | 人物寄存器 150 条；生成脚本不在本目录（未随旧仓库复制），文件本身是权威版本 |
| `build_world_data.py` / `build_world_data_v2.py` / `build_world_data_v3.py` | 世界域记录 v1–v3（历史版本，未进 `processed/`） | 小世界循环记录的早期生成器 |
| `build_world_tool_data.py` / `..._v7.py` / `..._v8.py` / `..._v9.py` | `processed/world_tool_records_v9.jsonl`（仅 v9 在用） | 工具调用格式的世界记录，v9 为当前版 |
| `build_self_report_data.py` / `..._full.py` / `..._marked.py` / `..._q5.py` / `..._v9.py` / `..._greeting_data.py` / `..._patch_data.py` / `..._patch_data2.py` | `processed/self_report_records_v9.jsonl`（仅 v9 在用） | 自述格式记录；其余为历史变体 |
| `build_probe_format_data.py` | `processed/probe_format_records.jsonl` | 探针格式记录 |
| `build_video_records.py` | `processed/video_making_records.jsonl` | 做视频经验 52 条（纯文字） |
| `build_move_repair_data.py` | （修复数据，未进 `processed/`） | 移动类修复记录 |
| `build_sft_wukong_data.py` | （SFT 数据，未进 `processed/`） | 悟空相关 SFT 记录 |
| `build_dpo_nanhai_data.py` / `build_dpo_wukong_data.py` / `build_dpo_wukong_data_v2.py` | （DPO 数据，未进 `processed/`） | DPO 偏好对；DPO 轮次已关闭，不再训练用 |
| `build_character_register_format_records` 等 | `processed/character_register_format_records.jsonl`、`carried_…`、`daytime_…`、`register_format_records.jsonl` | 寄存器格式记录由 `build_minimal_pairs.py` 与上述各 `build_*_data.py` 共同生成，见各脚本头注释 |

注：`processed/` 里实际在用的 8 个世界域 `.jsonl` 文件名，见
`training/dataset.py::WORLD_RECORD_FILES`。`build_probe_format_data.py`
生成的是早期探针记录，当前 `probe_format_records.jsonl` 以
`build_minimal_pairs.py` 的输出为准（脚本内 `append` 为追加写）。
