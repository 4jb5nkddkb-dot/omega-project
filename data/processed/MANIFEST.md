# processed/ 清单（MANIFEST）

计数方法：2026-11-21 实测，用本仓库 `tokenizer.CharTokenizer`
读 `generalist_vocab.json`（4619 字 + BOS/EOS/UNK = 4622 词）编码。
字符级分词器：1 个字符 = 1 个 token（词表外字符计为 1 个 UNK token），
所以“字符数 = token 数”不是笔误。`.jsonl` 的字符/token 数只计 `text` 字段。
sha256 取文件前 16 位 hex，仅作副本校验用。

## 词表

| 文件 | 内容 | 大小 |
| --- | --- | --- |
| `generalist_vocab.json` | `{"type": "char", "chars": [...]}`，4619 字，词表 4622 | 32,166 B，sha256 `df0d260447a01ff4` |

## 文本域

| 文件 | 字符数 | token 数 | 大小 / sha256 |
| --- | --- | --- | --- |
| `dialogue_pilot.txt`（对话） | 2,918 | 2,918 | 8,050 B，`920fc75fcc4f4020` |
| `knowledge_wiki.txt`（常识·维基） | 73,905 | 73,905 | 190,044 B，`c4e2edbb03cb5a41` |
| `knowledge_youxue.txt`（常识·幼学琼林） | 888 | 888 | 2,520 B，`a3d1351c4946dc82` |

## 世界域 .jsonl（`text` 字段计数）

| 文件 | 记录数 | 字符数 | token 数 | 词表外字符 | 大小 / sha256 |
| --- | --- | --- | --- | --- | --- |
| `world_tool_records_v9.jsonl` | 1,763 | 197,014 | 197,014 | 0 | 919,768 B，`e56665b5b254604b` |
| `self_report_records_v9.jsonl` | 1,200 | 118,065 | 118,065 | 0 | 969,554 B，`94aadeebb2c04536` |
| `probe_format_records.jsonl` | 722 | 54,651 | 54,651 | 0 | 434,635 B，`b8533b70e0ad210e` |
| `register_format_records.jsonl` | 220 | 17,040 | 17,040 | 0 | 137,519 B，`536cd24c9c4e4e7c` |
| `carried_register_format_records.jsonl` | 181 | 15,028 | 15,028 | 0 | 119,357 B，`d4f83678fb24bb77` |
| `daytime_register_format_records.jsonl` | 171 | 13,832 | 13,832 | 0 | 110,523 B，`b1c5bef9aabb9d22` |
| `character_register_format_records.jsonl` | 150 | 16,586 | 16,586 | 0 | 123,760 B，`1d272bc5260bb2f5` |
| `video_making_records.jsonl` | 52 | 2,512 | 2,512 | 0 | 5,934 B，`9ffb4f23aa186ed2` |
| **世界域合计** | **4,459** | **434,728** | **434,728** | **0** | — |

注意：`training/dataset.py` 的模块 docstring 里写的旧数（最小对 120 条、
地点寄存器 70 条等）是 2026-11-20 生成时的数字，文件后来又追加过，
以上实测计数为准。

## 原始语料（raw/，见 `data/README.md`）

| 文件 | 字符数 | token 数 | 大小 / sha256 |
| --- | --- | --- | --- |
| `raw/Xiyouji.txt` | 677,421 | 677,421 | 1,999,157 B，`3f2bf2164ee63ad7` |
| `raw/classics_rulin.txt` | 241,580 | 241,580 | 703,730 B，`50d73f7be7e83276` |

## 校验

全部 14 个复制文件与源文件逐字节比对一致（`cmp` 全过，2026-11-21）。
