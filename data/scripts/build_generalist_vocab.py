#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""全能新词表构建（试点用）。

哲学沿用项目本身的 from_text：词表 = 试点语料里出现过的全部字符，
不设频次门槛——试点自己的训练文本不应该被自己的词表打成 UNK。
再强制收录通用字符（哪怕在中文经典里罕见）：ASCII、数字、代码/数学
标点、中文标点、工具符。专科词表的教训就是反面：代码片段 29 个不同
字符里 17 个没有，def 全灭、大写 0/26，模型连代码都读不进来。

来源（GENERALIST_CORPUS.md 盘点过的本地试点语料）：
  西游记全文、仓库主线 26 个 .py（根目录＋tools＋tests，排除
  archive/legacy）、对话试点 40 轮、世界工具记录与自报记录的文本字段。
  明确不含：grok 包、用户粘贴的手机对话记录、旧检查点。

输出：generalist_vocab.json，格式与 CharTokenizer.to_meta 完全一致
  {"type": "char", "chars": [...]}（ID：0=BOS，1=EOS，2=UNK，3..=字符，
  按码点排序），训练器可直接当 tokenizer 元数据用。

用法：python3 build_generalist_vocab.py
  写出 generalist_vocab.json 与 GENERALIST_VOCAB_REPORT.md，
  并打印分词验证（各域抽样往返＋代码片段酸测）。
"""
import glob
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.expanduser("~/workspace/omega-chat"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HERE = Path(__file__).resolve().parent
MIRROR = Path(os.path.expanduser("~/workspace/omega-chat"))
OUT_JSON = HERE / "generalist_vocab.json"
OUT_REPORT = HERE / "GENERALIST_VOCAB_REPORT.md"

FORCED = (
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    "()[]{}:=+-*/_#'\".,<>!%@&|\\;?$`~^×÷±"
    "，。！？；：、·「」『』《》〈〉【】"
    "\n\t "
)


def read_text(path):
    return Path(path).read_text(encoding="utf-8")


def main():
    sources = {}
    sources["西游记"] = read_text(MIRROR / "Xiyouji.txt")
    main_py = [p for p in glob.glob(str(MIRROR / "**/*.py"), recursive=True)
               if "__pycache__" not in p
               and "/archive/" not in p and "/legacy/" not in p]
    sources["主线代码（26 文件）"] = "".join(read_text(p) for p in sorted(main_py))
    sources["对话试点"] = read_text(HERE / "generalist_corpus/dialogue_pilot.txt")
    world_text = []
    for name in ("world_tool_records_v8.jsonl", "self_report_records_full.jsonl"):
        for line in open(HERE / name, encoding="utf-8"):
            world_text.append(json.loads(line)["text"])
    sources["世界/自报记录文本"] = "\n".join(world_text)

    counts = Counter()
    for text in sources.values():
        counts.update(text)
    banned = set("\u200b\u200c\u200d\u2060\ufeff\u00ad")
    observed = {c for c in counts if c not in banned
                and (c in "\n\t " or ord(c) >= 32)}
    forced_new = [c for c in FORCED if c not in observed]
    chars = sorted(observed | set(FORCED))
    stoi = {c: i + 3 for i, c in enumerate(chars)}

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"type": "char", "chars": chars}, f, ensure_ascii=False)

    def encode(text):
        return [stoi.get(c, 2) for c in text]

    def decode(ids):
        return "".join(chars[i - 3] for i in ids if i >= 3)

    samples = {
        "经典": sources["西游记"][:60],
        "代码": "def add(a, b):\n    return a + b  # 加法",
        "对话": "用户：写个打招呼的代码。\n助手：print(\"你好\")，运行就打印你好。",
        "世界摘要": "世界状态。你在南海。在场人物孙悟空、唐僧、猪八戒、沙僧、白龙马、观音。可见物品紫金钵盂。随身物品无。时间第三步白天。",
        "工具调用": "《移动》花果山】《查看》】《拾取》蟠桃】《对话》孙悟空】",
    }
    acid = "def hello(name):\n    print(\"hi\", name)\n    x = 1 + 2 * 3\n    return x  # done\n"
    lines = []
    lines.append("# 全能新词表报告（试点）\n")
    lines.append(f"词表规模：{3 + len(chars)}（BOS/EOS/UNK ＋ {len(chars)} 字符）")
    lines.append(f"语料观测字符：{len(observed)}；强制新增：{len(forced_new)}"
                 f"（{''.join(forced_new)}）")
    lines.append("")
    lines.append("## 来源规模")
    for name, text in sources.items():
        unk = sum(1 for c in text if c not in stoi)
        lines.append(f"- {name}：{len(text)} 字，词表外 {unk} 字"
                     f"（{100.0 * unk / max(len(text), 1):.3f}%）")
    lines.append("")
    lines.append("## 分词验证（往返＝编码再解码还原原文）")
    for name, text in samples.items():
        rt = decode(encode(text))
        lines.append(f"- {name}：往返{'通过' if rt == text else '失败'}"
                     f"（{len(text)} 字，UNK {sum(1 for c in text if c not in stoi)}）")
    acid_unk = sum(1 for c in acid if c not in stoi)
    acid_rt = decode(encode(acid))
    lines.append(f"- 代码酸测（专科词表曾 17/29 缺失）：UNK {acid_unk}，"
                 f"往返{'通过' if acid_rt == acid else '失败'}")
    lines.append("")
    lines.append("## 说明")
    lines.append("- 试点词表小于 GENERALIST_PLAN.md 的 6000–8000 全轮目标，"
                 "因为目前只有一个经典在内；更多经典进全轮语料时词表随之长大，"
                 "这是预期，不是失败。")
    lines.append("- 频次门槛刻意不用：试点自己的训练文本不应被自己的词表打成 UNK。")
    report = "\n".join(lines)
    OUT_REPORT.write_text(report, encoding="utf-8")
    print(report)
    print(f"wrote {OUT_JSON} ({3 + len(chars)} tokens)")


if __name__ == "__main__":
    main()
