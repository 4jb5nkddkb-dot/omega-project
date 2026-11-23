#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""红楼梦：Gutenberg #24264 繁体本 → OpenCC t2s 转简体 → 按冻结词表整句过滤。

来源：Project Gutenberg eBook #24264（曹雪芹，公有领域），
https://www.gutenberg.org/ebooks/24264。转简体用 OpenCC t2s
（确定性转换，逐字/逐词对照，非人工改写）；过滤规则与《儒林外史》
（convert_rulin.py）相同：句子含任一词表外字则整句丢弃。
原文于 2026-11-23 自 Waterloo 镜像站
（mirror.csclub.uwaterloo.ca/gutenberg/）拉取，因 Gutenberg 主站
当晚维护（503）；书号经 Online Books Page 核对。
输出：hongloumeng_simplified.txt（转简后全文）、
      classics_honglou.txt（过滤后训练用句）。
运行：cd ~/workspace/dual_sft && python3 convert_honglou.py
"""
import json
import re

from opencc import OpenCC

HERE = "generalist_corpus"
SPLIT_RE = re.compile(r"(?<=[。！？；\n])")


def main():
    raw = open(f"{HERE}/hongloumeng_raw.txt", encoding="utf-8-sig").read()
    start = raw.find("*** START OF THE PROJECT GUTENBERG EBOOK")
    if start >= 0:
        raw = raw[raw.find("\n", start) + 1:]
    end = raw.find("*** END OF THE PROJECT GUTENBERG EBOOK")
    if end >= 0:
        raw = raw[:end]
    print(f"body chars (traditional): {len(raw)}")
    conv = OpenCC("t2s").convert(raw)
    with open(f"{HERE}/hongloumeng_simplified.txt", "w", encoding="utf-8") as f:
        f.write(conv)
    print(f"converted chars (simplified): {len(conv)}")
    vocab = set(json.load(open("generalist_vocab.json", encoding="utf-8"))["chars"])
    kept, dropped, total = [], 0, 0
    for chunk in SPLIT_RE.split(conv):
        s = chunk.strip()
        if not s:
            continue
        total += len(s)
        if len(s) < 4:
            dropped += 1
            continue
        if all(c in vocab for c in s):
            kept.append(s)
        else:
            dropped += 1
    with open(f"{HERE}/classics_honglou.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(kept) + "\n")
    print(f"sentences: kept {len(kept)} / dropped {dropped}; "
          f"kept chars {sum(len(s) for s in kept)} / {total} "
          f"({100 * sum(len(s) for s in kept) / max(total, 1):.1f}% survival)")


if __name__ == "__main__":
    main()
