#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""常识域词表过滤：句子级整句保留/丢弃。

规则（与 2026-10-30 定下的硬约束一致）：词表块内冻结，任何新文本
必须先按现有 4622 词表过滤；句子含任一词表外字符则整句丢弃，
绝不把生字换成 UNK 混进训练。
输入：wiki_knowledge_raw.json（A，CC BY-SA 4.0）、
      youxue_qionglin_raw.txt（B，Gutenberg 公版 #52269）。
输出：knowledge_wiki.txt、knowledge_youxue.txt、
      knowledge_provenance.md（出处与统计）。
"""
import json
import re

HERE = "generalist_corpus"
SPLIT_RE = re.compile(r"(?<=[。！？；\n])")


def load_vocab():
    with open("generalist_vocab.json", encoding="utf-8") as f:
        meta = json.load(f)
    return set(meta["chars"])


def keep_sentences(text, vocab, min_len=4):
    kept, dropped = [], 0
    total_chars = 0
    for chunk in SPLIT_RE.split(text):
        s = chunk.strip()
        if not s:
            continue
        total_chars += len(s)
        if len(s) < min_len:
            dropped += 1
            continue
        if all(c in vocab for c in s):
            kept.append(s)
        else:
            dropped += 1
    return kept, dropped, total_chars


def main():
    vocab = load_vocab()
    print(f"vocab chars: {len(vocab)}")

    # ---- A: 维基百科导言 ----
    with open(f"{HERE}/wiki_knowledge_raw.json", encoding="utf-8") as f:
        articles = json.load(f)
    wiki_kept, wiki_dropped, wiki_raw = [], 0, 0
    wiki_used_articles = 0
    prov_lines = []
    for a in articles:
        ext = (a.get("extract") or "").strip()
        if not ext:
            continue
        kept, dropped, raw = keep_sentences(ext, vocab)
        wiki_raw += raw
        wiki_dropped += dropped
        if kept:
            wiki_used_articles += 1
            wiki_kept.extend(kept)
            prov_lines.append(f"- {a['title']} — {a['url']}")
    with open(f"{HERE}/knowledge_wiki.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(wiki_kept) + "\n")
    wiki_chars = sum(len(s) for s in wiki_kept)
    print(f"wiki: {len(articles)} articles, {wiki_used_articles} kept any, "
          f"raw {wiki_raw} chars -> kept {len(wiki_kept)} sentences, "
          f"{wiki_chars} chars, dropped {wiki_dropped} sentences")

    # ---- B: 幼学琼林（Gutenberg #52269，公版） ----
    with open(f"{HERE}/youxue_qionglin_raw.txt", encoding="utf-8") as f:
        raw_text = f.read()
    start = raw_text.find("*** START OF THE PROJECT GUTENBERG EBOOK")
    if start >= 0:
        raw_text = raw_text[raw_text.find("\n", start) + 1:]
    # 去掉 START 标记后重复的 Title/Author 头块，正文从“卷一”起
    body = raw_text.find("\n卷一\n")
    if body >= 0:
        raw_text = raw_text[body + 1:]
    end = raw_text.find("*** END OF THE PROJECT GUTENBERG EBOOK")
    if end >= 0:
        raw_text = raw_text[:end]
    kept, dropped, raw_chars = keep_sentences(raw_text, vocab)
    with open(f"{HERE}/knowledge_youxue.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(kept) + "\n")
    yx_chars = sum(len(s) for s in kept)
    print(f"youxue: raw {raw_chars} chars -> kept {len(kept)} sentences, "
          f"{yx_chars} chars, dropped {dropped} sentences")

    # ---- 出处 ----
    with open(f"{HERE}/knowledge_provenance.md", "w",
              encoding="utf-8") as f:
        f.write("# 常识域出处（2026-11-11 落地，2026-11-12 续抓一批）\n\n")
        f.write("## A：中文维基百科条目导言（CC BY-SA 4.0）\n\n")
        f.write("来源：zh.wikipedia.org API 随机条目导言（exintro），"
                "CC BY-SA 4.0，可自由复制改作，条件是署名（附条目链接）"
                "并以同协议共享改作。\n\n")
        f.write(f"条目数：{len(articles)}，其中 {wiki_used_articles} 条"
                f"在词表过滤后至少保留一句；过滤后 {len(wiki_kept)} 句、"
                f"{wiki_chars} 字。\n\n")
        f.write("条目清单：\n\n")
        f.write("\n".join(prov_lines) + "\n\n")
        f.write("## B：《幼学琼林》（公有领域）\n\n")
        f.write("来源：Project Gutenberg eBook #52269，"
                "https://www.gutenberg.org/ebooks/52269，"
                "明程登吉撰，作者逝世逾百年，公有领域；"
                "Gutenberg 转录本依其许可自由使用。\n\n")
        f.write(f"过滤后 {len(kept)} 句、{yx_chars} 字"
                f"（繁体字多，词表外句子整句丢弃，留存率低是预期的）。"
                f"原文 raw 文件：youxue_qionglin_raw.txt。\n")


if __name__ == "__main__":
    main()
