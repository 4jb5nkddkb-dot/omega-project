#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""抓中文维基百科随机条目导言（常识域 A 来源，CC BY-SA 4.0）。

用 generator=random 分批取条目导言（exintro + explaintext），去重后
存 raw JSON 与条目清单（标题/pageid/URL，供署名出处）。
目标约 300 条；空导言与消歧义短条在过滤阶段自然落掉。
"""
import json
import time
import urllib.request
import urllib.parse

API = "https://zh.wikipedia.org/w/api.php"
UA = {"User-Agent": "Muse-agent/1.0 (generalist corpus sourcing; local research)"}
BATCHES = 15
PER_BATCH = 20


def fetch_batch():
    params = {
        "action": "query",
        "format": "json",
        "generator": "random",
        "grnnamespace": "0",
        "grnlimit": str(PER_BATCH),
        "prop": "extracts|info",
        "exintro": "1",
        "explaintext": "1",
        "inprop": "url",
    }
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def main():
    seen = {}
    for i in range(BATCHES):
        try:
            data = fetch_batch()
        except Exception as e:
            print(f"batch {i}: FAILED {e}")
            time.sleep(5)
            continue
        pages = data.get("query", {}).get("pages", {})
        n_new = 0
        for pid, p in pages.items():
            if pid in seen:
                continue
            seen[pid] = {
                "pageid": p.get("pageid"),
                "title": p.get("title"),
                "url": p.get("fullurl"),
                "extract": p.get("extract", ""),
            }
            n_new += 1
        print(f"batch {i}: {len(pages)} pages, {n_new} new, total {len(seen)}")
        time.sleep(1)
    out = list(seen.values())
    with open("generalist_corpus/wiki_knowledge_raw.json", "w",
              encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    total_chars = sum(len(p["extract"]) for p in out)
    nonempty = sum(1 for p in out if p["extract"].strip())
    print(f"saved {len(out)} articles, {nonempty} with extract, "
          f"{total_chars} extract chars")


if __name__ == "__main__":
    main()
