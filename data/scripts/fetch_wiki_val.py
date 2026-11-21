#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""抓 8 条新随机维基导言作常识域留出验证句（不进训练）。"""
import json
import urllib.request
import urllib.parse

API = "https://zh.wikipedia.org/w/api.php"
UA = {"User-Agent": "Muse-agent/1.0 (generalist eval; local research)"}
params = {
    "action": "query", "format": "json",
    "generator": "random", "grnnamespace": "0", "grnlimit": "8",
    "prop": "extracts|info", "exintro": "1", "explaintext": "1",
    "inprop": "url",
}
url = API + "?" + urllib.parse.urlencode(params)
req = urllib.request.Request(url, headers=UA)
with urllib.request.urlopen(req, timeout=30) as r:
    data = json.load(r)
pages = data.get("query", {}).get("pages", {})
train = json.load(open("generalist_corpus/wiki_knowledge_raw.json", encoding="utf-8"))
train_ids = {p["pageid"] for p in train}
kept = []
for pid, p in pages.items():
    if p.get("pageid") in train_ids:
        print("skip (already in train):", p.get("title"))
        continue
    kept.append({"pageid": p.get("pageid"), "title": p.get("title"),
                 "url": p.get("fullurl"), "extract": p.get("extract", "")})
with open("generalist_corpus/knowledge_val_heldout.json", "w", encoding="utf-8") as f:
    json.dump(kept, f, ensure_ascii=False, indent=1)
print(f"held out {len(kept)} fresh articles")
for k in kept:
    print(" -", k["title"], len(k["extract"]), "chars")
