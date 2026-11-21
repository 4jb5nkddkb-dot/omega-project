#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""全能新词表构建。

改编自旧仓 ``dual_sft/build_generalist_vocab.py``，逻辑与之一致，
只改输出位置与命令行接口：默认写出项目内
``data/processed/generalist_vocab.json``（规范位置，见 CONTRACT.md），
词表本身与旧脚本确定性相同（同一批源文件 → 同一个 4,622 词表）。

哲学沿用项目本身的 from_text：词表 = 试点语料里出现过的全部字符，
不设频次门槛——试点自己的训练文本不应该被自己的词表打成 UNK。
再强制收录通用字符（哪怕在中文经典里罕见）：ASCII、数字、代码/数学
标点、中文标点、工具符。专科词表的教训就是反面：代码片段 29 个不同
字符里 17 个没有，def 全灭、大写 0/26，模型连代码都读不进来。

来源（GENERALIST_CORPUS.md 盘点过的本地试点语料）：
西游记全文、仓库主线 26 个.py（根目录＋tools＋tests，排除
archive/legacy）、对话试点 40 轮、世界工具记录与自报记录的文本字段。
明确不含：grok 包、用户粘贴的手机对话记录、旧检查点。
注意：世界记录用的是 v8 版（world_tool_records_v8.jsonl /
self_report_records_full.jsonl），与旧脚本一致——换 v9 会改动词表，
如需升级请先确认词表变化并同步训练器。

输出：generalist_vocab.json，格式与 CharTokenizer.to_meta 完全一致
{"type": "char", "chars": [...]}（ID：0=BOS，1=EOS，2=UNK，3..=字符，
按码点排序），训练器可直接当 tokenizer 元数据用。

用法（在项目根目录）：
python3 -m tokenizer.build_vocab
python3 -m tokenizer.build_vocab --out data/processed/generalist_vocab.json
python3 -m tokenizer.build_vocab --out /tmp/vocab.json --report /tmp/report.md
源文件默认从旧仓只读路径取（--omega-chat / --dual-sft 可改）：
西游记与主线代码来自 ~/workspace/omega-chat，
对话试点与世界记录来自 ~/workspace/dual_sft。
"""
import argparse
import glob
import json
import os
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = PROJECT_ROOT / "data" / "processed" / "generalist_vocab.json"
DEFAULT_REPORT = PROJECT_ROOT / "data" / "processed" / "GENERALIST_VOCAB_REPORT.md"

FORCED = (
"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
"()[]{}:=+-*/_#'\".,<>!%@&|\\;?$`~^×÷±"
"，。！？；：、·「」『』《》〈〉"
"\n\t "
)


def read_text(path):
return Path(path).read_text(encoding="utf-8")


def collect_sources(omega_chat: Path, dual_sft: Path):
"""与旧脚本完全相同的源文件清单（只读，不改动源文件）。"""
sources = {}
sources["西游记"] = read_text(omega_chat / "Xiyouji.txt")
main_py = [p for p in glob.glob(str(omega_chat / "**/*.py"), recursive=True)
if "__pycache__" not in p
and "/archive/" not in p and "/legacy/" not in p]
sources["主线代码（26 文件）"] = "".join(read_text(p) for p in sorted(main_py))
sources["对话试点"] = read_text(
dual_sft / "generalist_corpus/dialogue_pilot.txt")
world_text = []
for name in ("world_tool_records_v8.jsonl", "self_report_records_full.jsonl"):
for line in open(dual_sft / name, encoding="utf-8"):
world_text.append(json.loads(line)["text"])
sources["世界/自报记录文本"] = "\n".join(world_text)
return sources


def build_chars(sources):
"""源文本 → 排序后的字符表（与旧脚本相同的确定性算法）。"""
counts = Counter()
for text in sources.values():
counts.update(text)
banned = set("\u200b\u200c\u200d\u2060\ufeff\u00ad")
observed = {c for c in counts if c not in banned
and (c in "\n\t " or ord(c) >= 32)}
forced_new = [c for c in FORCED if c not in observed]
chars = sorted(observed | set(FORCED))
return chars, observed, forced_new


def build_report(chars, observed, forced_new, sources):
stoi = {c: i + 3 for i, c in enumerate(chars)}

def encode(text):
return [stoi.get(c, 2) for c in text]

def decode(ids):
return "".join(chars[i - 3] for i in ids if i >= 3)

samples = {
"经典": sources["西游记"][:60],
"代码": "def add(a, b):\n return a + b # 加法",
"对话": "用户：写个打招呼的代码。\n助手：print(\"你好\")，运行就打印你好。",
"世界摘要": "世界状态。你在南海。在场人物孙悟空、唐僧、猪八戒、沙僧、白龙马、观音。可见物品紫金钵盂。随身物品无。时间第三步白天。",
"工具调用": "《移动》花果山】《查看》】《拾取》蟠桃】《对话》孙悟空】",
}
acid = "def hello(name):\n print(\"hi\", name)\n x = 1 + 2 * 3\n return x # done\n"
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
return "\n".join(lines)


def main():
ap = argparse.ArgumentParser(description="构建全能字符词表（确定性，与旧脚本同输出）")
ap.add_argument("--out", default=str(DEFAULT_OUT),
help="词表 JSON 输出路径（默认 data/processed/generalist_vocab.json）")
ap.add_argument("--report", default=str(DEFAULT_REPORT),
help="验证报告输出路径（默认 data/processed/GENERALIST_VOCAB_REPORT.md）")
ap.add_argument("--omega-chat",
default=os.path.expanduser("~/workspace/omega-chat"),
help="旧仓路径：西游记与主线代码来源（只读）")
ap.add_argument("--dual-sft",
default=os.path.expanduser("~/workspace/dual_sft"),
help="旧仓路径：对话试点与世界记录来源（只读）")
args = ap.parse_args()

sources = collect_sources(Path(args.omega_chat), Path(args.dual_sft))
chars, observed, forced_new = build_chars(sources)

out_path = Path(args.out)
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
json.dump({"type": "char", "chars": chars}, f, ensure_ascii=False)

report = build_report(chars, observed, forced_new, sources)
report_path = Path(args.report)
report_path.parent.mkdir(parents=True, exist_ok=True)
report_path.write_text(report, encoding="utf-8")
print(report)
print(f"wrote {out_path} ({3 + len(chars)} tokens)")


if __name__ == "__main__":
main()
