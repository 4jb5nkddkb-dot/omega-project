#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""做视频的经验 -> 世界域文本记录（motion-cli 分支，2026-11-16 用户定）。

规则：
- 只加文字，不加视频/图片文件（模型是纯文本，4622 字词表）。
- 句级过滤（与 filter_knowledge.py 同款）：句含任一词表外字符则整句丢弃，
  绝不把生字换成 UNK 混进训练。
- 每条记录 <= 126 字（含 BOS/EOS 进 128 字窗口），与寄存器记录同标准。
来源：~/workspace/motion-cli/README.md、RELEASE_NOTES、各集实测教训。
输出：video_making_records.jsonl（每行 {"text": ...}），由
generalist_pilot_train.py 的 domain_texts 接进世界域。
"""
import json
import re

HERE = "."
SPLIT_RE = re.compile(r"(?<=[。！？；\n])")

RECORDS = [
    "motion 是给 agent 的一双手：脑在 agent 那边，负责理解 skill、写分镜、判断画面；手只做三件确定的事：找本领、按故事清单出片、验片打分。",
    "一句话能出片的人是 agent，motion 是让这句话每次都落到同一个结果上的工具。",
    "motion 不生成画面，不写旁白，不判断画面好不好；画面与旁白由 agent 或用户提供，工具只保证合成与验片这两步确定、可复现、可打分。",
    "找本领：motion skill search short-form，按关键词找。",
    "出片：motion make --manifest story.json，按故事清单出一条 30 s 竖屏片。",
    "验片：motion video verify --json out.mp4，出机器可读的报告和分数。",
    "故事清单是脑、世界与手之间的契约：工具只按文件办事，不猜故事。",
    "清单谁都能写：agent、小世界、用户，写好后一条命令出片。",
    "清单字段 title 是故事名，仅作提示，可以不填。",
    "清单字段 seconds 是成片时长，不填默认 30 s。",
    "清单字段 width 与 height 是画面尺寸，不填默认 1080 与 1920。",
    "清单字段 fps 是每 s 播放的画面张数，不填默认 30。",
    "清单字段 bg_volume 是片段原声音量，0 到 1 之间，不填默认 0.25。",
    "清单字段 narr_delay_ms 管旁白晚多久出声，单位是 ms，不填默认 1000。",
    "清单字段 clips 是片段路径数组，必填，至少一个。",
    "清单字段 narration 是旁白音频路径，必填。",
    "清单字段 out 是输出路径，必填。",
    "清单字段 captions 是字幕行，每行有 style 与 text，时间 start 与 end 可选；style 只允许 Hook、Body、Payoff。",
    "清单里的相对路径按清单文件所在目录来算，绝对路径照常用。",
    "缺省 captions 或空数组时不烧字幕。",
    "字幕时间有两种写法：清单自带，每行都写 start 与 end，工具原样采用。",
    "字幕时间自动计算：每行都不写时间，只给 style 与 text，工具在合成前用 ffprobe 实测旁白时长，按字数均分算出每行时间。",
    "自动计时公式：每字时长等于旁白实测时长除以总字数，每行起始等于旁白延迟加上前面各行字数乘以每字时长，最后一行结束强制为成片时长。",
    "自动计时成功时打印字幕时间：自动计算，旁白实测时长。",
    "工具在内部生成 ASS 字幕：Hook 84 号字，Body 62 号字，Payoff 72 号字金色，用 Noto Sans CJK SC 字体。",
    "生成的 ASS 保留在输出文件同目录，与输出同名，后缀为 ass。",
    "清单报错：缺必填字段、JSON 写坏了、style 非法、时间不合法、混用带时间与不带时间的行时，工具以非零退出码退出，并在 stderr 里点出字段名或原因。",
    "验片命令：motion video verify out.mp4，可加 --expect 1080x1920 与 --fps 30。",
    "验片加 --json 时 stdout 只输出一个 JSON 对象，诊断走 stderr，含 score 与 verdict。",
    "验片分数第一项是分辨率，权重 0.35，实测宽高等于 --expect 才算过，默认 1080x1920。",
    "验片分数第二项是每 s 画面张数，权重 0.25，实测与 --fps 之差小于 0.05 才算过，默认 30。",
    "验片分数第三项是时长，权重 0.25，实测与 --seconds 之差不超过 0.10 s 才算过，默认 30。",
    "验片分数第四项是音频流存在，权重 0.15，有音频流才算过，只看存在，不听内容。",
    "verdict 为 pass 当且仅当四项全过，分数在 0 到 1 之间。",
    "分数是规格接地分，回流给世界的格式是 grounded_error 等于 1 减 score，reward 等于 score。",
    "接地记录由 verify_to_grounding.py 写进 grounding_ledger.jsonl，按 episode 与 video 去重。",
    "诚实说明：verify 只能看画面规格和音频流是否存在，听不到旁白，读不到烧进去的字幕。",
    "字幕与旁白的对齐程度、画面内容是否对题都不在分数里，不要把 score 说成视频质量全检。",
    "checks 里 caption_alignment 的 ok 为 null，detail 写 not_measured，意思是没测，不是通过，也不是没通过。",
    "合成配方：每个片段先缩放、再 concat，再用 subtitles 烧字幕；片段音频调音量后 concat。",
    "旁白先延迟，再补齐，再截到 TOTAL，TOTAL 是各片段 ffprobe 实测时长之和，不猜。",
    "背景音与旁白用 amix 混合，取第一个时长；编码用 libx264，音频用 aac 160k，加上 faststart。",
    "支持任意多个片段，不只 3 个；--clips 里多个路径用逗号分隔时请用绝对路径。",
    "教训：字幕必须逐字抄旁白，并按实测旁白时长计时；第一集的字幕与旁白对不上，就是因为字幕不是逐字抄的。",
    "教训：ffmpeg 的 silenceremove 没有 keep_silence 参数，只有 start 与 stop 相关参数。",
    "局限：JSON 处理器没有 true、false、null，Unicode 代理对会退化。",
    "局限：无字幕分支与相对路径清单还没有端到端跑通过。",
    "样例 examples/danaotiangong.json 可以复现第 3 集大闹天宫，验证通过。",
    "安装需要系统里有 ffmpeg 与 ffprobe，以及 Noto CJK 字体，字幕烧录用。",
    "开发命令：cargo test 跑单元测试，cargo clippy 检查，cargo build --release 编出二进制。",
    "局限：目前只在 Linux 上验证过，没有 Windows 交叉编译，也没有发过 release。",
    "局限：skill search 是子串匹配，不是语义检索。",
]


def load_vocab():
    with open("generalist_vocab.json", encoding="utf-8") as f:
        meta = json.load(f)
    return set(meta["chars"])


def main():
    vocab = load_vocab()
    print(f"vocab chars: {len(vocab)}, records authored: {len(RECORDS)}")
    kept_records, dropped_sents = [], []
    for i, rec in enumerate(RECORDS):
        parts = []
        for chunk in SPLIT_RE.split(rec):
            s = chunk.strip()
            if not s:
                continue
            if all(c in vocab for c in s):
                parts.append(s)
            else:
                oov = sorted({c for c in s if c not in vocab})
                dropped_sents.append((i, s, oov))
        text = "".join(parts)
        if not text:
            print(f"record {i}: ALL DROPPED: {rec}")
            continue
        if len(text) > 126:
            print(f"record {i}: TOO LONG ({len(text)} chars): {text[:60]}...")
            continue
        kept_records.append(text)
    print(f"kept records: {len(kept_records)}, dropped sentences: {len(dropped_sents)}")
    for i, s, oov in dropped_sents:
        print(f"  dropped r{i}: {s} | OOV: {oov}")
    total_chars = sum(len(t) for t in kept_records)
    print(f"total chars: {total_chars}")
    with open("video_making_records.jsonl", "w", encoding="utf-8") as f:
        for t in kept_records:
            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
    print("wrote video_making_records.jsonl")


if __name__ == "__main__":
    main()
