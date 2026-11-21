"""通用模型（generalist）语料域与分层采样（PyTorch 版）。

域与配比（v2，2026-11-13 起）：
    RATIOS = 经典 0.25 / 代码 0.20 / 对话 0.10 / 常识 0.15 / 推理 0.15 / 世界 0.15

数据来源（见 ~/workspace/dual_sft/GENERALIST_CORPUS.md）：
- 经典：西游记全文 + 《儒林外史》（Gutenberg #24032 繁体本经 OpenCC t2s
  转简体，均按 4622 词表整句过滤）。
- 代码：仓库 .py 文件（排除 __pycache__/archive/legacy）。
- 对话：dialogue_pilot.txt。
- 推理：现生成的中文算术/抄写/记忆格式（reasoning_text，种子固定，
  与协议记录同分布）。
- 常识：knowledge_wiki.txt（维基百科导言，CC BY-SA 4.0）+
  knowledge_youxue.txt（《幼学琼林》Gutenberg 公版），均已按词表整句过滤。
- 世界：world .jsonl 记录文件的 "text" 字段。
  明确不含：grok 包、用户粘贴的手机对话、旧检查点。

最小对（minimal-pair）记录（2026-11-20，build_minimal_pairs.py 生成，
针对性修“读可变字段并重复”：同一底本只改一个字段——地点/随身/昼夜，
答案随之变，模型若输出固定答案，7 个变体里 6 个错）：
- probe_format_records.jsonl：120 条；register_format_records.jsonl：70 条；
- carried_register_format_records.jsonl：30 条；
  daytime_register_format_records.jsonl：20 条；合计 240 条最小对记录；
  另有 character_register_format_records.jsonl（人物寄存器 150 条）与
  video_making_records.jsonl（做视频经验 52 条，2,512 字，纯文字）同属
  世界域。全部记录过词表检查（0 OOV）。

重要：语料在每次启动/断点续跑时都从磁盘文件重新构建（train_steps
--resume 会重建语料），所以启动后改语料文件会在续跑的尾段生效，
不只从下一块起生效（2026-11-14 第 13 块续跑实证）。

路径：默认用新布局（项目根目录 data/raw/ + data/processed/）；也可传
显式 paths，把训练器指向过渡期的 live 语料（~/workspace/dual_sft
+ ~/workspace/omega-chat），见 live_corpus_paths()。data/ 下文件
缺失时 get_dataloader/domain_texts 自动回落到 live 语料并打印说明。

PyTorch 化说明（与 NumPy 版 staging 文件的差别）：
- 语料加载/编码仍用 NumPy + 纯 Python（分词器是 NumPy 的，数据准备
  允许用 NumPy；合同：NumPy 只用于数据准备/分词，不用于模型）。
- domain_texts 返回 {域名: torch.LongTensor}；make_batch_fn 返回
  (inputs, targets, gate_targets) 三个 torch 张量（long/long/float）。
- 分层采样数学与 NumPy 版逐行一致：_batch_counts 最大余数法，
  每批各域条数固定，每域每批至少 1 条（batch>=6 时），域内窗口随机，
  批内顺序打乱。GeneralistDataset + StratifiedBatchSampler 把同一
  采样放进 torch DataLoader；训练器默认走 make_batch_fn（步数制，
  与 NumPy train_steps 的调用方式一致）。
"""
import glob
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

RATIOS = [("经典", 0.25), ("代码", 0.20), ("对话", 0.10),
          ("常识", 0.15), ("推理", 0.15), ("世界", 0.15)]  # v2（2026-11-13）

# 门控分流（2026-11-16，用户：双半脑，门按内容动态定百分比）：
# 左支吃推理类域，右支吃世界域；门的监督目标 t=1 走左，t=0 走右。
# 世界域只占 15%，右支吃得少——先跑起来看门分不分得开，不够再提配比。
GATE_LEFT_DOMAINS = ("经典", "代码", "对话", "常识", "推理")
GATE_RIGHT_DOMAINS = ("世界",)

# 世界域 .jsonl 记录文件（按 generalist_pilot_train.py 的加载顺序）。
WORLD_RECORD_FILES = (
    "world_tool_records_v9.jsonl",
    "self_report_records_v9.jsonl",
    "probe_format_records.jsonl",
    "register_format_records.jsonl",
    "carried_register_format_records.jsonl",
    "daytime_register_format_records.jsonl",
    "character_register_format_records.jsonl",
    "video_making_records.jsonl",
)

_ZH = "零一二三四五六七八九十"

__all__ = [
    "RATIOS", "GATE_LEFT_DOMAINS", "GATE_RIGHT_DOMAINS", "WORLD_RECORD_FILES",
    "project_root", "default_paths", "live_corpus_paths", "zh", "reasoning_text",
    "domain_texts", "make_batch_fn", "GeneralistDataset",
    "StratifiedBatchSampler", "get_dataloader",
]


def project_root():
    """项目根目录（本文件上两级：training/ -> llm-project/）。"""
    return Path(__file__).resolve().parent.parent


def default_paths(root=None):
    """新布局下的默认语料路径：域 -> 文件列表（代码域为 glob 模式）。

    经典：data/raw/Xiyouji.txt + data/processed/classics_rulin.txt
    代码：项目根目录下全部 .py（排除 __pycache__/archive/legacy）
    对话：data/processed/dialogue_pilot.txt
    常识：data/processed/knowledge_wiki.txt + knowledge_youxue.txt
    世界：data/processed/ 下的 WORLD_RECORD_FILES（.jsonl，取 "text" 字段）
    推理：现生成，不读文件。
    """
    root = Path(root) if root is not None else project_root()
    raw, proc = root / "data" / "raw", root / "data" / "processed"
    return {
        "经典": [raw / "Xiyouji.txt", proc / "classics_rulin.txt"],
        "代码": [str(root / "**" / "*.py")],
        "对话": [proc / "dialogue_pilot.txt"],
        "常识": [proc / "knowledge_wiki.txt", proc / "knowledge_youxue.txt"],
        "世界": [proc / name for name in WORLD_RECORD_FILES],
    }


def live_corpus_paths():
    """过渡期 live 语料路径（~/workspace/dual_sft + ~/workspace/omega-chat），
    与 generalist_pilot_train.py 的加载位置完全一致。训练器在切到新
    data/ 布局之前用这个。"""
    dual_sft = Path.home() / "workspace" / "dual_sft"
    mirror = Path.home() / "workspace" / "omega-chat"
    corpus = dual_sft / "generalist_corpus"
    return {
        "经典": [mirror / "Xiyouji.txt", corpus / "classics_rulin.txt"],
        "代码": [str(mirror / "**" / "*.py")],
        "对话": [corpus / "dialogue_pilot.txt"],
        "常识": [corpus / "knowledge_wiki.txt", corpus / "knowledge_youxue.txt"],
        "世界": [dual_sft / name for name in WORLD_RECORD_FILES],
    }


def _paths_available(paths):
    """default_paths() 指向的文件是否都在磁盘上（glob 模式算命中文件数）。"""
    for entries in paths.values():
        for entry in entries:
            entry = str(entry)
            if any(ch in entry for ch in "*?["):
                if not glob.glob(entry, recursive=True):
                    return False
            elif not Path(entry).exists():
                return False
    return True


def resolve_paths(paths=None, root=None):
    """选一套可用的语料路径：显式 paths > data/ 新布局 > live 语料。

    data/ 布局缺文件时回落到 live_corpus_paths() 并打印一行说明
    （Agent D 还在搬 data/，过渡期训练不断档）。
    """
    if paths is not None:
        return dict(paths)
    default = default_paths(root)
    if _paths_available(default):
        return default
    print("data/ 语料不完整，回落到 live 语料 "
          "(~/workspace/dual_sft + ~/workspace/omega-chat)")
    return live_corpus_paths()


def zh(n):
    """0-99 整数 -> 中文读法（与协议记录同款，不用阿拉伯数字）。"""
    if n < 10:
        return _ZH[n]
    tens, ones = divmod(n, 10)
    head = "十" if tens == 1 else _ZH[tens] + "十"
    return head + (_ZH[ones] if ones else "")


def reasoning_text(seed=7):
    """中文算术/抄写/记忆格式文本（现生成，种子固定，与协议同分布）。"""
    rng = np.random.default_rng(seed)
    lines = []
    for _ in range(400):
        a, b = int(rng.integers(1, 20)), int(rng.integers(1, 20))
        lines.append(f"问：{zh(a)}加{zh(b)}等于几？答案是{zh(a + b)}。")
    for _ in range(200):
        a, b = int(rng.integers(2, 10)), int(rng.integers(2, 10))
        lines.append(f"问：{zh(a)}乘{zh(b)}等于几？答案是{zh(a * b)}。")
    for _ in range(300):  # 减法（2026-11-12 补）：被减数 1..19，减数 ≤ 被减数，结果非负
        a = int(rng.integers(1, 20))
        b = int(rng.integers(1, a + 1))
        lines.append(f"问：{zh(a)}减{zh(b)}等于几？答案是{zh(a - b)}。")
    for _ in range(200):  # 除法（2026-11-12 补）：整除，商 1..9
        b = int(rng.integers(2, 10))
        k = int(rng.integers(1, 10))
        lines.append(f"问：{zh(b * k)}除以{zh(b)}等于几？答案是{zh(k)}。")
    for _ in range(200):
        w = rng.choice(["苹果", "花果山", "金箍棒", "蟠桃", "南海", "取经路"])
        lines.append(f"记住这个词：{w}。")
        lines.append(f"刚才让你记住什么。{w}。")
    return "\n".join(lines) + "\n"


def _read_text_files(entries):
    """读一批文本文件/ glob 模式，拼成一个字符串。"""
    parts = []
    for entry in entries:
        entry = str(entry)
        if any(ch in entry for ch in "*?["):
            files = sorted(glob.glob(entry, recursive=True))
        else:
            files = [entry]
        for f in files:
            p = Path(f)
            if "__pycache__" in p.parts or "/archive/" in str(p) or "/legacy/" in str(p):
                continue
            parts.append(p.read_text(encoding="utf-8"))
    return "".join(parts)


def _read_world_records(entries):
    """读一批世界域 .jsonl 记录文件，取每条的 "text" 字段拼成文本。"""
    world = []
    for entry in entries:
        with open(entry, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    world.append(json.loads(line)["text"])
    return "\n".join(world)


def domain_texts(tok, paths=None, root=None):
    """把 6 个域的文本编码成 token id 张量：{域名: torch.LongTensor}。

    paths: 域 -> 文件/glob 列表，缺省用 resolve_paths()（先 data/ 布局，
    缺文件回落 live 语料）。每个域至少 200 token，否则报错。每次调用
    都从磁盘重读（启动/续跑重建语料，见模块 docstring）。
    """
    paths = resolve_paths(paths, root)
    texts = {}
    texts["经典"] = _read_text_files(paths["经典"])
    texts["代码"] = _read_text_files(paths["代码"])
    texts["对话"] = _read_text_files(paths["对话"])
    texts["推理"] = reasoning_text()
    texts["常识"] = _read_text_files(paths["常识"])
    texts["世界"] = _read_world_records(paths["世界"])
    ids = {}
    for name, text in texts.items():
        arr = np.asarray(tok.encode(text), dtype=np.int64)
        assert len(arr) > 200, f"domain {name} too short: {len(arr)}"
        ids[name] = torch.from_numpy(arr)
        print(f"domain {name}: {len(text)} chars -> {len(arr)} tokens")
    return ids


def _batch_counts(weights, batch_size):
    """Largest-remainder counts: every domain appears in every batch in
    proportion (min 1 when batch_size >= n_domains). Fixes the old
    multinomial draw, where a 20%-weight domain could get 0 or 3 rows
    in an 8-row batch by chance.（与 NumPy 版逐行一致。）"""
    raw = np.asarray(weights, dtype=np.float64) * batch_size
    counts = np.floor(raw).astype(int)
    remainder = int(batch_size - counts.sum())
    order = np.argsort(-(raw - counts))
    for i in order[:remainder]:
        counts[i] += 1
    return counts


def _domain_gate_value(name):
    return 1.0 if name in GATE_LEFT_DOMAINS else 0.0


def make_batch_fn(ids, seq_len, batch_size=8, seed=7):
    """按 RATIOS 分层采样一个 batch_fn(rng) -> (inputs, targets, gate_targets)。

    每批各域条数固定（最大余数法，每域每批至少 1 条，batch>=6 时），
    域内窗口随机，批内顺序打乱。第三返回值 gate 是门的监督目标
    (B, S)：左支域 1.0，世界域 0.0；train_steps 只在 merge=gated 时
    把它传给 loss。注意：训练器传进来的 r 就是它自己的 rng；这里
    用独立 rng 抽域，保证配比不受训练 rng 影响。

    与 NumPy 版同一签名、同一采样数学；返回值改成 torch 张量
    （inputs/targets long，gate_targets float32），直接喂模型。
    """
    names = [n for n, _ in RATIOS]
    weights = np.asarray([w for _, w in RATIOS], dtype=np.float64)
    weights = weights / weights.sum()
    rng = np.random.default_rng(seed)
    counts = _batch_counts(weights, batch_size)

    def batch_fn(r=None):
        inputs, targets, gate = [], [], []
        for d, c in enumerate(counts):
            arr = ids[names[d]]
            t = _domain_gate_value(names[d])
            for _ in range(int(c)):
                s = int(rng.integers(0, len(arr) - seq_len - 1))
                inputs.append(torch.as_tensor(arr[s:s + seq_len], dtype=torch.long))
                targets.append(torch.as_tensor(arr[s + 1:s + seq_len + 1], dtype=torch.long))
                gate.append(torch.full((seq_len,), t, dtype=torch.float32))
        order = rng.permutation(len(inputs))
        return (torch.stack([inputs[i] for i in order]),
                torch.stack([targets[i] for i in order]),
                torch.stack([gate[i] for i in order]))

    return batch_fn


def _encode_index(domain_idx, start):
    """把 (域下标, 窗口起点) 编成一个 int，供 Sampler/Dataset 之间传递。"""
    return (int(domain_idx) << 32) | int(start)


def _decode_index(index):
    return int(index) >> 32, int(index) & 0xFFFFFFFF


class GeneralistDataset(Dataset):
    """一域一窗口的序列数据集：__getitem__ 取一个编码下标。

    下标由 StratifiedBatchSampler 产生（域下标 << 32 | 窗口起点），
    所以分层配比由 sampler 保证，dataset 本身只负责切窗口。
    __getitem__ 返回 (input, target, gate_target)，形状
    (S,) / (S,) / (S,)——input/target long，gate float32。
    """

    def __init__(self, ids, seq_len, seed=7):
        self.names = [n for n, _ in RATIOS]
        self.seq_len = int(seq_len)
        self.seed = seed
        self.ids = {}
        for name in self.names:
            arr = torch.as_tensor(ids[name], dtype=torch.long)
            if len(arr) < seq_len + 2:
                raise ValueError(f"domain {name} too short for seq_len {seq_len}")
            self.ids[name] = arr
        # 各域可用窗口数（分层 sampler 用它定每批的随机起点范围）。
        self.windows = {name: len(arr) - seq_len - 1 for name, arr in self.ids.items()}

    def __len__(self):
        return int(sum(self.windows.values()))

    def __getitem__(self, index):
        d, s = _decode_index(index)
        name = self.names[d]
        arr = self.ids[name]
        inputs = arr[s:s + self.seq_len]
        targets = arr[s + 1:s + self.seq_len + 1]
        gate = torch.full((self.seq_len,), _domain_gate_value(name),
                          dtype=torch.float32)
        return inputs, targets, gate


class StratifiedBatchSampler(Sampler):
    """按 RATIOS 分层抽批的 sampler：每批各域条数固定（最大余数法）。

    每批先按 _batch_counts 定各域条数，再在各域内随机取窗口起点，
    最后把批内顺序打乱——与 make_batch_fn 同一采样数学，只是
    以 DataLoader batch_sampler 的形式实现。num_batches 是每轮
    的批数（训练器步数制时一般不用 DataLoader，见 get_dataloader）。
    """

    def __init__(self, dataset, batch_size=8, seed=7, num_batches=100):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.num_batches = int(num_batches)
        names = dataset.names
        weights = np.asarray([w for _, w in RATIOS], dtype=np.float64)
        weights = weights / weights.sum()
        self.counts = _batch_counts(weights, self.batch_size)
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        for _ in range(self.num_batches):
            batch = []
            for d, c in enumerate(self.counts):
                name = self.dataset.names[d]
                for _ in range(int(c)):
                    s = int(self.rng.integers(0, self.dataset.windows[name]))
                    batch.append(_encode_index(d, s))
            order = self.rng.permutation(len(batch))
            yield [batch[i] for i in order]


def _collate(batch):
    inputs, targets, gate = zip(*batch)
    return (torch.stack(list(inputs)), torch.stack(list(targets)),
            torch.stack(list(gate)))


def _load_default_tokenizer():
    """从 data/processed/generalist_vocab.json（回落 live 版）建 CharTokenizer。"""
    from tokenizer import CharTokenizer
    candidates = [
        project_root() / "data" / "processed" / "generalist_vocab.json",
        Path.home() / "workspace" / "dual_sft" / "generalist_vocab.json",
    ]
    for path in candidates:
        if path.exists():
            meta = json.loads(path.read_text(encoding="utf-8"))
            assert meta["type"] == "char"
            return CharTokenizer(meta["chars"])
    raise FileNotFoundError(
        "generalist_vocab.json not found in data/processed/ or ~/workspace/dual_sft/")


def get_dataloader(ids=None, tok=None, paths=None, root=None, seq_len=128,
                   batch_size=8, seed=7, num_workers=0, num_batches=100,
                   pin_memory=False):
    """建一个分层采样的 DataLoader，每批返回 (inputs, targets, gate_targets)。

    ids 不给时用 tok（不给则自动加载默认词表）+ resolve_paths() 从磁盘
    建语料。num_batches 是每轮的批数——训练器是步数制（train() 默认
    走 make_batch_fn），DataLoader 主要供评测/抽查与 epoch 制调用；
    跑满 300 步请传 num_batches=300 或直接循环 dataloader。
    num_workers>0 时 dataset 的张量在子进程间共享（只读），安全。
    """
    if ids is None:
        if tok is None:
            tok = _load_default_tokenizer()
        ids = domain_texts(tok, paths=paths, root=root)
    dataset = GeneralistDataset(ids, seq_len=seq_len, seed=seed)
    sampler = StratifiedBatchSampler(dataset, batch_size=batch_size,
                                     seed=seed, num_batches=num_batches)
    return DataLoader(dataset, batch_sampler=sampler, collate_fn=_collate,
                      num_workers=num_workers, pin_memory=pin_memory)
