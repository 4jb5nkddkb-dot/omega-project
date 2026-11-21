# 检查点（checkpoints/）

本目录平时是空的：**检查点文件不进仓库**（`.gitignore` 排除 `*.npz`、`*.pt`）。
大文件只放网盘，仓库里只留名字、下载方式与转换方法。

## 命名

| 模式 | 含义 |
| --- | --- |
| `generalist2_s<N>.npz` | d512 单分支 generalist-v2，第 N 步（N 累计，每块 300 步） |
| `generalist3_d1024_s<N>.npz` | d1024 门控双半脑 generalist-v5，第 N 步 |
| `<stem>_last.npz` / `<stem>_last.pt` | 同名最优/最新检查点的断点续跑文件（参数 + Adam m/v + 随机数状态） |
| `.npz` → `.pt` | 同名转换：`generalist3_d1024_s600.npz` → `generalist3_d1024_s600.pt` |

注意：`.npz` 的 `meta.step` 是**当块步数**（每块 300 步），不是文件名里的
累计步数——`generalist3_d1024_s600.npz` 的 meta.step 是 300，
`extra.schedule_total_steps` 也是当块总数。续跑时按文件名累计步数认进度，
按 meta.step 认块内位置，别混淆。

## 下载（Google Drive）

| 检查点 | 说明 | Drive 文件 ID |
| --- | --- | --- |
| s5100（`generalist2_s5100.npz`） | **当前产品**，d512，loop 12/12 | `1Bl-J410NRRnwXl4AJH-eeScjYx2Hv2PQ` |
| s6000（`generalist2_s6000.npz`） | d512 备份 | `1oNPX3IciUOgg84BO525M4ck9iXHIj6QF` |
| s600（`generalist3_d1024_s600.npz`） | d1024 门控 v5 第 2 块结束 | 旧仓库 `~/workspace/dual_sft/out/` 有本地副本（约 1 GB），未单独传 Drive |

其余历史检查点在旧仓库 `~/workspace/dual_sft/out/`（本地，不进本仓库）。

## 产品规则（swap rule）

- **s5100 是当前产品**，它在旧配置 12 步 loop 里拿了 12/12。
- 只有**新块的 loop 成绩超过 12/12**（严格大于，不是等于）才能替换它。
  s6300（12/12）、s6900（6/12）、s7800（9/12）都没超过，s5100 留任；
  门控 v5 的 s600 门分得开（左 1.0000 / 世界 0.2500）但 loop 只有 3/12，
  也不发货。详见 `evaluation/README.md`。

## .npz → .pt 转换

```bash
# 转换器自包含（numpy + torch，不依赖项目代码），转完带 --verify 核对
# 注意：完整 .pt 约 1.03 GB，/tmp 是 512 MB tmpfs，装不下——
# 完整转换写到 home 磁盘（如 ~/workspace/convert-scratch/）；
# 只转模型参数（--no-optimizer，约 342 MB）可以放 /tmp。
python scripts/convert_npz_to_pt.py \
    ~/workspace/dual_sft/out/generalist3_d1024_s600.npz \
    ~/workspace/convert-scratch/generalist3_d1024_s600.pt --verify
```

转换是 1:1 的：`p_<name>` → `model[<name>]`，`m_/<v_` → `optimizer.m/v`，
`meta_json` → `config/step/extra/rng_state`。大文件
**不要转进 staging 目录**，放 `/tmp`（仅模型版）或 `checkpoints/`（git 忽略）
或 staging 外的 scratch 目录。
转换后的正确性由 `tests/test_parity.py` 验证（PyTorch 与 NumPy
在同一输入上的 logits 差异，期望 1e-5~1e-4 量级）。

**已验证（2026-11-21）**：s600 转换通过——74 个参数、85,542,235 标量、
step=300（当块步数）、优化器 t=300，转换前后全量 sha256 校验和一致
（`06c67809b27a4b87…`）。完整版在 `~/workspace/convert-scratch/generalist3_d1024_s600.pt`
（1,026,635,935 字节，含优化器），模型版在 `/tmp/generalist3_d1024_s600.pt`
（342,253,109 字节）。转换器另有目标磁盘余量预检：余量不足时直接报错，
不再写一半失败（初版曾在 /tmp 写满时于 close 处报 iostream error，
后加预检并修了一个 `param_count` 的 `.size` 方法/属性 bug）。
