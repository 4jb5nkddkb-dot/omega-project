"""评估包：因果探针、门控探针、冻结循环、Q0/Q1/Q2/Q3 探针集。

共用件：world_state（冻结的世界状态机副本）、scoring（判卷）、
sidecar（PyTorch 版 Sidecar）、_common（检查点/词表加载）。
各探针都可作脚本跑：python -m evaluation.<probe> --checkpoint <ckpt.pt>
"""
from evaluation.scoring import QUESTIONS, score

__all__ = ["QUESTIONS", "score"]
