"""pytest bootstrap：项目根目录进 sys.path，保证 `from tokenizer import …`、
`from model import …` 等绝对导入可用（CONTRACT.md 约定）。"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
