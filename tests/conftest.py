"""pytest 共享配置:确保项目根在 sys.path(便于 from tests.xxx import)。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
