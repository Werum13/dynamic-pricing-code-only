from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Elasticity.item_price_analyzer import analyze_item, main  # noqa: F401,E402

__all__ = ["analyze_item"]


if __name__ == "__main__":
    main()
