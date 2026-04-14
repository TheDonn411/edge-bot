from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_demo_dataset import main as build_demo_dataset
from demo_app.server import run_server


if __name__ == "__main__":
    build_demo_dataset()
    run_server(
        host=os.environ.get("EDGE_BOT_DEMO_HOST", "127.0.0.1"),
        port=int(os.environ.get("EDGE_BOT_DEMO_PORT", "8787")),
    )
