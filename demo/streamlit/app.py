"""Streamlit App Entrypoint for CompanionAI Conversation Practice Demo.

Notice: CompanionAI is a non-clinical AI conversation practice tool, not a human or therapist.
This is an educational practice demo — please use fictional/made-up messages only.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
ST_DIR = Path(__file__).parent
if str(ST_DIR) not in sys.path:
    sys.path.insert(0, str(ST_DIR))

from safety_panel import main, render_panel, render_safety_panel

__all__ = ["main", "render_safety_panel", "render_panel"]

if __name__ == "__main__":
    main()
