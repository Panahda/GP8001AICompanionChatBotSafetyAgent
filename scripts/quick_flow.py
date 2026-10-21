"""Same as `python scripts/run_flow.py --mode quick` (kept so older instructions still work).

    python scripts/quick_flow.py [--no-generate] [--provider mock] [--model <id>] ...
"""
import sys

import run_flow

if __name__ == "__main__":
    sys.argv[1:1] = ["--mode", "quick"]
    run_flow.main()
