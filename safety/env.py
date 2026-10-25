"""Zero-dependency .env loader for CompanionAI.

Loads key-value pairs from .env into os.environ without requiring python-dotenv.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union


def _find_env_file() -> Optional[Path]:
    """Find .env file in cwd or project root (parent of safety/)."""
    # Check cwd
    cwd_env = Path.cwd() / ".env"
    if cwd_env.is_file():
        return cwd_env

    # Check project root relative to this file
    root_env = Path(__file__).resolve().parent.parent / ".env"
    if root_env.is_file():
        return root_env

    return None


def parse_env_line(line: str) -> Optional[tuple[str, str]]:
    """Parse a single line from a .env file.
    
    Supports:
      KEY=VALUE
      export KEY=VALUE
      Quoted values: KEY="hello world" or KEY='hello'
      Comments: # comment or KEY=val # comment (for unquoted)
    """
    line = line.strip()
    if not line or line.startswith("#"):
        return None

    if line.startswith("export "):
        line = line[len("export "):].strip()

    if "=" not in line:
        return None

    key, val = line.split("=", 1)
    key = key.strip()
    val = val.strip()

    if not key:
        return None

    # Handle quotes and inline comments
    if val and val[0] in ('"', "'"):
        quote = val[0]
        r_idx = val.rfind(quote)
        if r_idx > 0:
            remainder = val[r_idx + 1:].strip()
            if remainder == "" or remainder.startswith("#"):
                val = val[1:r_idx]
            elif " #" in val:
                val = val.split(" #", 1)[0].rstrip()
        elif " #" in val:
            val = val.split(" #", 1)[0].rstrip()
    else:
        # Strip inline comments for unquoted values
        if " #" in val:
            val = val.split(" #", 1)[0].rstrip()

    return key, val


def load_env(env_path: Optional[Union[str, Path]] = None, override: bool = False) -> bool:
    """Loads key-value pairs from .env into os.environ if not already set.
    
    Returns True if .env was found and loaded, False otherwise.
    """
    if env_path is not None:
        target_path = Path(env_path)
    else:
        target_path = _find_env_file()

    if target_path is None or not target_path.is_file():
        return False

    try:
        with open(target_path, "r", encoding="utf-8-sig") as f:
            for line in f:
                parsed = parse_env_line(line)
                if parsed:
                    k, v = parsed
                    if override or k not in os.environ:
                        os.environ[k] = v
        return True
    except Exception:
        return False


# Auto-load on module import
load_env()
