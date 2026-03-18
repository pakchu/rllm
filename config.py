"""Project-level configuration values."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_SAVE_DIR_PATH = PROJECT_ROOT / "data"
DATA_SAVE_DIR_PATH.mkdir(parents=True, exist_ok=True)
DATA_SAVE_DIR = str(DATA_SAVE_DIR_PATH.resolve())

