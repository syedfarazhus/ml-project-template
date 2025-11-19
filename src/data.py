import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def load_data(path: str) -> pd.DataFrame:
    """Generic CSV loader. Later you'll swap this to your actual dataset."""
    full_path = RAW_DIR / path
    return pd.read_csv(full_path)
