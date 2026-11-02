import yaml
from pathlib import Path

def load_config(path: str | Path = "config.yaml") -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)
