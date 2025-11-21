from pathlib import Path
import json

EVAL_DIR = Path(__file__).parent
DATA_DIR = EVAL_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_fixture(name: str):
    path = DATA_DIR / f"{name}.json"
    with path.open() as f:
        return json.load(f)

