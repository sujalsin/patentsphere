import json
from pathlib import Path
from scipy.stats import ttest_rel


def paired_t_test(baseline_path: Path, rlaif_path: Path):
    with baseline_path.open() as f:
        baseline = json.load(f)["scores"]
    with rlaif_path.open() as f:
        rlaif = json.load(f)["scores"]

    t_stat, p_val = ttest_rel(baseline, rlaif)
    return {"t_stat": t_stat, "p_value": p_val}

