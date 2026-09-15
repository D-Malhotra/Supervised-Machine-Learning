"""A small results store shared across notebooks.

The original notebook accumulated `MSE_list_models`, `R_square_list_models` and
`model_list` as globals, which only works while everything lives in one file.
Splitting the work across notebooks means the comparison table needs somewhere
durable to read from, so each notebook writes its scores to results/metrics.csv
and the comparison notebook reads them back.
"""

from pathlib import Path

import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_PATH = RESULTS_DIR / "metrics.csv"

KEY_COLUMNS = ["model", "split"]


def load_results():
    """Return the stored results, or an empty frame if none exist yet."""
    if not RESULTS_PATH.exists():
        return pd.DataFrame(columns=KEY_COLUMNS)

    return pd.read_csv(RESULTS_PATH)


def record_result(model, split, **metrics):
    """Record one model/split score, replacing any previous entry for that key.

    Replacing rather than appending keeps the file stable when a notebook is
    re-run, so the comparison table never shows duplicates.
    """
    RESULTS_DIR.mkdir(exist_ok=True)

    results = load_results()
    row = {"model": model, "split": split, **metrics}

    if not results.empty:
        already_present = (results["model"] == model) & (results["split"] == split)
        results = results[~already_present]

    results = pd.concat([results, pd.DataFrame([row])], ignore_index=True)
    results = results.sort_values(KEY_COLUMNS).reset_index(drop=True)
    results.to_csv(RESULTS_PATH, index=False)

    return results


def clear_results():
    """Delete the stored results, for a clean re-run from notebook 01."""
    if RESULTS_PATH.exists():
        RESULTS_PATH.unlink()
