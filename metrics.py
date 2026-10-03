"""Metric helpers including TCE and convergence rounds."""

import numpy as np
from typing import List, Dict


def trust_calibration_error(R_history: List[np.ndarray],
                            malicious: List[int]) -> float:
    """TCE = (1/TN) sum_t sum_i |R_i^t - 1[i not in M]|."""
    if not R_history:
        return float("nan")
    malicious_set = set(malicious)
    total = 0.0
    count = 0
    for R in R_history:
        R = np.asarray(R)
        target = np.array([0.0 if i in malicious_set else 1.0
                           for i in range(R.shape[0])])
        total += np.abs(R - target).sum()
        count += R.shape[0]
    return total / max(count, 1)


def convergence_round(ga_history: List[float], threshold: float = 0.95) -> int:
    """First round at which GA >= threshold * final_GA."""
    if not ga_history:
        return 0
    final = ga_history[-1]
    target = threshold * final
    for t, ga in enumerate(ga_history, start=1):
        if ga >= target:
            return t
    return len(ga_history)


def summarize(results: List[Dict]) -> Dict:
    """Aggregate metrics over seeds: mean, std, 95% t-CI half-width."""
    import math
    keys = ["final_ga", "final_asr", "final_fpr", "final_tpr"]
    n = len(results)
    out = {}
    for k in keys:
        vals = np.array([r[k] for r in results], dtype=float)
        mean = vals.mean()
        std = vals.std(ddof=1) if n > 1 else 0.0
        # t_{0.975, n-1}
        t_table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
                   6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}
        t_val = t_table.get(n - 1, 1.96)
        half = t_val * std / math.sqrt(n) if n > 1 else 0.0
        out[k] = {"mean": mean, "std": std, "ci95": half, "values": vals.tolist()}
    return out
