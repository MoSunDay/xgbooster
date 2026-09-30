"""Rank-based metrics (tie-aware AUC, KS) and SHAP importance."""

import numpy as np


def _sorted_scores(y, p):
    y_arr = np.asarray(y, dtype=np.int64)
    p_arr = np.asarray(p, dtype=np.float64)
    order = np.argsort(p_arr, kind="mergesort")
    return y_arr[order], p_arr[order]


def auc(y, p) -> float:
    """Tie-aware AUC via the Mann-Whitney rank statistic."""
    y_s, p_s = _sorted_scores(y, p)
    n_pos = int(np.sum(y_s == 1))
    n_neg = int(np.sum(y_s == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    _, inverse, counts = np.unique(p_s, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    starts = ends - counts + 1
    avg_rank = (starts + ends) / 2.0
    ranks = avg_rank[inverse]
    rank_sum_pos = float(np.sum(ranks[y_s == 1]))
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def ks(y, p) -> float:
    """Max |CDF_pos - CDF_neg| evaluated at unique score thresholds."""
    y_s, p_s = _sorted_scores(y, p)
    n_pos = int(np.sum(y_s == 1))
    n_neg = int(np.sum(y_s == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    cdf_pos = np.cumsum(y_s == 1) / n_pos
    cdf_neg = np.cumsum(y_s == 0) / n_neg
    gap = np.abs(cdf_pos - cdf_neg)
    starts = np.unique(p_s, return_index=True)[1]
    ends = np.append(starts[1:] - 1, len(p_s) - 1)
    return float(np.max(gap[ends]))


def shap_importance(booster, dmatrix, schema: list) -> dict:
    """Mean absolute SHAP contribution per feature (bias column dropped)."""
    contribs = np.asarray(booster.predict(dmatrix, pred_contribs=True))
    names = [field["name"] for field in schema]
    mean_abs = np.mean(np.abs(contribs[:, : len(names)]), axis=0)
    return {name: float(value) for name, value in zip(names, mean_abs)}
