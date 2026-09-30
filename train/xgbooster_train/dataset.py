"""Deterministic synthetic dataset generation for the risk-score model."""

import numpy as np

CHANNELS = ("web", "app", "api")


def generate(seed: int = 42, n: int = 24000) -> tuple[list[dict], list[int]]:
    """Generate n deterministic (sample, label) pairs via default_rng(seed)."""
    rng = np.random.default_rng(seed)
    amount = np.clip(rng.lognormal(mean=4.0, sigma=1.0, size=n), 1.0, None)
    channel_idx = rng.integers(0, len(CHANNELS), size=n)
    hour = rng.integers(0, 24, size=n)
    is_new = rng.integers(0, 2, size=n)
    txn = rng.poisson(2.0, size=n)
    card_age = rng.integers(0, 2001, size=n)
    noise = rng.normal(0.0, 0.5, size=n)
    uni = rng.random(n)

    channel = np.asarray(CHANNELS, dtype=object)[channel_idx]
    night = ((hour >= 22) | (hour <= 4)).astype(np.float64)
    logit = (
        -4.8
        + 4.8 * is_new
        + 3.4 * (channel == "api")
        + 2.1 * (channel == "app")
        + 2.9 * np.log1p(amount) / 8.0
        + 2.6 * night
        + 0.19 * txn
        - 3.7 * np.minimum(card_age, 730) / 730.0
        + noise
    )
    prob = 1.0 / (1.0 + np.exp(-logit))
    labels = (uni < prob).astype(np.int64)

    samples = [
        {
            "amount": float(amount[i]),
            "channel": str(channel[i]),
            "hour": int(hour[i]),
            "is_new_user": int(is_new[i]),
            "txn_count_24h": int(txn[i]),
            "card_age_days": int(card_age[i]),
        }
        for i in range(n)
    ]
    return samples, labels.tolist()


def split(samples: list, labels: list, n_train: int = 18000, n_valid: int = 2000):
    """Deterministic split: train / tuning-valid / holdout (head/mid/tail)."""
    return (
        (samples[:n_train], labels[:n_train]),
        (samples[n_train:n_train + n_valid], labels[n_train:n_train + n_valid]),
        (samples[n_train + n_valid:], labels[n_train + n_valid:]),
    )
