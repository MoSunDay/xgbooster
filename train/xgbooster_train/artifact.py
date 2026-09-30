"""Write versioned model artifacts: model.ubj, manifest.json, holdout.csv."""

import csv
import json
import pathlib
from datetime import datetime, timezone

import xgboost as xgb

from xgbooster_train.features import n_features


def _objective_name(booster) -> str:
    config = json.loads(booster.save_config())
    return config["learner"]["objective"]["name"]


def _artifact_dir(models_dir, name: str, version: str) -> pathlib.Path:
    base = pathlib.Path(models_dir) / name / version
    if not base.exists():
        return base
    # 3-digit padding keeps lexicographic order == numeric order up to 999 same-second collisions; the Rust registry compares the suffix numerically regardless.
    suffix = 1
    while True:
        candidate = base.parent / f"{version}-{suffix:03d}"
        if not candidate.exists():
            return candidate
        suffix += 1


def write_artifact(models_dir, name: str, booster, schema: list, metrics: dict,
                   holdout_samples: list, holdout_labels: list,
                   holdout_scores) -> pathlib.Path:
    """Persist one model version directory following the artifact contract."""
    if not name:
        raise ValueError("artifact name must be non-empty")
    now = datetime.now(timezone.utc)
    version = now.strftime("%Y-%m-%dT%H%M%S")
    created_at = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    out_dir = _artifact_dir(models_dir, name, version)
    out_dir.mkdir(parents=True, exist_ok=True)

    booster.save_model(out_dir / "model.ubj")

    manifest = {
        "name": name,
        "version": out_dir.name,
        "created_at": created_at,
        "xgboost_version": xgb.__version__,
        "objective": _objective_name(booster),
        "n_features": n_features(schema),
        "feature_schema": schema,
        "metrics": metrics,
    }
    with open(out_dir / "manifest.json", "w", encoding="ascii") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
        fh.write("\n")

    with open(out_dir / "holdout.csv", "w", encoding="ascii", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow([f["name"] for f in schema] + ["label", "score"])
        for sample, label, score in zip(holdout_samples, holdout_labels,
                                        holdout_scores):
            row = [sample.get(f["name"]) for f in schema]
            writer.writerow(row + [int(label), "%.10g" % float(score)])
    return out_dir
