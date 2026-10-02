"""End-to-end pipeline: load -> split -> tune -> train -> evaluate -> write."""

import argparse
import pathlib

import numpy as np
import xgboost as xgb

from xgbooster_train import adult, artifact, dataset, evaluate, features

MODEL_NAME = "risk_score"
ADULT_MODEL_NAME = "adult_income"
ADULT_N_TRAIN = 28000
ADULT_N_VALID = 4561
NUM_BOOST_ROUND = 2000
TUNE_NUM_BOOST_ROUND = 300
EARLY_STOP_ROUNDS = 30

DEFAULT_PARAMS = {
    "max_depth": 6,
    "eta": 0.1,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "min_child_weight": 2,
    "reg_lambda": 1.0,
}


def _base_params(seed: int) -> dict:
    return {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "seed": seed,
    }


def tune_hyperparams(x_tr, y_tr, x_va, y_va, n_trials: int, seed: int) -> dict:
    """Optuna TPE search maximizing validation AUC; defaults when skipped."""
    if n_trials <= 0:
        return dict(DEFAULT_PARAMS)
    try:
        import optuna
    except ImportError:
        return dict(DEFAULT_PARAMS)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    dtrain = xgb.DMatrix(x_tr, label=y_tr)
    dvalid = xgb.DMatrix(x_va, label=y_va)

    def objective(trial):
        params = _base_params(seed)
        params.update({
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "eta": trial.suggest_float("eta", 0.02, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 8),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        })
        callbacks = [xgb.callback.EarlyStopping(rounds=EARLY_STOP_ROUNDS,
                                                save_best=True)]
        bst = xgb.train(params, dtrain, num_boost_round=TUNE_NUM_BOOST_ROUND,
                        evals=[(dvalid, "valid")], callbacks=callbacks)
        preds = bst.predict(dvalid,
                            iteration_range=(0, bst.best_iteration + 1))
        return evaluate.auc(y_va, preds)

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)
    best = study.best_params
    return {
        "max_depth": int(best["max_depth"]),
        "eta": float(best["eta"]),
        "subsample": float(best["subsample"]),
        "colsample_bytree": float(best["colsample_bytree"]),
        "min_child_weight": int(best["min_child_weight"]),
        "reg_lambda": float(best["reg_lambda"]),
    }


def train_final(x_tr, y_tr, x_va, y_va, params: dict, seed: int):
    """Train on the train rows with the tuning validation set as eval set."""
    full = _base_params(seed)
    full.update(params)
    dtrain = xgb.DMatrix(x_tr, label=y_tr)
    dvalid = xgb.DMatrix(x_va, label=y_va)
    callbacks = [xgb.callback.EarlyStopping(rounds=EARLY_STOP_ROUNDS,
                                            save_best=True)]
    bst = xgb.train(full, dtrain, num_boost_round=NUM_BOOST_ROUND,
                    evals=[(dvalid, "valid")], callbacks=callbacks)
    best_iteration = int(bst.best_iteration)
    bst = bst[: best_iteration + 1]
    return bst, best_iteration


def _require_file(path: pathlib.Path) -> pathlib.Path:
    if path.is_file():
        return path
    raise SystemExit(
        f"missing dataset file: {path}\n"
        f"download it from {adult.ADULT_URL} (gzip it in place), e.g.\n"
        f"  curl --proxy socks5h://127.0.0.1:1080 -o {path} "
        f"{adult.ADULT_URL}{path.name}"
    )


def _load_synthetic(seed: int):
    """Assemble the synthetic dataset: schema + train/valid/holdout splits."""
    schema = features.FEATURE_SCHEMA
    samples, labels = dataset.generate(seed=seed)
    (tr, va, ho) = dataset.split(samples, labels)
    return MODEL_NAME, schema, tr, va, ho


def _load_adult(datasets_dir: str):
    """Assemble the Adult dataset: train/valid from adult.data, holdout =
    the full independent adult.test file (no leakage into tuning)."""
    base = pathlib.Path(datasets_dir) / "adult"
    samples, labels = adult.load(_require_file(base / "adult.data.gz"))
    holdout = adult.load(_require_file(base / "adult.test.gz"))
    # 32561 rows total: head 28000 train, mid 4561 valid, empty tail.
    (tr, va, tail) = dataset.split(samples, labels,
                                   n_train=ADULT_N_TRAIN, n_valid=ADULT_N_VALID)
    if tail[0]:
        raise SystemExit(
            f"adult.data has {len(samples)} rows; expected exactly "
            f"{ADULT_N_TRAIN + ADULT_N_VALID} so the tail split is empty")
    # Schema derives from train rows only; unseen categories -> NaN.
    return ADULT_MODEL_NAME, adult.build_schema(tr[0]), tr, va, holdout


def _load_dataset(kind: str, datasets_dir: str, seed: int):
    """Dispatch on --dataset; returns (name, schema, train, valid, holdout)."""
    if kind == "adult":
        return _load_adult(datasets_dir)
    return _load_synthetic(seed)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train an XGBoost model and write a versioned artifact.")
    parser.add_argument("--models-dir", default="models",
                        help="root directory for model artifacts")
    parser.add_argument("--dataset", choices=("synthetic", "adult"),
                        default="synthetic",
                        help="dataset to train on (synthetic risk_score or "
                             "UCI Adult adult_income)")
    parser.add_argument("--datasets-dir", default="datasets",
                        help="root directory holding adult/*.gz datasets")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trials", type=int, default=12,
                        help="optuna trials (0 skips tuning)")
    args = parser.parse_args()

    name, schema, tr, va, ho = _load_dataset(args.dataset, args.datasets_dir,
                                             args.seed)
    (tr_samples, tr_labels), (va_samples, va_labels) = tr, va
    (ho_samples, ho_labels) = ho
    print(f"dataset: {args.dataset} (model {name})")
    x_tr = features.to_matrix(tr_samples, schema)
    x_va = features.to_matrix(va_samples, schema)
    x_ho = features.to_matrix(ho_samples, schema)
    y_tr = np.asarray(tr_labels, dtype=np.int64)
    y_va = np.asarray(va_labels, dtype=np.int64)
    y_ho = np.asarray(ho_labels, dtype=np.int64)
    print(f"train rows: {len(tr_samples)}, valid rows: {len(va_samples)}, "
          f"holdout rows: {len(ho_samples)}")

    tuned = tune_hyperparams(x_tr, y_tr, x_va, y_va, args.trials, args.seed)
    print(f"params: {tuned}")

    bst, best_iteration = train_final(x_tr, y_tr, x_va, y_va, tuned,
                                      args.seed)
    print(f"best_iteration: {best_iteration}")
    print(f"rounds kept: {bst.num_boosted_rounds()}")

    dhold = xgb.DMatrix(x_ho, label=y_ho)
    scores = bst.predict(dhold)
    hold_auc = evaluate.auc(y_ho, scores)
    hold_ks = evaluate.ks(y_ho, scores)
    print(f"holdout AUC: {hold_auc:.4f}")
    print(f"holdout KS: {hold_ks:.4f}")

    importance = evaluate.shap_importance(bst, dhold, schema)
    ranked = sorted(importance.items(), key=lambda kv: kv[1], reverse=True)
    for rank, (fname, value) in enumerate(ranked, start=1):
        print(f"shap[{rank}] {fname}: {value:.4f}")

    metrics = {"auc": float(hold_auc), "ks": float(hold_ks),
               "n_holdout": len(ho_samples)}
    out_dir = artifact.write_artifact(args.models_dir, name, bst, schema,
                                      metrics, ho_samples, ho_labels, scores)
    print(f"artifact: {out_dir}")


if __name__ == "__main__":
    main()
