# XGBoost 训练→制品→Rust FFI 推理全链路 MVP

首个代码变更:落地 Python 训练 + Rust 推理(xgboost==3.4.1)最小闭环。

- train/(Python,纯函数):`features.py` schema 驱动特征向量化 → `dataset.py` 合成数据(seed 固定) → `train.py`(Optuna TPE 调参 + 早停 + best_iteration 切片) → `evaluate.py`(AUC/KS/SHAP) → `artifact.py` 产出制品 `models/risk_score/<version>/{model.ubj, manifest.json, holdout.csv}`。
- infer/(Rust,纯函数式,unsafe 全部隔离在 ffi.rs 的 Drop 守卫内):libloading 运行时加载 `infer/lib/libxgboost.so`(从 pinned pip wheel 提取);registry 双层扫描多模型多版本、`name@version` 解析(latest 取字典序最大);axum HTTP:`GET /models` / `POST /predict` / `POST /admin/reload`(整体换新 Registry 原子热替换)。
- 一致性:holdout 4000 行 Python↔Rust 逐行对拍 max_abs_diff ≈ 5e-11(容差 1e-6);release 单次推理 ~157µs。
- 验证:`bash tests/consistency/run.sh` 端到端(构建→对拍→HTTP 冒烟)SMOKE OK。

已知取舍:制品目录即版本目录(时间戳可排序);未知类别按缺失(NaN)处理;v1 每模型 Mutex 串行化 predict。
