# 推理侧加固 + 训练侧指标修复

按上线评审结论落地的首轮加固:修复调参泄漏、FFI 生命周期隐患,补齐访问控制、版本校验与缺失特征可观测性。

- 训练侧(train/)
  - 调参与评估同集泄漏修复:`dataset.split` 改为 train(18000)/tuning-valid(2000)/holdout(4000) 三段切分;Optuna 调参与早停只用 tuning-valid,最终 AUC/KS/SHAP 与制品 manifest metrics 全部在从未参与调参的 holdout 上计算,消除乐观偏倚。
  - 版本号改秒级 `%Y-%m-%dT%H%M%S`,同秒冲突回退零填充后缀 `-01..-99`,字典序恒等于时间序。
  - 新增 `train/tests/test_features.py`(纯函数测试,无 class,无 pytest 依赖):bool→0/1、null→NaN、未知类别→NaN、类型错误、matrix 形状/dtype 等 13 项。
- 推理侧(infer/)
  - FFI 生命周期:`XgbBooster`/`XgbDMatrix` 持有 `Arc<Lib>` 取代裸 free 函数指针,Drop 经存活库调用释放,类型级消除悬垂指针可能。
  - xgboost 版本门禁:`load_registry` 加载每个制品前比对 `manifest.xgboost_version` 与运行库 `XGBoostVersion`,不一致即拒绝加载并输出告警。
  - 静默 NaN 消除:`/predict` 响应新增 `missing_features`/`unknown_categories`/`unexpected_fields`(空则省略),服务端同步 warn 日志——特征名打错(如 `Amount`)现在可见。
  - 访问控制:`XGBOOSTER_ADMIN_TOKEN` 环境变量启用后 `/admin/reload` 要求 Bearer 或 `X-Admin-Token`(常数时间比较);未设 token 且绑定非回环地址时拒绝启动。
  - 状态码语义:非法模型引用(如 `risk_score@`)由 404 改为 400,未知模型仍 404。
  - latest 选取:`pick_latest` 改数值感知比较(`-10` > `-9`),不依赖后缀零填充假设。
- 验证:`cargo test` 26 项全过;重训制品 `risk_score@2026-09-30T154712`(holdout AUC 0.9350 / KS 0.7286);`bash tests/consistency/run.sh` SMOKE OK(含 400/404/401 负向路径与 missing_features 断言,对拍 diff 3.7e-11)。
