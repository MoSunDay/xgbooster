# features

Commit: f2e6dee

## 能力组

- XGBoost 结构化判断/打分(MVP,已实现):输入结构化特征 → 输出风险分。训练侧 Python 产出制品,Rust 推理侧加载制品提供 HTTP 服务。训练三段切分(调参/早停与 holdout 分离),制品 metrics 为无泄漏 holdout 指标;推理侧带 admin 鉴权、xgboost 版本门禁与缺失特征上报;/predict 限流+并发上限(429+Retry-After)、缺失特征 warn 日志按 (模型,异常签名) 60s 限频、XGBOOSTER_STRICT_VERSION 严格版本门禁、非回环启动 guard 同时要求限流配置。
  - 训练入口(repo 根目录):`PYTHONPATH=train .venv/bin/python -m xgbooster_train.train --models-dir models --trials 12`(0 跳过调参)
  - 推理服务:`infer/target/release/xgbooster-infer --models-dir models --lib infer/lib/libxgboost.so [--addr 127.0.0.1:8080]`
  - HTTP:`GET /models`、`POST /predict {"model":"risk_score[@version]","features":{...}}`、`POST /admin/reload`
  - 环境变量:`XGBOOSTER_ADMIN_TOKEN`(admin 鉴权 token)、`XGBOOSTER_RATE_LIMIT_RPS`(/predict 令牌桶速率,float >0 启用)、`XGBOOSTER_RATE_BURST`(突发容量,默认 max(1, rps))、`XGBOOSTER_MAX_INFLIGHT`(/predict 并发上限)、`XGBOOSTER_STRICT_VERSION`(manifest 缺 xgboost_version 时拒绝加载的严格门禁)
  - 端到端验证:`bash tests/consistency/run.sh`(构建+对拍+HTTP 冒烟)、`bash tests/e2e/run.sh`(HTTP 契约、热重载换血、版本门禁故障路径、限流恢复)
  - 规则层 / LLM 难例级联:仍为规划,未实现。

## changelog

变更时间线归档于 [changelog/](changelog/) 目录(按日期分目录,每条变更含背景、影响面与验证记录)。
