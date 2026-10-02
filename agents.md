# agents

Commit: f2e6dee

> 状态说明:首个实现已落地(XGBoost 训练→制品→Rust FFI 推理全链路 MVP),以下概述已按实际实现复核修正。级联架构中的规则层与 LLM 难例层仍为规划。

## 仓库概览

- 目标:构建基于 XGBoost 的复杂判断/决策能力(输入结构化特征 → 输出分类/打分判断)。
- 级联架构方向:规则层兜底明显 case(规划)→ XGBoost 处理可结构化高频流量(已实现,<1ms)→ LLM 处理低置信度难例并回流标注(规划)。
- 实现技术选型(已落地):**Python 训练 + Rust 推理**
  - 训练侧 `train/`(纯函数,无 class):features(schema 驱动向量化)/dataset(合成数据)/train(Optuna TPE + 早停)/evaluate(AUC/KS/SHAP)/artifact
  - 推理侧 `infer/`(纯函数式;unsafe 仅限 ffi.rs 的 FFI Drop 守卫):**libxgboost.so + FFI 运行时 dlopen**(非 m2cgen 生成源码——制品为 XGBoost 原生 `model.ubj`,支持运行时动态加载多模型与热更新)
  - 两侧唯一耦合点:`models/` 制品目录 = `models/<name>/<version>/{model.ubj, manifest.json, holdout.csv}`;`manifest.json` 的 `feature_schema` 是特征语义单一事实来源(Python 写入,Rust schema 驱动提取)
  - 环境约束备注:xgboost==3.4.1(pinned,wheel 内 libxgboost.so 提取至 `infer/lib/`);`train/pyproject.toml` 为声明式规格,实际运行用仓库根 `.venv` + `PYTHONPATH=train`
- 推理硬化:`XGBOOSTER_ADMIN_TOKEN`/`XGBOOSTER_RATE_LIMIT_RPS`/`XGBOOSTER_RATE_BURST`/`XGBOOSTER_MAX_INFLIGHT`/`XGBOOSTER_STRICT_VERSION` 环境变量控制鉴权、限流、并发上限与严格版本门禁;绑定非回环地址时启动 guard 要求鉴权 + 限流配置齐备,否则拒绝启动。
- 远端与提交约定:origin=git@github.com:MoSunDay/xgbooster.git,推送走 SSH(github.com 经 socks5h 127.0.0.1:1080 代理;本环境无 HTTPS 凭证,HTTPS push 会无提示挂起);提交作者统一为 MoSunDay <MoSunDay@users.noreply.github.com>;82MB 的 infer/lib/libxgboost.so 已进 git 历史(GitHub 大文件警告,移除需 LFS 或历史重写);本地分支 backup/pre-owner-rewrite 为作者重写前历史,确认后可删。

## 模块索引

- `train/xgbooster_train/`:训练管线。入口 `train.py`(`PYTHONPATH=train .venv/bin/python -m xgbooster_train.train --models-dir models [--trials 12]`,默认 synthetic risk_score;`--dataset adult` 训练 UI 真实数据集,制品名 `adult_income`)。`adult.py` 为 Adult 数据装载与 schema 推导(纯函数)。
- `datasets/adult/`:UCI Adult 真实数据集 gzip 入库(adult.data.gz / adult.test.gz,勿改动内容;语义与重下载见其 README.md)。
- `infer/src/`:推理服务。`ffi.rs`(libxgboost C API 薄封装)、`features.rs`(manifest 解析 + schema 驱动 JSON→特征向量)、`registry.rs`(多模型多版本扫描/解析/热替换)、`guard.rs`(令牌桶限流+并发上限)与 `throttle.rs`(warn 日志限频)、`predict.rs`(判断入口纯函数)、`http.rs`(axum 路由)、`main.rs`(组装)。
- `infer/tests/consistency.rs` + `tests/consistency/run.sh`:Python↔Rust 预测一致性(遍历 models/ 下全部模型,逐模型最新版 holdout 对拍,容差 1e-6)与逐模型 HTTP 冒烟。
- `tests/e2e/`:端到端契约与运维语义套件(`run.sh` 编排):`http_contract.sh` 验证 /models 结构、版本钉扎与 latest 解析、7 种 400 分支及提取 notes 内容与顺序;`reload_gates.sh` 在 mktemp 制品副本上验证热增/热删 reload 换血、xgboost_version 门禁(500 保旧、strict 拒启/非 strict 放行)与限流令牌恢复;`lib.sh` 为共享助手。
- `infer/lib/`:从 pinned wheel 提取的 libxgboost.so 及其伴随库(勿手改)。

## 相关文档

- [features/index.md](features/index.md)
