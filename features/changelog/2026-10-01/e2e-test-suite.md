# 端到端测试套件:HTTP 契约 + 热重载/版本门禁/限流恢复

在 tests/consistency/run.sh(构建+对拍+冒烟)之外补齐 e2e 语义验证;新增 tests/e2e/ 套件,不改动 infer/、train/、tests/consistency/ 与 models/。

## 背景

既有冒烟只覆盖单版本 happy path 与少量负向状态码,以下语义缺口无 e2e 覆盖:热重载真实语义(增删制品后 registry 换血、latest 切换与回退、@version 可达性)、版本门禁故障路径(manifest 版本不匹配时 reload 500 且保旧 registry 继续服务;strict 模式缺 `xgboost_version` 拒绝启动)、完整请求契约(坏 JSON/数组 body/model 空与错型/features 非对象/非法引用/特征类型错误等 400 分支,以及三种提取 notes 的内容与顺序)、限流 429 后的令牌恢复。

## 文件

- `tests/e2e/lib.sh`(179 行):共享助手。HTTP 断言(`expect_code`/`expect_contains`/`expect_error_contains`(JSON 解码后匹配,规避引号转义)/`http_code`/`http_body`)、数值感知版本选取 `latest_version`/`oldest_version`(镜像 `registry::pick_latest` 的 `-NN` 后缀数值比较)、`version_count`、服务器启停 `srv_start`/`srv_wait_ready`/`srv_stop`(失败时自动附带服务日志尾部)、`assert_py`(内联 python 断言,argv 透传)、`ensure_models`(models/ 为 gitignore,空目录时等待 .venv 并训练兜底,镜像 consistency 行为)。source 时统一 `unset XGBOOSTER_*` 防调用方环境泄漏。
- `tests/e2e/run.sh`(34 行):编排器。cargo build(debug)→ ensure_models → 依次执行两个 stage → `E2E OK`;trap 兜底清理服务器。
- `tests/e2e/http_contract.sh`(161 行):对仓库 `models/` 只读,单服务器 127.0.0.1:18091。覆盖:/models 结构(key 有序、最新/最旧在列、metrics 字段);版本钉扎 `risk_score@<最旧>`/`@<最新>` 与 latest 解析(未钉扎与钉扎最新同 booster 分数恒等、干净请求省略 notes 字段);7 种 400 分支(坏 JSON/数组 body/model 空/model 错型/features 非对象/`risk_score@` 非法引用/特征类型错误)逐一断言错误消息;404(未知模型、未知钉扎版本);`missing_features`(按 manifest schema 序派生断言)、`unknown_categories`(`channel=blockchain`)、`unexpected_fields`(字母序)。
- `tests/e2e/reload_gates.sh`(231 行):全程 `mktemp -d` 制品副本,**不污染仓库 models/**(结尾校验版本数不变)。覆盖:基线 3 版本;热增(`--trials 0 --seed 7` 训练新制品 → reload 前 @new 404 不可见 → reload 后 models=4、@new 200 且 HTTP 分数 vs holdout 首行 diff<1e-6、latest 切换到 @new);热删(reload 收敛回 3 版本、@new 404、latest 回退);版本门禁(`xgboost_version` 改 9.9.9 → reload 500 且错误含 "refusing to load"、旧 registry 继续服务、恢复后 reload 200);strict 门禁(删 `xgboost_version` 后 `XGBOOSTER_STRICT_VERSION=1` 拒启且日志含 "no xgboost_version";非 strict 警告放行可服务);限流恢复(RPS=2 BURST=1:首请求 200 → 立即二连 429 → sleep 0.8s 令牌补充后 200)。

## 设计决策

- 端口 18091–18094,避开 consistency 的 18097–18099;单 stage 单服务器,trap 兜底清理。
- manifest 编辑用 python(写回 `indent=2, sort_keys=True`),保持制品格式一致;`version` 字段不动(必须等于目录名)。
- 限流恢复用 BURST=1/RPS=2/sleep 0.8s(>0.5s 单令牌补充期),时序确定;MAX_INFLIGHT 并发 429 不做 e2e(毫秒级推理必然 flaky,已由 guard.rs 单测覆盖)。
- 测试脚本内 python 片段均为纯函数式,无 class,与仓库工程规则一致。

## 运行方式

`bash tests/e2e/run.sh`(依赖仓库根 .venv 与 models/ 制品,空制品时自动训练兜底)。

## 验证

首跑全绿:`E2E OK`;热增制品分数对拍 diff 7.2e-13;仓库 `models/risk_score` 保持 3 版本未变。提交前全量回归:cargo test 43 项(42 单测 + 1 consistency)、train python 测试 13 项、`bash tests/consistency/run.sh` SMOKE OK、`bash tests/e2e/run.sh` E2E OK(复跑两次,文档变更后仍全绿)。
