# /predict 准入控制与遗留加固

公网暴露前评审清单的第二轮补齐:/predict 限流与并发上限、warn 日志限频、strict 版本门禁开关,并把非回环启动 guard 从"仅要求 admin token"扩展到"同时要求限流配置"。

## 背景

上一轮加固(inference-hardening.md)后仍遗留四项公网前的洞:

- 无准入控制:/predict 没有速率与并发上限,单实例可被单一调用方打满。
- 日志无节流:缺失特征 warn 日志按请求逐条输出,异常流量(如批量打错特征名)可撑爆日志。
- 排序理论边界:同秒冲突后缀 `-01..-99` 零填充只保证到 99 的字典序=数值序;冒烟脚本 `run.sh` 仍用朴素 `sort` 选 latest,与 Rust 数值感知 `pick_latest` 存在语义偏差面。
- strict 缺失:manifest 缺 `xgboost_version` 时仅 warn 后放行,公网场景应允许配置为直接拒绝加载。

## 变更列表

环境变量(infer/):

| 环境变量 | 含义 | 默认 |
| --- | --- | --- |
| `XGBOOSTER_ADMIN_TOKEN` | `/admin/reload` 要求 Bearer / `X-Admin-Token`(常数时间比较) | 未设(回环免鉴权) |
| `XGBOOSTER_RATE_LIMIT_RPS` | `/predict` 令牌桶速率(请求/秒,float >0 启用) | 未设(不限流) |
| `XGBOOSTER_RATE_BURST` | 令牌桶突发容量(≥1) | `max(1, rps)` |
| `XGBOOSTER_MAX_INFLIGHT` | `/predict` 并发上限(≥1) | 未设(不限制) |
| `XGBOOSTER_STRICT_VERSION` | `1`/`true`/`yes`/`on`:manifest 缺 `xgboost_version` 即拒绝加载 | 未设(warn 放行) |

- 限流与并发上限(新增 `infer/src/guard.rs`):令牌桶按速率补充令牌、突发容量封顶;并发闸门限制同时在飞的 `/predict` 数。超限统一 429 并携带 `Retry-After: 1`,仅作用于 `POST /predict`,`GET /models` 等只读路径不受影响。
- 429 契约:超速率 → `429 {"error":"rate limit exceeded"}`;超并发 → `429 {"error":"too many concurrent predict requests"}`;两者响应头均带 `Retry-After: 1`。
- 非回环启动 guard 扩展:绑定非回环地址时,`XGBOOSTER_ADMIN_TOKEN` + `XGBOOSTER_RATE_LIMIT_RPS` + `XGBOOSTER_MAX_INFLIGHT` 三者必须同时设置,否则在绑定端口前直接退出(非零),错误信息列出缺失的环境变量名。
- warn 日志限频(新增 `infer/src/throttle.rs`):缺失特征 / 未知类别等 warn 按 (模型, 异常签名) 每 60s 至多输出一条;跟踪表超过 1024 个 key 时整体清空,防止无界内存增长。响应体中的 `missing_features` 等字段不受限频影响,仍逐请求如实返回。
- strict 版本门禁:`XGBOOSTER_STRICT_VERSION=1` 后 manifest 缺 `xgboost_version` 的制品拒绝加载(默认行为不变:warn 放行);版本不匹配仍然始终拒绝。
- 排序对齐:冒烟脚本 `run.sh` 的 latest 选取改为与 Rust `pick_latest` 同语义的数值感知比较(尾缀 `-<纯数字>` 按数值比较,否则字典序);`artifact.py` 同秒冲突后缀由 2 位改 3 位零填充(`-001..-999`),字典序=数值序的保证范围扩到 999 次同秒冲突。
- 回环默认行为不变:未设置任何 `XGBOOSTER_*` 变量时限流 / 并发上限 / strict 全部关闭,本地开发与既有冒烟主路径不受影响。

## 验证

- `bash tests/consistency/run.sh` SMOKE OK,新增负向断言:绑定 `0.0.0.0` 且未设防护变量时拒绝启动(非零退出,日志含 `XGBOOSTER_ADMIN_TOKEN` 等缺失变量名);`XGBOOSTER_RATE_LIMIT_RPS=0.001` + `XGBOOSTER_RATE_BURST=1` 下首笔 200、紧随第二笔 429,且 429 响应携带 `Retry-After: 1`;`XGBOOSTER_STRICT_VERSION=1` 下既有制品(manifest 记录 `xgboost_version: 3.4.1`)仍正常加载(`/models` 成功)。
- latest 选取逻辑以样例对拍 Rust `pick_latest` 语义:`[..., "-01", "-02"]` → `-02`、`["a-9","a-10"]` → `a-10`、`["2026-09-30T1547","2026-09-30T1528"]` → `2026-09-30T1547`,含无后缀名与带后缀名混合、`a-10` vs `a-dev` 等边界,结论一致。
