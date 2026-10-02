# e2e 套件收尾加固(strict 拒启孤儿回收 + 限流恢复时长派生)

## 背景

835bdfd 上线评审遗留两项可延后 TODO,本次收尾:

1. `tests/e2e/reload_gates.sh` 的 strict 拒启分支直接前台运行 `XGBOOSTER_STRICT_VERSION=1 "$BIN" ...`:若服务因回归误启动,前台命令永不退出导致整套 e2e 悬挂,且该进程不经 `srv_start`,不受 trap 兜底清理,残留孤儿占用端口。
2. 限流恢复段硬编码 `sleep 0.8`(注释注明 rps=2 时 0.5s 补充一令牌):等待时长与 `XGBOOSTER_RATE_LIMIT_RPS=2` 魔数耦合,日后调参 RPS 必须同步改 sleep 魔数,否则时序断言失真。

另注明:835bdfd 的提交标题笔误已随 amend 修正,无内容变更(amend 改写哈希,正文与树不变)。

## 变更列表

仅改 `tests/e2e/reload_gates.sh`(+8 行,231 → 239 行),`infer/`、`train/`、`models/`、其余测试脚本零改动。

- strict 拒启分支改经 `timeout -k 2 10` 托管启动:若回归导致服务误启动,由 timeout 收割孤儿(KILL 兜底 2s)并以 rc 124 返回,脚本显式 `fail "strict mode kept serving; timeout killed orphan"` 而非悬挂;原先的 rc==0 拒启断言与 `"no xgboost_version"` 日志断言原样保留。
- 限流段 RPS/BURST 提为 `RL_RPS`/`RL_BURST` 变量,等待时长按 `1/RL_RPS + 0.3s` margin 用 awk 派生为 `RL_RECOVERY`;默认参数下仍为 0.80s,与原行为等价,但调参 RL_RPS 时等待时长自动随之修正。

## 设计决策

- timeout 选 `-k 2 10`:10s 覆盖正常启动路径(实际毫秒级即拒绝退出),误启动时 TERM 后 2s KILL 兜底,rc 124 语义化"进程存活被杀"。
- 三段式 rc 断言(124 → kept serving;0 → must refuse;其余非 0 → 通过)把"误启动"与"未拒绝"两种回归都转为显式 fail,不再依赖前台命令自退。
- margin 取固定 0.3s 而非按比例派生:吸收服务器调度抖动即可,公式简单可读;`XGBOOSTER_RATE_LIMIT_RPS="$RL_RPS"` 带引号整体作为单个 NAME=value 参数传给 `env`,合法且防分词。

## 验证

- `bash -n tests/e2e/reload_gates.sh` 语法检查通过。
- 仓库根 `bash tests/e2e/run.sh` 全绿:`== E2E OK ==`(debug 构建后两 stage 顺序通过;strict 拒启日志、非 strict 放行、限流 200→429→(等待 0.80s)→200 恢复均按预期)。
- 热增制品分数对拍 diff=7.24e-13;套件收尾校验仓库 `models/risk_score` 保持 3 版本未变(`ls models/risk_score | wc -l` = 3)。
- `wc -l tests/e2e/reload_gates.sh` = 239 行,满足 ≤400 行限制。
