# 接入 UCI Adult 真实数据集(训练→制品→推理一致性全流程)

## 背景

MVP 至今仅有 synthetic 风控数据(risk_score)。为验证 schema 驱动管线对"真实脏数据"的适配能力,接入 UCI Adult(census income)数据集:真实分布、分类特征带缺失值(`?`)、测试集含训练未见类别,恰好覆盖此前面向合成数据从未走过的代码路径(缺失值向量化、空 CSV 单元格对拍、多模型注册表)。

## 变更列表

- 新增 `train/xgbooster_train/adult.py`(114 行,纯函数):`parse_line`(strip、15 字段校验、`?`→None、数值列转 int、标签剥尾点映射 0/1)、`load`(gzip 逐行,跳过空行与 `|` 垃圾行)、`build_schema`(仅从给定样本推导,分类 mapping = 排序去重后按序编号,与 `features.FEATURE_SCHEMA` 同构)。
- `train/xgbooster_train/train.py`(145 → 200 行):数据装配抽为 `_load_dataset` 按 `--dataset {synthetic,adult}` 分发(synthetic 默认,行为不变);新增 `--datasets-dir`(默认 datasets)。adult 路径:adult.data 头 28000 训练 / 中 4561 调参验证(尾段必须为空,否则报错),adult.test 全量 16281 作独立 holdout;schema 仅用训练段推导;模型名 `adult_income`。管线主体(tune/train_final/evaluate/write_artifact)原样参数化复用。
- 新增 `train/tests/test_adult.py`(172 行,自研 runner,仿 test_features.py):10 项,覆盖 `?`→None、尾点标签、字段数/标签/数值列 ValueError、`|` 垃圾行、schema 确定性与排序、空 mapping、vectorize 对 adult schema 的兼容(None/未知类别→NaN)。
- `infer/tests/consistency.rs`(159 → 187 行):从硬编码 risk_score 改为 `discover_models` 遍历 models/ 下所有含有效版本的模型,逐模型 `check_model`(最新版本 holdout 逐行对拍),输出每模型 n_rows/max_abs_diff;无制品或无 lib 时整体 SKIP 语义保持。
- `tests/consistency/run.sh`(240 → 263 行):HTTP 冒烟从单 risk_score 改为遍历全部模型(`GET /models` 断言与磁盘模型目录集合一致;每模型取最新版首行构造 /predict 对拍);latest 版本选择与请求构造提为 `latest_version_of`/`build_request` 函数。负例分支仍用 risk_score(兜底训练保证其存在)。
- `tests/e2e/http_contract.sh`:`/models` 数量断言从"risk_score 版本数"扩为"models/ 下全部版本目录总数",并断言 name 集合与磁盘模型目录一致(不弱化:仍精确计数 + 排序 + latest/oldest 钉扎)。reload_gates 在 mktemp 副本内只操作 risk_score,零改动。
- 新增 `datasets/adult/README.md`(55 行):来源 URL、行数/标签分布、15 字段语义表、`?` 缺失约定、gzip 入库说明与 curl 重下载命令。
- `features/index.md` / `agents.md`:训练入口补 `--dataset adult` 示例与 datasets/、adult.py 最小索引。

## 设计决策

- **schema 仅从 train 段推导**:调参/早停用的 valid 与最终 holdout 都不得参与类别表构造,避免任何测试信息泄漏进特征编码;test 独有类别(如 Holand-Netherlands)双侧统一按未知类别→NaN(Python `mapping.get(v, NaN)`,Rust `unknown_categorical→NaN`),语义一致故对拍可行。
- **空单元格对拍路径**:adult holdout 1221 行含缺失(workclass/occupation/native_country),`holdout.csv` 由 csv writer 写成空单元格。Rust 一致性测试的 `cell_value` 原有实现恰好兼容:数值列空串 parse 失败→`Value::Null`→NaN;分类列空串→未知类别→NaN——与 Python 侧 None→NaN 完全等价,无需修改 features.rs。
- **test 集作独立 holdout**:Adult 官方 train/test 天然分离,16281 行全量入 holdout,制品 metrics 无泄漏;训练段内部再切 28000/4561 供调参与早停(复用现有确定性头/中切片 `dataset.split`)。
- **fnlwgt 照用**:人口普查权重列虽非个体因果特征,但作为真实数据基线保留原始列序与语义,不做特征筛选(14 列全量入模);SHAP 显示其贡献最低(0.109),佐证无害。
- **`|` 垃圾行按注释跳过**:adult.test 首行 `|1x3 Cross validator` 不满足 15 字段,parse_line 对空行与 `|` 前缀行返回 None,其余格式错误仍 ValueError(不静默吞坏行)。
- **模型名 adult_income 与 risk_score 并存**:registry 本就多模型多版本,一致性/冒烟/e2e 相应通用化,验证"制品目录即契约"在多模型下成立。

## 验证

- 训练:`PYTHONPATH=train .venv/bin/python -m xgbooster_train.train --models-dir models --dataset adult --trials 12` → **holdout AUC 0.9273 / KS 0.6895**(16281 行,best_iteration 312/313 rounds,重复运行参数与指标一致;SHAP top:marital_status 0.763、age 0.703、capital_gain 0.477),落在 Adult 合理区间 0.90~0.93;制品 `models/adult_income/2026-10-02T033642/`。
- Python 测试:`train/tests/test_features.py` 13 项 + `train/tests/test_adult.py` 10 项全绿。
- `bash tests/consistency/run.sh` SMOKE OK:一致性对拍 adult_income n_rows=16281 max_abs_diff=4.998e-11、risk_score n_rows=4000 max_abs_diff=4.997e-11(容差 1e-6);HTTP 冒烟两模型逐个 POST 对拍(diff 3.4e-13 / 3.7e-11),/models 列出全部 2 模型。
- `cd infer && cargo test`:43 通过(42 单元 + 1 consistency),0 失败。
- `bash tests/e2e/run.sh` E2E OK(两 stage 全绿,`/models` 断言改为按 models/ 全部版本目录总数精确计数(验证时 5 版本 × 2 模型)后通过;仓库 models/ 的 risk_score 保持 3 版本未变)。
