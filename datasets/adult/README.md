# UCI Adult 数据集(gzip 入库)

来源:<https://archive.ics.uci.edu/ml/machine-learning-databases/adult/>
(Adult / census income,UCI Machine Learning Repository)。

## 文件与规模

| 文件 | 行数 | 说明 |
| --- | --- | --- |
| `adult.data.gz` | 32561 条 + 1 个尾部空行 | 训练/调参数据 |
| `adult.test.gz` | 首行垃圾 `\|1x3 Cross validator` + 16281 条 + 1 个尾部空行 | 独立测试集(本仓库作 holdout) |

标签分布(`>50K` 为正类 1,`<=50K` 为负类 0):

- `adult.data`:>50K 7841,<=50K 24720
- `adult.test`:>50K. 3846,<=50K. 12435

## 15 字段语义(前 14 列为特征,末列为标签)

| # | snake_case 名 | 类型 | UCI 原名 |
| --- | --- | --- | --- |
| 1 | age | number | age |
| 2 | workclass | categorical | workclass |
| 3 | fnlwgt | number | fnlwgt |
| 4 | education | categorical | education |
| 5 | education_num | number | education-num |
| 6 | marital_status | categorical | marital-status |
| 7 | occupation | categorical | occupation |
| 8 | relationship | categorical | relationship |
| 9 | race | categorical | race |
| 10 | sex | categorical | sex |
| 11 | capital_gain | number | capital-gain |
| 12 | capital_loss | number | capital-loss |
| 13 | hours_per_week | number | hours-per-week |
| 14 | native_country | categorical | native-country |
| 15 | (标签) | 0/1 | >50K / <=50K |

## 约定

- 每字段以逗号分隔并带前后空格,解析时需 strip;`?`(即原文 ` ?`)表示缺失值,统一解析为 None(向量化为 NaN)。
- `adult.test` 标签带尾部 `.`(如 `<=50K.`),解析时剥掉;首行 `|1x3 Cross validator` 为垃圾行,跳过。
- 缺失仅出现在 workclass / occupation / native_country 三个分类列。
- 两个文件均以 gzip 文本入库(共约 600KB),提交进 git,保证 `train/tests/test_adult.py` 与训练可离线复现;**勿改动文件内容**。

## 重下载

若文件缺失或校验不符,可重新下载(gzip 原样,本环境经 socks5 代理):

```bash
cd datasets/adult
curl --proxy socks5h://127.0.0.1:1080 -O https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data.gz
curl --proxy socks5h://127.0.0.1:1080 -O https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test.gz
```

参考大小:`adult.data.gz` 408980 B,`adult.test.gz` 205265 B。
