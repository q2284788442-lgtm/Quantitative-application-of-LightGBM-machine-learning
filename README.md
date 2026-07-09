# LightGBM 指数期货方向预测教学项目

本仓库是一个使用 LightGBM 做中证 1000 指数方向预测，并映射到 IM 股指期货当月连续合约上的量化教学示例。

项目目标是演示一个可复现的机器学习量化流程：构造日线特征、训练二分类模型、生成多空信号、执行期货回测，并输出模型、资金曲线和绩效报告。它不是实盘投资建议。

## 策略逻辑

- 特征：使用 T 日收盘后已经可获得的 `000852.XSHG` 日线 OHLCV/money 数据。
- 标签：预测 T+1 日开盘到 T+2 日开盘的收益方向。
- 多空信号：`p_up - 0.5 > 0` 做多，`p_up - 0.5 < 0` 做空。
- 持仓规则：连续同向继续持仓；反向信号先平旧仓，再反向开仓。
- 执行标的：IM 当月连续合约。
- 手数：1 手。
- 初始资金：1,000,000 元。

## 数据区间

- 指数日线数据：`2019-01-01` 至 `2026-06-30`。
- 训练标签截止：`2025-06-30`，即训练样本的退出日不晚于 `2025-06-30`。
- 观察回测区间：`2025-07-01` 至 `2026-06-30`。

## 当前固定参数

```python
PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.06,
    "n_estimators": 45,
    "max_depth": 4,
    "num_leaves": 14,
    "min_child_samples": 40,
    "min_split_gain": 0.05,
    "reg_lambda": 30.0,
    "reg_alpha": 0.0,
    "feature_fraction": 1.0,
    "bagging_fraction": 1.0,
    "bagging_freq": 0,
    "random_state": 2026,
    "n_jobs": -1,
    "verbosity": -1,
}
```



## 目录结构

```text
.
├── data/
│   ├── 1000_dayK_20190101_20260630.csv
│   ├── metadata/
│   │   └── futures_contract_meta.csv
│   └── raw/
│       └── im_60m/
│           └── IM_monthly_continuous_60m_20220722_20260630.csv
├── scripts/
│   └── run_final_lightgbm_im_daily_static.py
├── train.py
├── requirements.txt
└── README.md
```

生成的模型、图表、回测明细和报告会写入 `outputs/`，该目录已在 `.gitignore` 中忽略。

## 安装依赖

```bash
pip install -r requirements.txt
```

## 运行

推荐直接运行：

```bash
python train.py
```

也可以运行兼容脚本：

```bash
python scripts/run_final_lightgbm_im_daily_static.py
```

运行后会生成：

```text
outputs/final_lightgbm_im_daily_static_no_rf/
├── summary.json
├── lightgbm_model.txt
├── feature_importance.csv
├── train_metrics.csv
├── oos_metrics.csv
├── train_equity_curve.csv
├── oos_equity_curve.csv
├── train_equity_curve.png
└── oos_equity_curve.png
```

## 成本假设

合约参数来自 `data/metadata/futures_contract_meta.csv`：

- 初始资金：1,000,000 元。
- 手数：1 手。
- 合约乘数：200。
- 开仓手续费率：`0.000023`。
- 平仓手续费率：`0.000023`。
- 单边滑点：0.4 点。
- 最小跳动：0.2 点。

## 风险说明

本项目仅用于教学和研究，不构成投资建议。历史回测结果不代表未来收益。机器学习模型可能出现过拟合、样本外失效、交易成本低估、流动性冲击和数据口径偏差等问题。

## feature 分支补充说明

`feature` 分支在当前教学项目基础上，补充了每日自动化运行相关功能：

* 使用 AkShare 更新中证 1000 日 K 数据，并写入现有 CSV。
* 增加收盘保护，避免盘中把当天未收盘日 K 写入训练数据。
* 每日更新数据后自动重新训练 LightGBM，并输出最新预测信号。
* 新增 `latest_signal.csv` 和 `latest_signal.json`，用于查看最新特征日对应的方向预测。
* 将 `min_child_samples` 改为按训练样本数动态计算。
* 移除原来的 OOS 输出，保留训练期结果和最新信号输出。
* 增加 Windows 任务计划脚本，可用于每天收盘后定时运行。

该分支主要补充的是数据更新、自动训练、信号输出和定时运行流程，不改变项目作为量化教学示例的定位。

