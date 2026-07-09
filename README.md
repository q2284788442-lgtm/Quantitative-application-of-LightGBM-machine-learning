# LightGBM IM 日线方向预测项目

本项目使用 `000852.XSHG`（中证 1000 指数）日 K 数据构造技术特征，训练 LightGBM 二分类模型，并在每天收盘后输出下一段开盘到开盘收益方向的预测信号。

当前版本面向日常自动运行：收盘后通过 AkShare 更新中证 1000 日 K CSV，重新训练模型，保存最新权重文件，并写出最新一行预测信号。项目仅用于研究和教学，不构成投资建议。

## 预测目标

特征使用 `T` 日收盘后可获得的中证 1000 日 K 数据：

```text
date, code, open, high, low, close, volume, money
```

标签定义为：

```text
T+1 交易日开盘 -> T+2 交易日开盘 的收益方向
```

例如在 `2026-07-09` 收盘后，模型预测的是：

```text
2026-07-10 开盘 -> 2026-07-13 开盘
```

注意这里的 `T+2` 是交易日，不是自然日。

## 当前流程

1. 从 `data/1000_dayK_20190101_20260630.csv` 读取中证 1000 日 K。
2. 构造日线技术特征。
3. 只使用已有完整标签的样本训练模型，即 `target_exit_date <= 最新可用日K日期`。
4. 使用最新完整特征行生成实时预测信号。
5. 保存模型、特征重要性、训练期回测和最新信号。

当前版本已经移除 OOS 输出，不再生成 `oos_*` 文件。

## 数据文件

```text
data/
  1000_dayK_20190101_20260630.csv
  metadata/
    futures_contract_meta.csv
  raw/
    im_60m/
      IM_monthly_continuous_60m_20220722_20260630.csv
```

说明：

- `1000_dayK_20190101_20260630.csv` 是模型训练和最新预测的核心输入。
- 文件名保留历史命名，即使后续追加到 `2026-06-30` 之后也仍然使用这个路径。
- `IM_monthly_continuous_60m_20220722_20260630.csv` 只用于训练期 IM 回测，不影响最新方向信号生成。
- AkShare 更新脚本会在写入前备份原 CSV 到 `data/backups/`。

## 安装依赖

推荐使用当前项目环境：

```powershell
D:\anaconda3\envs\py39\python.exe -m pip install -r requirements.txt
```

依赖包括：

```text
akshare
lightgbm
matplotlib
numpy
pandas
scikit-learn
```

## 手动运行

只训练和生成最新信号，不更新 AkShare 数据：

```powershell
D:\anaconda3\envs\py39\python.exe train.py
```

只更新中证 1000 日 K CSV：

```powershell
D:\anaconda3\envs\py39\python.exe scripts\update_1000_dayk_akshare.py
```

完整每日流程：先更新 CSV，再训练并生成最新信号：

```powershell
D:\anaconda3\envs\py39\python.exe scripts\run_daily_update_and_train.py
```

盘中测试时建议使用 dry-run，避免写入 CSV：

```powershell
D:\anaconda3\envs\py39\python.exe scripts\update_1000_dayk_akshare.py --dry-run
```

## 收盘保护

AkShare 在盘中可能返回当天临时日线。为避免把未收盘数据写入训练集，更新脚本默认启用保护：

```text
本地时间 16:00 前，自动丢弃当天日期的数据行。
```

如需修改保护时间：

```powershell
D:\anaconda3\envs\py39\python.exe scripts\run_daily_update_and_train.py --ready-time 16:30
```

不建议盘中使用 `--allow-intraday-row`，除非你明确知道数据已经是最终收盘数据。

## Windows 任务计划

已提供注册脚本：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\register_daily_update_task.ps1 -At 18:30
```

默认任务：

```text
任务名: LightGBM IM Daily Update Train
时间: 每天 18:30
动作: D:\anaconda3\envs\py39\python.exe scripts\run_daily_update_and_train.py
```

任务运行日志写入：

```text
outputs/daily_runs/
```

如果 AkShare 或行情源临时断连，日志里会保留完整错误信息。脚本内置重试和备用接口，但行情源不可用时不会强行写入 CSV。

## 输出文件

训练和预测输出目录：

```text
outputs/final_lightgbm_im_daily_static_no_rf/
```

主要文件：

```text
summary.json
latest_signal.csv
latest_signal.json
lightgbm_model.txt
lightgbm_model_YYYYMMDD.txt
feature_importance.csv
train_predictions.csv
train_metrics.csv
train_equity_curve.csv
train_trade_log.csv
train_monthly_returns.csv
train_equity_curve.png
```

最新信号重点看：

```text
latest_signal.csv
latest_signal.json
```

关键字段：

```text
feature_date      预测使用的特征日期
p_up              模型预测上涨概率
score             p_up - 0.5
target_position   1 为做多，-1 为做空
direction         up_or_long 或 down_or_short
```

当最新行还没有未来标签时，`target_date`、`target_next_day_return`、`target_up` 会是空值，这是正常情况。

## 参数规则

LightGBM 主要参数位于：

```text
scripts/run_final_lightgbm_im_daily_static.py
```

`min_child_samples` 不再固定为 `40`，而是在训练样本筛选完成后动态计算：

```text
min_child_samples = round_half_up(train_rows * 0.025)
```

例如：

```text
1595 * 0.025 = 39.875 -> 40
1754 * 0.025 = 43.85  -> 44
```

## AkShare 数据源

默认更新接口：

```text
ak.stock_zh_index_daily_em(symbol="csi000852")
```

备用接口：

```text
ak.index_zh_a_hist(symbol="000852", period="daily")
```

更新后的字段会统一映射为项目 CSV 格式：

```text
date, code, open, high, low, close, volume, money
```

其中 `amount` 或 `成交额` 会映射为 `money`。

## 风险说明

本项目不保证预测准确率或交易收益。历史回测不代表未来表现。模型可能受到过拟合、样本外失效、数据源异常、交易成本估计偏差、滑点、流动性和执行延迟等影响。

正式使用前，应先检查：

```text
outputs/daily_runs/
outputs/final_lightgbm_im_daily_static_no_rf/latest_signal.json
data/1000_dayK_20190101_20260630.csv
```

确认数据日期、最新信号日期和任务日志都符合预期后，再做后续决策。
