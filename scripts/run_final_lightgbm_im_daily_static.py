"""
Final no-RF LightGBM teaching example.

This script builds daily 000852.XSHG index features, trains one LightGBM
classifier with a fixed parameter set, writes the latest live signal, and
backtests the training-period signal on IM current-month continuous futures data.

Rules:
  - Daily training uses all complete labeled rows whose target exit date
    is not later than the latest available feature date.
  - The latest complete feature row is also scored as the live signal.
  - No China 10Y yield file or interest-rate feature is used.
  - The label uses T+1 open to T+2 open return, aligned with the
    open-rebalance state-holding futures strategy.
  - p_up - 0.5 > 0 means long; p_up - 0.5 < 0 means short.
  - Same direction keeps the position; reverse signals close then open.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Dict, Iterable, List

import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss, roc_auc_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
INDEX_FILE = ROOT / "data" / "1000_dayK_20190101_20260630.csv"
IM_FILE = ROOT / "data" / "raw" / "im_60m" / "IM_monthly_continuous_60m_20220722_20260630.csv"
META_FILE = ROOT / "data" / "metadata" / "futures_contract_meta.csv"
OUT_DIR = ROOT / "outputs" / "final_lightgbm_im_daily_static_no_rf"

INDEX_CODE = "000852.XSHG"
INITIAL_CASH = 1_000_000.0
MIN_CHILD_SAMPLES_RATE = Decimal("0.025")

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

FEATURE_COLUMNS: List[str] = [
    "ret_1d",
    "ret_5d",
    "ret_20d",
    "close_ma_gap_5",
    "close_ma_gap_20",
    "close_ma_gap_60",
    "ma_ratio_5_20",
    "ma_ratio_20_60",
    "macd_dif_12_26_9",
    "macd_hist_12_26_9",
    "macd_dea_slope_20",
    "rsi_14",
    "atr_pct_14",
    "volatility_5",
    "volatility_20",
    "range_pct",
    "body_pct",
    "close_location",
    "volume_ratio_5",
    "volume_ratio_20",
    "money_ratio_5",
    "money_ratio_20",
]


@dataclass(frozen=True)
class ContractMeta:
    multiplier: float
    margin_rate: float
    open_commission: float
    close_commission: float
    close_today_commission: float
    slippage_points_per_side: float
    tick_size: float


def json_default(obj):
    if obj is pd.NaT:
        return None
    if isinstance(obj, (pd.Timestamp, np.datetime64)):
        if pd.isna(obj):
            return None
        return pd.Timestamp(obj).isoformat()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if pd.isna(obj):
        return None
    raise TypeError(f"{type(obj).__name__} is not JSON serializable")


def json_sanitize(obj):
    if obj is pd.NaT:
        return None
    if isinstance(obj, dict):
        return {str(key): json_sanitize(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_sanitize(value) for value in obj]
    if isinstance(obj, np.ndarray):
        return [json_sanitize(value) for value in obj.tolist()]
    if isinstance(obj, (pd.Timestamp, np.datetime64)):
        if pd.isna(obj):
            return None
        return pd.Timestamp(obj).isoformat()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        if pd.isna(obj) or not np.isfinite(obj):
            return None
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    try:
        if pd.isna(obj):
            return None
    except (TypeError, ValueError):
        pass
    return obj


def write_json(path: Path, data: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_sanitize(data), ensure_ascii=False, indent=2, allow_nan=False, default=json_default),
        encoding="utf-8",
    )


def calc_min_child_samples(train_rows: int) -> int:
    if train_rows <= 0:
        raise ValueError("Cannot calculate min_child_samples for an empty training set.")
    value = Decimal(train_rows) * MIN_CHILD_SAMPLES_RATE
    return max(1, int(value.quantize(Decimal("1"), rounding=ROUND_HALF_UP)))


def training_params(train_rows: int) -> Dict[str, object]:
    params = dict(PARAMS)
    params["min_child_samples"] = calc_min_child_samples(train_rows)
    return params


def rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).rolling(window, min_periods=window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    prev_close = close.shift(1)
    true_range = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(window, min_periods=window).mean()


def load_contract_meta() -> ContractMeta:
    meta = pd.read_csv(META_FILE)
    row = meta[meta["symbol"].astype(str).eq("IM")].iloc[0]
    return ContractMeta(
        multiplier=float(row["multiplier"]),
        margin_rate=float(row["margin_rate"]),
        open_commission=float(row["open_commission"]),
        close_commission=float(row["close_commission"]),
        close_today_commission=float(row["close_today_commission"]),
        slippage_points_per_side=float(row["slippage_points_per_side"]),
        tick_size=float(row["tick_size"]),
    )


def load_index_data() -> pd.DataFrame:
    df = pd.read_csv(INDEX_FILE, parse_dates=["date"])
    expected = {"date", "code", "open", "high", "low", "close", "volume", "money"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Index file missing columns: {sorted(missing)}")
    df = df[df["code"].astype(str).eq(INDEX_CODE)].sort_values("date").reset_index(drop=True)
    if df.empty:
        raise ValueError(f"No rows found for {INDEX_CODE}")
    return df


def build_feature_panel(index_df: pd.DataFrame) -> pd.DataFrame:
    df = index_df.copy().sort_values("date").reset_index(drop=True)
    close = df["close"].astype(float)
    open_ = df["open"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)
    money = df["money"].astype(float)
    ret = close.pct_change()

    for n in [1, 5, 20]:
        df[f"ret_{n}d"] = close / close.shift(n) - 1.0

    ma5 = close.rolling(5, min_periods=5).mean()
    ma20 = close.rolling(20, min_periods=20).mean()
    ma60 = close.rolling(60, min_periods=60).mean()
    df["close_ma_gap_5"] = close / ma5 - 1.0
    df["close_ma_gap_20"] = close / ma20 - 1.0
    df["close_ma_gap_60"] = close / ma60 - 1.0
    df["ma_ratio_5_20"] = ma5 / ma20 - 1.0
    df["ma_ratio_20_60"] = ma20 / ma60 - 1.0

    ema12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False, min_periods=9).mean()
    df["macd_dif_12_26_9"] = dif
    df["macd_hist_12_26_9"] = dif - dea
    df["macd_dea_slope_20"] = dea - dea.shift(20)

    df["rsi_14"] = rsi(close, 14)
    df["atr_pct_14"] = atr(high, low, close, 14) / close
    df["volatility_5"] = ret.rolling(5, min_periods=5).std()
    df["volatility_20"] = ret.rolling(20, min_periods=20).std()
    df["range_pct"] = (high - low) / open_
    df["body_pct"] = (close - open_) / open_
    df["close_location"] = np.where((high - low) == 0, np.nan, (close - low) / (high - low))
    df["volume_ratio_5"] = volume / volume.rolling(5, min_periods=5).mean()
    df["volume_ratio_20"] = volume / volume.rolling(20, min_periods=20).mean()
    df["money_ratio_5"] = money / money.rolling(5, min_periods=5).mean()
    df["money_ratio_20"] = money / money.rolling(20, min_periods=20).mean()

    df["feature_date"] = df["date"]
    df["target_entry_date"] = df["date"].shift(-1)
    df["target_exit_date"] = df["date"].shift(-2)
    df["target_entry_open"] = df["open"].shift(-1)
    df["target_exit_open"] = df["open"].shift(-2)
    df["target_open_to_next_open_return"] = (
        df["target_exit_open"] / df["target_entry_open"] - 1.0
    )

    # Backward-compatible names used by prediction and report code.
    # target_date is the executable entry date. The return is entry open -> next open.
    df["target_date"] = df["target_entry_date"]
    df["target_next_day_open"] = df["target_entry_open"]
    df["target_next_day_return"] = df["target_open_to_next_open_return"]

    valid_target = df["target_open_to_next_open_return"].notna()
    df["target_up"] = np.nan
    df.loc[valid_target, "target_up"] = (
        df.loc[valid_target, "target_open_to_next_open_return"] > 0
    ).astype(float)
    df["feature_complete"] = df[FEATURE_COLUMNS].notna().all(axis=1).astype(int)
    return df


def load_im_daily_execution() -> pd.DataFrame:
    im = pd.read_csv(IM_FILE, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    expected = {"date", "code", "open", "high", "low", "close", "volume"}
    missing = expected - set(im.columns)
    if missing:
        raise ValueError(f"IM file missing columns: {sorted(missing)}")
    im["trade_date"] = im["date"].dt.normalize()
    first = im.groupby("trade_date", as_index=False).first()[["trade_date", "date", "code", "open", "high", "low"]]
    last = im.groupby("trade_date", as_index=False).last()[["trade_date", "date", "close"]]
    daily = first.merge(last, on="trade_date", how="inner", suffixes=("_open_time", "_close_time"))
    daily = daily.rename(columns={"date_open_time": "open_datetime", "date_close_time": "close_datetime"})
    daily["roll_flag"] = daily["code"].astype(str).ne(daily["code"].astype(str).shift(1)).astype(int)
    daily.loc[daily.index[0], "roll_flag"] = 0
    return daily.sort_values("trade_date").reset_index(drop=True)


def select_rows(panel: pd.DataFrame, start: pd.Timestamp | None, end: pd.Timestamp) -> pd.DataFrame:
    mask = (
        panel["feature_complete"].eq(1)
        & panel["target_up"].notna()
        & panel["target_exit_date"].le(end)
    )
    if start is not None:
        mask &= panel["target_entry_date"].ge(start)
    return panel.loc[mask].copy()


def predict(model: lgb.LGBMClassifier, rows: pd.DataFrame) -> pd.DataFrame:
    class_index = list(model.classes_).index(1)
    p_up = model.predict_proba(rows[FEATURE_COLUMNS].astype(float))[:, class_index]
    out = rows[["feature_date", "target_date", "target_next_day_return", "target_up"]].copy()
    out["p_up"] = p_up
    out["score"] = out["p_up"] - 0.5

    targets = []
    previous = 0
    for score in out["score"].astype(float):
        if score > 0:
            target = 1
        elif score < 0:
            target = -1
        else:
            target = previous
        targets.append(target)
        previous = target
    out["target_position"] = targets
    return out


def latest_complete_feature_rows(panel: pd.DataFrame) -> pd.DataFrame:
    complete = panel[panel["feature_complete"].eq(1)].copy()
    if complete.empty:
        raise ValueError("No complete feature row is available for latest prediction.")
    latest_feature_date = complete["feature_date"].max()
    return complete[complete["feature_date"].eq(latest_feature_date)].copy()


def predict_latest_signal(model: lgb.LGBMClassifier, panel: pd.DataFrame) -> pd.DataFrame:
    latest_rows = latest_complete_feature_rows(panel)
    signal = predict(model, latest_rows)
    signal["prediction_horizon"] = "next_trading_open_to_second_next_trading_open"
    signal["direction"] = np.select(
        [signal["target_position"].eq(1), signal["target_position"].eq(-1)],
        ["up_or_long", "down_or_short"],
        default="flat",
    )
    signal["is_labeled_training_row"] = signal["target_up"].notna()
    return signal


def safe_auc(y: Iterable[int], p: Iterable[float]) -> float:
    y_arr = np.asarray(list(y), dtype=int)
    p_arr = np.asarray(list(p), dtype=float)
    if len(np.unique(y_arr)) < 2:
        return float("nan")
    return float(roc_auc_score(y_arr, p_arr))


def classification_metrics(name: str, pred: pd.DataFrame) -> Dict[str, float]:
    y = pred["target_up"].astype(int).to_numpy()
    p = pred["p_up"].astype(float).to_numpy()
    hard = (p >= 0.5).astype(int)
    return {
        "sample": name,
        "rows": int(len(pred)),
        "auc": safe_auc(y, p),
        "accuracy": float(accuracy_score(y, hard)),
        "balanced_accuracy": float(balanced_accuracy_score(y, hard)),
        "logloss": float(log_loss(y, np.clip(p, 1e-12, 1 - 1e-12), labels=[0, 1])),
        "target_up_rate": float(np.mean(y)),
        "probability_mean": float(np.mean(p)),
        "probability_std": float(np.std(p, ddof=0)),
        "long_signal_ratio": float(np.mean(pred["target_position"].eq(1))),
        "short_signal_ratio": float(np.mean(pred["target_position"].eq(-1))),
    }


def trade_cost(price: float, lots: int, rate: float, meta: ContractMeta) -> float:
    commission = abs(lots) * price * meta.multiplier * rate
    slippage = abs(lots) * meta.slippage_points_per_side * meta.multiplier
    return float(commission + slippage)


def backtest(
    name: str,
    execution: pd.DataFrame,
    target_by_date: Dict[pd.Timestamp, int],
    meta: ContractMeta,
    default_target: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    equity = INITIAL_CASH
    position = 0
    prev_close = np.nan
    prev_code = None
    rows = []
    trades = []

    for row in execution.itertuples(index=False):
        date = pd.Timestamp(row.trade_date)
        open_price = float(row.open)
        close_price = float(row.close)
        code = str(row.code)
        desired = int(target_by_date.get(date, position if default_target is None else default_target))

        gap_pnl = 0.0 if np.isnan(prev_close) else position * (open_price - prev_close) * meta.multiplier
        cost = 0.0

        if prev_code is not None and code != prev_code and position != 0:
            close_cost = trade_cost(open_price, 1, meta.close_commission, meta)
            cost += close_cost
            trades.append(
                {
                    "strategy": name,
                    "datetime": row.open_datetime,
                    "trade_date": date,
                    "code": code,
                    "action": "close",
                    "position_before": position,
                    "position_after": 0,
                    "price": open_price,
                    "cost": close_cost,
                    "reason": "roll_close",
                }
            )
            position = 0

        if desired != position:
            if position != 0:
                close_cost = trade_cost(open_price, 1, meta.close_commission, meta)
                cost += close_cost
                trades.append(
                    {
                        "strategy": name,
                        "datetime": row.open_datetime,
                        "trade_date": date,
                        "code": code,
                        "action": "close",
                        "position_before": position,
                        "position_after": 0,
                        "price": open_price,
                        "cost": close_cost,
                        "reason": "signal_reverse_close",
                    }
                )
                position = 0
            if desired != 0:
                open_cost = trade_cost(open_price, 1, meta.open_commission, meta)
                cost += open_cost
                trades.append(
                    {
                        "strategy": name,
                        "datetime": row.open_datetime,
                        "trade_date": date,
                        "code": code,
                        "action": "open",
                        "position_before": 0,
                        "position_after": desired,
                        "price": open_price,
                        "cost": open_cost,
                        "reason": "signal_open",
                    }
                )
                position = desired

        intraday_pnl = position * (close_price - open_price) * meta.multiplier
        net_pnl = gap_pnl + intraday_pnl - cost
        equity += net_pnl
        rows.append(
            {
                "strategy": name,
                "datetime": row.close_datetime,
                "trade_date": date,
                "code": code,
                "open": open_price,
                "close": close_price,
                "target_position": desired,
                "position": position,
                "gap_pnl": gap_pnl,
                "intraday_pnl": intraday_pnl,
                "cost": cost,
                "net_pnl": net_pnl,
                "equity": equity,
            }
        )
        prev_close = close_price
        prev_code = code

    if position != 0 and rows:
        last = execution.iloc[-1]
        final_cost = trade_cost(float(last["close"]), 1, meta.close_commission, meta)
        equity -= final_cost
        trades.append(
            {
                "strategy": name,
                "datetime": last["close_datetime"],
                "trade_date": last["trade_date"],
                "code": last["code"],
                "action": "close",
                "position_before": position,
                "position_after": 0,
                "price": float(last["close"]),
                "cost": final_cost,
                "reason": "final_close",
            }
        )
        rows[-1]["cost"] += final_cost
        rows[-1]["net_pnl"] -= final_cost
        rows[-1]["equity"] = equity

    curve = pd.DataFrame(rows)
    trade_log = pd.DataFrame(trades)
    metrics = strategy_metrics(name, curve, trade_log)
    return curve, trade_log, metrics


def strategy_metrics(name: str, curve: pd.DataFrame, trade_log: pd.DataFrame) -> Dict[str, object]:
    equity = curve["equity"].astype(float)
    returns = equity.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    drawdown = equity / equity.cummax() - 1.0
    annual_vol = float(returns.std(ddof=0) * np.sqrt(244)) if len(returns) else 0.0
    sharpe = 0.0 if annual_vol == 0 else float((returns.mean() * 244) / annual_vol)
    daily_win_rate = float(curve["net_pnl"].gt(0).mean()) if len(curve) else 0.0
    return {
        "strategy": name,
        "rows": int(len(curve)),
        "final_equity": float(equity.iloc[-1]),
        "net_profit": float(equity.iloc[-1] - INITIAL_CASH),
        "total_return": float(equity.iloc[-1] / INITIAL_CASH - 1.0),
        "sharpe": sharpe,
        "max_drawdown": float(-drawdown.min()),
        "daily_win_rate": daily_win_rate,
        "turnover_count": int(len(trade_log)),
        "long_holding_ratio": float(curve["position"].gt(0).mean()),
        "short_holding_ratio": float(curve["position"].lt(0).mean()),
        "total_cost": float(curve["cost"].sum()),
        "sum_net_pnl": float(curve["net_pnl"].sum()),
        "reconciliation_error": float(curve["net_pnl"].sum() - (equity.iloc[-1] - INITIAL_CASH)),
    }


def monthly_metrics(curves: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for strategy, group in curves.groupby("strategy"):
        data = group.copy()
        data["month"] = pd.to_datetime(data["trade_date"]).dt.to_period("M").astype(str)
        start_equity = INITIAL_CASH
        for month, m in data.groupby("month"):
            end_equity = float(m["equity"].iloc[-1])
            rows.append(
                {
                    "strategy": strategy,
                    "month": month,
                    "start_equity": start_equity,
                    "end_equity": end_equity,
                    "net_profit": end_equity - start_equity,
                    "monthly_return": end_equity / start_equity - 1.0,
                }
            )
            start_equity = end_equity
    return pd.DataFrame(rows)


def plot_equity(curves: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 7))
    for strategy, group in curves.groupby("strategy"):
        ax.plot(pd.to_datetime(group["trade_date"]), group["equity"], label=strategy, linewidth=1.8)
    ax.axhline(INITIAL_CASH, color="black", linestyle="--", linewidth=1.0, label="initial_cash")
    ax.set_title("LightGBM Daily Static IM Backtest")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run_sample(
    label: str,
    execution: pd.DataFrame,
    predictions: pd.DataFrame,
    meta: ContractMeta,
) -> Dict[str, object]:
    target_by_date = {
        pd.Timestamp(row.target_date).normalize(): int(row.target_position)
        for row in predictions.itertuples(index=False)
    }
    model_curve, model_trades, model_metrics = backtest(f"model_{label}", execution, target_by_date, meta)
    long_curve, long_trades, long_metrics = backtest(
        f"always_long_{label}", execution, {}, meta, default_target=1
    )
    short_curve, short_trades, short_metrics = backtest(
        f"always_short_{label}", execution, {}, meta, default_target=-1
    )
    curves = pd.concat([model_curve, long_curve, short_curve], ignore_index=True)
    trades = pd.concat([model_trades, long_trades, short_trades], ignore_index=True)
    monthly = monthly_metrics(curves)
    metrics = pd.DataFrame([model_metrics, long_metrics, short_metrics])
    return {"curves": curves, "trades": trades, "monthly": monthly, "metrics": metrics}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta = load_contract_meta()
    panel = build_feature_panel(load_index_data())
    im_daily = load_im_daily_execution()

    latest_feature_rows = latest_complete_feature_rows(panel)
    latest_feature_date = pd.Timestamp(latest_feature_rows["feature_date"].max())
    train_target_end = latest_feature_date
    train = select_rows(panel, start=None, end=train_target_end)
    if train.empty:
        raise ValueError("Train sample is empty.")
    if train["target_exit_date"].max() > train_target_end:
        raise ValueError("Training data uses a target after the latest available feature date.")

    active_params = training_params(len(train))
    model = lgb.LGBMClassifier(**active_params)
    model.fit(train[FEATURE_COLUMNS].astype(float), train["target_up"].astype(int))
    train_pred = predict(model, train)
    latest_signal = predict_latest_signal(model, panel)

    train_execution = im_daily[
        im_daily["trade_date"].between(pd.Timestamp("2022-07-22"), train_target_end)
    ].copy()
    train_result = run_sample("train", train_execution, train_pred, meta)

    train_pred.to_csv(OUT_DIR / "train_predictions.csv", index=False, encoding="utf-8")
    latest_signal.to_csv(OUT_DIR / "latest_signal.csv", index=False, encoding="utf-8")
    write_json(OUT_DIR / "latest_signal.json", latest_signal.iloc[0].to_dict())

    train_result["curves"].to_csv(OUT_DIR / "train_equity_curve.csv", index=False, encoding="utf-8")
    train_result["trades"].to_csv(OUT_DIR / "train_trade_log.csv", index=False, encoding="utf-8")
    train_result["monthly"].to_csv(OUT_DIR / "train_monthly_returns.csv", index=False, encoding="utf-8")
    train_result["metrics"].to_csv(OUT_DIR / "train_metrics.csv", index=False, encoding="utf-8")
    plot_equity(train_result["curves"], OUT_DIR / "train_equity_curve.png")

    model.booster_.save_model(str(OUT_DIR / "lightgbm_model.txt"))
    model.booster_.save_model(str(OUT_DIR / f"lightgbm_model_{latest_feature_date:%Y%m%d}.txt"))
    pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "gain": model.booster_.feature_importance(importance_type="gain"),
            "split": model.booster_.feature_importance(importance_type="split"),
        }
    ).to_csv(OUT_DIR / "feature_importance.csv", index=False, encoding="utf-8")

    summary = {
        "project": "LightGBM daily static IM teaching example",
        "warning": "Latest-row signal is a forecast for the next trading open to the second next trading open; that latest row is not used as a labeled training row unless its future target is already known.",
        "no_rf_features": True,
        "params": active_params,
        "min_child_samples_rule": f"round_half_up(train_rows * {MIN_CHILD_SAMPLES_RATE})",
        "feature_count": len(FEATURE_COLUMNS),
        "feature_columns": FEATURE_COLUMNS,
        "target_definition": "For features at T close, label T+1 open -> T+2 open return.",
        "latest_feature_date": latest_feature_date,
        "train_target_end": train_target_end,
        "train_target_entry_start": train["target_entry_date"].min(),
        "train_target_entry_end": train["target_entry_date"].max(),
        "train_target_exit_start": train["target_exit_date"].min(),
        "train_target_exit_end": train["target_exit_date"].max(),
        "train_rows": int(len(train)),
        "latest_signal": latest_signal.iloc[0].to_dict(),
        "actual_tree_count": int(model.booster_.num_trees()),
        "classification": {
            "train": classification_metrics("train", train_pred),
        },
        "backtest": {
            "train": train_result["metrics"].to_dict("records"),
        },
        "contract": meta.__dict__,
    }
    write_json(OUT_DIR / "summary.json", summary)
    print(json.dumps(json_sanitize(summary), ensure_ascii=False, indent=2, allow_nan=False, default=json_default))


if __name__ == "__main__":
    main()
