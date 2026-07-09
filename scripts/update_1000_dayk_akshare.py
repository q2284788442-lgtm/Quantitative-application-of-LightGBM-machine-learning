"""Update the CSI 1000 daily index CSV with AkShare data."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Dict

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data" / "1000_dayK_20190101_20260630.csv"
DEFAULT_SYMBOL = "csi000852"
DEFAULT_CODE = "000852.XSHG"
DEFAULT_START_DATE = "20190101"
DEFAULT_END_DATE = "20500101"
DEFAULT_READY_TIME = "16:00"
OUTPUT_COLUMNS = ["date", "code", "open", "high", "low", "close", "volume", "money"]


def fallback_plain_symbol(symbol: str) -> str:
    lowered = symbol.lower()
    for prefix in ["csi", "sh", "sz", "bj"]:
        if lowered.startswith(prefix):
            return symbol[len(prefix) :]
    return symbol


def fetch_akshare_index_daily(
    symbol: str,
    start_date: str,
    end_date: str,
    retries: int = 3,
    retry_sleep: float = 3.0,
) -> pd.DataFrame:
    try:
        import akshare as ak
    except ImportError as exc:
        raise RuntimeError(
            "akshare is not installed. Run `pip install -r requirements.txt` first."
        ) from exc

    methods = [
        (
            f"stock_zh_index_daily_em({symbol})",
            lambda: ak.stock_zh_index_daily_em(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
            ),
        ),
        (
            f"index_zh_a_hist({fallback_plain_symbol(symbol)})",
            lambda: ak.index_zh_a_hist(
                symbol=fallback_plain_symbol(symbol),
                period="daily",
                start_date=start_date,
                end_date=end_date,
            ),
        ),
    ]
    errors = []
    for attempt in range(1, retries + 1):
        for name, fetcher in methods:
            try:
                df = fetcher()
                if not df.empty:
                    return df
                errors.append(f"{name}: empty dataframe")
            except Exception as exc:  # noqa: BLE001 - keep upstream data-source errors visible
                errors.append(f"{name}: {type(exc).__name__}: {exc}")
        if attempt < retries:
            time.sleep(retry_sleep)

    raise RuntimeError("AkShare fetch failed after retries. " + " | ".join(errors[-6:]))


def normalize_akshare_daily(df: pd.DataFrame, code: str) -> pd.DataFrame:
    column_map = {
        "date": "date",
        "日期": "date",
        "open": "open",
        "开盘": "open",
        "high": "high",
        "最高": "high",
        "low": "low",
        "最低": "low",
        "close": "close",
        "收盘": "close",
        "volume": "volume",
        "成交量": "volume",
        "amount": "money",
        "成交额": "money",
    }
    renamed = df.rename(columns={col: column_map[col] for col in df.columns if col in column_map})
    missing = set(OUTPUT_COLUMNS) - {"code"} - set(renamed.columns)
    if missing:
        raise ValueError(f"AkShare data missing columns: {sorted(missing)}")

    out = renamed[["date", "open", "high", "low", "close", "volume", "money"]].copy()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out["code"] = code
    for col in ["open", "high", "low", "close", "volume", "money"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out[OUTPUT_COLUMNS].dropna(subset=["date", "open", "high", "low", "close"])
    out = out.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)
    return out


def normalize_project_csv(df: pd.DataFrame) -> pd.DataFrame:
    missing = set(OUTPUT_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Existing index CSV missing columns: {sorted(missing)}")

    out = df[OUTPUT_COLUMNS].copy()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out["code"] = out["code"].astype(str)
    for col in ["open", "high", "low", "close", "volume", "money"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.drop_duplicates(["date", "code"], keep="last").sort_values("date").reset_index(drop=True)


def parse_ready_time(value: str) -> tuple[int, int]:
    parts = value.split(":")
    if len(parts) != 2:
        raise ValueError("ready_time must use HH:MM format.")
    hour, minute = int(parts[0]), int(parts[1])
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError("ready_time must use HH:MM format.")
    return hour, minute


def drop_unready_today_rows(
    df: pd.DataFrame,
    ready_time: str,
    allow_intraday_row: bool,
    now: datetime | None = None,
) -> tuple[pd.DataFrame, int]:
    if allow_intraday_row:
        return df, 0

    current = now or datetime.now()
    hour, minute = parse_ready_time(ready_time)
    ready_at = current.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if current >= ready_at:
        return df, 0

    today = pd.Timestamp(current.date())
    keep = pd.to_datetime(df["date"]).dt.normalize().lt(today)
    dropped = int((~keep).sum())
    return df.loc[keep].copy(), dropped


def write_project_csv(df: pd.DataFrame, path: Path) -> None:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False, encoding="utf-8")


def backup_existing_csv(path: Path) -> Path:
    backup_dir = path.parent / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"{path.stem}_{stamp}{path.suffix}"
    shutil.copy2(path, backup_path)
    return backup_path


def update_index_csv(
    output: Path = DEFAULT_OUTPUT,
    symbol: str = DEFAULT_SYMBOL,
    code: str = DEFAULT_CODE,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    dry_run: bool = False,
    backup: bool = True,
    retries: int = 3,
    retry_sleep: float = 3.0,
    ready_time: str = DEFAULT_READY_TIME,
    allow_intraday_row: bool = False,
) -> Dict[str, object]:
    fetched = normalize_akshare_daily(
        fetch_akshare_index_daily(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            retries=retries,
            retry_sleep=retry_sleep,
        ),
        code,
    )
    fetched, dropped_unready_today_rows = drop_unready_today_rows(
        fetched,
        ready_time=ready_time,
        allow_intraday_row=allow_intraday_row,
    )
    if fetched.empty:
        raise ValueError(f"AkShare returned no rows for {symbol}.")

    existing = (
        normalize_project_csv(pd.read_csv(output))
        if output.exists()
        else pd.DataFrame(columns=OUTPUT_COLUMNS)
    )
    existing_other_codes = existing[~existing["code"].astype(str).eq(code)].copy()
    updated = pd.concat([existing_other_codes, fetched], ignore_index=True)
    updated = updated.drop_duplicates(["date", "code"], keep="last").sort_values(["date", "code"]).reset_index(drop=True)

    changed = True
    if output.exists() and not existing.empty:
        comparable_existing = existing[OUTPUT_COLUMNS].reset_index(drop=True)
        comparable_updated = updated[OUTPUT_COLUMNS].reset_index(drop=True)
        changed = not comparable_existing.equals(comparable_updated)

    backup_path = None
    if changed and not dry_run:
        if backup and output.exists():
            backup_path = backup_existing_csv(output)
        write_project_csv(updated, output)

    latest = updated[updated["code"].astype(str).eq(code)]["date"].max()
    return {
        "output": str(output),
        "symbol": symbol,
        "code": code,
        "start_date": start_date,
        "end_date": end_date,
        "rows": int(len(updated[updated["code"].astype(str).eq(code)])),
        "latest_date": None if pd.isna(latest) else pd.Timestamp(latest).strftime("%Y-%m-%d"),
        "changed": changed,
        "dry_run": dry_run,
        "backup_path": None if backup_path is None else str(backup_path),
        "retries": retries,
        "ready_time": ready_time,
        "allow_intraday_row": allow_intraday_row,
        "dropped_unready_today_rows": dropped_unready_today_rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update 000852.XSHG daily index CSV from AkShare.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL)
    parser.add_argument("--code", default=DEFAULT_CODE)
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-backup", action="store_true")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=3.0)
    parser.add_argument(
        "--ready-time",
        default=DEFAULT_READY_TIME,
        help="Before this local HH:MM time, same-day rows are treated as intraday and dropped.",
    )
    parser.add_argument(
        "--allow-intraday-row",
        action="store_true",
        help="Allow writing same-day rows even before --ready-time.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = update_index_csv(
        output=args.output,
        symbol=args.symbol,
        code=args.code,
        start_date=args.start_date,
        end_date=args.end_date,
        dry_run=args.dry_run,
        backup=not args.no_backup,
        retries=args.retries,
        retry_sleep=args.retry_sleep,
        ready_time=args.ready_time,
        allow_intraday_row=args.allow_intraday_row,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
