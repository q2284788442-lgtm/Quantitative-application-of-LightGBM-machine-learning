"""Daily production entry point: update index data, then retrain and score."""

from __future__ import annotations

import argparse
import contextlib
import sys
from datetime import datetime
from pathlib import Path
from typing import TextIO


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "outputs" / "daily_runs"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_final_lightgbm_im_daily_static import main as train_main
from scripts.update_1000_dayk_akshare import update_index_csv


class Tee:
    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update AkShare data and run daily LightGBM training.")
    parser.add_argument("--skip-update", action="store_true", help="Only run training.")
    parser.add_argument("--skip-train", action="store_true", help="Only update the CSV.")
    parser.add_argument("--dry-run-update", action="store_true", help="Fetch AkShare data but do not write the CSV.")
    parser.add_argument("--log-dir", type=Path, default=LOG_DIR)
    parser.add_argument(
        "--ready-time",
        default="16:00",
        help="Before this local HH:MM time, same-day AkShare rows are dropped.",
    )
    parser.add_argument(
        "--allow-intraday-row",
        action="store_true",
        help="Allow same-day AkShare rows before --ready-time.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.log_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.log_dir / f"daily_update_train_{datetime.now():%Y%m%d_%H%M%S}.log"

    with log_path.open("w", encoding="utf-8") as log_file:
        tee_out = Tee(sys.__stdout__, log_file)
        tee_err = Tee(sys.__stderr__, log_file)
        with contextlib.redirect_stdout(tee_out), contextlib.redirect_stderr(tee_err):
            print(f"[daily] project_root={ROOT}")
            print(f"[daily] log_path={log_path}")
            print(f"[daily] started_at={datetime.now().isoformat(timespec='seconds')}")

            if not args.skip_update:
                print("[daily] updating index CSV from AkShare")
                result = update_index_csv(
                    dry_run=args.dry_run_update,
                    ready_time=args.ready_time,
                    allow_intraday_row=args.allow_intraday_row,
                )
                print(f"[daily] update_result={result}")
            else:
                print("[daily] skip_update=True")

            if not args.skip_train:
                print("[daily] running training and latest signal generation")
                train_main()
            else:
                print("[daily] skip_train=True")

            print(f"[daily] finished_at={datetime.now().isoformat(timespec='seconds')}")


if __name__ == "__main__":
    main()
