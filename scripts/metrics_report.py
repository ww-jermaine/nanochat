#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from nanochat.metrics_report import generate_html_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a self-contained HTML GPU metrics report from nvidia-smi CSVs.")
    parser.add_argument("--metrics-dir", type=Path, default=None, help="Directory containing gpu_*.csv files (default: $NANOCHAT_BASE_DIR/metrics).")
    parser.add_argument("--out", type=Path, default=None, help="Output HTML path (default: <metrics-dir>/report.html).")
    args = parser.parse_args()

    out = generate_html_report(metrics_dir=args.metrics_dir, out_path=args.out)
    print(f"Wrote report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

