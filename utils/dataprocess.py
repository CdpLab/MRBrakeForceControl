# -*- coding: utf-8 -*-


from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


COLUMN_ALIASES = {
    "I": "current", "i": "current", "Current": "current", "电流": "current", "电流(A)": "current",
    "F": "force", "f": "force", "Force": "force", "制动力": "force", "输出力": "force", "力(N)": "force",
    "t": "time", "Time": "time", "时间": "time", "时间(s)": "time",
}


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [COLUMN_ALIASES.get(str(c).strip(), str(c).strip()) for c in df.columns]
    return df


def _read_any(p: Path) -> pd.DataFrame:
    suf = p.suffix.lower()
    if suf == ".csv":
        for enc in ("utf-8-sig", "utf-8", "gbk"):
            try:
                return pd.read_csv(p, encoding=enc)
            except Exception:
                continue
        return pd.read_csv(p, encoding="latin1")
    if suf in (".xlsx", ".xls"):
        return pd.read_excel(p)
    raise ValueError(f"Unsupported: {p.name}")


def _crop(df: pd.DataFrame, crop_head: int, crop_tail: int) -> pd.DataFrame:
    n = len(df)
    start = min(max(crop_head, 0), n)
    end = n - min(max(crop_tail, 0), n)
    if end <= start:
        return df.iloc[0:0].copy()
    return df.iloc[start:end].copy()


def _clip_outliers(series: pd.Series, z_th: float = 6.0) -> pd.Series:

    x = series.to_numpy(dtype=float)
    mu = np.nanmean(x)
    sd = np.nanstd(x) + 1e-12
    z = (x - mu) / sd
    x = np.where(np.abs(z) > z_th, np.nan, x)
    return pd.Series(x).interpolate(limit_direction="both")


def _smooth_ma(series: pd.Series, win: int = 5) -> pd.Series:

    if win <= 1:
        return series
    return series.rolling(win, center=True, min_periods=1).mean()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=str, default="data/raw", help="Raw data folder")
    parser.add_argument("--out_csv", type=str, default="data/processed/force_control_dataset.csv", help="Output merged csv")
    parser.add_argument("--crop_head", type=int, default=0, help="Crop N samples at beginning")
    parser.add_argument("--crop_tail", type=int, default=0, help="Crop N samples at end")
    parser.add_argument("--smooth_win", type=int, default=1, help="Moving average window (1 means no smoothing)")
    parser.add_argument("--clip_z", type=float, default=0.0, help="Z-score outlier clipping threshold (0 disables)")
    parser.add_argument("--add_file_id", action="store_true", help="Add source filename column")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(f"raw_dir not found: {raw_dir.resolve()}")

    out_csv = Path(args.out_csv)
    _safe_mkdir(out_csv.parent)

    files = sorted([p for p in raw_dir.iterdir() if p.is_file() and p.suffix.lower() in (".csv", ".xlsx", ".xls")])
    if not files:
        print(f"[WARN] No raw files in {raw_dir.resolve()}")
        return

    merged = []
    for p in files:
        try:
            df = _read_any(p)
            df = _normalize_columns(df)


            if "current" not in df.columns or "force" not in df.columns:
                print(f"[SKIP] {p.name}: missing 'current' or 'force' columns. Got: {list(df.columns)}")
                continue


            keep_cols = [c for c in ["time", "current", "force"] if c in df.columns]
            df = df[keep_cols].copy()


            if "time" not in df.columns:
                df.insert(0, "time", np.arange(len(df), dtype=float))

            # 裁剪
            df = _crop(df, args.crop_head, args.crop_tail)

            # 数值化 & 清洗
            df["current"] = pd.to_numeric(df["current"], errors="coerce")
            df["force"] = pd.to_numeric(df["force"], errors="coerce")
            df["time"] = pd.to_numeric(df["time"], errors="coerce")
            df = df.dropna(subset=["current", "force", "time"]).reset_index(drop=True)

            if len(df) < 5:
                print(f"[SKIP] {p.name}: too few valid rows after cleaning.")
                continue

            # 去极端值
            if args.clip_z and args.clip_z > 0:
                df["current"] = _clip_outliers(df["current"], z_th=args.clip_z)
                df["force"] = _clip_outliers(df["force"], z_th=args.clip_z)

            # 平滑
            if args.smooth_win and args.smooth_win > 1:
                df["current"] = _smooth_ma(df["current"], win=args.smooth_win)
                df["force"] = _smooth_ma(df["force"], win=args.smooth_win)

            # 加文件来源信息，方便按文件名对齐/分组训练
            if args.add_file_id:
                df.insert(0, "file", p.stem)

            merged.append(df)
            print(f"[OK] {p.name}: {len(df)} rows")

        except Exception as e:
            print(f"[FAIL] {p.name}: {e}")

    if not merged:
        print("[ERROR] No valid data merged.")
        return

    out = pd.concat(merged, axis=0, ignore_index=True)

    # 最终保存
    out.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[DONE] merged csv saved to: {out_csv.resolve()}  (rows={len(out)})")


if __name__ == "__main__":
    main()
