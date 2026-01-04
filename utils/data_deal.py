# -*- coding: utf-8 -*-


from __future__ import annotations

import argparse
import shutil
from pathlib import Path
import pandas as pd



COLUMN_ALIASES = {
    # current
    "I": "current",
    "i": "current",
    "current(A)": "current",
    "Current": "current",
    "电流": "current",
    "电流(A)": "current",
    # force
    "F": "force",
    "f": "force",
    "Force": "force",
    "force(N)": "force",
    "输出力": "force",
    "制动力": "force",
    "力(N)": "force",
    # time
    "t": "time",
    "Time": "time",
    "time(s)": "time",
    "时间": "time",
    "时间(s)": "time",
    # displacement / speed
    "x": "disp",
    "displacement": "disp",
    "位移": "disp",
    "v": "speed",
    "speed": "speed",
    "速度": "speed",
}


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = []
    for c in df.columns:
        c2 = str(c).strip()
        cols.append(COLUMN_ALIASES.get(c2, c2))
    df.columns = cols
    return df


def _load_any(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":

        for enc in ("utf-8-sig", "utf-8", "gbk"):
            try:
                return pd.read_csv(path, encoding=enc)
            except Exception:
                continue

        return pd.read_csv(path, encoding="latin1")
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file type: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, default="The dataset we collected", help="Folder of raw files")
    parser.add_argument("--dst", type=str, default="data/raw", help="Output folder for organized raw files")
    parser.add_argument("--rewrite", action="store_true", help="Rewrite files with normalized column names")
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    if not src.exists():
        raise FileNotFoundError(f"Source folder not found: {src.resolve()}")

    _safe_mkdir(dst)

    exts = (".csv", ".xlsx", ".xls")
    files = [p for p in src.rglob("*") if p.is_file() and p.suffix.lower() in exts]

    if not files:
        print(f"[WARN] No data files found in: {src.resolve()}")
        return

    print(f"[INFO] Found {len(files)} files. Copying to: {dst.resolve()}")

    for p in files:
        out = dst / p.name
        shutil.copy2(p, out)
        print(f"  - copied: {p.name}")

        if args.rewrite:
            try:
                df = _load_any(out)
                df = _normalize_columns(df)

                out_csv = out.with_suffix(".csv")
                df.to_csv(out_csv, index=False, encoding="utf-8-sig")
                if out_csv != out:
                    out.unlink(missing_ok=True)
                print(f"    -> rewritten as: {out_csv.name}")
            except Exception as e:
                print(f"[WARN] Failed to normalize {out.name}: {e}")

    print("[DONE] Raw dataset organized.")


if __name__ == "__main__":
    main()
