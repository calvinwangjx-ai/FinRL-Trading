"""Build an offline SQLite DB from the data bundled in the repo.

  * data/finrl_trading.7z           -> price_data table (~1.28M daily rows, ~500 tickers, 2015-06 .. 2025-10)
  * data/fundamental_data_full.csv  -> fundamental_data table (what ml_bucket_selection.py reads)

Usage:
    pip install py7zr
    python examples/offline_rolling_backtest/prepare_db.py --out /path/to/work/finrl_trading.db
"""
import argparse, os, sqlite3, tempfile, shutil
import pandas as pd
import py7zr

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output SQLite path (keep it outside the repo; ~450 MB)")
    a = ap.parse_args()
    out = os.path.abspath(a.out); os.makedirs(os.path.dirname(out), exist_ok=True)

    tmp = tempfile.mkdtemp(dir=os.path.dirname(out))
    try:
        with py7zr.SevenZipFile(os.path.join(ROOT, "data", "finrl_trading.7z")) as z:
            z.extractall(tmp)
        shutil.move(os.path.join(tmp, "finrl_trading.db"), out)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    fund = pd.read_csv(os.path.join(ROOT, "data", "fundamental_data_full.csv"), low_memory=False)
    with sqlite3.connect(out) as conn:
        fund.to_sql("fundamental_data", conn, if_exists="replace", index=False)
        n_px = conn.execute("SELECT COUNT(*) FROM price_data").fetchone()[0]
    print(f"Wrote {out}: price_data={n_px:,} rows, fundamental_data={len(fund):,} rows "
          f"({fund.ticker.nunique()} tickers, {fund.datadate.min()} .. {fund.datadate.max()})")


if __name__ == "__main__":
    main()
