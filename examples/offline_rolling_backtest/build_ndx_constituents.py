"""Build point-in-time NASDAQ-100 membership in the `date,tickers` format of
data/sp500_historical_constituents.csv, from the yearly change files of
https://github.com/jmccarrell/n100tickers (src/nasdaq_100_ticker_history/n100-ticker-changes-YYYY.yaml).

The YAML is read as data with yaml.BaseLoader (no code from that repo is imported). BaseLoader keeps
every value a string; safe_load would turn the ticker `ON` (ON Semiconductor) into the boolean True.

Usage:
    git clone --depth 1 https://github.com/jmccarrell/n100tickers /path/to/n100tickers
    python -I examples/offline_rolling_backtest/build_ndx_constituents.py \
        --src /path/to/n100tickers/src/nasdaq_100_ticker_history --out /path/to/work/ndx_constituents.csv \
        [--db /path/to/work/finrl_trading.db]      # optional: print fundamentals/price coverage
"""
import argparse, os, sqlite3
import pandas as pd
import yaml

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--first-year", type=int, default=2014); ap.add_argument("--last-year", type=int, default=2026)
    ap.add_argument("--db", default=None)
    a = ap.parse_args()

    snap = {}
    for y in range(a.first_year, a.last_year + 1):
        with open(os.path.join(a.src, f"n100-ticker-changes-{y}.yaml")) as fh:
            d = yaml.load(fh, Loader=yaml.BaseLoader)
        cur = set(d["tickers_on_Jan_1"]); snap[f"{y}-01-01"] = set(cur)
        for dt in sorted(d.get("changes", {})):
            for op, t in d["changes"][dt].items():
                if op == "union": cur = cur | set(t)
                elif op == "difference": cur = cur - set(t)
                else: raise ValueError(f"unknown op {op} on {dt}")
            snap[str(dt)] = set(cur)
    df = pd.DataFrame({"date": list(snap), "tickers": [",".join(sorted(v)) for v in snap.values()]}).sort_values("date")
    df.to_csv(a.out, index=False)
    print(f"{len(df)} snapshots {df.date.min()} .. {df.date.max()} -> {a.out}")

    if a.db:
        fund = set(pd.read_csv(os.path.join(REPO, "data", "fundamental_data_full.csv"), usecols=["ticker"]).ticker)
        with sqlite3.connect(a.db) as c:
            px = {r[0] for r in c.execute("SELECT DISTINCT ticker FROM price_data")}
        hist = df.set_index(pd.to_datetime(df.date))
        for td in pd.date_range("2018-03-01", "2025-09-01", freq="12MS"):
            m = set(hist[hist.index <= td].iloc[-1].tickers.split(","))
            print(f"{td.date()} NDX={len(m)} with_fundamentals={len(m & fund)} with_prices={len(m & px)}")


if __name__ == "__main__":
    main()
