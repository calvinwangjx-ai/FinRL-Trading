"""Rolling quarterly backtest of ML bucket selection (2018-03 -> 2025-10) on the bundled offline prices.

Reads the q_*/ folders written by run_rolling_selection.sh and compares, with a 0.1% trading cost:
  * ML top-25% per bucket (by best-model rank), equal weight, rebalanced only on the quarterly dates
  * equal weight over every stock the model scored (quarterly)
  * ML top-25% through BacktestEngine.run_backtest's default path (forward-fills weights -> daily rebalance)
  * SPY / QQQ buy & hold (the bundled DB only has them from 2020-12-01)
Picks without a price in the bundled DB on the rebalance date are dropped (see coverage.csv).

Note: BacktestEngine._create_bt_strategy uses bt.algos.RunAfterDate(first_signal), which is strictly-after in
bt and skips the first rebalance; the quarterly runs here anchor it one trading day earlier instead.

Usage:
    python examples/offline_rolling_backtest/backtest_rolling.py --run-dir <out_dir> --db <db> [--title NASDAQ-100]
Writes coverage.csv, equity_curves.csv, weights_top25.csv and rolling_backtest.png into --run-dir.
"""
import argparse, glob, os, sqlite3, sys
import numpy as np, pandas as pd
import bt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
_ap.add_argument("--run-dir", required=True); _ap.add_argument("--db", required=True)
_ap.add_argument("--title", default="S&P 500")
_args = _ap.parse_args()
R, DB = os.path.abspath(_args.run_dir), os.path.abspath(_args.db)
sys.path.insert(0, ROOT)
from src.backtest.backtest_engine import BacktestEngine, BacktestConfig

TD_MAP = {"03-31": ("06-01", 0), "06-30": ("09-01", 0), "09-30": ("12-01", 0), "12-31": ("03-01", 1)}
def tradedate(d):
    md, dy = TD_MAP[d[5:]]; return pd.Timestamp(f"{int(d[:4]) + dy}-{md}")

# ---- collect quarterly predictions ----
preds = []
for qdir in sorted(glob.glob(f"{R}/q_*")):
    f = glob.glob(f"{qdir}/sp500_ml_bucket_predictions_*.csv")
    if not f: print("MISSING predictions:", qdir); continue
    p = pd.read_csv(f[0]); p["td"] = tradedate(qdir.split("q_")[1]); preds.append(p)
preds = pd.concat(preds)
print(f"{preds['td'].nunique()} rebalance dates: {preds['td'].min().date()} -> {preds['td'].max().date()}")

# ---- prices ----
tick = sorted(set(preds["tic"]) | {"SPY", "QQQ"})
conn = sqlite3.connect(DB)
px = pd.read_sql(f"SELECT ticker tic, date, adj_close FROM price_data WHERE date >= '2018-02-01' "
                 f"AND ticker IN ({','.join('?' * len(tick))})", conn, params=tick)
conn.close()
wide = px.pivot(index="date", columns="tic", values="adj_close"); wide.index = pd.to_datetime(wide.index)
wide = wide.sort_index()
wide = wide[wide.index >= wide.index[wide.index < "2018-03-01"][-1]]  # keep 1 day before first rebalance
bench = wide[["SPY", "QQQ"]]; prices = wide.drop(columns=["SPY", "QQQ"])
END = prices.index.max()

# ---- weights: top-25% per bucket (by best-model rank) vs all scored stocks, equal weight ----
def build(top):
    rows, cov = {}, []
    for td, g in preds.groupby("td"):
        d = prices.index[prices.index.searchsorted(td)]
        if top:
            g = g[g["rank_best"] <= np.ceil(g.groupby("bucket")["tic"].transform("size") * 0.25)]
        ok = [t for t in g["tic"] if t in prices.columns and pd.notna(prices.at[d, t])]
        cov.append((d.date(), len(g), len(ok)))
        rows[d] = pd.Series(1.0 / len(ok), index=ok)
    w = pd.DataFrame(rows).T.fillna(0.0).sort_index()
    return w, pd.DataFrame(cov, columns=["rebalance", "picked", "priced"])

prices_bt = prices.ffill().bfill()  # bfill only affects pre-listing dates, where weight is always 0

def run_quarterly(name, w):
    eng = BacktestEngine(BacktestConfig(start_date="2018-03-01", end_date=str(END.date()), benchmark_tickers=[]))
    p = prices_bt[w.columns]
    # Same algo stack as BacktestEngine._create_bt_strategy, but its RunAfterDate(first_signal)
    # is strictly-after in bt and would skip the first rebalance; anchor it one day earlier.
    strat = bt.Strategy(name, [
        bt.algos.RunAfterDate(p.index[0]),
        bt.algos.RunOnDate(*w.index.tolist()),               # rebalance only on signal dates
        bt.algos.SelectThese(list(w.columns)),
        bt.algos.WeighTarget(w),
        bt.algos.Rebalance(),
    ])
    res = bt.run(eng._build_bt_backtest(strat, p, p))
    v = res[name].prices
    return v[v.index >= w.index[0]]

def run_engine_default(name, w):
    eng = BacktestEngine(BacktestConfig(start_date="2018-03-01", end_date=str(END.date()), benchmark_tickers=[]))
    v = eng.run_backtest(name, prices_bt[w.columns].copy(), w).portfolio_values
    return v[v.index >= w.index[0]]

def stats(v):
    v = v.dropna(); r = v.pct_change().dropna(); yrs = (v.index[-1] - v.index[0]).days / 365.25
    cum = v.iloc[-1] / v.iloc[0]; ann = cum ** (1 / yrs) - 1; vol = r.std() * np.sqrt(252)
    mdd = (v / v.cummax() - 1).min()
    return dict(Cumulative=cum, AnnReturn=ann, AnnVol=vol, Sharpe=r.mean() / r.std() * np.sqrt(252),
                MaxDD=mdd, Calmar=ann / abs(mdd), WinRate=(r > 0).mean())

w_top, cov_top = build(True); w_all, cov_all = build(False)
cov = cov_top.merge(cov_all, on="rebalance", suffixes=("_top25", "_all"))
cov.to_csv(f"{R}/coverage.csv", index=False)
print("\nPrice coverage per rebalance (picked vs priced):"); print(cov.to_string(index=False))

curves = {
    "ML Top-25% (quarterly rebal)": run_quarterly("ML_Top25_Q", w_top),
    "EW all scored (quarterly rebal)": run_quarterly("EW_All_Q", w_all),
    "ML Top-25% (engine default: daily rebal)": run_engine_default("ML_Top25_D", w_top),
}
fmt = {"Cumulative": "{:.2f}x", "AnnReturn": "{:.2%}", "AnnVol": "{:.2%}", "Sharpe": "{:.2f}",
       "MaxDD": "{:.2%}", "Calmar": "{:.2f}", "WinRate": "{:.2%}"}
def table(cs):
    t = pd.DataFrame({k: stats(v) for k, v in cs.items()}).T
    return t.to_string(formatters={k: v.format for k, v in fmt.items()})

print(f"\n=== Full period {curves[next(iter(curves))].index[0].date()} -> {END.date()} ===")
print(table(curves))

bs = bench.dropna().index[0]
sub = {k: v[v.index >= bs] for k, v in curves.items()}
sub.update({f"{b} buy & hold": bench[b].dropna() for b in ["SPY", "QQQ"]})
print(f"\n=== Benchmark window {bs.date()} -> {END.date()} (SPY/QQQ data starts here) ===")
print(table(sub))

# per-year returns
yr = pd.DataFrame({k: v.resample("YE").last().pct_change() for k, v in curves.items()})
first = {k: v.resample("YE").last().iloc[0] / v.iloc[0] - 1 for k, v in curves.items()}
yr.iloc[0] = pd.Series(first)
yr.index = yr.index.year
for b in ["SPY", "QQQ"]:
    s = bench[b].dropna(); y = s.resample("YE").last().pct_change(); y.iloc[0] = s.resample("YE").last().iloc[0] / s.iloc[0] - 1
    y.index = y.index.year; yr[f"{b}"] = y
print("\n=== Calendar-year returns (2018 from Mar 1; 2025 to", END.date(), "; SPY/QQQ 2020 = Dec only) ===")
print(yr.to_string(float_format=lambda x: f"{x:+.1%}"))

pd.DataFrame(curves).to_csv(f"{R}/equity_curves.csv")
w_top.to_csv(f"{R}/weights_top25.csv")

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(11, 5.5))
for k, v in curves.items():
    ax.plot(v.index, v / v.iloc[0], label=k, lw=1.6 if "quarterly" in k else 1, ls="-" if "quarterly" in k else ":")
for b, c in [("SPY", "gray"), ("QQQ", "black")]:
    s = bench[b].dropna(); anchor = curves["ML Top-25% (quarterly rebal)"]; a = anchor.loc[s.index[0]] / anchor.iloc[0]
    ax.plot(s.index, s / s.iloc[0] * a, "--", color=c, lw=1, label=f"{b} (from {s.index[0].date()}, rebased to strategy)")
for d in w_top.index: ax.axvline(d, color="#ddd", lw=.5, zorder=0)
ax.set_yscale("log"); ax.set_ylabel("Growth of $1 (log)"); ax.grid(alpha=.3); ax.legend(fontsize=8)
ax.set_title(f"Rolling quarterly ML stock selection ({_args.title}), {curves[next(iter(curves))].index[0].date()} → {END.date()}")
fig.tight_layout(); fig.savefig(f"{R}/rolling_backtest.png", dpi=130)
print(f"\nChart: {R}/rolling_backtest.png")
