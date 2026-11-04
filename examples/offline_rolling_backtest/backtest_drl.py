"""Backtest DRL-allocated ML picks vs equal weight, using the project's bt-based engine pieces.

Reads the q_*/ folders written by run_drl.sh and compares, with a 0.1% trading cost:
  * DRL allocator: the chosen model's daily weights (rebalanced daily), with and without trading cost
  * equal weight on exactly the stocks the DRL used each quarter (rebalanced quarterly)
  * the equal-weight curve of the selection run (backtest_rolling.py), and SPY / QQQ from 2020-12-01

Usage:
    python examples/offline_rolling_backtest/backtest_drl.py --run-dir <drl_out_dir> --db <db> \
        --selection-run-dir <selection_out_dir> [--title NASDAQ-100]
Writes per_quarter.csv, equity_curves.csv and drl_backtest.png into --run-dir.
"""
import argparse, glob, json, os, sqlite3, sys
import numpy as np, pandas as pd, bt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
_ap.add_argument("--run-dir", required=True); _ap.add_argument("--db", required=True)
_ap.add_argument("--selection-run-dir", required=True); _ap.add_argument("--title", default="NASDAQ-100")
_args = _ap.parse_args()
D, DB, SEL = (os.path.abspath(p) for p in (_args.run_dir, _args.db, _args.selection_run_dir))
sys.path.insert(0, ROOT)
from src.backtest.backtest_engine import BacktestEngine, BacktestConfig

# ---- DRL daily weights + per-quarter info ----
infos, wparts = [], []
for q in sorted(glob.glob(f"{D}/q_*")):
    if not os.path.exists(f"{q}/info.json"): print("MISSING", q); continue
    i = json.load(open(f"{q}/info.json")); infos.append(i)
    if os.path.exists(f"{q}/drl_weights.csv"):
        wparts.append(pd.read_csv(f"{q}/drl_weights.csv", index_col=0, parse_dates=True))
info = pd.DataFrame(infos).sort_values("trade_date").drop_duplicates("trade_date").reset_index(drop=True)
w_drl = pd.concat(wparts).sort_index().fillna(0.0)
w_drl = w_drl[~w_drl.index.duplicated()]
print(f"{len(info)} quarters, DRL weight rows {len(w_drl)}: {w_drl.index.min().date()} -> {w_drl.index.max().date()}")

# ---- prices ----
tick = sorted(set(w_drl.columns) | {"SPY", "QQQ"})
conn = sqlite3.connect(DB)
px = pd.read_sql(f"SELECT ticker tic, date, adj_close FROM price_data WHERE date >= '2018-02-01' "
                 f"AND ticker IN ({','.join('?' * len(tick))})", conn, params=tick)
conn.close()
wide = px.pivot(index="date", columns="tic", values="adj_close"); wide.index = pd.to_datetime(wide.index)
wide = wide.sort_index(); wide = wide[wide.index >= wide.index[wide.index < "2018-03-01"][-1]]
bench = wide[["SPY", "QQQ"]]; prices = wide.drop(columns=["SPY", "QQQ"]).ffill().bfill()

# equal weight on exactly the stocks the DRL used each quarter (first DRL row of each quarter)
reb = pd.to_datetime(info["trade_date"])
w_ew = {}
for d in reb:
    first = w_drl.index[w_drl.index >= d][0]
    used = w_drl.columns[w_drl.loc[first] > 0]
    w_ew[first] = pd.Series(1 / len(used), index=used)
w_ew = pd.DataFrame(w_ew).T.reindex(columns=w_drl.columns).fillna(0.0)

def run(name, w, cost=0.001):
    eng = BacktestEngine(BacktestConfig(start_date="2018-03-01", end_date=str(prices.index[-1].date()),
                                        benchmark_tickers=[], transaction_cost=cost))
    p = prices[w.columns]
    strat = bt.Strategy(name, [bt.algos.RunAfterDate(p.index[0]), bt.algos.RunOnDate(*w.index.tolist()),
                               bt.algos.SelectThese(list(w.columns)), bt.algos.WeighTarget(w), bt.algos.Rebalance()])
    v = bt.run(eng._build_bt_backtest(strat, p, p))[name].prices
    return v[v.index >= w.index[0]]

def stats(v):
    v = v.dropna(); r = v.pct_change().dropna(); yrs = (v.index[-1] - v.index[0]).days / 365.25
    cum = v.iloc[-1] / v.iloc[0]; ann = cum ** (1 / yrs) - 1; mdd = (v / v.cummax() - 1).min()
    return dict(Cumulative=cum, AnnReturn=ann, AnnVol=r.std() * np.sqrt(252), Sharpe=r.mean() / r.std() * np.sqrt(252),
                MaxDD=mdd, Calmar=ann / abs(mdd), WinRate=(r > 0).mean())

prev = pd.read_csv(f"{SEL}/equity_curves.csv", index_col=0, parse_dates=True)["ML Top-25% (quarterly rebal)"]
curves = {
    "DRL allocator (daily weights, 0.1% cost)": run("DRL", w_drl),
    "DRL allocator (no trading cost)": run("DRL_nocost", w_drl, cost=0.0),
    "Equal weight, same stocks (quarterly)": run("EW_same", w_ew),
    "Equal weight, all priced picks (prev. run)": prev.dropna(),
}
fmt = {"Cumulative": "{:.2f}x", "AnnReturn": "{:.2%}", "AnnVol": "{:.2%}", "Sharpe": "{:.2f}",
       "MaxDD": "{:.2%}", "Calmar": "{:.2f}", "WinRate": "{:.2%}"}
def table(cs):
    t = pd.DataFrame({k: stats(v) for k, v in cs.items()}).T
    return t.to_string(formatters={k: v.format for k, v in fmt.items()})

end = prices.index[-1].date()
print(f"\n=== Full period 2018-03-01 -> {end} ===\n" + table(curves))
bs = bench.dropna().index[0]
sub = {k: v[v.index >= bs] for k, v in curves.items()}
sub.update({f"{b} buy & hold": bench[b].dropna() for b in ["SPY", "QQQ"]})
print(f"\n=== Benchmark window {bs.date()} -> {end} ===\n" + table(sub))

yr = {}
for k, v in {**curves, **{b: bench[b].dropna() for b in ["SPY", "QQQ"]}}.items():
    ye = v.resample("YE").last(); y = ye.pct_change(); y.iloc[0] = ye.iloc[0] / v.iloc[0] - 1; y.index = y.index.year; yr[k] = y
print("\n=== Calendar-year returns ===\n" + pd.DataFrame(yr).to_string(float_format=lambda x: f"{x:+.1%}"))

# quarter-level diagnostics
q = info[["trade_date", "n_picks", "n_used", "best_model", "avg_max_weight", "avg_daily_turnover"]].copy()
q["val_best"] = info.apply(lambda r: r["val_return"][r["best_model"]] - 1, axis=1)
q["val_EW"] = info["val_return_equal_weight"] - 1
d_drl, d_ew = curves["DRL allocator (daily weights, 0.1% cost)"], curves["Equal weight, same stocks (quarterly)"]
nxt = list(pd.to_datetime(q.trade_date)[1:]) + [d_drl.index[-1] + pd.Timedelta(days=1)]
def qret(v, a, b):
    s = v[(v.index >= a) & (v.index < b)]; e = v[v.index < b]
    return e.iloc[-1] / s.iloc[0] - 1 if len(s) else np.nan
q["trade_DRL"] = [qret(d_drl, pd.Timestamp(a), b) for a, b in zip(q.trade_date, nxt)]
q["trade_EW"] = [qret(d_ew, pd.Timestamp(a), b) for a, b in zip(q.trade_date, nxt)]
pd.set_option("display.width", 200)
print("\n=== Per quarter ===")
print(q.to_string(index=False, formatters={c: "{:+.1%}".format for c in ["val_best", "val_EW", "trade_DRL", "trade_EW"]} |
      {"avg_max_weight": "{:.1%}".format, "avg_daily_turnover": "{:.1%}".format}))
print(f"\nBest-model counts: {q.best_model.value_counts().to_dict()}")
print(f"DRL beat same-stock EW in {(q.trade_DRL > q.trade_EW).sum()}/{len(q)} quarters; "
      f"validation picked a model that beat EW on validation in {(q.val_best > q.val_EW).sum()}/{len(q)}")
q.to_csv(f"{D}/per_quarter.csv", index=False); pd.DataFrame(curves).to_csv(f"{D}/equity_curves.csv")

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(11, 5.5))
style = {"DRL allocator (daily weights, 0.1% cost)": dict(lw=1.8), "DRL allocator (no trading cost)": dict(lw=1, ls=":"),
         "Equal weight, same stocks (quarterly)": dict(lw=1.6), "Equal weight, all priced picks (prev. run)": dict(lw=1, ls="-.")}
for k, v in curves.items(): ax.plot(v.index, v / v.iloc[0], label=k, **style[k])
anchor = curves["Equal weight, same stocks (quarterly)"]
for b, c in [("SPY", "gray"), ("QQQ", "black")]:
    s = bench[b].dropna(); a = anchor.loc[s.index[0]] / anchor.iloc[0]
    ax.plot(s.index, s / s.iloc[0] * a, "--", color=c, lw=1, label=f"{b} (from {s.index[0].date()}, rebased to EW)")
ax.set_yscale("log"); ax.set_ylabel("Growth of $1 (log)"); ax.grid(alpha=.3); ax.legend(fontsize=8)
ax.set_title(f"{_args.title} ML picks: DRL allocator vs equal weight, 2018-03-01 → {end}")
fig.tight_layout(); fig.savefig(f"{D}/drl_backtest.png", dpi=130); print(f"\nChart: {D}/drl_backtest.png")
