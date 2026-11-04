"""
DRL allocator for one rebalance quarter — a port of src/strategies/fundamental_portfolio_drl.py + rl_model.py.

Faithful to the repo:
  * universe = that quarter's ML stock picks
  * data window  = [trade_date - 3y, next_trade_date)            (max_rolling_window = 1095 days)
  * FinRL technical indicators (config.INDICATORS) via stockstats, 252-day covariance lookback
  * train on [trade_date - 3y, trade_date - 1y), validate on [trade_date - 1y, trade_date)  (testing_window = 365 days)
  * A2C / PPO / DDPG with the repo's CPU hyper-parameters & timesteps; best = highest validation cumulative return
  * best model trades [trade_date, next_trade_date) and emits DAILY softmax weights
  * env: FinRL StockPortfolioEnv logic (softmax weights, reward = portfolio value), with the repo's
    custom state = per-stock vols + top-K(10) covariance eigenvalues + per-stock indicators

Deviations (forced or documented):
  * the repo's custom env (k_eig) is not in the repo -> re-implemented from its state_space formula/comments
  * reward_scaling=1e-4 is applied (repo passes it; stock FinRL env ignores it)
  * prices are split/dividend-adjusted (repo used unadjusted prccd for 'close')
  * indicators z-scored with TRAINING-window stats only (repo's zscore helper uses all data)
  * stocks without full price history over the window are excluded (repo would drop dates instead,
    which empties the training set for recent IPOs)
"""
import argparse, json, os, random, sqlite3, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import torch
torch.set_num_threads(1)
import gymnasium as gym
from gymnasium import spaces
from stockstats import StockDataFrame as Sdf
from stable_baselines3 import A2C, PPO, DDPG

INDICATORS = ["macd", "boll_ub", "boll_lb", "rsi_30", "cci_30", "dx_30", "close_30_sma", "close_60_sma"]
K_EIG, LOOKBACK = 10, 252
MAX_ROLL, TEST_WIN = pd.Timedelta(days=1095), pd.Timedelta(days=365)
TIMESTEPS = {"a2c": 50_000, "ppo": 80_000, "ddpg": 50_000}


def seed_all(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


class PortfolioEnv(gym.Env):
    """FinRL StockPortfolioEnv mechanics with a flat state: vols + top-K eigenvalues + indicators."""
    def __init__(self, days, reward_scaling=1e-4, initial_amount=1_000_000):
        self.days, self.n = days, days[0]["close"].shape[0]
        self.reward_scaling, self.initial_amount = reward_scaling, initial_amount
        dim = days[0]["state"].shape[0]
        self.action_space = spaces.Box(low=0, high=1, shape=(self.n,))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(dim,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        self.day, self.value = 0, self.initial_amount
        self.returns, self.weights = [0.0], [np.full(self.n, 1 / self.n)]
        return self.days[0]["state"], {}

    @staticmethod
    def softmax(a):
        e = np.exp(a - np.max(a)); return e / e.sum()

    def step(self, action):
        if self.day >= len(self.days) - 1:
            return self.days[self.day]["state"], 0.0, True, False, {}
        w = self.softmax(np.asarray(action, dtype=np.float64))
        prev = self.days[self.day]["close"]; self.day += 1; cur = self.days[self.day]["close"]
        r = float(np.sum((cur / prev - 1) * w))
        self.value *= 1 + r
        self.returns.append(r); self.weights.append(w)
        done = self.day >= len(self.days) - 1
        return self.days[self.day]["state"], self.value * self.reward_scaling, done, False, {}


def build_days(df, dates, ind_mu, ind_sd):
    """Per-day dict(state, close). df: long frame with close + indicators + cov."""
    out = []
    for d in dates:
        g = df[df.date == d]
        cov = g["cov"].iloc[0]
        vols = np.sqrt(np.clip(np.diag(cov), 0, None) * 252)
        eig = np.sort(np.linalg.eigvalsh(cov * 252))[::-1][:K_EIG]
        eig = np.pad(eig, (0, K_EIG - len(eig)))
        ind = ((g[INDICATORS].values - ind_mu) / ind_sd).astype(np.float64)
        state = np.concatenate([vols, eig, ind.T.ravel()]).astype(np.float32)
        out.append({"state": np.nan_to_num(state), "close": g["close"].values.astype(np.float64)})
    return out


def run_episode(model, days):
    env = PortfolioEnv(days); obs, _ = env.reset(); done = False
    while not done:
        a, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, _ = env.step(a)
    return env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True); ap.add_argument("--tickers", required=True)
    ap.add_argument("--trade-date", required=True); ap.add_argument("--next-date", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); seed_all(42); t0 = time.time()
    td, nd = pd.Timestamp(a.trade_date), pd.Timestamp(a.next_date)
    tick = a.tickers.split(","); start = td - MAX_ROLL

    conn = sqlite3.connect(a.db)
    px = pd.read_sql(f"SELECT ticker tic, date, open, high, low, close, adj_close, volume FROM price_data "
                     f"WHERE ticker IN ({','.join('?' * len(tick))}) AND date >= ? AND date <= ?",
                     conn, params=tick + [str(start.date()), str(nd.date())])
    conn.close()
    px["date"] = pd.to_datetime(px["date"])
    f = px["adj_close"] / px["close"]
    for c in ["open", "high", "low"]: px[c] = px[c] * f
    px["close"] = px["adj_close"]
    px = px.drop(columns="adj_close").drop_duplicates(["tic", "date"])

    # keep stocks with full coverage of the window (dates where all kept stocks trade)
    all_dates = np.sort(px.date.unique())
    cnt = px.groupby("tic").date.nunique()
    keep = sorted(cnt[cnt >= 0.98 * len(all_dates)].index)
    dropped = sorted(set(tick) - set(keep))
    px = px[px.tic.isin(keep)]
    full = px.groupby("date").tic.nunique(); full = full[full == len(keep)].index
    px = px[px.date.isin(full)].sort_values(["tic", "date"])
    info = {"trade_date": a.trade_date, "n_picks": len(tick), "n_used": len(keep), "dropped": dropped}
    if len(keep) < 2:
        json.dump({**info, "error": "too few stocks"}, open(f"{a.out}/info.json", "w")); return

    # FinRL FeatureEngineer.add_technical_indicator (stockstats, per ticker)
    parts = []
    for t, g in px.groupby("tic"):
        s = Sdf.retype(g.drop(columns="tic").set_index("date").copy())
        gi = g.copy()
        for ind in INDICATORS: gi[ind] = s[ind].values
        parts.append(gi)
    df = pd.concat(parts).sort_values(["date", "tic"]).reset_index(drop=True)
    df[INDICATORS] = df[INDICATORS].replace([np.inf, -np.inf], np.nan).groupby(df.tic).ffill().fillna(0)

    # 252-day covariance of daily returns (repo: pivot close over lookback window)
    wide = df.pivot(index="date", columns="tic", values="close")[keep]
    rets = wide.pct_change()
    dates = wide.index
    covs = {dates[i]: rets.iloc[i - LOOKBACK + 1:i + 1].cov().values for i in range(LOOKBACK, len(dates))}
    df = df[df.date.isin(covs)].copy(); df["cov"] = df.date.map(covs)

    d_all = np.sort(df.date.unique())
    tr = [d for d in d_all if d < td - TEST_WIN]
    va = [d for d in d_all if td - TEST_WIN <= d < td]
    te = [d for d in d_all if td <= d < nd]
    trm = df[df.date.isin(tr)]
    mu, sd = trm[INDICATORS].mean().values, trm[INDICATORS].std().replace(0, 1).values
    days_tr, days_va, days_te = (build_days(df, x, mu, sd) for x in (tr, va, te))
    info.update(n_train_days=len(tr), n_val_days=len(va), n_trade_days=len(te))

    env = PortfolioEnv(days_tr)
    models = {
        "a2c": A2C("MlpPolicy", env, n_steps=5, ent_coef=0.005, learning_rate=0.0002, seed=42, verbose=0, device="cpu"),
        "ppo": PPO("MlpPolicy", env, n_steps=2048, ent_coef=0.005, learning_rate=0.0001, batch_size=128, seed=42, verbose=0, device="cpu"),
        "ddpg": DDPG("MlpPolicy", env, batch_size=128, buffer_size=50000, learning_rate=0.001, seed=42, verbose=0, device="cpu"),
    }
    val = {}
    for k, m in models.items():
        t1 = time.time(); m.learn(total_timesteps=TIMESTEPS[k])
        e = run_episode(m, days_va); val[k] = float(np.prod(1 + np.array(e.returns)))
        info[f"train_sec_{k}"] = round(time.time() - t1)
    best = max(val, key=val.get)
    cv = np.array([d["close"] for d in days_va])
    ew_val = float(np.prod(1 + (cv[1:] / cv[:-1] - 1).mean(axis=1)))
    info.update(val_return=val, val_return_equal_weight=ew_val, best_model=best)

    # trade quarter: weight decided at close of day t (from state t) is held from close t to close t+1
    m = models[best]; rows = {}
    for d, day in zip(te, days_te):
        act, _ = m.predict(day["state"], deterministic=True)
        rows[d] = PortfolioEnv.softmax(np.asarray(act, dtype=np.float64))
    w = pd.DataFrame(rows, index=keep).T; w.index.name = "date"
    w.to_csv(f"{a.out}/drl_weights.csv")
    info["elapsed_sec"] = round(time.time() - t0)
    info["avg_max_weight"] = float(w.max(1).mean()); info["avg_daily_turnover"] = float(w.diff().abs().sum(1).iloc[1:].mean()) if len(w) > 1 else 0.0
    json.dump(info, open(f"{a.out}/info.json", "w"), indent=2, default=str)
    print(json.dumps(info, default=str))


if __name__ == "__main__":
    main()
