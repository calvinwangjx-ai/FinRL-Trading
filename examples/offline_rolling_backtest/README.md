# Offline rolling quarterly backtest (ML selection + DRL allocator)

Scripts that run the ML stock-selection pipeline (`src/strategies/ml_bucket_selection.py`) as a rolling,
point-in-time quarterly backtest from 2018-03-01 to 2025-10-16, then re-run the same picks with a DRL
allocator. Everything uses **only the data bundled in the repo** (`data/finrl_trading.7z`,
`data/fundamental_data_full.csv`, `data/sp500_historical_constituents.csv`), so it works without
Yahoo / FMP access or API keys.

| Script | What it does |
|---|---|
| `prepare_db.py` | Extracts `data/finrl_trading.7z` (daily prices) and loads `fundamental_data_full.csv` into one SQLite DB |
| `build_ndx_constituents.py` | Builds point-in-time NASDAQ-100 membership (`date,tickers`) from [jmccarrell/n100tickers](https://github.com/jmccarrell/n100tickers) |
| `run_rolling_selection.sh` | Runs `ml_bucket_selection.py` once per quarter (31 rebalances), each trained only on data available then |
| `backtest_rolling.py` | Top-25% per bucket, equal weight, quarterly rebalance vs. equal weight on all scored stocks vs. SPY/QQQ |
| `drl_quarter.py` | One quarter of the DRL allocator — a port of `src/strategies/fundamental_portfolio_drl.py` + `rl_model.py` |
| `run_drl.sh` | Runs `drl_quarter.py` for every rebalance date on the selection run's picks |
| `backtest_drl.py` | DRL daily weights vs. equal weight on the same stocks vs. SPY/QQQ |

## Usage

Keep the work directory outside the repo (the DB is ~450 MB).

```bash
pip install -r requirements.txt py7zr openpyxl pyyaml   # torch is only needed for the DRL step
W=/path/to/work

python examples/offline_rolling_backtest/prepare_db.py --out $W/finrl_trading.db

# S&P 500 universe (~25-30 min with 4 cores; NASDAQ-100 below takes ~10 min)
examples/offline_rolling_backtest/run_rolling_selection.sh $W/finrl_trading.db $W/sp500
python examples/offline_rolling_backtest/backtest_rolling.py --run-dir $W/sp500 --db $W/finrl_trading.db

# NASDAQ-100 universe
git clone --depth 1 https://github.com/jmccarrell/n100tickers $W/n100tickers
python -I examples/offline_rolling_backtest/build_ndx_constituents.py \
    --src $W/n100tickers/src/nasdaq_100_ticker_history --out $W/ndx_constituents.csv
examples/offline_rolling_backtest/run_rolling_selection.sh $W/finrl_trading.db $W/ndx $W/ndx_constituents.csv
python examples/offline_rolling_backtest/backtest_rolling.py --run-dir $W/ndx --db $W/finrl_trading.db --title NASDAQ-100

# DRL allocator on the NASDAQ-100 picks (~1h40m with 4 cores)
pip install torch stable-baselines3 gymnasium stockstats
examples/offline_rolling_backtest/run_drl.sh $W/finrl_trading.db $W/ndx $W/ndx_drl
python examples/offline_rolling_backtest/backtest_drl.py --run-dir $W/ndx_drl --db $W/finrl_trading.db --selection-run-dir $W/ndx
```

Both runners are resumable (finished quarters are skipped) and pin each process to one thread: several
multi-threaded XGBoost/LightGBM/sklearn processes sharing the same cores ran more than 10x slower.

## Method

* **Rebalance dates.** For each quarter-end datadate D, `--infer-date D --val-cutoff <D − 1 quarter>`; picks
  are bought on D's tradedate (first trading day on or after 03-01 / 06-01 / 09-01 / 12-01) and held to the next one.
* **Portfolio.** Top 25% of each sector bucket by the best model's rank, equal weight, 0.1% transaction cost.
  Picks with no price in the bundled DB on the rebalance date are dropped (`coverage.csv`).
* **NASDAQ-100.** `--universe nasdaq100` is not used: it calls the FMP API and filters history to *current*
  members (survivorship bias). Instead `run_rolling_selection.sh` creates a scratch project root whose
  `data/sp500_historical_constituents.csv` is the NASDAQ-100 membership file and whose `src` links to this
  repo's `src`; the selection script is unmodified (its logs still say "SP500").
* **DRL allocator.** Same windows, indicators, models, hyper-parameters and timesteps as the repo:
  data from trade date − 3y, 252-day covariance, FinRL `INDICATORS`, A2C (50k) / PPO (80k) / DDPG (50k)
  trained on [−3y, −1y), the model with the best validation return on [−1y, trade date) trades the quarter
  with daily softmax weights. Deviations, because the repo's version can't run as-is (its inputs and its
  custom `StockPortfolioEnv(k_eig=...)` are not in the repo):
  * the custom env is re-implemented from its `state_space` formula: per-stock vols + top-10 covariance
    eigenvalues + per-stock indicators; FinRL's step/reward/softmax logic is kept
  * `reward_scaling=1e-4` is applied (stock FinRL's env ignores it)
  * prices are split/dividend adjusted; indicators are z-scored with training-window statistics only
  * stocks without full price history over the window are excluded (0–1 per quarter)

## Results (2018-03-01 → 2025-10-16, 31 rebalances)

| Strategy | Cumulative | Ann. return | Sharpe | Max DD |
|---|---|---|---|---|
| S&P 500 — ML top-25%, equal weight | 2.20x | 10.9% | 0.62 | −38.7% |
| S&P 500 — equal weight, all scored stocks | 2.48x | 12.6% | 0.72 | −37.8% |
| NASDAQ-100 — ML top-25%, equal weight | 3.91x | 19.6% | 0.89 | −33.2% |
| NASDAQ-100 — equal weight, all scored stocks | 3.03x | 15.7% | 0.78 | −32.3% |
| NASDAQ-100 — ML top-25%, DRL allocator (0.1% cost) | 3.99x | 19.9% | 0.89 | −32.9% |
| NASDAQ-100 — ML top-25%, DRL allocator (no cost) | 4.26x | 20.9% | 0.92 | −32.7% |
| NASDAQ-100 — ML top-25%, equal weight on the DRL's stocks | 4.04x | 20.1% | 0.91 | −33.0% |

SPY / QQQ are only in the bundled DB from 2020-12-01; over 2020-12-01 → 2025-10-16 they returned 1.96x / 2.05x,
vs. 2.28x for NASDAQ-100 ML top-25% (equal weight) and 1.70x for the S&P 500 version.

Observations:
* ML selection added value on NASDAQ-100 (vs. equal weight over all scored members) but not on the S&P 500.
* The DRL allocator was indistinguishable from equal weight: FinRL's env bounds actions to [0, 1] before a
  softmax, so no stock can get more than e ≈ 2.7x another's weight. Average max weight was 5–10%,
  DDPG (chosen in 15/31 quarters) usually ended up producing the same weights every day, and the model that
  won validation beat equal weight in the following quarter in only 18/31 quarters. Daily rebalancing costs
  (~1%/yr) offset its ~0.8%/yr gross edge.

## Caveats

* **Survivorship bias.** The bundled price DB holds ~500 recent tickers, so stocks that later left the index
  are missing: only 62–87% of S&P 500 picks were priced in 2018–2019 (98–100% by 2025).
* **NASDAQ-100 fundamentals coverage.** `fundamental_data_full.csv` only covers stocks that were ever in the
  S&P 500, so ~10–17 NASDAQ-100 members per quarter (mostly foreign listings: ASML, MELI, PDD, BIDU, JD, AZN, …)
  can't be scored. Meta's history only starts at datadate 2022-06-30 (as `META`). The NASDAQ-100 has almost
  no `real_assets` stocks, so that bucket's model trains on ~20 samples.
* The last quarter is partial (prices end 2025-10-16). Results are single-seed (42).
* Not comparable to the README's 5.98x "Rolling Strategy", which uses a different universe/allocation setup.
