"""Tests for BacktestEngine rebalance scheduling.

Uses tiny synthetic price frames so no network / database access is needed.
"""
import os
import sys
import types

import numpy as np
import pandas as pd
import pytest

bt = pytest.importorskip("bt")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# backtest_engine imports src.data.data_fetcher at module level, which pulls in
# optional network dependencies (yfinance, ...). Stub it if it cannot load.
try:
    import src.data.data_fetcher  # noqa: F401
except Exception:
    stub = types.ModuleType("src.data.data_fetcher")
    stub.fetch_price_data = lambda *a, **k: None
    sys.modules["src.data.data_fetcher"] = stub

from src.backtest.backtest_engine import BacktestConfig, BacktestEngine  # noqa: E402


def _prices(n_days=10):
    idx = pd.bdate_range("2018-03-01", periods=n_days)
    rng = np.arange(n_days, dtype=float)
    return pd.DataFrame(
        {"AAA": 100.0 + rng, "BBB": 50.0 * (1.02 ** rng)}, index=idx
    )


def _engine(**kwargs):
    cfg = BacktestConfig(
        start_date="2018-03-01",
        end_date="2018-03-31",
        initial_capital=100_000.0,
        transaction_cost=0.0,
        **kwargs,
    )
    return BacktestEngine(cfg)


def _run_strategy(engine, prices, weights, **kwargs):
    strat = engine._create_bt_strategy("s", weights, **kwargs)
    res = bt.run(engine._build_bt_backtest(strat, prices, prices))
    return res


def _trade_dates(res):
    tx = res.get_transactions("s")
    return sorted(set(tx.index.get_level_values("Date")))


def test_sparse_weights_trade_on_first_signal_date():
    prices = _prices()
    d1, d2 = prices.index[2], prices.index[6]
    weights = pd.DataFrame(
        {"AAA": [1.0, 0.0], "BBB": [0.0, 1.0]}, index=[d1, d2]
    )

    res = _run_strategy(_engine(), prices, weights)
    positions = res.backtests["s"].strategy.positions

    # Nothing held before the first signal date ...
    assert (positions.loc[: prices.index[1]] == 0).all().all()
    # ... positions opened ON the first signal date (not one period later).
    assert positions.loc[d1, "AAA"] > 0
    assert positions.loc[d1, "BBB"] == 0
    # ... and switched on the second signal date.
    assert positions.loc[d2, "AAA"] == 0
    assert positions.loc[d2, "BBB"] > 0

    assert _trade_dates(res) == [d1, d2]


def test_run_backtest_default_rebalances_daily():
    prices = _prices()
    d1, d2 = prices.index[2], prices.index[6]
    weights = pd.DataFrame(
        {"AAA": [0.5, 0.2], "BBB": [0.5, 0.8]}, index=[d1, d2]
    )
    engine = _engine(integer_positions=False)
    result = engine.run_backtest("s", prices, weights)
    # Fully invested from the first signal date onward.
    assert result.portfolio_values.loc[d1:].nunique() > 1

    # Default path: weights are forward-filled and re-targeted daily, so the
    # drifting BBB/AAA mix produces trades on every day from d1 on.
    ws = weights.reindex(prices.index).ffill().fillna(0.0)
    res = _run_strategy(engine, prices, ws)
    assert _trade_dates(res) == list(prices.index[2:])


def test_rebalance_on_signal_dates_only():
    prices = _prices()
    d1, d2 = prices.index[2], prices.index[6]
    # Second signal dated on a Saturday -> executed the following Monday.
    sat = pd.Timestamp("2018-03-10")
    monday = pd.Timestamp("2018-03-12")
    assert monday in prices.index
    weights = pd.DataFrame(
        {"AAA": [0.5, 0.2, 1.0], "BBB": [0.5, 0.8, 0.0]}, index=[d1, d2, sat]
    )
    engine = _engine(integer_positions=False, rebalance_on_signal_dates_only=True)

    captured = {}
    orig = engine._create_bt_strategy

    def spy(name, ws, rebalance_dates=None):
        captured["rebalance_dates"] = rebalance_dates
        captured["ws"] = ws
        return orig(name, ws, rebalance_dates=rebalance_dates)

    engine._create_bt_strategy = spy
    engine.run_backtest("s", prices, weights)

    assert captured["rebalance_dates"] == [d1, d2, monday]
    # The Saturday signal's weights are carried to Monday.
    assert captured["ws"].loc[monday, "AAA"] == pytest.approx(1.0)

    res = _run_strategy(
        engine, prices, captured["ws"],
        rebalance_dates=captured["rebalance_dates"],
    )
    assert _trade_dates(res) == [d1, d2, monday]
    positions = res.backtests["s"].strategy.positions
    assert positions.loc[d1, "AAA"] > 0 and positions.loc[d1, "BBB"] > 0
    assert positions.loc[monday, "BBB"] == 0
