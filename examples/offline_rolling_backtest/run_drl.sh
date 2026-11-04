#!/usr/bin/env bash
# DRL allocator over the rolling ML picks: one drl_quarter.py run per rebalance date, on the priced
# top-25% picks that backtest_rolling.py wrote to <selection_out_dir>/weights_top25.csv.
#
# Usage:
#   run_drl.sh <db> <selection_out_dir> <out_dir> [parallel]
#
# Each quarter trains A2C (50k) + PPO (80k) + DDPG (50k timesteps) on CPU, ~10-12 min single-threaded
# (DDPG is ~90% of it); 31 quarters with 4 in parallel take ~1h40m. Resumable: quarters whose
# q_<date>/info.json exists are skipped.
# Needs: pip install torch stable-baselines3 gymnasium stockstats
set -euo pipefail
DB=$(realpath "$1"); SEL=$(realpath "$2"); OUT=$(mkdir -p "$3" && realpath "$3"); PAR=${4:-4}
HERE=$(cd "$(dirname "$0")" && pwd)

python3 - "$SEL/weights_top25.csv" "$DB" > "$OUT/jobs.txt" <<'EOF'
import sys, sqlite3, pandas as pd
w = pd.read_csv(sys.argv[1], index_col=0, parse_dates=True)
with sqlite3.connect(sys.argv[2]) as c:
    last = pd.Timestamp(c.execute("SELECT MAX(date) FROM price_data").fetchone()[0])
dates = list(w.index) + [last + pd.Timedelta(days=1)]
for d, n in zip(dates[:-1], dates[1:]):
    print(d.date(), n.date(), ",".join(w.columns[w.loc[d] > 0]))
EOF

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export DB OUT HERE
xargs -P "$PAR" -L 1 bash -c '
    q="$OUT/q_$0"
    if [[ -f "$q/info.json" ]]; then echo "skip $0 (done)"; exit 0; fi
    mkdir -p "$q"
    python3 "$HERE/drl_quarter.py" --db "$DB" --trade-date "$0" --next-date "$1" --tickers "$2" \
        --out "$q" > "$q/log.txt" 2>&1
    echo "done $0 rc=$?"' < "$OUT/jobs.txt"
