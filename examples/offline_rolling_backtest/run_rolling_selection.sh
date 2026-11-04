#!/usr/bin/env bash
# Rolling quarterly ML stock selection: one run of src/strategies/ml_bucket_selection.py per rebalance
# date (2018-03-01 .. 2025-09-01), each trained only on data available at that date.
#
# Usage:
#   run_rolling_selection.sh <db> <out_dir> [constituents_csv] [parallel]
#
#   <db>               SQLite DB from prepare_db.py
#   <out_dir>          one sub-folder q_<infer-datadate>/ per quarter is written here (resumable: finished
#                      quarters are skipped)
#   constituents_csv   optional point-in-time membership (date,tickers) for another universe, e.g. the
#                      NASDAQ-100 file from build_ndx_constituents.py. ml_bucket_selection.py always reads
#                      <project_root>/data/sp500_historical_constituents.csv when no --universe is given, so a
#                      scratch project root is created under <out_dir>/_root with src -> this repo's src and
#                      that file swapped in (its logs will still say "SP500"). The repo is not modified.
#                      (--universe nasdaq100 is not used: it calls the FMP API and keeps only CURRENT members.)
#   parallel           concurrent runs (default 4). Each run is pinned to 1 thread: several multi-threaded
#                      XGBoost/LightGBM/sklearn processes on the same cores slow down by >10x.
set -euo pipefail
DB=$(realpath "$1"); OUT=$(mkdir -p "$2" && realpath "$2"); CONST=${3:-}; PAR=${4:-4}
REPO=$(cd "$(dirname "$0")/../.." && pwd)

ROOT="$REPO"
if [[ -n "$CONST" ]]; then
    ROOT="$OUT/_root"; mkdir -p "$ROOT/data"
    ln -sfn "$REPO/src" "$ROOT/src"
    cp "$CONST" "$ROOT/data/sp500_historical_constituents.csv"
fi

# val-cutoff / infer-date pairs: infer datadate D (traded at D's tradedate), validation ends one quarter earlier
python3 - > "$OUT/jobs.txt" <<'EOF'
import pandas as pd
qe = pd.date_range("2017-09-30", "2025-06-30", freq="QE")
for v, i in zip(qe[:-1], qe[1:]): print(v.date(), i.date())
EOF

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=1
export ROOT DB OUT
cd "$ROOT"
xargs -P "$PAR" -L 1 bash -c '
    q="$OUT/q_$1"
    if ls "$q"/*_ml_bucket_predictions_*.csv >/dev/null 2>&1; then echo "skip $1 (done)"; exit 0; fi
    mkdir -p "$q"
    python3 "$ROOT/src/strategies/ml_bucket_selection.py" --db "$DB" \
        --val-cutoff "$0" --infer-date "$1" --output-dir "$q" > "$q/log.txt" 2>&1
    echo "done $1 rc=$?"' < "$OUT/jobs.txt"
