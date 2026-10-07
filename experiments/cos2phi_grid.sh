#!/usr/bin/env bash
# The pre-registered grid (prereg/PREREG_cos2phi_20261007.md), one JSON per cell in results/cos2phi/.
# Usage: experiments/cos2phi_grid.sh <distance> <eta> [<eta> ...]     e.g.  experiments/cos2phi_grid.sh 3 1 27.5
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
d=$1; shift
for eta in "$@"; do
  for p in 0.01 0.03; do
    for code in css xzzx; do
      for basis in x z; do
        lookup=0; [ "$d" -eq 3 ] && lookup=10000000
        $PY experiments/cos2phi_pilot.py --d "$d" --eta "$eta" --p "$p" --code "$code" --basis "$basis" --lookup "$lookup" \
            --out "results/cos2phi/d${d}_eta${eta}_p${p}_${code}_${basis}.json"
      done
    done
  done
done
