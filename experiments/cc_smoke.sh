#!/usr/bin/env bash
# Claude Code agent, smoke: two questions, every arm, trial 1. Bills the Claude subscription, not the API.
# Run from the repository root with .env filled (vendor + Jevlin keys for the wrapper) and the dataset
# built (scripts/jevlin_bench/build_public_dataset.py --download) and frozen (experiments/catalog_freeze.py freeze).
# Not official OpenBenchmarks scoring: public rows, a different agent; scores are not comparable to the board.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-}"
[ -z "$PY" ] && { [ -x .venv/bin/python ] && PY=.venv/bin/python || PY=python3; }

: "${DATASET:=datasets/company-research/ob-company-websearch-v1-converted.json}"
: "${ARMS:=perplexity,exa_instant,parallel_fast,jevlin_fast,jevlin_balanced,jevlin_single,jevlin_fanout_raw,jevlin_fanout_merged,jevlin_cli}"
: "${QUERY_LIMIT:=2}"
: "${MODEL:=claude-sonnet-5-5}"
: "${OUT_DIR:=runs/cc-smoke}"
mkdir -p "$OUT_DIR"

"$PY" experiments/catalog_freeze.py verify
SLUG="$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["dataset"]["slug"])' "$DATASET")"

PYTHONPATH=scripts "$PY" scripts/jevlin_bench/cc_run_company.py \
  --dataset "$DATASET" --arms "$ARMS" --query-limit "$QUERY_LIMIT" --model "$MODEL" \
  --trial-index 1 --seed 7 --out-dir "$OUT_DIR"

# One trial: the judge's completeness gate is satisfied with --trials 1.
PYTHONPATH=scripts "$PY" scripts/judge_company_research.py \
  --runs "$OUT_DIR/$SLUG" --trials 1 --output "$OUT_DIR/leaderboard.json"
