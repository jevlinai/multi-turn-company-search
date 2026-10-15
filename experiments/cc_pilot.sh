#!/usr/bin/env bash
# Claude Code agent, pilot: every question of the frozen set, every arm, PASSES passes. Pass N writes
# trial N with seed N, so the three trials the judge expects are three separate passes — run a cache
# TTL (15 min) apart at least, never concurrently, so Jevlin's cache cannot answer trial 2 for free.
# Bills the Claude subscription; a spent usage window is waited out and retried by the driver.
# Rerunning resumes an interrupted pass (rows judged on something are kept; the rest re-run).
# Not official OpenBenchmarks scoring: public rows, a different agent; scores are not comparable to the board.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-}"
[ -z "$PY" ] && { [ -x .venv/bin/python ] && PY=.venv/bin/python || PY=python3; }

: "${DATASET:=datasets/company-research/ob-company-websearch-v1-converted.json}"
: "${ARMS:=perplexity,exa_instant,parallel_fast,jevlin_fast,jevlin_balanced,jevlin_single,jevlin_fanout_raw,jevlin_fanout_merged,jevlin_cli}"
: "${MODEL:=claude-sonnet-5-5}"
: "${PASSES:=3}"
: "${OUT_DIR:=runs/cc-pilot}"
: "${RATE_LIMIT_WAIT:=1200}"
: "${PASS_GAP_S:=900}"   # at least the gateway cache TTL between passes
mkdir -p "$OUT_DIR"

"$PY" experiments/catalog_freeze.py verify
SLUG="$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["dataset"]["slug"])' "$DATASET")"

for pass in $(seq 1 "$PASSES"); do
  SUMMARY="$OUT_DIR/cc-pass-$pass.json"
  RESUME=()
  if [ -f "$SUMMARY" ]; then
    cp "$SUMMARY" "$OUT_DIR/cc-pass-$pass.before-resume-$(date -u +%Y%m%dT%H%M%SZ).json"
    RESUME=(--resume-from "$SUMMARY")
  fi
  echo "== pass $pass / $PASSES  trial=$pass seed=$pass model=$MODEL arms=$ARMS ${RESUME:+(resuming)}"
  if ! PYTHONPATH=scripts "$PY" scripts/jevlin_bench/cc_run_company.py \
      --dataset "$DATASET" --arms "$ARMS" --model "$MODEL" --trial-index "$pass" --seed "$pass" \
      --rate-limit-wait "$RATE_LIMIT_WAIT" --abort-after-run-errors 5 --abort-after-hop-errors 12 \
      ${RESUME[@]+"${RESUME[@]}"} --out-dir "$OUT_DIR" --summary "$SUMMARY"; then
    echo "pass $pass aborted; stopping. Rerun cc_pilot.sh to resume."
    exit 1
  fi
  if [ "$pass" -lt "$PASSES" ] && [ "$PASS_GAP_S" -gt 0 ]; then
    echo "== waiting ${PASS_GAP_S}s before pass $((pass + 1)) (gateway cache TTL)"
    sleep "$PASS_GAP_S"
  fi
done

PYTHONPATH=scripts "$PY" scripts/judge_company_research.py \
  --runs "$OUT_DIR/$SLUG" --trials "$PASSES" --output "$OUT_DIR/leaderboard.json"
