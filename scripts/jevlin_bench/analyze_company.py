#!/usr/bin/env python3
"""Per-arm report of Claude Code company-search passes, for the forms experiment.

  python scripts/jevlin_bench/analyze_company.py runs/cc-pilot/cc-pass-1.json [more summaries]

Reads the pass summaries written by cc_run_company.py (and, through each row's `artifact`, the
searches' Jevlin metadata) and prints one line per arm. The five numbers that decide the forms
experiment, `jevlin_balanced_forms` against `jevlin_balanced`, are:

  F1 (the harness's own judge), searches per trial, observed cost per trial (`cost_usd`, with its
  `vendor_cost_usd` and `model_cost_usd` parts reported beside it), hits shown per search, and from the metadata the mean
  `coverage.reformulation_only` and the share of searches that carried a form.

A trial that errored is counted with the F1 the judge gave it (zero) like any other row; the report
does not drop rows, so both arms are held to the same rule.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def _mean(values: list[float]) -> float | None:
    return round(mean(values), 4) if values else None


def load_rows(summaries: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in summaries:
        rows += json.loads(path.read_text()).get("rows") or []
    return rows


def search_metadata(row: dict[str, Any]) -> list[dict[str, Any]]:
    """The Jevlin metadata of each search of a row; empty when the artifact is gone."""
    try:
        artifact = json.loads(Path(row["artifact"]).read_text())
    except (OSError, ValueError, KeyError):
        return []
    return [((s.get("call") or {}).get("metadata") or {}) for s in (artifact.get("result") or {}).get("searches") or []]


def arm_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    searches = [meta for row in rows for meta in search_metadata(row)]
    with_form = [m for m in searches if m.get("jevlin_forms_sent")]
    reformulation_only = [
        float(m["jevlin_coverage"]["reformulation_only"]) for m in searches
        if isinstance(m.get("jevlin_coverage"), dict) and isinstance(m["jevlin_coverage"].get("reformulation_only"), (int, float))
    ]
    shown = [int(m["jevlin_hits_shown"]) for m in searches if m.get("jevlin_hits_shown") is not None]
    return {
        "trials": len(rows),
        "f1": _mean([float(r["f1"]) for r in rows]),
        "searches_per_trial": _mean([float(r["search_count"]) for r in rows]),
        # The sum hides which side moved: a forms arm doubles search spend and changes the
        # reading load, so the vendor and model parts are reported beside it.
        "cost_usd_per_trial": _mean([float(r["cost_usd"]) for r in rows if r.get("cost_usd") is not None]),
        "vendor_cost_usd_per_trial": _mean([float(r["vendor_cost_usd"]) for r in rows if r.get("vendor_cost_usd") is not None]),
        "model_cost_usd_per_trial": _mean([float(r["model_cost_usd"]) for r in rows if r.get("model_cost_usd") is not None]),
        "hits_shown_per_search": _mean([float(n) for n in shown]),
        "mean_reformulation_only": _mean(reformulation_only),
        "share_searches_with_form": round(len(with_form) / len(searches), 4) if searches else None,
    }


def report(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_arm[str(row["arm"])].append(row)
    return {arm: arm_report(arm_rows) for arm, arm_rows in sorted(by_arm.items())}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("summaries", nargs="+", type=Path, help="cc-pass-<n>.json files")
    parser.add_argument("--arms", default="", help="comma list of arms to report; default all")
    args = parser.parse_args(argv)
    out = report(load_rows(args.summaries))
    wanted = [a.strip() for a in args.arms.split(",") if a.strip()]
    if wanted:
        out = {arm: out[arm] for arm in wanted if arm in out}
    json.dump(out, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
