#!/usr/bin/env python3
"""The one command a headless Claude Code session may run: `web_search "<query>"`.

The driver (cc_run_company.py) writes a tiny shell script per job that sets CC_* and execs this
file with the harness's Python. The session sees only this command's stdout, which is exactly the
tool output the harness's own agent loop shows the model (`agent.py`):
{"vendor", "query", "hits": [{url, title, snippet, metadata}], "error"}. Every call is recorded as
a hop file carrying the whole `VendorCall` (redacted request, raw response, attempts, cost,
telemetry), which the driver turns into the artifact's `result.searches[]`.

Budgets enforced here, because a wrapper can enforce them: the total search budget (the (N+1)th
call answers "search budget exhausted; answer now", the harness's words). Not enforceable here:
searches per turn — a wrapper sees a sequence of calls, not turn boundaries.

Environment set by the generated script:
  CC_ROOT          the fork's root (for .env and the company_research package)
  CC_ARM           vendor key (a harness vendor or a Jevlin arm); a forms arm gets the job's previous
                   query as `search_queries` (see previous_query)
  CC_HOP_DIR       where hop files go (search-01.json, search-02.json, ...)
  CC_MAX_SEARCHES  total budget (14, the harness's AgentConfig.max_searches)
  CC_MAX_RESULTS   hits shown per search (10, the harness's AgentConfig.max_results)
  CC_FIXTURE_PATH  tests and plumbing checks only: with CC_ARM=jevlin_fixture, serve this merged body
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(os.environ.get("CC_ROOT") or Path(__file__).resolve().parents[2])
sys.path.insert(0, str(ROOT / "scripts"))

import jevlin_core.jevlin as core  # noqa: E402
from company_research import jevlin_vendor, vendors  # noqa: E402
from company_research.vendors import VendorCall  # noqa: E402

BUDGET_EXHAUSTED = {"error": "search budget exhausted; answer now"}  # agent.py, verbatim
DEFAULT_MAX_SEARCHES = 14
DEFAULT_MAX_RESULTS = 10


def load_environment() -> None:
    """The same two files the harness's own entry point reads; values are never printed."""
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env.local")
    load_dotenv(ROOT / ".env")


def hop_files(hop_dir: Path) -> list[Path]:
    return sorted(hop_dir.glob("search-*.json"))


def emit(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
    sys.stdout.flush()


def previous_query(hop_dir: Path) -> str | None:
    """The query of the job's most recent hop that can be read; None before the first search."""
    for path in reversed(hop_files(hop_dir)):
        try:
            query = str(json.loads(path.read_text()).get("query") or "").strip()
        except (OSError, ValueError, AttributeError):
            continue
        if query:
            return query
    return None


def forms_for(arm: str, query: str, hop_dir: Path) -> list[str]:
    """The forms a search carries: for a forms arm, the previous query of this job when it differs
    from the current one (case-insensitively). Exactly one, not the history: the experiment is
    whether the reformulation chain's last link helps. The first search of a job carries none, so it
    is byte-identical to the twin arm's and shares its gateway cache key."""
    if not core.is_forms_arm(arm):
        return []
    previous = previous_query(hop_dir)
    if previous is None or previous.casefold() == query.casefold():
        return []
    return [previous]


def run_search(arm: str, query: str, max_results: int, search_queries: list[str] | None = None) -> VendorCall:
    if arm == jevlin_vendor.FIXTURE_VENDOR_KEY:
        payload = json.loads(Path(os.environ["CC_FIXTURE_PATH"]).read_text())
        return jevlin_vendor.fixture_call(payload, query, max_results=max_results)
    if search_queries:
        return jevlin_vendor.search_call(arm, query, max_results=max_results, search_queries=search_queries)
    kwargs: dict = {"max_results": max_results}
    if arm == "perplexity":
        kwargs["search_context_size"] = "low"  # search-only board (agent.py: low without fetch)
    return vendors.search(arm, query, **kwargs)


def main(argv: list[str]) -> int:
    if len(argv) != 2 or not argv[1].strip():
        emit({"error": 'usage: web_search "your search query" (exactly one quoted argument)'})
        return 64
    query = argv[1].strip()
    arm = (os.environ.get("CC_ARM") or "").strip()
    hop_dir = Path(os.environ.get("CC_HOP_DIR") or "")
    max_searches = int(os.environ.get("CC_MAX_SEARCHES") or DEFAULT_MAX_SEARCHES)
    max_results = int(os.environ.get("CC_MAX_RESULTS") or DEFAULT_MAX_RESULTS)
    if not arm or not str(hop_dir):
        emit({"error": "web_search is not configured for this session"})
        return 70
    hop_dir.mkdir(parents=True, exist_ok=True)
    done = len(hop_files(hop_dir))
    if done >= max_searches:
        (hop_dir / "budget_exhausted").write_text(str(done) + "\n")
        emit(BUDGET_EXHAUSTED)
        return 0
    load_environment()
    search_queries = forms_for(arm, query, hop_dir)
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    started = time.perf_counter()
    try:
        call = run_search(arm, query, max_results, search_queries)
    except Exception as exc:  # noqa: BLE001 — any failure is a recorded, shown error hop, never a crash
        call = VendorCall(status="error", latency_ms=round((time.perf_counter() - started) * 1000),
                          raw_request={"vendor": arm, "query": query}, raw_response=None,
                          error=f"{type(exc).__name__}: {exc}", attempts=[], cost_usd=0.0, hits=[])
    wall_ms = round((time.perf_counter() - started) * 1000)
    seq = done + 1
    shown = {"vendor": arm, "query": query, "hits": call.hits or [], "error": call.error}
    hop = {
        "seq": seq,
        "query": query,
        "vendor": arm,
        "search_queries": search_queries,
        "started_at": started_at,
        "wall_ms": wall_ms,
        "call": asdict(call),
        "shown": shown,
    }
    (hop_dir / f"search-{seq:02d}.json").write_text(json.dumps(hop, ensure_ascii=False, default=str))
    emit(shown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
