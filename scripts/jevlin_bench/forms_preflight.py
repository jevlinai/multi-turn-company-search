"""Pre-flight for the forms arm: does the router accept `search_queries`?

A router without Jevlin-core PR #55 answers a request carrying `search_queries` with
`400 malformed_json`. Without this check a rollback, or a run pointed at another host, would execute
`jevlin_balanced_forms` and either fail every second search or, worse, be read as a result. The
probe is one balanced search with one form and `max_results` 1; the router bills each extra form as a
query, so it costs about $0.08 of gateway spend. It is
accepted only on a 200 whose body carries `coverage.by_form`, the router's own evidence that it ran
the forms; a 200 without it is a router that ignored the field.

Importable (`probe`, `require_forms_router`) and runnable: `python forms_preflight.py` probes the
router named by JEVLIN_API_URL with JEVLIN_API_KEY (read from the environment or the fork's .env;
values are never printed).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import jevlin_core.jevlin as core  # noqa: E402

PROBE_QUERY = "companies backed by Insight Partners founded in Colorado"
PROBE_FORM = "Colorado startups with Insight Partners as an investor"
PROBE_COST_USD = 0.08
PR = "Jevlin-core PR #55 (multi-form `search_queries`)"


def probe_body() -> dict[str, Any]:
    return {"query": PROBE_QUERY, "tier": "balanced", "search_queries": [PROBE_FORM], "view": "merged", "max_results": 1}


def probe() -> tuple[bool, str]:
    """(accepted, detail). One request, no retry: a refusal is the answer, not a fault to ride out."""
    from dotenv import load_dotenv

    from company_research.vendors import _request

    load_dotenv(ROOT / ".env.local")
    load_dotenv(ROOT / ".env")
    try:
        key = core.api_key()
    except RuntimeError as exc:
        return False, str(exc)
    call = _request(
        method="POST", url=core.api_url(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        body=probe_body(), timeout=int(core.timeout_s()), retries=0,
    )
    if call.status != "ok":
        return False, str(call.error or call.status)[:300]
    payload = call.raw_response if isinstance(call.raw_response, dict) else {}
    coverage = payload.get("coverage")
    if not isinstance(coverage, dict) or "by_form" not in coverage:
        return False, "200 without coverage.by_form: the router ignored search_queries"
    return True, "accepted: 200 with coverage.by_form"


def require_forms_router(arms: list[str], probe_fn=probe) -> str | None:
    """None when no arm needs forms or the router accepts them; otherwise the refusal message."""
    if not any(core.is_forms_arm(arm) for arm in arms):
        return None
    accepted, detail = probe_fn()
    if accepted:
        return None
    return (f"refusing to start: the router did not accept a multi-form request ({detail}). "
            f"The forms arm needs a router with {PR}; without it the arm would run without its forms. "
            f"The probe is one balanced query plus one form, billed as two queries (about ${PROBE_COST_USD:.2f}).")


def main() -> int:
    accepted, detail = probe()
    print(f"forms pre-flight against {os.environ.get('JEVLIN_API_URL') or core.DEFAULT_API_URL}: {detail}")
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
