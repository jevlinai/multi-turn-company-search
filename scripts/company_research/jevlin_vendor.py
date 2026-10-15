"""Jevlin arms as company-search vendors: the `VendorCall` glue over the vendored core.

The arm table, request body, presentations and telemetry live in `scripts/jevlin_core/jevlin.py`
(vendored from the coding-agents fork at f01b032). This module is the part that belongs to this
harness: it sends the request through `vendors._request` (so the host's retry, redaction and
raw-response rules apply to Jevlin exactly as to every other vendor), shows the model hits in the
host's shape — `{url, title, snippet[:4000], metadata}` — converts the gateway's observed
`usage.cost_micros` to the `cost_usd` the judge sums, and keeps the Jevlin telemetry on the call.

What the model sees per hit is url / title / snippet with an empty `metadata`, as on the coding
suite: the gateway's per-result `found_by` vote is deliberately not shown, so Jevlin's advantage can
appear only through which pages it puts first and what their snippets say. The search-level
telemetry (request id, tier, providers, cache hit, pending arms, per-arm statuses, upstream calls,
merged total) is on `JevlinCall.metadata`, and the whole gateway body is `raw_response`, which the
trial artifact keeps verbatim.

The Jevlin arms are search-only. They have no native fetch, and the harness's neutral HTTP /
Playwright fetcher is deliberately not wired for them: the search + fetch board is out of scope for
this adaptation, and `runner.RunnerConfig` refuses to plan a search_and_fetch run that includes a
vendor without a fetch path.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import jevlin_core.jevlin as core

from .vendors import VendorCall, VendorSpec, _hit, _request

# `vendors._hit` cuts every vendor's snippet at this many characters; the core is told the same
# number so the hit it builds and the hit the host builds agree.
SNIPPET_CHARS = 4000

FIXTURE_VENDOR_KEY = "jevlin_fixture"
# Tests and the live plumbing check: the merged presentation over a fixture body, no provider
# called. Not registered in VENDORS, so it can never run in a benchmark pass.
FIXTURE_SPEC = VendorSpec(
    FIXTURE_VENDOR_KEY, "jevlin-fixture", "Jevlin fixture (tests only)", "jevlin-fixture",
    "FIXTURE merged-response body, no request", (), None, False, None, None,
    {"presentation": "merged", "fixture": True},
)


@dataclass
class JevlinCall(VendorCall):
    """A VendorCall plus the gateway telemetry. `asdict` serialises it like any other call."""

    metadata: dict[str, Any] = field(default_factory=dict)


def cost_usd_from_micros(usage: Any) -> float | None:
    """Observed gateway cost in dollars; None when the body carries no figure (unknown, not free)."""
    micros = (usage or {}).get("cost_micros") if isinstance(usage, dict) else None
    try:
        value = float(micros)
    except (TypeError, ValueError):
        return None
    return round(value / 1_000_000, 6) if value >= 0 else None


def _host_hits(core_hits: list[dict[str, str]]) -> list[dict[str, Any]]:
    return [_hit(h["url"], h["title"], h["snippet"]) for h in core_hits]


def finish_http_call(presentation: str, call: VendorCall, max_results: int) -> JevlinCall:
    """Turn a completed `_request` into the Jevlin VendorCall: hits, observed cost, telemetry."""
    payload = call.raw_response if isinstance(call.raw_response, dict) else {}
    hits: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {}
    cost: float | None = 0.0
    if call.status == "ok":
        hits = _host_hits(core.PRESENTATIONS[presentation](payload, max_results=max_results, snippet_chars=SNIPPET_CHARS))
        metadata = core.jevlin_meta(payload, hits, presentation=presentation)
        cost = cost_usd_from_micros(payload.get("usage"))
    return JevlinCall(
        status=call.status, latency_ms=call.latency_ms, raw_request=call.raw_request,
        raw_response=call.raw_response, error=call.error, attempts=call.attempts,
        cost_usd=cost, hits=hits, page=None, metadata=metadata,
    )


def http_call(vendor_key: str, query: str, *, max_results: int) -> JevlinCall:
    arm = core.HTTP_ARMS[vendor_key]
    # Refused before any request: a missing key or an unpinned fan-out arm raises here, as the
    # harness's other vendors raise on a missing credential.
    body = core.request_body(vendor_key, query, max_results)
    key = core.api_key()
    call = _request(
        method="POST", url=core.api_url(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        body=body, timeout=int(core.timeout_s()),
    )
    return finish_http_call(arm.presentation, call, max_results)


def cli_call(query: str, *, max_results: int) -> JevlinCall:
    """One `jevlin search --stdin`, as a VendorCall built by hand (a subprocess, not an HTTP request)."""
    res = core.run_cli(query, max_results)
    raw_request = {"method": "CLI", "url": " ".join(res.argv), "headers": {}, "body": res.request, "params": None}
    if res.exception:
        return JevlinCall(
            status="error", latency_ms=res.elapsed_ms, raw_request=raw_request, raw_response=None,
            error=res.exception, attempts=[{"attempt": 1, "status_code": None, "latency_ms": res.elapsed_ms, "error": res.exception}],
            cost_usd=0.0, hits=[], metadata={"jevlin_presentation": "cli"},
        )
    envelope = core.cli_envelope(res.stdout)
    raw_response = {"exit_code": res.returncode, "body": envelope, "stderr": res.stderr[:2000]}
    attempts = [{"attempt": 1, "status_code": envelope.get("http_status"), "latency_ms": res.elapsed_ms, "exit_code": res.returncode}]
    error = core.cli_error(envelope, returncode=res.returncode, stderr=res.stderr)
    if error:
        return JevlinCall(
            status="error", latency_ms=res.elapsed_ms, raw_request=raw_request, raw_response=raw_response,
            error=error, attempts=attempts, cost_usd=0.0, hits=[],
            metadata={"jevlin_presentation": "cli", "jevlin_http_status": envelope.get("http_status")},
        )
    hits = _host_hits(core.present_cli(envelope, max_results=max_results, snippet_chars=SNIPPET_CHARS))
    result = envelope.get("result") if isinstance(envelope.get("result"), dict) else {}
    return JevlinCall(
        status="ok", latency_ms=res.elapsed_ms, raw_request=raw_request, raw_response=raw_response,
        error=None, attempts=attempts, cost_usd=cost_usd_from_micros(result.get("usage")), hits=hits,
        metadata=core.cli_meta(envelope, hits),
    )


def fixture_call(payload: dict[str, Any], query: str, *, max_results: int) -> JevlinCall:
    """The merged presentation over a fixture body: the adapter path without a provider."""
    started = time.perf_counter()
    call = VendorCall(
        status="ok", latency_ms=0,
        raw_request={"method": "FIXTURE", "url": "fixture", "headers": {}, "body": {"query": query, "max_results": max_results, "view": "merged"}, "params": None},
        raw_response=payload, error=None, attempts=[{"attempt": 1, "status_code": 200, "latency_ms": 0}], cost_usd=None,
    )
    out = finish_http_call("merged", call, max_results)
    out.latency_ms = round((time.perf_counter() - started) * 1000)
    return out


def search_call(vendor_key: str, query: str, *, max_results: int = 10) -> JevlinCall:
    """`vendors.search()` for a Jevlin arm. Returns a complete VendorCall (hits and cost set)."""
    if vendor_key == core.CLI_ARM:
        return cli_call(query, max_results=max_results)
    return http_call(vendor_key, query, max_results=max_results)


def parse_hits(vendor_key: str, payload: Any, max_results: int) -> list[dict[str, Any]]:
    """`vendors._parse_hits()` for a Jevlin arm: the arm's presentation over a raw response."""
    payload = payload if isinstance(payload, dict) else {}
    if vendor_key == core.CLI_ARM:
        envelope = payload.get("body") if isinstance(payload.get("body"), dict) else payload
        return _host_hits(core.present_cli(envelope, max_results=max_results, snippet_chars=SNIPPET_CHARS))
    presentation = "merged" if vendor_key == FIXTURE_VENDOR_KEY else core.HTTP_ARMS[vendor_key].presentation
    return _host_hits(core.PRESENTATIONS[presentation](payload, max_results=max_results, snippet_chars=SNIPPET_CHARS))
