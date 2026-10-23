"""Jevlin arms: request shape, presentations, telemetry and the CLI shell-out. Benchmark-only code.

Vendored from `jevlinai/web-search-for-coding-agents` @ `f01b032` (branch `jevlin-backend`,
`scripts/coding_search/jevlin.py`). Copied, not imported, per the shared conventions, so that the
arms behave the same on every suite.

Jevlin is a multi-provider search gateway: one `POST /v1/search` fans out to a tier-defined set of
upstream providers and returns every provider's results side by side; with `view:"merged"` it also
returns one deduplicated list of pages ordered by how many providers returned each page. This
module translates that into url / title / snippet hits and keeps, in metadata, what a harness's own
fields cannot say: which upstream providers ran, what each cost, and whether Jevlin's cache answered.

Nothing here changes Jevlin. Each arm is one request shape against the public API, or one
invocation of the public `jevlin` CLI.

| arm                    | request                                   | model sees                                         |
|------------------------|-------------------------------------------|----------------------------------------------------|
| jevlin_fast            | tier=fast, tenant's default providers     | Jevlin's merged order, top N                       |
| jevlin_balanced        | tier=balanced, default providers          | merged order, top N                                |
| jevlin_balanced_forms  | tier=balanced, default providers, plus    | merged order across all forms, top N               |
|                        | search_queries=[the job's previous query] |                                                    |
| jevlin_balanced_split  | tier=balanced, default providers, plus    | the router's WHOLE merged list (up to 40): the     |
|                        | excerpt_mode="split"                      | top 10 long, the rest one line each                |
| jevlin_balanced_w16    | tier=balanced, default providers, plus    | merged order, ALL 16 pages (deviation from top N)  |
|                        | merged_limit=16                           |                                                    |
| jevlin_balanced_w24    | tier=balanced, default providers, plus    | merged order, ALL 24 pages (deviation from top N)  |
|                        | merged_limit=24                           |                                                    |
| jevlin_max             | tier=max, default providers               | merged order, top N (carried; not a company arm)   |
| jevlin_single          | providers=[JEVLIN_SINGLE_PROVIDER]        | that provider's own order, top N                   |
| jevlin_fanout_raw      | tier=balanced, providers=JEVLIN_PROVIDERS | round-robin over providers, exact-URL dedup, top N |
| jevlin_fanout_merged   | tier=balanced, providers=JEVLIN_PROVIDERS | merged order, top N                                |
| jevlin_cli             | `jevlin search --stdin`, no tier, no view | the CLI's own client-side merge, top N             |

`jevlin_balanced_forms` is the twin of `jevlin_balanced` that differs only in the forms it sends: the
caller supplies the agent's previous query and the router runs it beside the current one in the same
call, returning one merged list. It measures whether carrying the reformulation chain helps; with no
form to send (a job's first search) its request is byte-identical to `jevlin_balanced`'s. It needs a
router that accepts `search_queries` (Jevlin-core PR #55).

`jevlin_balanced_split` is the twin of `jevlin_balanced` that differs only in one request field,
`excerpt_mode:"split"` (valid with view merged and tier balanced or max), and in what the model is
shown. In that mode the router's merged list holds up to 40 pages: the top 10 carry longer selected
excerpts and the rest are compact entries (`compact: true`: a title, a URL, about 100 characters).
INTERNAL-ADAPTATION DEVIATION from the upstream "top N" rule (every other arm cuts the list at N):
this arm shows the whole list, because the compact tail is the thing under test and a top-N cut would
hide it. The long entries keep the host's hit format; each compact entry is one line,
"title | url | text". It needs a router that knows `excerpt_mode`; the driver probes for that first
(`split_preflight.py`).

`jevlin_fanout_*` refuse to run without `JEVLIN_PROVIDERS`: the ablation must pin the set so a
configuration change on the server cannot move it between passes. The tier arms deliberately do
not pin, because they measure what a customer gets.

Identical to the source: the arm table; `request_body` (`view:"merged"` always, tier / providers
per arm); pinned-provider handling; `present_merged`; `present_raw`; `arm_summary`; `jevlin_meta`;
the CLI argv (`search -config <bench> --stdin -no-flush`), environment, stdin request, envelope
reading, client-merge presentation and inferred `no_usable_result`.

Local extensions, not in the coding-agents fork at `SOURCE_COMMIT`: the forms arm (`Arm.forms`,
`clean_forms`, `request_body(search_queries=...)`, and `forms_sent` / `merged_limit` / `coverage` in
`jevlin_meta`) was added here for the B3-only forms experiment; the split arm (`Arm.excerpt_mode`,
`request_body`, `present_split`, `compact_line` / `shown_hits` / `split_stats`, and the split keys of
`jevlin_meta`) was added for the same-day excerpt_mode test and is mirrored on the fork's
`feat/split-view-arm` branch (not yet on `jevlin-backend`). Both are additive: every other arm's request body, presentation and
metadata are unchanged, a forms arm with nothing to send is byte-identical to `jevlin_balanced`, and a
split request is `jevlin_balanced`'s plus one field. Because neither landed on the fork's
`jevlin-backend` first, `SOURCE_COMMIT` stays the base and `LOCAL_CHANGES` records the divergence; a
re-vendor has to carry it or retire it.

Changed, and why:
  1. The snippet cap is a parameter. The coding harness cuts snippets at 1,200 characters; the
     company harness cuts at 4,000 (`company_research.vendors._hit`). The conventions bind each
     fork to its host's cap, so `present_*` take `snippet_chars` (default: the coding value).
  2. The coupling to the coding harness is gone. The source called back into its `search.py` for
     the HTTP transport (`vendor_call`), hit statistics (`_search_meta`) and raw-dump clipping
     (`clip_raw`). Here the host adapter — `company_research/jevlin_vendor.py` — supplies the
     transport (`vendors._request`) and builds the host's `VendorCall`; this module exposes the
     pure pieces and `run_cli`, which only runs the subprocess.
  3. Backend classes became an arm table plus functions, because this harness has no backend
     objects. The ordering, dedup, snippet choice and telemetry are unchanged.

Environment:
  JEVLIN_API_KEY          bearer key for the HTTP arms (a portal key; agent keys are restricted
                          to the fast tier)
  JEVLIN_API_URL          base URL, default https://api.jevlin.ai
  JEVLIN_PROVIDERS        comma list, required by jevlin_fanout_*
  JEVLIN_SINGLE_PROVIDER  provider name for jevlin_single, default exa
  JEVLIN_BIN              path to the jevlin executable for jevlin_cli
  JEVLIN_CLI_CONFIG       the bench client config for jevlin_cli (never JEVLIN_CONFIG)
  JEVLIN_TIMEOUT_S        HTTP / CLI timeout, default 45 (Jevlin's own fan-out deadline is 30 s in
                          the deployed config)
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass
from typing import Any

SOURCE_COMMIT = "f01b032"
LOCAL_CHANGES = (
    "forms: Arm.forms, clean_forms, request_body(search_queries=...), jevlin_meta forms_sent/merged_limit/coverage",
    "split: Arm.excerpt_mode, request_body(excerpt_mode), present_split, compact_line/shown_hits/split_stats, jevlin_meta split stats",
)
DEFAULT_API_URL = "https://api.jevlin.ai"
SEARCH_PATH = "/v1/search"
DEFAULT_TIMEOUT_S = 45.0
# The coding harness's cap, kept as the default so the function signatures stay comparable.
DEFAULT_SNIPPET_CHARS = 1200
# Statuses the gateway assigns an arm it never called. Everything else was an upstream request,
# billed or not.
NOT_CALLED = frozenset({"skipped"})
# The one value of the request field `excerpt_mode` this module sends.
EXCERPT_MODE_SPLIT = "split"
# Stamped into the notes of every split-arm job, so the label travels with the data.
SPLIT_DEVIATION = (
    "deviation (internal adaptation): jevlin_balanced_split shows the model the router's whole merged "
    "list, not the upstream top-N cut"
)


def api_url() -> str:
    base = (os.environ.get("JEVLIN_API_URL") or DEFAULT_API_URL).strip().rstrip("/")
    return base + SEARCH_PATH


def timeout_s() -> float:
    raw = (os.environ.get("JEVLIN_TIMEOUT_S") or "").strip()
    try:
        return float(raw) if raw else DEFAULT_TIMEOUT_S
    except ValueError:
        return DEFAULT_TIMEOUT_S


def api_key() -> str:
    key = (os.environ.get("JEVLIN_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("JEVLIN_API_KEY is required for the Jevlin HTTP arms")
    return key


def pinned_providers() -> list[str]:
    raw = os.environ.get("JEVLIN_PROVIDERS") or ""
    out: list[str] = []
    for item in raw.split(","):
        name = item.strip()
        if name and name not in out:
            out.append(name)
    return out


# ── the arm table ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Arm:
    name: str
    tier: str | None
    presentation: str  # "merged" | "raw"
    pin_providers: bool = False
    single_provider: bool = False
    forms: bool = False  # sends the caller's `search_queries` beside `query`
    excerpt_mode: str | None = None  # sends `excerpt_mode` (only "split" is used here)
    merged_limit: int | None = None  # sends `merged_limit`: the router's merged window for this arm
    shown: int | None = None  # hits shown to the model, in place of the harness's max_results cut

    def providers(self) -> list[str]:
        if self.single_provider:
            return [(os.environ.get("JEVLIN_SINGLE_PROVIDER") or "exa").strip()]
        if not self.pin_providers:
            return []
        names = pinned_providers()
        if not names:
            raise RuntimeError(f"JEVLIN_PROVIDERS must name the pinned provider set for {self.name}")
        return names


HTTP_ARMS: dict[str, Arm] = {
    "jevlin_fast": Arm("jevlin_fast", "fast", "merged"),
    "jevlin_balanced": Arm("jevlin_balanced", "balanced", "merged"),
    # The balanced twin that carries the previous query as a second form. Unpinned, like
    # jevlin_balanced: it measures what a customer gets.
    "jevlin_balanced_forms": Arm("jevlin_balanced_forms", "balanced", "merged", forms=True),
    # The balanced twin that asks the router for the split excerpt view and shows the whole list.
    # Unpinned, like jevlin_balanced.
    "jevlin_balanced_split": Arm("jevlin_balanced_split", "balanced", "split", excerpt_mode=EXCERPT_MODE_SPLIT),
    # The balanced twins that ask for a wider merged window and show all of it (the 2026-10-23 window
    # experiment: today's text, more pages). Same request otherwise; the cache key ignores merged_limit.
    "jevlin_balanced_w16": Arm("jevlin_balanced_w16", "balanced", "merged", merged_limit=16, shown=16),
    "jevlin_balanced_w24": Arm("jevlin_balanced_w24", "balanced", "merged", merged_limit=24, shown=24),
    "jevlin_max": Arm("jevlin_max", "max", "merged"),
    # One provider through Jevlin: the plumbing without the fan-out. Compared with the same
    # provider's direct runner this isolates what the gateway adds or costs on its own:
    # authentication, citation canonicalisation, metering, the 30 s deadline.
    "jevlin_single": Arm("jevlin_single", None, "raw", single_provider=True),
    "jevlin_fanout_raw": Arm("jevlin_fanout_raw", "balanced", "raw", pin_providers=True),
    "jevlin_fanout_merged": Arm("jevlin_fanout_merged", "balanced", "merged", pin_providers=True),
}
CLI_ARM = "jevlin_cli"


MAX_FORMS = 4  # the router's cap on `search_queries`


def is_forms_arm(arm_name: str) -> bool:
    arm = HTTP_ARMS.get(arm_name)
    return bool(arm and arm.forms)


def is_split_arm(arm_name: str) -> bool:
    arm = HTTP_ARMS.get(arm_name)
    return bool(arm and arm.excerpt_mode == EXCERPT_MODE_SPLIT)


def clean_forms(query: str, search_queries: Any) -> list[str]:
    """The forms worth sending: not the query itself (case-insensitively), no duplicates, at most four.

    The router drops those too; they are dropped here so the hop record shows what was meaningfully sent.
    """
    seen = {query.strip().casefold()}
    out: list[str] = []
    for form in search_queries or ():
        text = str(form).strip()
        if not text or text.casefold() in seen:
            continue
        seen.add(text.casefold())
        out.append(text)
        if len(out) >= MAX_FORMS:
            break
    return out


def request_body(arm_name: str, query: str, max_results: int, *, search_queries: Any = ()) -> dict[str, Any]:
    """The HTTP request for an arm. Raises RuntimeError when a fan-out arm is unpinned.

    `search_queries` is sent only by a forms arm and only when something is left after cleaning, so
    every other arm's body, and a forms arm's first search of a job, are unchanged. `excerpt_mode` is
    sent by the split arm on every search and by no other: its body is `jevlin_balanced`'s plus that field.
    """
    arm = HTTP_ARMS[arm_name]
    # view:"merged" adds `merged` and `indexes` and changes nothing else in the response, so every
    # arm asks for it and the raw presentation reads `candidates` from the same body.
    body: dict[str, Any] = {"query": query, "max_results": int(max_results), "view": "merged"}
    if arm.tier:
        body["tier"] = arm.tier
    providers = arm.providers()
    if providers:
        body["providers"] = providers
    forms = clean_forms(query, search_queries) if arm.forms else []
    if forms:
        body["search_queries"] = forms
    if arm.excerpt_mode:
        body["excerpt_mode"] = arm.excerpt_mode
    if arm.merged_limit:
        body["merged_limit"] = int(arm.merged_limit)
    return body


def shown_for(arm_name: str, max_results: int) -> int:
    """How many hits the model is shown for an HTTP arm: the arm's own `shown`, else the harness cut."""
    arm = HTTP_ARMS.get(arm_name)
    return int(arm.shown) if arm and arm.shown else int(max_results)


# ── presentation: Jevlin response → url / title / snippet hits ─────────────


def _hit(url: str, title: Any, snippet: Any, snippet_chars: int) -> dict[str, str]:
    return {"url": url, "title": str(title or ""), "snippet": str(snippet or "")[:snippet_chars]}


def present_merged(payload: Any, *, max_results: int, snippet_chars: int = DEFAULT_SNIPPET_CHARS) -> list[dict[str, str]]:
    """Jevlin's `merged[]`, in Jevlin's order, cut at max_results.

    The order is the gateway's: pages returned by more providers first, then best per-provider
    rank, then provider order. The excerpt is the longest snippet any provider gave for the page;
    when only `excerpts` (quoted evidence) exist they are joined.
    """
    hits: list[dict[str, str]] = []
    seen: set[str] = set()
    for page in (payload or {}).get("merged") or []:
        if not isinstance(page, dict):
            continue
        url = str(page.get("url") or "")
        if not url or url in seen:
            continue
        seen.add(url)
        snippet = page.get("excerpt") or "\n".join(
            part for part in (page.get("excerpts") or []) if isinstance(part, str)
        )
        hits.append(_hit(url, page.get("title"), snippet, snippet_chars))
        if len(hits) >= max_results:
            break
    return hits


def present_split(payload: Any, *, max_results: int | None = None, snippet_chars: int = DEFAULT_SNIPPET_CHARS) -> list[dict[str, Any]]:
    """The split view's `merged[]`: EVERY entry, in the router's order, with no top-N cut.

    INTERNAL-ADAPTATION DEVIATION from the upstream rule that a harness shows the model the first N
    hits (see `SPLIT_DEVIATION`). `max_results` is accepted so the presentations share a signature and
    is deliberately unused: it is still sent to the router as its per-provider count, but it does not
    cut what the model is shown.

    Entries are read as `present_merged` reads them (same URL dedup, same excerpt choice, same
    character cap), so the long entries (the top 10) are identical to what `jevlin_balanced` would
    show for the same page. An entry the router marked `compact: true` additionally carries
    `compact: True`; `shown_hits` turns those into one line each. Entries without a URL cannot be cited
    and are skipped, which `jevlin_merged_total` against the entries shown makes visible.
    """
    del max_results
    hits: list[dict[str, Any]] = []
    seen: set[str] = set()
    for page in (payload or {}).get("merged") or []:
        if not isinstance(page, dict):
            continue
        url = str(page.get("url") or "")
        if not url or url in seen:
            continue
        seen.add(url)
        snippet = page.get("excerpt") or "\n".join(
            part for part in (page.get("excerpts") or []) if isinstance(part, str)
        )
        hit: dict[str, Any] = _hit(url, page.get("title"), snippet, snippet_chars)
        if page.get("compact") is True:
            hit["compact"] = True
        hits.append(hit)
    return hits


def present_raw(payload: Any, *, max_results: int, snippet_chars: int = DEFAULT_SNIPPET_CHARS) -> list[dict[str, str]]:
    """Fan-out with the least processing that yields one list.

    Round-robin over the providers that answered, in the gateway's dispatch order: every
    provider's first result, then every provider's second, and so on. Duplicates are dropped only
    on the exact URL string the providers returned (each adapter has already canonicalised its own
    citations). No vote count, no re-ranking, no snippet choice: the first provider to show a page
    supplies its title and snippet.
    """
    lists: list[list[dict[str, Any]]] = []
    for cand in (payload or {}).get("candidates") or []:
        if not isinstance(cand, dict) or cand.get("status") != "ok":
            continue
        cits = [c for c in (cand.get("citations") or []) if isinstance(c, dict) and c.get("url")]
        if cits:
            lists.append(cits)
    hits: list[dict[str, str]] = []
    seen: set[str] = set()
    depth = max((len(cits) for cits in lists), default=0)
    for rank in range(depth):
        for cits in lists:
            if rank >= len(cits):
                continue
            cit = cits[rank]
            url = str(cit.get("url") or "")
            if not url or url in seen:
                continue
            seen.add(url)
            hits.append(_hit(url, cit.get("title"), cit.get("snippet"), snippet_chars))
            if len(hits) >= max_results:
                return hits
    return hits


PRESENTATIONS = {"merged": present_merged, "raw": present_raw, "split": present_split}


# ── the split view as the model reads it ───────────────────────────────────


def _one_line(text: Any) -> str:
    return " ".join(str(text or "").split())


def is_compact(hit: Any) -> bool:
    return isinstance(hit, dict) and hit.get("compact") is True


def compact_line(hit: dict[str, Any]) -> str:
    """One line for a compact entry: title | url | the short text (empty parts left out)."""
    parts = (_one_line(hit.get("title")), _one_line(hit.get("url")), _one_line(hit.get("snippet")))
    return " | ".join(part for part in parts if part)


def shown_hits(hits: list[dict[str, Any]]) -> list[Any]:
    """The list the model reads: a long entry stays the host's hit object, a compact one becomes its line.

    Router order is kept. A hit list with no compact entry (every other arm) comes back unchanged.
    """
    return [compact_line(hit) if is_compact(hit) else hit for hit in hits]


def split_stats(shown: list[Any]) -> dict[str, int]:
    """What one split search put in front of the model: entries, how many compact, characters.

    `characters` is the length of the JSON the model receives for `hits` (long entries as objects,
    compact ones as strings), serialised exactly as the tool output is, so it is a measure of the
    reading load rather than of the router's payload.
    """
    return {
        "jevlin_split_entries_shown": len(shown),
        "jevlin_split_compact_shown": sum(1 for item in shown if isinstance(item, str)),
        "jevlin_split_chars_shown": len(json.dumps(shown, ensure_ascii=False, default=str)),
    }


# ── metadata: what a harness's hop fields cannot carry ─────────────────────


def arm_summary(payload: Any) -> list[dict[str, Any]]:
    """One row per dispatched arm: provider, status, results, cost, latency."""
    reasons: dict[str, str] = {}
    indexes = (payload or {}).get("indexes") or {}
    for row in indexes.get("by_provider") or []:
        if isinstance(row, dict) and row.get("provider") and row.get("reason"):
            reasons[str(row["provider"])] = str(row["reason"])
    arms: list[dict[str, Any]] = []
    for cand in (payload or {}).get("candidates") or []:
        if not isinstance(cand, dict):
            continue
        latency = cand.get("latency") if isinstance(cand.get("latency"), dict) else {}
        provider = str(cand.get("provider") or "")
        arms.append({
            "provider": provider,
            "kind": cand.get("kind"),
            "status": cand.get("status"),
            "reason": reasons.get(provider),
            "results": len(cand.get("citations") or []),
            "has_answer": bool(cand.get("answer")),
            "cost_micros": cand.get("cost_micros"),
            "latency_ms": latency.get("total_ms"),
            "error": cand.get("error"),
        })
    return arms


def jevlin_meta(
    payload: Any, hits: list[dict[str, Any]], *, presentation: str, forms_sent: Any = (), split: dict[str, int] | None = None,
) -> dict[str, Any]:
    decision = (payload or {}).get("decision") or {}
    usage = (payload or {}).get("usage") or {}
    arms = arm_summary(payload)
    merged = (payload or {}).get("merged")
    coverage = (payload or {}).get("coverage")
    meta = {
        "jevlin_request_id": str((payload or {}).get("request_id") or ""),
        "jevlin_presentation": presentation,
        "jevlin_tier": decision.get("tier"),
        "jevlin_mode": decision.get("mode"),
        "jevlin_provider_set": decision.get("provider_set"),
        "jevlin_providers": list(decision.get("providers") or []),
        "jevlin_trimmed": list(decision.get("trimmed") or []),
        "jevlin_requested": bool(decision.get("requested")),
        "jevlin_router_version": decision.get("router_version"),
        "jevlin_ranker_version": decision.get("ranker_version"),
        "jevlin_cost_micros": usage.get("cost_micros"),
        "jevlin_latency_ms": usage.get("latency_ms"),
        "jevlin_cache_hit": bool(usage.get("cache_hit")),
        "jevlin_deadline_hit": bool(usage.get("deadline_hit")),
        "jevlin_pending": int(usage.get("pending") or 0),
        "jevlin_no_usable_result": bool((payload or {}).get("no_usable_result")),
        "jevlin_chosen": (payload or {}).get("chosen"),
        "jevlin_upstream_calls": sum(1 for arm in arms if arm["status"] not in NOT_CALLED),
        "jevlin_arms_ok": sum(1 for arm in arms if arm["status"] == "ok"),
        "jevlin_arms": arms,
        "jevlin_merged_total": len(merged) if isinstance(merged, list) else None,
        "jevlin_hits_shown": len(hits),
        # What the forms arm sent (empty for every other arm and for a job's first search), the
        # router's own cut of the merged list, and its per-form coverage (`by_form`,
        # `reformulation_only`) when it answered a multi-form request.
        "jevlin_forms_sent": list(forms_sent or ()),
        "jevlin_merged_limit": (payload or {}).get("merged_limit"),
        "jevlin_coverage": coverage if isinstance(coverage, dict) else None,
    }
    # The split arm's reading load (`split_stats`); absent for every other arm, so their metadata is unchanged.
    if split:
        meta.update(split)
    return meta


# ── the CLI arm: production Jevlin as an agent receives it ──────────────────
#
# `jevlin search --stdin`, the customer-facing client. The CLI chooses no tier (the account's
# default applies), asks for no view, merges the router's candidates itself with its own URL rule,
# and reports `ok: true` whenever the protocol succeeded, even if every provider failed. All of that
# is what an agent gets in production, so none of it is corrected here. The model sees the client's
# merged list, top N. The CLI reads its router URL from its own config, not from this process's
# environment; the bench config written by experiments/jevlin-cli/setup.sh decides where these
# searches go.


def cli_config_path() -> str:
    # Its own variable, not JEVLIN_CONFIG: an operator's shell commonly exports JEVLIN_CONFIG for
    # their personal client, and python-dotenv never overrides a variable the shell already set, so
    # a JEVLIN_CONFIG line in .env would be silently ignored and the arm would run on the operator's
    # own config.
    return (os.environ.get("JEVLIN_CLI_CONFIG") or "").strip()


def cli_argv() -> list[str]:
    # -config: the bench config, passed explicitly so inherited environment cannot redirect the
    # arm. -no-flush: do not spawn the client's background mining flush after the search; the bench
    # config disables mining anyway, this keeps stray processes off the run host. Neither changes
    # the search.
    exe = (os.environ.get("JEVLIN_BIN") or "jevlin").strip()
    argv = [exe, "search"]
    if cli_config_path():
        argv += ["-config", cli_config_path()]
    return argv + ["--stdin", "-no-flush"]


def cli_env() -> dict[str, str]:
    env = dict(os.environ)
    if cli_config_path():
        env["JEVLIN_CONFIG"] = cli_config_path()
    return env


def cli_request(query: str, max_results: int) -> dict[str, Any]:
    return {"version": 1, "query": query, "max_results": int(max_results)}


@dataclass
class CliResult:
    argv: list[str]
    request: dict[str, Any]
    returncode: int | None
    stdout: str
    stderr: str
    elapsed_ms: int
    exception: str | None  # "OSError: ..." / "TimeoutExpired: ..." when the process never answered


def run_cli(query: str, max_results: int) -> CliResult:
    """Run `jevlin search --stdin` once. Never raises; a failed launch is reported in `exception`."""
    argv = cli_argv()
    request = cli_request(query, max_results)
    started = time.perf_counter()
    try:
        proc = subprocess.run(
            argv,
            input=json.dumps(request, ensure_ascii=False),
            capture_output=True,
            text=True,
            timeout=timeout_s(),
            check=False,
            env=cli_env(),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return CliResult(argv, request, None, "", "", int((time.perf_counter() - started) * 1000), f"{type(exc).__name__}: {exc}")
    return CliResult(argv, request, proc.returncode, proc.stdout or "", proc.stderr or "", int((time.perf_counter() - started) * 1000), None)


def cli_envelope(stdout: str) -> dict[str, Any]:
    try:
        envelope: Any = json.loads(stdout) if stdout.strip() else {}
    except ValueError:
        envelope = {"raw": stdout[:500]}
    if not isinstance(envelope, dict):
        envelope = {"raw": str(envelope)[:500]}
    return envelope


def cli_error(envelope: dict[str, Any], *, returncode: int | None, stderr: str) -> str | None:
    """The client's error when the envelope is not `ok: true`; None when the search succeeded."""
    if envelope.get("ok") is True:
        return None
    err = envelope.get("error") if isinstance(envelope.get("error"), dict) else {}
    code = str(err.get("code") or envelope.get("code") or envelope.get("status") or "")
    message = str(err.get("message") or envelope.get("message") or (stderr or "").strip()[:300])
    return f"jevlin search --stdin exit {returncode}: {code} {message}".strip()


def present_cli(envelope: dict[str, Any], *, max_results: int, snippet_chars: int = DEFAULT_SNIPPET_CHARS) -> list[dict[str, str]]:
    """The client's own merged list, top N."""
    result = envelope.get("result") if isinstance(envelope.get("result"), dict) else {}
    merged = result.get("merged") if isinstance(result.get("merged"), list) else []
    hits: list[dict[str, str]] = []
    seen: set[str] = set()
    for page in merged:
        if not isinstance(page, dict):
            continue
        url = str(page.get("url") or "")
        if not url or url in seen:
            continue
        seen.add(url)
        hits.append(_hit(url, page.get("title"), page.get("snippet"), snippet_chars))
        if len(hits) >= max_results:
            break
    return hits


def cli_meta(envelope: dict[str, Any], hits: list[dict[str, Any]]) -> dict[str, Any]:
    result = envelope.get("result") if isinstance(envelope.get("result"), dict) else {}
    merged = result.get("merged") if isinstance(result.get("merged"), list) else []
    candidates = result.get("candidates") if isinstance(result.get("candidates"), list) else None
    decision = result.get("decision") if isinstance(result.get("decision"), dict) else {}
    usage = result.get("usage") if isinstance(result.get("usage"), dict) else {}
    arms = [
        {
            "provider": c.get("provider"),
            "kind": c.get("kind"),
            "status": c.get("status"),
            "results": len(c.get("citations") or []),
            "cost_micros": c.get("cost_micros"),
            "latency_ms": c.get("latency_ms"),
            "error": c.get("error"),
        }
        for c in (candidates or [])
        if isinstance(c, dict)
    ]
    return {
        "jevlin_request_id": str(result.get("request_id") or envelope.get("request_id") or ""),
        "jevlin_presentation": "cli",
        "jevlin_client_merge": True,
        "jevlin_tier": decision.get("tier"),
        "jevlin_providers": list(decision.get("providers") or []),
        "jevlin_trimmed": list(decision.get("trimmed") or []),
        "jevlin_cost_micros": usage.get("cost_micros"),
        "jevlin_latency_ms": usage.get("latency_ms") or result.get("latency_ms"),
        "jevlin_cache_hit": bool(usage.get("cache_hit")),
        "jevlin_pending": int(usage.get("pending") or 0),
        # The envelope has no no_usable_result flag; infer it when the candidates are present, as
        # they are in the CLI's default view.
        "jevlin_no_usable_result": bool(not merged and arms and all(arm["status"] != "ok" for arm in arms)),
        "jevlin_upstream_calls": sum(1 for arm in arms if arm["status"] not in NOT_CALLED) if arms else None,
        "jevlin_arms_ok": sum(1 for arm in arms if arm["status"] == "ok") if arms else None,
        "jevlin_arms": arms,
        "jevlin_merged_total": len(merged),
        "jevlin_hits_shown": len(hits),
        "jevlin_mining": envelope.get("mining"),
        "jevlin_http_status": envelope.get("http_status"),
    }


# The six arms the shared conventions name for every suite, plus the forms and split experiments;
# jevlin_max is carried in the table for comparability but is not a company arm.
COMPANY_ARMS: tuple[str, ...] = (
    "jevlin_fast",
    "jevlin_balanced",
    "jevlin_balanced_forms",
    "jevlin_balanced_split",
    "jevlin_balanced_w16",
    "jevlin_balanced_w24",
    "jevlin_single",
    "jevlin_fanout_raw",
    "jevlin_fanout_merged",
    "jevlin_cli",
)
