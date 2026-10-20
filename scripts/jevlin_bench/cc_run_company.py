#!/usr/bin/env python3
"""Run the company-search questions with headless Claude Code as the agent, search-only board.

Same dataset, judge, budgets, hit shape and output schema as the harness's own runner; a different
agent. Each (question, arm, trial) job is one `claude -p` session under the user's subscription,
started with an empty environment (no API key, no JEVLIN_*, no hooks, skills or MCP), whose only
permitted tool is one generated shell command, `web_search "<query>"`, that calls the harness's
vendor adapter for the arm (cc_search.py) and records every hop. The session's final message is
the company set; it is parsed with the harness's own strict parser, scored with the harness's own
deterministic judge, and written as a `multi-turn-company-search-run-v1` artifact at the harness's
path, so `scripts/judge_company_research.py` reads it unchanged.

  PYTHONPATH=scripts python scripts/jevlin_bench/cc_run_company.py \
      --dataset datasets/company-research/ob-company-websearch-v1-converted.json \
      --arms perplexity,exa_instant,parallel_fast,jevlin_fast,jevlin_balanced \
      --trial-index 1 --seed 1 --out-dir runs/cc-smoke

Enforced, by the wrapper or by Claude Code flags: the total search budget (14), the hit count and
snippet cap (10 × 4,000 through the vendor adapter), one tool and one command, no API key, the
reasoning effort (`--effort medium`), the output schema (`--json-schema`, Claude Code's structured
output; the tolerant text parser is the fallback and a parse failure scores as an empty set for
every arm alike), "no successful search → error" (the harness's rule). Not enforceable from the
outside and recorded as protocol deviations in every artifact: two searches per turn, the eight-turn
cap, the final-turn tool withdrawal and its system line, max_output_tokens=6000, and the API's
strict decoding of the schema (Claude Code validates client-side).

A subscription window that is exhausted shows up as a Claude Code error; the driver waits
--rate-limit-wait seconds and retries the same job up to --rate-limit-retries times before
counting it as a run-level failure. The pass aborts after consecutive run-level failures and after
a run of failed searches counted per backend family (all Jevlin arms together). --resume-from
keeps the rows of an interrupted pass that were judged on something and re-runs the rest.
Trials run one at a time (concurrency 1): the harness's default of three concurrent trials would
let Jevlin's 15-minute cache answer trials 2 and 3 of a question for free.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import stat
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import forms_preflight  # noqa: E402
from company_research import jevlin_vendor  # noqa: E402
from company_research.agent import COMPANY_SET_SCHEMA, OUTPUT_SCHEMA_VERSION, PARSER_VERSION, AgentConfig  # noqa: E402
from company_research.dataset import load_artifact  # noqa: E402
from company_research.judge import score_artifact  # noqa: E402
from company_research.parser import parse_company_set  # noqa: E402
from company_research.vendors import VENDORS, required_env  # noqa: E402

DEFAULT_MODEL = "claude-sonnet-5-5"
DEFAULT_EFFORT = "medium"  # the harness's reasoning.effort
RESEARCH_MODE = "search_only"
SCHEMA_VERSION = "multi-turn-company-search-run-v1"
PROTOCOL_VERSION = "company-research-agent-v2+claude-code"
RATE_LIMIT_RE = re.compile(r"rate.?limit|usage limit|limit reached|hit your limit|resets at|too many requests|overloaded|\b529\b", re.I)

# The harness's system prompt (agent.py SYSTEM_PROMPT) with its tool sentences rewritten for a shell
# command and the output schema stated, because Claude Code has no API-side strict schema. The
# evidence, precision and stopping rules are the harness's words, unchanged. Identical for every arm.
SYSTEM_PROMPT = """You are a company research agent. Identify companies that satisfy ALL
constraints in the user's question. Your only research tool is the shell command
`{command} "<query>"`, run with Bash exactly as written (full path, one quoted argument). It searches through the benchmark vendor and returns JSON: {{"vendor", "query", "hits": [{{"url", "title", "snippet", "metadata"}}], "error"}}. Use it to form focused research queries for missing evidence. There is no page-fetch tool and no other tool: do not run any other command, and do not read, write or fetch anything. You may call the search command at most {max_searches} times in total; after that it answers "search budget exhausted; answer now". The reference protocol allows at most two searches per turn over at most eight turns; keep to that pace.

Write queries in plain words only. A query must not contain the characters $ ` \\ | ; & < > ( ) ! or quotes: the command is refused before it runs and the search is lost. Write money as "USD 100 million" or "100 million dollars", never "$100 million". If a search command is refused, rewrite the query without those characters and call it again.

Do not use prior knowledge as evidence. Do not include a company unless the
collected evidence supports every constraint. Prefer precision over padding.
Return each company once under its canonical public name, include its canonical
domain when known, cite the URLs supporting it, and briefly state the evidence.
An empty company list is valid when the evidence is insufficient.

Your final message must be exactly one JSON object and nothing else (no prose before or after it, no code fence) of this shape:
{{"companies": [{{"name": "<canonical public name>", "domain": "<canonical domain or null>", "cited_urls": ["<URL from a search result>"], "evidence": [{{"url": "<URL from a search result>", "claim": "<what it shows>"}}]}}]}}"""

ENFORCED = [
    "total search budget (wrapper: the (N+1)th call answers 'search budget exhausted; answer now')",
    "hits per search and snippet cap (vendor adapter: max_results, 4000 chars)",
    "one tool (Bash) and one permitted command, named by absolute path",
    "empty session environment: no API key, no JEVLIN_*, no hooks/skills/MCP (--safe-mode, --strict-mcp-config)",
    "no successful vendor search -> trial status 'error' (the harness's rule)",
    "parse failure -> empty prediction (the harness's rule), equally for every arm",
]
NOT_ENFORCED = [
    "two searches per turn (a wrapper sees calls, not turn boundaries)",
    "eight-turn cap and the final-turn tool withdrawal with its 'This is the final turn' system line (no --max-turns in Claude Code)",
    "max_output_tokens=6000 per turn (not exposed)",
    "API-enforced strict JSON schema on every turn (Claude Code's --json-schema validates the final output client-side; the prompt states the schema)",
    "the model itself: Claude Code's agent and context, not gpt-5.6-sol through the Responses API",
]


# ── Claude Code session plumbing (ported from the coding fork's cc_run.py) ───


def resolve_claude(claude_bin: str) -> str:
    """Absolute path of the launcher, found on the parent's PATH once; the session gets no such PATH."""
    found = shutil.which(claude_bin) if not os.path.isabs(claude_bin) else claude_bin
    return os.path.abspath(found or claude_bin)


def claude_version(claude_bin: str) -> str:
    try:
        return subprocess.run([resolve_claude(claude_bin), "--version"], capture_output=True, text=True, timeout=30, check=False).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"


def clean_env(claude_bin: str, tools_dir: Path | None = None) -> dict[str, str]:
    """The session's environment: enough to run claude and the wrapper, nothing else.

    No ANTHROPIC_API_KEY (the subscription login must be used), no JEVLIN_* or vendor keys (the
    wrapper loads .env itself), no CLAUDECODE markers from a parent session, no hooks or skills
    (--safe-mode). The job's tools directory leads PATH so the model can type `web_search "..."`.
    """
    # The directory of the `claude` launcher as found on PATH, not resolved through its symlink:
    # the symlink points into the npm package, while `node` lives beside the launcher.
    claude_dir = str(Path(resolve_claude(claude_bin)).parent)
    dirs = ([str(tools_dir)] if tools_dir else []) + [claude_dir, "/usr/bin", "/bin", "/usr/sbin", "/sbin"]
    path = ":".join(dict.fromkeys(dirs))
    env = {
        "HOME": os.environ.get("HOME", ""),
        "USER": os.environ.get("USER", ""),
        "LOGNAME": os.environ.get("LOGNAME", os.environ.get("USER", "")),
        "SHELL": "/bin/sh",
        "TERM": "xterm-256color",
        "LANG": os.environ.get("LANG", "en_US.UTF-8"),
        "TMPDIR": os.environ.get("TMPDIR", "/tmp"),
        "PATH": path,
        "DISABLE_AUTOUPDATER": "1",
    }
    return {k: v for k, v in env.items() if v}


def write_wrapper(tools_dir: Path, *, arm: str, hop_dir: Path, max_searches: int, max_results: int, python: str, fixture: str | None) -> Path:
    tools_dir.mkdir(parents=True, exist_ok=True)
    wrapper = tools_dir / "web_search"
    lines = [
        "#!/bin/sh",
        f"export CC_ROOT='{ROOT}'",
        f"export CC_ARM='{arm}'",
        f"export CC_HOP_DIR='{hop_dir}'",
        f"export CC_MAX_SEARCHES='{max_searches}'",
        f"export CC_MAX_RESULTS='{max_results}'",
    ]
    if fixture:
        lines.append(f"export CC_FIXTURE_PATH='{fixture}'")
    lines.append(f"exec '{python}' '{HERE / 'cc_search.py'}' \"$@\"")
    wrapper.write_text("\n".join(lines) + "\n")
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return wrapper


def claude_argv(claude_bin: str, *, prompt: str, system: str, wrapper: Path, model: str, effort: str | None, json_schema: bool) -> list[str]:
    argv = [
        resolve_claude(claude_bin), "-p", prompt,
        "--model", model,
        "--output-format", "json",
        "--append-system-prompt", system,
        "--tools", "Bash",
        # Both spellings of the one permitted command: the absolute path (first, so tooling that
        # reads the pattern finds the file) and the bare name the model might type, which resolves
        # through the session PATH to the same file.
        "--allowedTools", f"Bash({wrapper} *)", f"Bash({wrapper.name} *)",
        "--disallowedTools", "WebSearch,WebFetch,Read,Write,Edit,Glob,Grep,Task,TodoWrite,NotebookEdit",
        "--permission-mode", "dontAsk",
        "--permission-prompts", "none",
        "--safe-mode",
        "--strict-mcp-config",
        "--disable-slash-commands",
        "--no-session-persistence",
    ]
    if effort:
        argv += ["--effort", effort]
    if json_schema:
        argv += ["--json-schema", json.dumps(COMPANY_SET_SCHEMA, separators=(",", ":"))]
    return argv


def parse_result(stdout: str) -> dict[str, Any]:
    text = (stdout or "").strip()
    if not text:
        return {}
    try:
        data = json.loads(text)
    except ValueError:
        # stream or noise before the object: take the last JSON object line
        for line in reversed(text.splitlines()):
            line = line.strip()
            if line.startswith("{"):
                try:
                    data = json.loads(line)
                    break
                except ValueError:
                    continue
        else:
            return {"_unparsed": text[:2000]}
    if isinstance(data, list):
        results = [item for item in data if isinstance(item, dict) and item.get("type") == "result"]
        data = results[-1] if results else (data[-1] if data and isinstance(data[-1], dict) else {})
    return data if isinstance(data, dict) else {}


def is_rate_limited(result: dict[str, Any], stderr: str, returncode: int) -> bool:
    text = " ".join(str(result.get(k) or "") for k in ("result", "error", "subtype", "_unparsed")) + " " + (stderr or "")
    return bool(result.get("is_error") or returncode != 0) and bool(RATE_LIMIT_RE.search(text))


def usage_tokens(result: dict[str, Any]) -> dict[str, int]:
    usage = result.get("usage") if isinstance(result.get("usage"), dict) else {}
    cache_read = int(usage.get("cache_read_input_tokens") or 0)
    cache_write = int(usage.get("cache_creation_input_tokens") or 0)
    uncached = int(usage.get("input_tokens") or 0)
    output = int(usage.get("output_tokens") or 0)
    return {
        "model_input_tokens": uncached + cache_read + cache_write,
        "model_cached_input_tokens": cache_read,
        "model_cache_write_input_tokens": cache_write,
        "model_output_tokens": output,
        "model_total_tokens": uncached + cache_read + cache_write + output,
    }


# ── the final answer ─────────────────────────────────────────────────────────

_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.S)


def extract_company_json(text: str) -> str | None:
    """The company-set object from a free-text final message, or None.

    Tolerant on the outside, strict inside: fences are stripped and the outermost object tried
    first; failing that, every `{` is tried as the start of a JSON object and the first one that
    decodes and carries a `companies` key wins. What comes back is re-serialised, so the harness's
    strict parser sees only JSON.
    """
    raw = (text or "").strip()
    if not raw:
        return None
    candidates = [m.group(1).strip() for m in _FENCE_RE.finditer(raw)] + [raw]
    decoder = json.JSONDecoder()
    for candidate in candidates:
        first, last = candidate.find("{"), candidate.rfind("}")
        if first != -1 and last > first:
            try:
                obj = json.loads(candidate[first:last + 1])
                if isinstance(obj, dict) and "companies" in obj:
                    return json.dumps(obj, ensure_ascii=False)
            except ValueError:
                pass
        idx = candidate.find("{")
        while idx != -1:
            try:
                obj, _ = decoder.raw_decode(candidate, idx)
                if isinstance(obj, dict) and "companies" in obj:
                    return json.dumps(obj, ensure_ascii=False)
            except ValueError:
                pass
            idx = candidate.find("{", idx + 1)
    return None


def final_company_set(result: dict[str, Any]) -> tuple[str, bool]:
    """(final response text, whether Claude Code's structured output supplied it)."""
    structured = result.get("structured_output")
    if isinstance(structured, dict):
        return json.dumps(structured, ensure_ascii=False), True
    return str(result.get("result") or ""), False


def load_hops(hop_dir: Path) -> list[dict[str, Any]]:
    hops = []
    for path in sorted(hop_dir.glob("search-*.json")):
        try:
            hops.append(json.loads(path.read_text()))
        except ValueError:
            continue
    return hops


# ── one job ──────────────────────────────────────────────────────────────────


def artifact_path(out_dir: Path, slug: str, arm: str, case_key: str, trial_index: int) -> Path:
    # The harness's layout (runner._artifact_path), so judge_company_research.py reads it unchanged.
    return out_dir / slug / RESEARCH_MODE / arm / case_key / f"trial-{trial_index}.json"


def vendor_public_dict(arm: str) -> dict[str, Any]:
    if arm == jevlin_vendor.FIXTURE_VENDOR_KEY:
        return jevlin_vendor.FIXTURE_SPEC.public_dict()
    return VENDORS[arm].public_dict()


def run_job(case: dict[str, Any], arm: str, trial_index: int, *, args, dataset: dict[str, Any], jobs_dir: Path, version: str, fixture: str | None) -> tuple[dict[str, Any], dict[str, Any]]:
    case_key = str(case["case_key"])
    question = str(case["question"])
    job_dir = jobs_dir / f"trial-{trial_index}" / case_key / arm
    if job_dir.exists():
        shutil.rmtree(job_dir)
    hop_dir = job_dir / "hops"
    hop_dir.mkdir(parents=True)
    # One command path for the whole run, rewritten before each job with that job's arm and hop
    # directory. The model is told the absolute path: Claude Code's Bash runs through the user's
    # shell, which rebuilds PATH from their profile, so a bare name is not reliably found.
    wrapper = write_wrapper(jobs_dir / "bin", arm=arm, hop_dir=hop_dir, max_searches=args.max_searches,
                            max_results=args.max_results, python=sys.executable, fixture=fixture)
    system = SYSTEM_PROMPT.format(max_searches=args.max_searches, command=wrapper)
    (job_dir / "prompt.txt").write_text(question)
    (job_dir / "system.txt").write_text(system)
    argv = claude_argv(args.claude_bin, prompt=question, system=system, wrapper=wrapper, model=args.model,
                       effort=args.effort or None, json_schema=args.json_schema)
    (job_dir / "argv.json").write_text(json.dumps(argv, indent=1))
    env = clean_env(args.claude_bin, tools_dir=wrapper.parent)
    attempts = 0
    waited_s = 0
    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    result: dict[str, Any] = {}
    stderr = ""
    returncode = 0
    while True:
        attempts += 1
        try:
            proc = subprocess.run(argv, cwd=job_dir, env=env, capture_output=True, text=True, timeout=args.job_timeout, check=False)
            stdout, stderr, returncode = proc.stdout, proc.stderr, proc.returncode
        except subprocess.TimeoutExpired:
            stdout, stderr, returncode = "", f"claude timed out after {args.job_timeout}s", 124
        except OSError as exc:
            stdout, stderr, returncode = "", f"{type(exc).__name__}: {exc}", 127
        (job_dir / f"claude.attempt-{attempts}.stdout.json").write_text(stdout or "")
        (job_dir / f"claude.attempt-{attempts}.stderr.txt").write_text(stderr or "")
        result = parse_result(stdout)
        if is_rate_limited(result, stderr, returncode) and attempts <= args.rate_limit_retries:
            print(f"  rate limited on {case_key} {arm} trial {trial_index}; waiting {args.rate_limit_wait}s (attempt {attempts})", flush=True)
            for path in hop_dir.glob("*"):
                path.unlink()
            time.sleep(args.rate_limit_wait)
            waited_s += args.rate_limit_wait
            continue
        break
    latency_ms = int((time.perf_counter() - started) * 1000) - waited_s * 1000

    run_error: str | None = None
    if returncode != 0 and not result:
        run_error = f"claude exit {returncode}: {(stderr or '').strip()[:500]}"
    elif result.get("is_error"):
        run_error = f"claude error ({result.get('subtype')}): {str(result.get('result') or result.get('error') or '')[:500]}"
    elif not result:
        run_error = "claude produced no JSON result"

    hops = load_hops(hop_dir)
    searches = [{
        # Turn boundaries are not observable from outside the session: 0 marks "unknown", and
        # call_index is the wrapper's sequence number.
        "turn_index": 0,
        "call_index": int(hop.get("seq") or i + 1),
        "tool_call_id": f"bash-{int(hop.get('seq') or i + 1):02d}",
        "query": str(hop.get("query") or ""),
        "call": hop.get("call") or {},
    } for i, hop in enumerate(hops)]
    successful = sum(1 for s in searches if (s["call"] or {}).get("status") == "ok")
    failed = sum(1 for s in searches if (s["call"] or {}).get("status") != "ok")

    final_text, used_structured = final_company_set(result) if not run_error else ("", False)
    status = "ok"
    error: str | None = None
    parsed: list[dict[str, Any]] = []
    parse_error: str | None = None
    notes: list[str] = []
    if run_error:
        status, error = "error", run_error
    elif not final_text.strip():
        # agent.py: "model returned no structured output"
        status, error = "error", "RuntimeError: model returned no final message"
    elif successful == 0:
        # agent.py: a final answer without a successful vendor web search is an error
        status, error = "error", "RuntimeError: model returned without a successful vendor web search"
    if status == "ok":
        text = final_text if used_structured else extract_company_json(final_text)
        if text is None:
            parse_error = "final response is not JSON: no company-set object found in the final message"
        else:
            try:
                parsed = [asdict(company) for company in parse_company_set(text)]
            except ValueError as exc:
                parse_error = str(exc)
    if (hop_dir / "budget_exhausted").exists():
        notes.append("search budget exhausted")
    if not hops:
        notes.append("no search was made")
    denials = result.get("permission_denials") if isinstance(result.get("permission_denials"), list) else []
    for denial in denials[:10]:
        notes.append(f"tool denied: {json.dumps(denial, default=str)[:160]}")
    if result.get("subtype") and result.get("subtype") != "success":
        notes.append(f"claude subtype: {result.get('subtype')}")
    if parse_error:
        notes.append(f"parse error: {parse_error[:160]}")
    if error:
        notes.append(error)

    tokens = usage_tokens(result)
    model_cost = result.get("total_cost_usd")
    agent_config = AgentConfig(
        model=args.model, reasoning_effort=args.effort or "none", max_searches=args.max_searches, max_results=args.max_results,
        max_fetches=0, max_fetches_per_turn=None,
        model_input_usd_per_million=None, model_cached_input_usd_per_million=None,
        model_cache_write_usd_per_million=None, model_output_usd_per_million=None,
    )
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "dataset": {"slug": dataset["dataset"]["slug"], "content_sha256": dataset["content_sha256"]},
        "research_mode": RESEARCH_MODE,
        "vendor_key": arm,
        "vendor": vendor_public_dict(arm),
        "case": {"case_key": case_key, "question": question, "gold": case["gold"]},
        "trial_index": trial_index,
        # The protocol's values as the reference states them; `agent.not_enforced` names the ones
        # this runner cannot impose on a Claude Code session.
        "agent_config": asdict(agent_config),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "result": {
            "question": question,
            "vendor_key": arm,
            "final_response": final_text,
            "status": status,
            "error": error,
            "latency_ms": latency_ms,
            **tokens,
            "model_turn_count": int(result.get("num_turns") or 0),
            "searches": searches,
            "fetches": [],
            "raw_model_response": {
                "protocol_version": PROTOCOL_VERSION,
                "vendor_key": arm,
                "responses": [result] if result else [],
                "turn_count": int(result.get("num_turns") or 0),
                "usage": tokens,
            },
            # API-equivalent dollars reported by Claude Code; the subscription is what is billed.
            "model_cost_usd": float(model_cost) if isinstance(model_cost, (int, float)) else None,
        },
        "parsed_companies": parsed,
        "parse_error": parse_error,
        "agent": {
            "kind": "claude-code",
            "version": version,
            "model": args.model,
            "effort": args.effort or None,
            "structured_output": {"requested": bool(args.json_schema), "used": used_structured},
            "output_schema_version": OUTPUT_SCHEMA_VERSION,
            "parser_version": PARSER_VERSION,
            "session_id": result.get("session_id"),
            "num_turns": result.get("num_turns"),
            "duration_ms": result.get("duration_ms"),
            "duration_api_ms": result.get("duration_api_ms"),
            "total_cost_usd": result.get("total_cost_usd"),
            "subtype": result.get("subtype"),
            "attempts": attempts,
            "rate_limit_wait_s": waited_s,
            "model_usage": result.get("modelUsage"),
            "permission_denials": len(denials),
            "system_prompt_sha256": hashlib.sha256(system.encode("utf-8")).hexdigest(),
            "started_at": started_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "job_dir": str(job_dir),
            "enforced": ENFORCED,
            "not_enforced": NOT_ENFORCED,
            "notes": notes,
        },
        "trajectory": [{
            "seq": s["call_index"],
            "query": s["query"],
            "status": (s["call"] or {}).get("status"),
            "n_hits": len((s["call"] or {}).get("hits") or []),
            "error": (s["call"] or {}).get("error"),
            "latency_ms": (s["call"] or {}).get("latency_ms"),
            "cost_usd": (s["call"] or {}).get("cost_usd"),
            "cache_hit": ((s["call"] or {}).get("metadata") or {}).get("jevlin_cache_hit"),
            "upstream_calls": ((s["call"] or {}).get("metadata") or {}).get("jevlin_upstream_calls"),
            "pending": ((s["call"] or {}).get("metadata") or {}).get("jevlin_pending"),
            "request_id": ((s["call"] or {}).get("metadata") or {}).get("jevlin_request_id"),
        } for s in searches],
    }
    path = artifact_path(args.out_dir, dataset["dataset"]["slug"], arm, case_key, trial_index)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(artifact, indent=2, ensure_ascii=False, default=str) + "\n")
    temporary.replace(path)

    score = score_artifact(artifact)  # the upstream judge, untouched
    vendor_costs = [(s["call"] or {}).get("cost_usd") for s in searches]
    row = {
        "case_key": case_key,
        "arm": arm,
        "trial_index": trial_index,
        "artifact": str(path),
        "status": status,
        "error": run_error,  # run-level failure only; protocol errors are in the artifact's result.error
        "result_error": error,
        "parse_error": parse_error,
        "search_count": len(searches),
        "successful_searches": successful,
        "failed_searches": failed,
        "queries": [s["query"] for s in searches],
        "n_companies": len(parsed),
        "true_positive": score["true_positive"],
        "false_positive": score["false_positive"],
        "false_negative": score["false_negative"],
        "precision": score["precision"],
        "recall": score["recall"],
        "f1": score["f1"],
        "exact_set_accuracy": score["exact_set_accuracy"],
        "cited_return_rate": score["cited_return_rate"],
        "cost_usd": score["cost_usd"],
        "vendor_cost_usd": sum(float(c) for c in vendor_costs if c is not None) if vendor_costs and all(c is not None for c in vendor_costs) else None,
        "model_cost_usd": artifact["result"]["model_cost_usd"],
        "cache_hits": sum(1 for t in artifact["trajectory"] if t["cache_hit"]),
        "latency_ms": latency_ms,
        "num_turns": result.get("num_turns"),
        "attempts": attempts,
        "rate_limit_wait_s": waited_s,
        "structured_output_used": used_structured,
        "notes": notes,
    }
    return artifact, row


# ── the pass ─────────────────────────────────────────────────────────────────


def judged_on_nothing(row: dict[str, Any]) -> bool:
    """A job with no successful search was judged on nothing: not done. That is every search failing
    (a gateway storm) and also no search being made at all (a refused command — the pre-fix
    dollar-sign rows of the 2026-10-16 pilot, which the first rule kept on resume and so froze the
    error in place). The harness marks both as errors, so an `ok` row always has a search behind it."""
    succ = row.get("successful_searches")
    if succ is not None:
        return int(succ or 0) == 0 and row.get("status") != "ok"
    return int(row.get("search_count") or 0) > 0 and int(row.get("failed_searches") or 0) == int(row.get("search_count") or 0)


def load_resume_rows(path: Path | None) -> tuple[list[dict[str, Any]], set[tuple[str, str, int]], int]:
    """Rows of an earlier, interrupted pass worth keeping: no run error, artifact present, judged on something."""
    if path is None or not path.is_file():
        return [], set(), 0
    summary = json.loads(path.read_text())
    kept, dropped = [], 0
    for row in summary.get("rows") or []:
        if row.get("error") or not Path(str(row.get("artifact") or "")).is_file() or judged_on_nothing(row):
            dropped += 1
            continue
        kept.append(row)
    keys = {(str(r["case_key"]), str(r["arm"]), int(r["trial_index"])) for r in kept}
    return kept, keys, dropped


def select_cases(dataset: dict[str, Any], *, case_keys: list[str], offset: int, limit: int | None) -> list[dict[str, Any]]:
    cases = list(dataset["cases"])
    if case_keys:
        by_key = {str(c["case_key"]): c for c in cases}
        missing = [k for k in case_keys if k not in by_key]
        if missing:
            raise SystemExit(f"unknown case keys: {missing}")
        cases = [by_key[k] for k in case_keys]
    end = None if limit is None else offset + limit
    selected = cases[offset:end]
    if not selected:
        raise SystemExit("case selection is empty")
    return selected


# Gateway cache key classes. The cache is keyed by query + tier + provider set + filters, so every
# arm that asks for tier balanced with the account's (pinned) provider set — jevlin_balanced, both
# fan-out arms and the CLI's default config — shares one key class; `fast` and `single` are their own;
# direct vendors have no shared cache. The smoke on 2026-10-16 saw 7 of 112 searches answered from
# cache between balanced-tier arms of the same question (the agent wrote the same first query), so a
# pass can be ordered by key class with a TTL wait before every new balanced-tier block.
SHARED_KEY_CLASSES = ("balanced", "fast", "single")


def cache_key_class(arm: str) -> str:
    if arm == "jevlin_fast":
        return "fast"
    if arm == "jevlin_single":
        return "single"
    if arm.startswith("jevlin"):
        return "balanced"
    return f"direct:{arm}"


def block_order(arms: list[str], trial_index: int) -> list[str]:
    """Arm blocks with the balanced-tier arms spread as far apart as the others allow; rotated by
    trial so no arm always runs first or last."""
    balanced = [a for a in arms if cache_key_class(a) == "balanced"]
    others = [a for a in arms if cache_key_class(a) != "balanced"]
    if balanced:
        r = (trial_index - 1) % len(balanced)
        balanced = balanced[r:] + balanced[:r]
    if others:
        r = (trial_index - 1) % len(others)
        others = others[r:] + others[:r]
    if len(balanced) <= 1:
        return balanced + others
    gaps = len(balanced) - 1
    per, extra = divmod(len(others), gaps)
    out, i = [balanced[0]], 0
    for g in range(gaps):
        n = per + (1 if g < extra else 0)
        out += others[i:i + n]
        i += n
        out.append(balanced[g + 1])
    return out + others[i:]


def plan_jobs(cases: list[dict[str, Any]], arms: list[str], *, seed: int | None, trial_index: int, order: str) -> tuple[list[tuple[dict[str, Any], str]], list[str]]:
    """`shuffle`: every (question, arm) pair in one seeded shuffle (the original order). `key-class`:
    one block per arm in `block_order`, questions shuffled by seed within each block."""
    if order == "shuffle":
        jobs = [(case, arm) for case in cases for arm in arms]
        if seed is not None:
            random.Random(seed).shuffle(jobs)
        return jobs, list(arms)
    blocks = block_order(arms, trial_index)
    jobs = []
    for arm in blocks:
        ordered = list(cases)
        if seed is not None:
            random.Random(f"{seed}:{arm}").shuffle(ordered)
        jobs += [(case, arm) for case in ordered]
    return jobs, blocks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True, help="company-research-dataset-v1 JSON (validated and content-addressed on load)")
    parser.add_argument("--arms", required=True, help="comma list of vendor keys (harness vendors and Jevlin arms)")
    parser.add_argument("--cases", default="", help="comma list of case_keys; default all")
    parser.add_argument("--query-offset", type=int, default=0)
    parser.add_argument("--query-limit", type=int, default=None)
    parser.add_argument("--trial-index", type=int, default=1, help="the trial this pass writes (1..N); one pass per trial")
    parser.add_argument("--seed", type=int, default=None, help="shuffle the (question, arm) order with this seed")
    parser.add_argument("--order", choices=("shuffle", "key-class"), default="shuffle",
                        help="shuffle: one seeded shuffle of every (question, arm) pair; key-class: one block per arm, balanced-tier arms spread "
                             "apart and a --cache-gap-min wait before each new block of a shared gateway cache key class")
    parser.add_argument("--cache-gap-min", type=float, default=16.0, help="key-class order: minutes since the previous block of the same key class ended")
    parser.add_argument("--out-dir", type=Path, required=True, help="runs root; artifacts at <out-dir>/<slug>/search_only/<arm>/<case>/trial-<n>.json")
    parser.add_argument("--summary", type=Path, default=None, help="pass summary JSON; default <out-dir>/cc-pass-<trial>.json")
    parser.add_argument("--jobs-dir", type=Path, default=None, help="per-job work dirs (wrapper, hops, claude output); default <out-dir>/jobs")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--effort", default=DEFAULT_EFFORT, help="Claude Code --effort (the harness's reasoning.effort is medium); '' to omit")
    parser.add_argument("--no-json-schema", dest="json_schema", action="store_false", help="do not pass the company-set schema as --json-schema (text parsing only)")
    parser.add_argument("--max-searches", type=int, default=AgentConfig().max_searches)
    parser.add_argument("--max-results", type=int, default=AgentConfig().max_results)
    parser.add_argument("--job-timeout", type=int, default=900, help="seconds per Claude Code session")
    parser.add_argument("--rate-limit-wait", type=int, default=1200)
    parser.add_argument("--rate-limit-retries", type=int, default=12)
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--abort-after-run-errors", type=int, default=5)
    parser.add_argument("--abort-after-hop-errors", type=int, default=12,
                        help="Stop the pass after this many consecutive failed searches across jobs, counted per backend family (all Jevlin arms together); 0 disables.")
    parser.add_argument("--claude-bin", default="claude")
    parser.add_argument("--fixture", default=None, help="tests and plumbing checks only: merged-response fixture served to the jevlin_fixture arm")
    parser.add_argument("--plan-only", action="store_true", help="print the plan (jobs, required env) and exit without starting a session")
    args = parser.parse_args(argv)

    # Absolute paths throughout: the session's working directory is the job directory, the model is
    # told the command's path, and the allow-rule is built from the same string.
    args.dataset = args.dataset.resolve()
    args.out_dir = args.out_dir.resolve()
    args.summary = (args.summary or (args.out_dir / f"cc-pass-{args.trial_index}.json")).resolve()
    args.jobs_dir = (args.jobs_dir or (args.out_dir / "jobs")).resolve()
    if args.resume_from is not None:
        args.resume_from = args.resume_from.resolve()
    if args.fixture:
        args.fixture = str(Path(args.fixture).resolve())
    if args.trial_index <= 0:
        parser.error("--trial-index must be positive")

    dataset, summary_info = load_artifact(args.dataset)  # validates schema and content hash
    arms = [item.strip() for item in args.arms.split(",") if item.strip()]
    unknown = [a for a in arms if a not in VENDORS and a != jevlin_vendor.FIXTURE_VENDOR_KEY]
    if unknown:
        parser.error(f"unknown vendors: {unknown}")
    if jevlin_vendor.FIXTURE_VENDOR_KEY in arms and not args.fixture:
        print("warning: jevlin_fixture without --fixture: every search of that arm will fail (a storm stand-in)", flush=True)
    case_keys = [k.strip() for k in args.cases.split(",") if k.strip()]
    cases = select_cases(dataset, case_keys=case_keys, offset=args.query_offset, limit=args.query_limit)
    jobs, blocks = plan_jobs(cases, arms, seed=args.seed, trial_index=args.trial_index, order=args.order)
    args.block_order = blocks
    env_needed = required_env([a for a in arms if a in VENDORS])
    if args.plan_only:
        print(json.dumps({
            "mode": "plan-only", "dataset_slug": summary_info.dataset_slug, "dataset_sha256": summary_info.content_sha256,
            "trial_index": args.trial_index, "seed": args.seed, "arms": arms, "question_count": len(cases), "planned_jobs": len(jobs),
            "order": args.order, "block_order": blocks if args.order == "key-class" else None,
            "cache_gap_min": args.cache_gap_min if args.order == "key-class" else None,
            "model": args.model, "effort": args.effort or None, "json_schema": args.json_schema,
            "max_searches": args.max_searches, "max_results": args.max_results, "concurrency": 1,
            "required_env_for_the_wrapper": env_needed, "out_dir": str(args.out_dir), "summary": str(args.summary),
        }, indent=2))
        return 0

    # A forms arm on a router without multi-form support would run without its forms: probe once
    # (one balanced search) before any session starts.
    refusal = forms_preflight.require_forms_router(arms)
    if refusal:
        print(refusal, flush=True)
        return 2

    version = claude_version(args.claude_bin)
    resumed_rows, done_keys, dropped = load_resume_rows(args.resume_from)
    if args.resume_from is not None:
        before = len(jobs)
        jobs = [job for job in jobs if (str(job[0]["case_key"]), job[1], args.trial_index) not in done_keys]
        print(f"resume: keeping {len(resumed_rows)} rows from {args.resume_from} ({dropped} dropped: run error, missing artifact or every search failed); "
              f"{before - len(jobs)} jobs skipped, {len(jobs)} to run", flush=True)
    print(f"agent=claude-code version={version} model={args.model} effort={args.effort or '-'} json_schema={args.json_schema} "
          f"dataset={summary_info.dataset_slug} {summary_info.content_sha256[:19]} arms={arms} trial={args.trial_index} n={len(jobs)} seed={args.seed} "
          f"order={args.order}" + (f" blocks={blocks} cache_gap_min={args.cache_gap_min}" if args.order == "key-class" else ""), flush=True)

    rows: list[dict[str, Any]] = list(resumed_rows)
    aborted = False
    consecutive = 0
    # Consecutive failed searches, counted per backend family (all Jevlin arms share one gateway
    # and one account; each direct vendor is its own), so a gateway refusing every call is noticed
    # even though direct-vendor jobs between them keep succeeding.
    hop_errors_in_a_row: dict[str, int] = {}
    # key-class order: the arm and end time of the last job in each shared cache key class, so a new
    # block of the same class starts only after the gateway TTL has passed since the previous one.
    last_in_class: dict[str, tuple[str, datetime]] = {}
    cache_waits_s = 0.0
    started = datetime.now(timezone.utc)
    planned = len(jobs) + len(resumed_rows)
    for case, arm in jobs:
        klass = cache_key_class(arm)
        if args.order == "key-class" and klass in SHARED_KEY_CLASSES and klass in last_in_class and last_in_class[klass][0] != arm:
            wait_s = args.cache_gap_min * 60 - (datetime.now(timezone.utc) - last_in_class[klass][1]).total_seconds()
            if wait_s > 0:
                print(f"== waiting {wait_s / 60:.1f} min before the {arm} block: '{klass}' cache key class, last used by {last_in_class[klass][0]}", flush=True)
                time.sleep(wait_s)
                cache_waits_s += wait_s
        _, row = run_job(case, arm, args.trial_index, args=args, dataset=dataset, jobs_dir=args.jobs_dir, version=version, fixture=args.fixture)
        last_in_class[klass] = (arm, datetime.now(timezone.utc))
        rows.append(row)
        print(f"{row['case_key']} {arm} trial={args.trial_index} {row['status']} f1={row['f1']:.2f} p={row['precision']:.2f} r={row['recall']:.2f} "
              f"searches={row['search_count']}" + (f" ({row['failed_searches']} failed)" if row['failed_searches'] else "") +
              f" turns={row['num_turns']} cost=${row['model_cost_usd'] if row['model_cost_usd'] is not None else '?'} "
              f"{(row.get('error') or row.get('result_error') or row.get('parse_error') or '')[:120]}", flush=True)
        consecutive = consecutive + 1 if row.get("error") else 0
        if args.abort_after_run_errors and consecutive >= args.abort_after_run_errors:
            print(f"ABORTING PASS: {consecutive} consecutive run-level failures; last: {str(row.get('error') or '')[:200]}", flush=True)
            aborted = True
            break
        # Failed searches are data for one job; a run of them across jobs is a vendor or gateway
        # refusing every call (a 429 or 402 storm), and every further job would be judged on nothing.
        family = "jevlin" if arm.startswith("jevlin") else arm
        for status in [s["status"] for s in _trajectory_statuses(row)]:
            hop_errors_in_a_row[family] = hop_errors_in_a_row.get(family, 0) + 1 if status != "ok" else 0
        if args.abort_after_hop_errors and hop_errors_in_a_row.get(family, 0) >= args.abort_after_hop_errors:
            print(f"ABORTING PASS: {hop_errors_in_a_row[family]} consecutive failed searches on {family}", flush=True)
            aborted = True
            break
        args.cache_waits_s = cache_waits_s
        _write_summary(args, rows, arms, version, summary_info, aborted=False, partial=True, started=started, planned=planned)
    args.cache_waits_s = cache_waits_s
    _write_summary(args, rows, arms, version, summary_info, aborted=aborted, partial=False, started=started, planned=planned)
    ok = sum(1 for r in rows if r["status"] == "ok")
    mean_f1 = sum(r["f1"] for r in rows) / len(rows) if rows else 0.0
    print(f"rows={len(rows)} ok={ok} mean_f1={mean_f1:.3f} wrote {args.summary}", flush=True)
    return 1 if aborted else 0


def _trajectory_statuses(row: dict[str, Any]) -> list[dict[str, Any]]:
    # The row keeps counts, not the hop list; rebuild the status sequence it implies: successes
    # then failures would hide an interleaving, so read the artifact when it exists.
    try:
        artifact = json.loads(Path(row["artifact"]).read_text())
        return [{"status": t.get("status")} for t in artifact.get("trajectory") or []]
    except (OSError, ValueError, KeyError):
        return [{"status": "ok"}] * int(row.get("successful_searches") or 0) + [{"status": "error"}] * int(row.get("failed_searches") or 0)


def _write_summary(args, rows, arms, version, summary_info, *, aborted, partial, started, planned) -> None:
    rows = sorted(rows, key=lambda r: (r["case_key"], r["arm"], r["trial_index"]))
    summary = {
        "schema_version": "jevlin-cc-company-pass-v1",
        "label": "Jevlin × OpenBenchmarks Multi-Turn Company Search — Claude Code adaptation (not official OpenBenchmarks scoring)",
        "dataset": {"slug": summary_info.dataset_slug, "content_sha256": summary_info.content_sha256, "path": str(args.dataset)},
        "research_mode": RESEARCH_MODE,
        "trial_index": args.trial_index,
        "shuffle_seed": args.seed,
        "order": args.order,
        "block_order": getattr(args, "block_order", None) if args.order == "key-class" else None,
        "cache_gap_min": args.cache_gap_min if args.order == "key-class" else None,
        "cache_waits_s": round(getattr(args, "cache_waits_s", 0.0), 1),
        "arms": list(arms),
        "model": args.model,
        "agent": {"kind": "claude-code", "version": version, "effort": args.effort or None, "json_schema": args.json_schema,
                  "system_prompt": SYSTEM_PROMPT.format(max_searches=args.max_searches, command=str(args.jobs_dir / "bin" / "web_search")),
                  "enforced": ENFORCED, "not_enforced": NOT_ENFORCED},
        "concurrency": 1,
        "max_searches": args.max_searches,
        "max_results": args.max_results,
        "resumed_from": str(args.resume_from) if args.resume_from else None,
        "aborted": aborted,
        "partial": partial,
        "planned": planned,
        "n": len(rows),
        "ok": sum(1 for r in rows if r["status"] == "ok"),
        "run_errors": sum(1 for r in rows if r.get("error")),
        "parse_errors": sum(1 for r in rows if r.get("parse_error")),
        "mean_f1": round(sum(r["f1"] for r in rows) / len(rows), 4) if rows else None,
        "started_at": started.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "written_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "out_dir": str(args.out_dir),
        "jobs_dir": str(args.jobs_dir),
        "rows": rows,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
