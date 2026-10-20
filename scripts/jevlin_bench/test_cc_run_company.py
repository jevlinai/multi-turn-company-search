"""The headless-Claude-Code company runner, offline, against a fake `claude` binary.

The fake reads the allowed tool pattern from its arguments, runs the generated web_search command
(which calls cc_search.py, which serves the merged fixture for the jevlin_fixture arm), and prints
a Claude Code-shaped JSON result. Modes: a prose-wrapped fenced company set (the tolerant parser),
Claude Code structured output, prose with no JSON (a parse failure), an answer without any search
(the harness's "no successful search" error), and a rate-limit error that clears on retry.

Run with PYTHONPATH=scripts from the repository root:
  python -m unittest discover -s scripts/jevlin_bench -p 'test_*.py'
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(HERE))

import cc_run_company as cc  # noqa: E402
import forms_preflight  # noqa: E402
from company_research.dataset import build_artifact, load_artifact  # noqa: E402
from company_research.judge import build_leaderboard, load_artifacts  # noqa: E402

FIXTURE = ROOT / "scripts" / "company_research" / "testdata" / "jevlin" / "merged_response.json"
VERSIONS_URL = "https://servicenow.com/docs/r/api-reference/rest-apis/versions-api.html"
ANSWER = {"companies": [{"name": "Acme", "domain": "acme.example", "cited_urls": [VERSIONS_URL],
                         "evidence": [{"url": VERSIONS_URL, "claim": "Founded in 2021 in Boulder"}]}]}

FAKE_CLAUDE = r'''#!/bin/bash
# Fake claude: find the wrapper in --allowedTools "Bash(<path> *)", search per the mode, answer.
# Paths are baked in: the driver starts claude with an empty environment, as it must.
MODE_FILE="@@MODE@@"
ANSWER_FILE="@@ANSWER@@"
ARGV_FILE="@@ARGV@@"
wrapper=""
prev=""
for a in "$@"; do
  if [ "$prev" = "--allowedTools" ]; then wrapper="${a#Bash(}"; wrapper="${wrapper% \*)}"; fi
  prev="$a"
done
if [ "$1" = "--version" ]; then echo "9.9.9 (Claude Code fake)"; exit 0; fi
printf '%s\n' "$@" > "$ARGV_FILE"
# the session must not see an API key, a Jevlin variable or a parent-session marker
if [ -n "${ANTHROPIC_API_KEY:-}" ] || [ -n "${JEVLIN_CONFIG:-}" ] || [ -n "${JEVLIN_API_KEY:-}" ] || [ -n "${CLAUDECODE:-}" ]; then
  echo '{"type":"result","is_error":true,"result":"leaked env"}'; exit 0
fi
mode=$(cat "$MODE_FILE" 2>/dev/null || echo ok)
if [ "$mode" = "ratelimit-once" ]; then
  echo "ok" > "$MODE_FILE"
  echo '{"type":"result","subtype":"error","is_error":true,"result":"You have hit your limit · resets at 3pm","num_turns":0,"usage":{"input_tokens":0,"output_tokens":0}}'
  exit 1
fi
if [ "$mode" != "no-search" ]; then
  "$wrapper" "Insight Partners portfolio Colorado Front Range founded 2013" > "$(dirname "$wrapper")/shown.json"
  "$wrapper" "second query" > /dev/null
fi
python3 - "$mode" "$ANSWER_FILE" <<'PY'
import json, sys
mode, answer = sys.argv[1], open(sys.argv[2]).read().strip()
result = {"type": "result", "subtype": "success", "is_error": False, "num_turns": 4, "duration_ms": 1234, "duration_api_ms": 900,
          "session_id": "sess_fake", "total_cost_usd": 0.0123,
          "usage": {"input_tokens": 800, "cache_creation_input_tokens": 100, "cache_read_input_tokens": 100, "output_tokens": 60},
          "modelUsage": {"claude-test": {"inputTokens": 1000, "outputTokens": 60}}, "permission_denials": []}
if mode == "ok":
    result["result"] = "Here is the set I can support:\n\n```json\n" + answer + "\n```\n"
elif mode == "bare":
    result["result"] = answer
elif mode == "structured":
    result["result"] = "Done."
    result["structured_output"] = json.loads(answer)
elif mode == "prose":
    result["result"] = "I could not find evidence for every constraint, so I am not naming any company."
elif mode == "no-search":
    result["result"] = answer
print(json.dumps(result))
PY
'''


class Driver(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.mode_file = self.root / "mode"
        self.mode_file.write_text("ok")
        self.answer_file = self.root / "answer.json"
        self.answer_file.write_text(json.dumps(ANSWER))
        self.argv_file = self.root / "claude-argv.txt"
        self.fake = self.root / "claude"
        self.fake.write_text(FAKE_CLAUDE.replace("@@MODE@@", str(self.mode_file)).replace("@@ANSWER@@", str(self.answer_file)).replace("@@ARGV@@", str(self.argv_file)))
        self.fake.chmod(0o755)
        self.dataset = self.root / "dataset.json"
        payload = build_artifact({"method": "test", "items": [{
            "id": "q-1", "question": "Which companies count Insight Partners among their investors and were founded after 2020?",
            "family": "investor", "constraint_count": 2,
            "clauses": ["count Insight Partners among their investors", "were founded after 2020"],
            "answers": ["Acme"], "answer_domains": {"Acme": "acme.example"}, "answer_aliases": {"Acme": ["Acme Inc."]},
        }]}, dataset_slug="cc-test", dataset_name="CC test dataset")
        self.dataset.write_text(json.dumps(payload))
        self.dataset_hash = payload["content_sha256"]
        self.out = self.root / "out"
        # The parent process carries secrets and a parent-session marker; none may reach the fake claude.
        self.env = {"ANTHROPIC_API_KEY": "must-not-leak", "JEVLIN_API_KEY": "must-not-leak",
                    "JEVLIN_CONFIG": "/home/op/.jevlin/jevlin.toml", "CLAUDECODE": "1"}

    def tearDown(self):
        self.tmp.cleanup()

    def _base(self, *extra, fixture: Path | None = FIXTURE):
        argv = ["--dataset", str(self.dataset), "--arms", "jevlin_fixture", "--trial-index", "1", "--seed", "1",
                "--out-dir", str(self.out), "--claude-bin", str(self.fake), "--rate-limit-wait", "0", "--model", "claude-test", *extra]
        if fixture is not None:
            argv += ["--fixture", str(fixture)]
        return argv

    def _run(self, *extra, fixture: Path | None = FIXTURE):
        with unittest_env(self.env), contextlib.redirect_stdout(io.StringIO()):
            code = cc.main(self._base(*extra, fixture=fixture))
        summary = json.loads((self.out / "cc-pass-1.json").read_text())
        row = summary["rows"][0] if summary["rows"] else None
        artifact = json.loads(Path(row["artifact"]).read_text()) if row and Path(row["artifact"]).is_file() else None
        return code, summary, row, artifact

    def test_one_job_end_to_end_is_a_valid_artifact_the_judge_scores(self):
        code, summary, row, artifact = self._run()
        self.assertEqual(code, 0)
        self.assertEqual(summary["n"], 1)
        self.assertEqual(summary["agent"]["kind"], "claude-code")
        self.assertIn("not official", summary["label"])
        self.assertEqual(row["status"], "ok")
        self.assertIsNone(row["error"])
        self.assertEqual(row["f1"], 1.0)
        self.assertEqual(row["search_count"], 2)
        self.assertEqual(row["failed_searches"], 0)
        self.assertEqual(row["vendor_cost_usd"], 0.02)
        self.assertEqual(row["cost_usd"], 0.02 + 0.0123)
        # the artifact is at the harness's path and in the harness's schema
        self.assertEqual(Path(row["artifact"]), (self.out / "cc-test" / "search_only" / "jevlin_fixture" / "q-1" / "trial-1.json").resolve())
        self.assertEqual(artifact["schema_version"], "multi-turn-company-search-run-v1")
        self.assertEqual(artifact["dataset"], {"slug": "cc-test", "content_sha256": self.dataset_hash})
        self.assertEqual(artifact["research_mode"], "search_only")
        self.assertEqual(artifact["vendor_key"], "jevlin_fixture")
        self.assertEqual(artifact["vendor"]["key"], "jevlin_fixture")
        self.assertEqual(artifact["case"]["case_key"], "q-1")
        self.assertEqual(artifact["case"]["gold"][0]["name"], "Acme")
        self.assertEqual(artifact["trial_index"], 1)
        self.assertEqual(artifact["agent_config"]["max_searches"], 14)
        self.assertEqual(artifact["agent_config"]["max_fetches"], 0)
        result = artifact["result"]
        self.assertEqual(result["status"], "ok")
        self.assertIsNone(result["error"])
        self.assertEqual(len(result["searches"]), 2)
        self.assertEqual(result["searches"][0]["query"], "Insight Partners portfolio Colorado Front Range founded 2013")
        call = result["searches"][0]["call"]
        self.assertEqual(call["status"], "ok")
        self.assertEqual(call["cost_usd"], 0.01)
        self.assertEqual(call["raw_response"]["request_id"], "req_fixture_0001")  # the whole gateway body
        self.assertEqual(call["metadata"]["jevlin_upstream_calls"], 4)
        self.assertEqual(set(call["hits"][0]), {"url", "title", "snippet", "metadata"})
        self.assertEqual(result["fetches"], [])
        self.assertEqual(result["model_turn_count"], 4)
        self.assertEqual(result["model_cost_usd"], 0.0123)
        self.assertEqual(result["model_input_tokens"], 1000)
        self.assertEqual(result["model_cached_input_tokens"], 100)
        self.assertEqual(result["model_output_tokens"], 60)
        self.assertEqual(result["raw_model_response"]["responses"][0]["session_id"], "sess_fake")
        self.assertEqual(artifact["parsed_companies"][0]["name"], "Acme")
        self.assertEqual(artifact["parsed_companies"][0]["cited_urls"], [VERSIONS_URL])
        self.assertIsNone(artifact["parse_error"])
        agent = artifact["agent"]
        self.assertEqual(agent["version"], "9.9.9 (Claude Code fake)")
        self.assertEqual(agent["structured_output"], {"requested": True, "used": False})
        self.assertEqual(agent["effort"], "medium")
        self.assertTrue(agent["not_enforced"] and agent["enforced"])
        self.assertEqual(len(artifact["trajectory"]), 2)
        self.assertEqual(artifact["trajectory"][0]["request_id"], "req_fixture_0001")
        self.assertFalse(artifact["trajectory"][0]["cache_hit"])
        # the upstream judge, through its own loader and completeness gate
        board = build_leaderboard(load_artifacts(self.out / "cc-test"), expected_trials=1)["boards"]["search_only"]
        self.assertEqual(board[0]["vendor_key"], "jevlin_fixture")
        self.assertEqual(board[0]["f1"]["mean_pct"], 100.0)
        self.assertEqual(board[0]["mean_cost_usd"], 0.0323)
        # what the model was shown: the harness's tool-output shape, verbatim
        shown = json.loads((self.out / "jobs" / "bin" / "shown.json").read_text())
        self.assertEqual(set(shown), {"vendor", "query", "hits", "error"})
        self.assertIsNone(shown["error"])
        self.assertEqual(shown["vendor"], "jevlin_fixture")
        self.assertEqual(len(shown["hits"]), 4)
        # the prompt names the command by absolute path; the session got the flags
        self.assertIn(str(self.out / "jobs" / "bin" / "web_search"), (Path(agent["job_dir"]) / "system.txt").read_text())
        argv = self.argv_file.read_text().splitlines()
        for flag in ("--safe-mode", "--strict-mcp-config", "--disable-slash-commands", "--no-session-persistence", "--json-schema", "--effort"):
            self.assertIn(flag, argv)
        self.assertEqual(argv[argv.index("--effort") + 1], "medium")
        self.assertIn('"companies"', argv[argv.index("--json-schema") + 1])
        self.assertEqual(argv[argv.index("--tools") + 1], "Bash")
        self.assertEqual(argv[argv.index("-p") + 1], "Which companies count Insight Partners among their investors and were founded after 2020?")

    def test_structured_output_is_preferred_when_claude_code_supplies_it(self):
        self.mode_file.write_text("structured")
        code, _, row, artifact = self._run()
        self.assertEqual(code, 0)
        self.assertEqual(row["f1"], 1.0)
        self.assertTrue(row["structured_output_used"])
        self.assertEqual(artifact["agent"]["structured_output"], {"requested": True, "used": True})
        self.assertEqual(artifact["parsed_companies"][0]["name"], "Acme")
        self.assertEqual(json.loads(artifact["result"]["final_response"]), ANSWER)

    def test_bare_json_final_message_parses(self):
        self.mode_file.write_text("bare")
        code, _, row, _ = self._run("--no-json-schema", "--effort", "")
        self.assertEqual(code, 0)
        self.assertEqual(row["f1"], 1.0)
        argv = self.argv_file.read_text().splitlines()
        self.assertNotIn("--json-schema", argv)
        self.assertNotIn("--effort", argv)

    def test_parse_failure_scores_as_an_empty_set_not_a_run_error(self):
        self.mode_file.write_text("prose")
        code, _, row, artifact = self._run()
        self.assertEqual(code, 0)
        self.assertEqual(row["status"], "ok")
        self.assertIsNone(row["error"])
        self.assertIsNotNone(row["parse_error"])
        self.assertEqual(artifact["parsed_companies"], [])
        self.assertEqual(artifact["result"]["status"], "ok")
        self.assertEqual(row["f1"], 0.0)
        self.assertEqual(row["recall"], 0.0)
        self.assertTrue(any(n.startswith("parse error") for n in row["notes"]))
        # the judge keeps the row in the denominator as an empty prediction
        board = build_leaderboard(load_artifacts(self.out / "cc-test"), expected_trials=1)["boards"]["search_only"]
        self.assertEqual(board[0]["f1"]["mean_pct"], 0.0)

    def test_an_answer_without_a_successful_search_is_a_protocol_error(self):
        self.mode_file.write_text("no-search")
        code, _, row, artifact = self._run()
        self.assertEqual(code, 0)
        self.assertIsNone(row["error"])  # not a run-level failure
        self.assertEqual(row["status"], "error")
        self.assertEqual(artifact["result"]["status"], "error")
        self.assertIn("without a successful vendor web search", artifact["result"]["error"])
        self.assertEqual(artifact["parsed_companies"], [])
        self.assertEqual(row["f1"], 0.0)
        self.assertIn("no search was made", row["notes"])

    def test_relative_paths_are_made_absolute_for_prompt_and_allow_rule(self):
        cwd = os.getcwd()
        os.chdir(self.root)
        try:
            argv = ["--dataset", "dataset.json", "--arms", "jevlin_fixture", "--out-dir", "rel-out", "--claude-bin", str(self.fake),
                    "--fixture", str(FIXTURE), "--model", "claude-test"]
            with unittest_env(self.env):
                code = cc.main(argv)
            summary = json.loads((self.root / "rel-out" / "cc-pass-1.json").read_text())
        finally:
            os.chdir(cwd)
        self.assertEqual(code, 0)
        row = summary["rows"][0]
        self.assertEqual(row["f1"], 1.0)
        self.assertTrue(Path(row["artifact"]).is_absolute())
        artifact = json.loads(Path(row["artifact"]).read_text())
        system = (Path(artifact["agent"]["job_dir"]) / "system.txt").read_text()
        self.assertIn(str(self.root / "rel-out" / "jobs" / "bin" / "web_search"), system)
        self.assertIn(str(self.root / "rel-out" / "jobs" / "bin" / "web_search"), summary["agent"]["system_prompt"])

    def test_rate_limit_is_waited_out_and_retried(self):
        self.mode_file.write_text("ratelimit-once")
        code, _, row, artifact = self._run("--rate-limit-retries", "2")
        self.assertEqual(code, 0)
        self.assertEqual(row["f1"], 1.0)
        self.assertEqual(row["attempts"], 2)
        self.assertEqual(artifact["agent"]["attempts"], 2)

    def test_rate_limit_exhausted_is_a_run_error_and_aborts(self):
        self.mode_file.write_text("ratelimit-once")
        code, summary, row, artifact = self._run("--rate-limit-retries", "0", "--abort-after-run-errors", "1")
        self.assertEqual(code, 1)
        self.assertTrue(summary["aborted"])
        self.assertIn("claude error", row["error"])
        self.assertEqual(artifact["result"]["status"], "error")
        self.assertEqual(row["f1"], 0.0)

    def test_a_run_of_failed_searches_aborts_the_pass(self):
        # A missing fixture makes every search of the jevlin_fixture arm fail: a 429 storm stand-in.
        code, summary, row, artifact = self._run("--abort-after-hop-errors", "2", fixture=self.root / "missing.json")
        self.assertEqual(code, 1)
        self.assertTrue(summary["aborted"])
        self.assertIsNone(row["error"])  # the job itself ran; its searches failed
        self.assertEqual(row["search_count"], 2)
        self.assertEqual(row["failed_searches"], 2)
        self.assertEqual(artifact["result"]["status"], "error")  # no successful search: the harness's rule
        self.assertTrue(all(s["call"]["status"] == "error" for s in artifact["result"]["searches"]))
        self.assertEqual(artifact["result"]["searches"][0]["call"]["cost_usd"], 0.0)
        self.assertTrue(cc.judged_on_nothing(row))

    def test_resume_reruns_jobs_judged_on_nothing_and_keeps_the_rest(self):
        summary_path = self.out / "cc-pass-1.json"
        with unittest_env(self.env):
            cc.main(self._base("--abort-after-hop-errors", "0", fixture=self.root / "missing.json"))  # a storm: every search fails
            first = json.loads(summary_path.read_text())
            self.assertTrue(cc.judged_on_nothing(first["rows"][0]))
            code = cc.main(self._base("--resume-from", str(summary_path)))  # the fixture is back
        second = json.loads(summary_path.read_text())
        self.assertEqual(code, 0)
        self.assertEqual(second["n"], 1)
        self.assertEqual(second["rows"][0]["f1"], 1.0)  # re-run, not kept
        # a judged row is kept: the fake is not invoked again even though it would now fail
        self.argv_file.unlink()
        self.mode_file.write_text("prose")
        with unittest_env(self.env):
            code = cc.main(self._base("--resume-from", str(summary_path)))
        third = json.loads(summary_path.read_text())
        self.assertEqual(code, 0)
        self.assertEqual(third["n"], 1)
        self.assertEqual(third["rows"][0]["f1"], 1.0)
        self.assertFalse(self.argv_file.exists())

    def test_budget_is_enforced_by_the_wrapper(self):
        hop_dir = self.root / "hops"
        hop_dir.mkdir()
        wrapper = cc.write_wrapper(self.root / "tools", arm="jevlin_fixture", hop_dir=hop_dir, max_searches=2,
                                   max_results=10, python=sys.executable, fixture=str(FIXTURE))
        outputs = [subprocess.run([str(wrapper), f"query {i}"], capture_output=True, text=True, check=False).stdout for i in range(3)]
        self.assertEqual(len(list(hop_dir.glob("search-*.json"))), 2)
        first = json.loads(outputs[0])
        self.assertEqual(set(first), {"vendor", "query", "hits", "error"})
        self.assertEqual(len(first["hits"]), 4)
        self.assertEqual(json.loads(outputs[2]), {"error": "search budget exhausted; answer now"})
        self.assertTrue((hop_dir / "budget_exhausted").exists())
        hop = json.loads((hop_dir / "search-01.json").read_text())
        self.assertEqual(hop["call"]["raw_response"]["request_id"], "req_fixture_0001")
        self.assertEqual(hop["call"]["metadata"]["jevlin_merged_total"], 4)
        usage = subprocess.run([str(wrapper)], capture_output=True, text=True, check=False)
        self.assertEqual(usage.returncode, 64)

    def test_max_results_is_passed_through_the_wrapper(self):
        hop_dir = self.root / "hops2"
        hop_dir.mkdir()
        wrapper = cc.write_wrapper(self.root / "tools2", arm="jevlin_fixture", hop_dir=hop_dir, max_searches=14,
                                   max_results=2, python=sys.executable, fixture=str(FIXTURE))
        shown = json.loads(subprocess.run([str(wrapper), "q"], capture_output=True, text=True, check=False).stdout)
        self.assertEqual(len(shown["hits"]), 2)

    def test_clean_env_has_no_secrets_or_parent_markers(self):
        with unittest_env({"ANTHROPIC_API_KEY": "x", "JEVLIN_API_KEY": "y", "CLAUDECODE": "1", "JEVLIN_CONFIG": "z", "EXA_API_KEY": "w"}):
            env = cc.clean_env(str(self.fake))
        for key in ("ANTHROPIC_API_KEY", "JEVLIN_API_KEY", "JEVLIN_CONFIG", "CLAUDECODE", "EXA_API_KEY"):
            self.assertNotIn(key, env)
        self.assertIn("HOME", env)
        self.assertIn(str(self.fake.parent), env["PATH"])
        tools = self.root / "tools"
        self.assertTrue(cc.clean_env(str(self.fake), tools_dir=tools)["PATH"].startswith(str(tools) + ":"))

    def test_claude_argv_names_both_spellings_and_the_isolation_flags(self):
        wrapper = self.root / "tools" / "web_search"
        argv = cc.claude_argv(str(self.fake), prompt="p", system="s", wrapper=wrapper, model="m", effort="medium", json_schema=True)
        i = argv.index("--allowedTools")
        self.assertEqual(argv[i + 1], f"Bash({wrapper} *)")
        self.assertEqual(argv[i + 2], "Bash(web_search *)")
        for flag in ("--safe-mode", "--strict-mcp-config", "--disable-slash-commands", "--no-session-persistence", "--permission-mode", "--permission-prompts"):
            self.assertIn(flag, argv)
        self.assertEqual(argv[argv.index("--permission-prompts") + 1], "none")
        self.assertIn("WebSearch,WebFetch,Read,Write,Edit,Glob,Grep,Task,TodoWrite,NotebookEdit", argv)
        schema = json.loads(argv[argv.index("--json-schema") + 1])
        self.assertEqual(schema["required"], ["companies"])
        bare = cc.claude_argv(str(self.fake), prompt="p", system="s", wrapper=wrapper, model="m", effort=None, json_schema=False)
        self.assertNotIn("--json-schema", bare)
        self.assertNotIn("--effort", bare)

    def test_extract_company_json_is_tolerant_outside_and_strict_inside(self):
        answer = json.dumps(ANSWER)
        self.assertEqual(json.loads(cc.extract_company_json(answer)), ANSWER)
        self.assertEqual(json.loads(cc.extract_company_json(f"Here you go:\n```json\n{answer}\n```\nDone.")), ANSWER)
        self.assertEqual(json.loads(cc.extract_company_json(f"Summary first {{not json}} then {answer} and a trailing brace }}")), ANSWER)
        nested = {"companies": [{"name": "Br{ace} Co", "domain": None, "cited_urls": [], "evidence": [{"url": "https://a.test", "claim": "has } in it"}]}]}
        self.assertEqual(json.loads(cc.extract_company_json("x " + json.dumps(nested) + " y")), nested)
        self.assertIsNone(cc.extract_company_json("no json here"))
        self.assertIsNone(cc.extract_company_json('{"other": 1}'))
        self.assertIsNone(cc.extract_company_json(""))
        self.assertIsNone(cc.extract_company_json('{"companies": [unterminated'))

    def test_plan_only_starts_no_session(self):
        with unittest_env(self.env):
            code = cc.main(self._base("--plan-only"))
        self.assertEqual(code, 0)
        self.assertFalse((self.out / "jobs").exists())
        self.assertFalse(self.argv_file.exists())

    def test_dataset_is_validated_on_load(self):
        self.dataset.write_text(json.dumps({"schema_version": "company-research-dataset-v1", "cases": []}))
        with self.assertRaises(Exception):
            with unittest_env(self.env):
                cc.main(self._base("--plan-only"))
        load_artifact  # the loader is the harness's own; this test only pins that it is on the path


class unittest_env:
    """Temporarily set environment variables."""

    def __init__(self, values: dict[str, str]):
        self.values = values
        self.saved: dict[str, str | None] = {}

    def __enter__(self):
        for key, value in self.values.items():
            self.saved[key] = os.environ.get(key)
            os.environ[key] = value
        return self

    def __exit__(self, *exc):
        for key, old in self.saved.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old
        return False


class JobOrder(unittest.TestCase):
    ARMS = ["perplexity", "exa_instant", "parallel_fast", "jevlin_fast", "jevlin_balanced", "jevlin_single",
            "jevlin_fanout_raw", "jevlin_fanout_merged", "jevlin_cli"]
    CASES = [{"case_key": f"c{i}"} for i in range(4)]

    def test_cache_key_classes(self):
        self.assertEqual(cc.cache_key_class("jevlin_fast"), "fast")
        self.assertEqual(cc.cache_key_class("jevlin_single"), "single")
        for arm in ("jevlin_balanced", "jevlin_fanout_raw", "jevlin_fanout_merged", "jevlin_cli"):
            self.assertEqual(cc.cache_key_class(arm), "balanced")
        self.assertEqual(cc.cache_key_class("exa_instant"), "direct:exa_instant")

    def test_forms_arm_shares_the_balanced_key_class_with_its_twin(self):
        # Its first search of a job sends no forms, so it is byte-identical to jevlin_balanced's and
        # hits the same gateway cache key: the cache-gap wait between the two blocks must apply.
        self.assertEqual(cc.cache_key_class("jevlin_balanced_forms"), "balanced")
        self.assertEqual(cc.cache_key_class("jevlin_balanced_forms"), cc.cache_key_class("jevlin_balanced"))

    def test_forms_arm_is_spread_with_the_other_balanced_tier_arms(self):
        arms = ["jevlin_balanced", "jevlin_balanced_forms", "exa_instant", "perplexity"]
        for trial in (1, 2, 3):
            blocks = cc.block_order(arms, trial)
            self.assertEqual(sorted(blocks), sorted(arms))
            at = [i for i, a in enumerate(blocks) if cc.cache_key_class(a) == "balanced"]
            self.assertEqual(len(at), 2)
            self.assertGreaterEqual(at[1] - at[0], 2, blocks)

    def test_key_class_order_runs_arm_blocks_with_balanced_tier_arms_apart(self):
        jobs, blocks = cc.plan_jobs(self.CASES, self.ARMS, seed=1, trial_index=1, order="key-class")
        self.assertEqual(len(jobs), 36)
        seq = [arm for _, arm in jobs]
        self.assertEqual([a for i, a in enumerate(seq) if i == 0 or seq[i - 1] != a], blocks, "each arm is one contiguous block")
        self.assertEqual(sorted(blocks), sorted(self.ARMS))
        balanced_positions = [i for i, a in enumerate(blocks) if cc.cache_key_class(a) == "balanced"]
        self.assertTrue(all(b - a >= 2 for a, b in zip(balanced_positions, balanced_positions[1:])), blocks)
        # Every (question, arm) pair exactly once, the same set the shuffle order produces.
        shuffled, _ = cc.plan_jobs(self.CASES, self.ARMS, seed=7, trial_index=1, order="shuffle")
        self.assertEqual(sorted((c["case_key"], a) for c, a in shuffled), sorted((c["case_key"], a) for c, a in jobs))

    def test_block_order_rotates_by_trial_and_shuffle_is_unchanged(self):
        b1 = cc.block_order(self.ARMS, 1)
        b2 = cc.block_order(self.ARMS, 2)
        self.assertNotEqual(b1, b2)
        self.assertEqual(sorted(b1), sorted(b2))
        self.assertEqual(cc.block_order(["exa_instant", "jevlin_balanced"], 1), ["jevlin_balanced", "exa_instant"])
        self.assertEqual(cc.block_order(["exa_instant", "perplexity"], 3), ["exa_instant", "perplexity"])
        jobs, blocks = cc.plan_jobs(self.CASES, self.ARMS, seed=7, trial_index=1, order="shuffle")
        self.assertEqual(blocks, self.ARMS)
        expected = [(c, a) for c in self.CASES for a in self.ARMS]
        import random
        random.Random(7).shuffle(expected)
        self.assertEqual(jobs, expected)


class FormsPreflight(unittest.TestCase):
    def test_only_a_forms_arm_needs_the_probe(self):
        def fail():
            raise AssertionError("probe must not run")
        self.assertIsNone(forms_preflight.require_forms_router(["jevlin_balanced", "exa_instant"], probe_fn=fail))

    def test_accepting_router_passes_and_a_refusing_one_blocks_naming_the_pr_and_the_cost(self):
        arms = ["jevlin_balanced", "jevlin_balanced_forms"]
        self.assertIsNone(forms_preflight.require_forms_router(arms, probe_fn=lambda: (True, "accepted")))
        message = forms_preflight.require_forms_router(arms, probe_fn=lambda: (False, "HTTP 400: malformed_json"))
        self.assertIn("PR #55", message)
        self.assertIn("malformed_json", message)
        self.assertIn("$0.08", message)

    def test_probe_accepts_only_a_200_with_coverage_by_form(self):
        def call(status, payload, error=None):
            return SimpleNamespace(status=status, raw_response=payload, error=error)
        cases = [
            (call("ok", {"coverage": {"by_form": []}}), True),
            (call("ok", {"merged": []}), False),
            (call("error", {"error": "malformed_json"}, "HTTP 400: malformed_json"), False),
        ]
        for response, accepted in cases:
            with unittest_env({"JEVLIN_API_KEY": "k"}), mock.patch("company_research.vendors._request", return_value=response) as req, \
                    mock.patch("dotenv.load_dotenv"):
                self.assertEqual(forms_preflight.probe()[0], accepted)
            body = req.call_args.kwargs["body"]
            self.assertEqual((body["tier"], body["view"], body["max_results"]), ("balanced", "merged", 1))
            self.assertEqual(len(body["search_queries"]), 1)
            self.assertEqual(req.call_args.kwargs["retries"], 0)

    def test_driver_refuses_to_start_a_forms_pass_against_a_router_without_forms(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "dataset.json"
            dataset.write_text(json.dumps(build_artifact({"method": "t", "items": [{
                "id": "q-1", "question": "Which companies were founded after 2020?", "family": "other",
                "constraint_count": 1, "clauses": ["founded after 2020"], "answers": ["Acme"]}]},
                dataset_slug="t", dataset_name="T")))
            argv = ["--dataset", str(dataset), "--arms", "jevlin_balanced,jevlin_balanced_forms", "--out-dir", str(Path(tmp) / "out"),
                    "--claude-bin", "/nonexistent/claude"]
            out = io.StringIO()
            with mock.patch.object(forms_preflight, "probe", return_value=(False, "HTTP 400: malformed_json")), \
                    mock.patch.object(cc, "run_job", side_effect=AssertionError("no session may start")), contextlib.redirect_stdout(out):
                self.assertEqual(cc.main(argv), 2)
            self.assertIn("PR #55", out.getvalue())
            self.assertFalse((Path(tmp) / "out" / "cc-pass-1.json").exists())
            # plan-only never probes (no spend)
            with mock.patch.object(forms_preflight, "probe", side_effect=AssertionError("no probe")), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cc.main([*argv, "--plan-only"]), 0)


class ResumeRule(unittest.TestCase):
    def test_no_successful_search_is_judged_on_nothing_whether_searches_failed_or_were_never_made(self):
        storm = {"status": "error", "search_count": 3, "successful_searches": 0, "failed_searches": 3}
        refused = {"status": "error", "result_error": "RuntimeError: model returned without a successful vendor web search",
                   "search_count": 0, "successful_searches": 0, "failed_searches": 0}
        judged = {"status": "ok", "search_count": 5, "successful_searches": 4, "failed_searches": 1}
        self.assertTrue(cc.judged_on_nothing(storm))
        self.assertTrue(cc.judged_on_nothing(refused), "zero searches made: the row must re-run on resume")
        self.assertFalse(cc.judged_on_nothing(judged))
        # rows written before successful_searches existed fall back to the all-failed rule
        self.assertTrue(cc.judged_on_nothing({"status": "error", "search_count": 2, "failed_searches": 2}))
        self.assertFalse(cc.judged_on_nothing({"status": "error", "search_count": 0, "failed_searches": 0}))


if __name__ == "__main__":
    unittest.main()
