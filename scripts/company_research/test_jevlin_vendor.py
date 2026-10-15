"""Offline contracts for the Jevlin arms in this harness: registry, request shape, presentation,
cost conversion, telemetry, failures, the CLI arm and the fixture path. No network: the HTTP arms
are exercised by patching the adapter's `_request`; the CLI arm by patching `subprocess.run`.
Fixtures are the coding-agents fork's (f01b032), copied under testdata/jevlin/."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import jevlin_core.jevlin as core
from company_research import jevlin_vendor, vendors
from company_research.agent import AgentConfig
from company_research.dataset import build_artifact
from company_research.runner import RunnerConfig
from company_research.vendors import VENDORS, VendorCall, _parse_hits, fetch, search

HERE = Path(__file__).resolve().parent
MERGED = json.loads((HERE / "testdata" / "jevlin" / "merged_response.json").read_text())
CLI_OK = json.loads((HERE / "testdata" / "jevlin" / "cli_envelope.json").read_text())

VERSIONS_URL = "https://servicenow.com/docs/r/api-reference/rest-apis/versions-api.html"
SHARED_URL = "https://servicenow.com/docs/r/api-reference/rest-apis/cdm-shared-libraries-api.html"
CDM_URL = "https://developer.servicenow.com/dev.do/reference/api/latest/rest/cdm-api"
THREAD_URL = "https://community.servicenow.com/cdm-snapshots-thread"

ENV = {
    "JEVLIN_API_KEY": "test-key",
    "JEVLIN_API_URL": "https://jevlin.test",
    "JEVLIN_PROVIDERS": "exa,brave,serpapi,tinyfish,you",
}
QUERY = "companies backed by Insight Partners founded 2013-2015 Colorado"


def _env(**extra: str):
    return patch.dict(os.environ, {**ENV, **extra}, clear=False)


def _ok(payload, *, latency_ms: int = 12) -> VendorCall:
    return VendorCall(
        status="ok", latency_ms=latency_ms,
        raw_request={"method": "POST", "url": "https://jevlin.test/v1/search", "headers": {"Authorization": vendors.REDACTED}, "body": {}, "params": None},
        raw_response=payload, error=None, attempts=[{"attempt": 1, "status_code": 200, "latency_ms": latency_ms}], cost_usd=None,
    )


def _completed(stdout: str, *, code: int = 0, stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=["jevlin", "search", "--stdin"], returncode=code, stdout=stdout, stderr=stderr)


class Registration(unittest.TestCase):
    def test_the_six_company_arms_are_registered_search_only(self):
        self.assertEqual(core.COMPANY_ARMS, ("jevlin_fast", "jevlin_balanced", "jevlin_single", "jevlin_fanout_raw", "jevlin_fanout_merged", "jevlin_cli"))
        for name in core.COMPANY_ARMS:
            with self.subTest(arm=name):
                spec = VENDORS[name]
                self.assertEqual(spec.key, name)
                self.assertIsNone(spec.search_unit_cost_usd)  # observed from the gateway, never a rate card
                self.assertFalse(spec.native_fetch)
                self.assertFalse(spec.custom_fetch)
                self.assertIn("JEVLIN_API_KEY", spec.env_keys)
                self.assertIsInstance(spec.public_dict()["env_keys"], list)
                # fetch is unavailable without any network call
                call = fetch(name, "https://example.com", objective="x")
                self.assertEqual(call.status, "unavailable")
        self.assertEqual(VENDORS["jevlin_fanout_raw"].env_keys, ("JEVLIN_API_KEY", "JEVLIN_PROVIDERS"))
        self.assertEqual(VENDORS["jevlin_cli"].env_keys, ("JEVLIN_API_KEY", "JEVLIN_BIN", "JEVLIN_CLI_CONFIG"))
        self.assertNotIn("jevlin_max", VENDORS)
        self.assertNotIn(jevlin_vendor.FIXTURE_VENDOR_KEY, VENDORS)

    def test_upstream_roster_is_untouched(self):
        self.assertTrue(VENDORS["perplexity"].custom_fetch)
        self.assertEqual(VENDORS["exa_instant"].search_unit_cost_usd, 0.007)
        self.assertEqual(VENDORS["parallel_fast"].search_unit_cost_usd, 0.001)
        self.assertEqual(vendors._hit("u", "t", "x" * 5000)["snippet"], "x" * jevlin_vendor.SNIPPET_CHARS)

    def test_vendored_core_names_its_source_commit(self):
        self.assertEqual(core.SOURCE_COMMIT, "f01b032")
        self.assertIn("f01b032", core.__doc__)
        self.assertIn("web-search-for-coding-agents", core.__doc__)

    def test_search_and_fetch_plan_refuses_a_search_only_vendor(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "dataset.json"
            dataset.write_text(json.dumps(build_artifact({"method": "t", "items": [{
                "id": "q-1", "question": "Find companies founded after 2020", "family": "other",
                "constraint_count": 1, "clauses": ["founded after 2020"], "answers": ["Acme"]}]},
                dataset_slug="t", dataset_name="T")))
            with self.assertRaisesRegex(ValueError, "fetch path"):
                RunnerConfig(dataset_path=dataset, vendor_keys=("exa_instant", "jevlin_fast"), research_mode="search_and_fetch").validate()
            RunnerConfig(dataset_path=dataset, vendor_keys=("jevlin_fast",), research_mode="search_only",
                         agent=AgentConfig(max_fetches=0, max_fetches_per_turn=None)).validate()


class RequestShape(unittest.TestCase):
    def _capture(self, arm: str, *, max_results: int = 10, **env: str):
        with _env(**env), patch("company_research.jevlin_vendor._request", return_value=_ok(MERGED)) as req:
            call = search(arm, QUERY, max_results=max_results)
        self.assertEqual(req.call_count, 1)
        return call, req.call_args.kwargs

    def test_tier_arms_send_tier_view_merged_and_no_providers(self):
        for arm, tier in (("jevlin_fast", "fast"), ("jevlin_balanced", "balanced")):
            with self.subTest(arm=arm):
                _, kw = self._capture(arm)
                self.assertEqual((kw["method"], kw["url"]), ("POST", "https://jevlin.test/v1/search"))
                self.assertEqual(kw["body"], {"query": QUERY, "max_results": 10, "view": "merged", "tier": tier})
                self.assertEqual(kw["headers"]["Authorization"], "Bearer test-key")
                self.assertEqual(kw["timeout"], 45)

    def test_fanout_arms_pin_the_provider_set(self):
        for arm in ("jevlin_fanout_raw", "jevlin_fanout_merged"):
            with self.subTest(arm=arm):
                _, kw = self._capture(arm, max_results=5)
                self.assertEqual(kw["body"], {"query": QUERY, "max_results": 5, "view": "merged", "tier": "balanced",
                                              "providers": ["exa", "brave", "serpapi", "tinyfish", "you"]})

    def test_fanout_arms_refuse_without_a_pinned_set(self):
        for arm in ("jevlin_fanout_raw", "jevlin_fanout_merged"):
            with self.subTest(arm=arm), _env(JEVLIN_PROVIDERS=""), patch("company_research.jevlin_vendor._request") as req:
                with self.assertRaisesRegex(RuntimeError, "JEVLIN_PROVIDERS"):
                    search(arm, QUERY)
                self.assertEqual(req.call_count, 0)

    def test_single_arm_names_one_provider_and_no_tier(self):
        _, kw = self._capture("jevlin_single")
        self.assertEqual(kw["body"]["providers"], ["exa"])
        self.assertNotIn("tier", kw["body"])
        _, kw = self._capture("jevlin_single", JEVLIN_SINGLE_PROVIDER="brave")
        self.assertEqual(kw["body"]["providers"], ["brave"])

    def test_missing_key_is_refused_before_any_call(self):
        with patch.dict(os.environ, {"JEVLIN_API_KEY": ""}), patch("company_research.jevlin_vendor._request") as req:
            with self.assertRaisesRegex(RuntimeError, "JEVLIN_API_KEY"):
                search("jevlin_balanced", QUERY)
            self.assertEqual(req.call_count, 0)

    def test_timeout_and_url_env_overrides(self):
        _, kw = self._capture("jevlin_balanced", JEVLIN_TIMEOUT_S="12", JEVLIN_API_URL="https://beta.jevlin.test/")
        self.assertEqual(kw["timeout"], 12)
        self.assertEqual(kw["url"], "https://beta.jevlin.test/v1/search")

    def test_the_hosts_retry_rule_applies_to_jevlin(self):
        # A 429 is retried by vendors._request for every vendor alike (3 retries, backoff); the
        # coding fork's "no retries" stance is replaced by the host harness's rule.
        class Resp:
            def __init__(self, status, body):
                self.status_code, self._body, self.ok = status, body, status < 400
            def json(self):
                return self._body
        answers = [Resp(429, {"error": "rate limit"}), Resp(429, {"error": "rate limit"}), Resp(200, MERGED)]
        real = vendors._request

        def patched(**kw):
            return real(**kw, request_fn=lambda **_: answers.pop(0), sleep_fn=lambda s: None)

        with _env(), patch("company_research.jevlin_vendor._request", side_effect=patched):
            call = search("jevlin_fast", QUERY)
        self.assertEqual(call.status, "ok")
        self.assertEqual([a["status_code"] for a in call.attempts], [429, 429, 200])
        self.assertEqual(call.hits[0]["url"], VERSIONS_URL)


class Presentation(unittest.TestCase):
    def _search(self, arm: str, payload=MERGED, **kwargs):
        with _env(), patch("company_research.jevlin_vendor._request", return_value=_ok(payload)):
            return search(arm, QUERY, **kwargs)

    def test_merged_arms_show_jevlin_order_in_the_host_hit_shape(self):
        for arm in ("jevlin_fast", "jevlin_balanced", "jevlin_fanout_merged"):
            with self.subTest(arm=arm):
                call = self._search(arm)
                self.assertEqual([h["url"] for h in call.hits], [VERSIONS_URL, SHARED_URL, CDM_URL, THREAD_URL])
                for hit in call.hits:
                    self.assertEqual(set(hit), {"url", "title", "snippet", "metadata"})
                    self.assertEqual(hit["metadata"], {})  # no found_by vote shown to the model
                self.assertTrue(call.hits[0]["snippet"].startswith("The Versions API provides endpoints"))
        self.assertEqual(len(self._search("jevlin_balanced", max_results=2).hits), 2)

    def test_raw_arms_round_robin_ok_candidates_and_dedup_exact_urls(self):
        for arm in ("jevlin_single", "jevlin_fanout_raw"):
            with self.subTest(arm=arm):
                call = self._search(arm)
                # exa#1, brave#1, exa#2, (brave#2 duplicate), exa#3, (brave#3 duplicate)
                self.assertEqual([h["url"] for h in call.hits], [VERSIONS_URL, CDM_URL, SHARED_URL, THREAD_URL])
                # the first provider to show a page supplies its snippet: exa's shorter one, not brave's
                self.assertTrue(call.hits[0]["snippet"].startswith("PUT /api/sn_cdm/versions/publish"))

    def test_snippets_are_cut_at_the_hosts_cap_not_the_coding_cap(self):
        payload = {"merged": [{"url": "https://a.test/x", "excerpts": ["one", "two"]}, {"url": "https://a.test/y", "excerpt": "z" * 5000}],
                   "usage": {"cost_micros": 1}}
        call = self._search("jevlin_fast", payload)
        self.assertEqual(call.hits[0]["snippet"], "one\ntwo")
        self.assertEqual(len(call.hits[1]["snippet"]), 4000)
        self.assertNotEqual(core.DEFAULT_SNIPPET_CHARS, 4000)  # the coding value is 1,200; the parameter is what differs

    def test_raw_ignores_pending_failed_and_skipped_arms(self):
        payload = {"candidates": [
            {"provider": "a", "status": "pending", "citations": [{"url": "https://p.test"}]},
            {"provider": "b", "status": "timeout", "citations": [{"url": "https://t.test"}]},
            {"provider": "c", "status": "ok", "citations": [{"url": "https://ok.test", "title": "T"}]},
        ]}
        self.assertEqual([h["url"] for h in self._search("jevlin_fanout_raw", payload).hits], ["https://ok.test"])

    def test_parse_hits_dispatcher_branch(self):
        self.assertEqual(_parse_hits("jevlin_fanout_raw", MERGED, 10)[1]["url"], CDM_URL)
        self.assertEqual(_parse_hits("jevlin_fanout_merged", MERGED, 10)[1]["url"], SHARED_URL)
        self.assertEqual([h["url"] for h in _parse_hits("jevlin_cli", {"body": CLI_OK}, 10)], [VERSIONS_URL, CDM_URL, THREAD_URL])
        self.assertEqual(len(_parse_hits("jevlin_cli", CLI_OK, 2)), 2)
        self.assertEqual(_parse_hits("jevlin_fixture", MERGED, 1)[0]["url"], VERSIONS_URL)
        self.assertEqual(_parse_hits("jevlin_fast", "not a dict", 10), [])


class CostAndTelemetry(unittest.TestCase):
    def _search(self, arm: str, payload=MERGED):
        with _env(), patch("company_research.jevlin_vendor._request", return_value=_ok(payload)):
            return search(arm, QUERY)

    def test_cost_is_observed_micros_converted_to_dollars(self):
        call = self._search("jevlin_fanout_merged")
        self.assertEqual(call.cost_usd, 0.01)  # usage.cost_micros 10000
        self.assertEqual(jevlin_vendor.cost_usd_from_micros({"cost_micros": 42_000}), 0.042)
        self.assertIsNone(jevlin_vendor.cost_usd_from_micros({}))         # unknown, not free
        self.assertIsNone(jevlin_vendor.cost_usd_from_micros(None))
        self.assertIsNone(jevlin_vendor.cost_usd_from_micros({"cost_micros": "n/a"}))

    def test_telemetry_rides_on_the_call_and_the_body_is_kept_whole(self):
        call = self._search("jevlin_fanout_merged")
        meta = call.metadata
        self.assertEqual(meta["jevlin_request_id"], "req_fixture_0001")
        self.assertEqual(meta["jevlin_tier"], "balanced")
        self.assertEqual(meta["jevlin_providers"], ["exa", "brave", "serpapi", "tinyfish", "you"])
        self.assertTrue(meta["jevlin_requested"])
        self.assertEqual(meta["jevlin_cost_micros"], 10000)
        self.assertFalse(meta["jevlin_cache_hit"])
        self.assertEqual(meta["jevlin_pending"], 1)
        # exa, brave, serpapi (pending), tinyfish (timeout) were called; `you` was skipped
        self.assertEqual(meta["jevlin_upstream_calls"], 4)
        self.assertEqual(meta["jevlin_arms_ok"], 2)
        self.assertEqual(meta["jevlin_merged_total"], 4)
        self.assertEqual(meta["jevlin_hits_shown"], 4)
        self.assertEqual(meta["jevlin_router_version"], "none")
        self.assertEqual([a for a in meta["jevlin_arms"] if a["provider"] == "you"][0]["reason"], "no_credential")
        serialised = asdict(call)
        self.assertEqual(serialised["raw_response"], MERGED)
        self.assertEqual(serialised["metadata"]["jevlin_request_id"], "req_fixture_0001")
        self.assertEqual(serialised["raw_request"]["headers"]["Authorization"], vendors.REDACTED)

    def test_cache_hit_is_recorded_with_its_zero_cost(self):
        payload = {**MERGED, "usage": {"cost_micros": 0, "cache_hit": True, "pending": 0}}
        call = self._search("jevlin_balanced", payload)
        self.assertEqual(call.cost_usd, 0.0)
        self.assertTrue(call.metadata["jevlin_cache_hit"])

    def test_no_usable_result_is_an_empty_ok_call_not_an_error(self):
        payload = {
            "request_id": "req_dead",
            "candidates": [{"provider": "exa", "status": "timeout"}, {"provider": "brave", "status": "error", "error": "upstream 500"}],
            "decision": {"tier": "balanced", "providers": ["exa", "brave"]},
            "usage": {"cost_micros": 7000, "cache_hit": False},
            "no_usable_result": True,
            "merged": [],
        }
        call = self._search("jevlin_balanced", payload)
        self.assertEqual(call.status, "ok")
        self.assertEqual(call.hits, [])
        self.assertEqual(call.cost_usd, 0.007)
        self.assertTrue(call.metadata["jevlin_no_usable_result"])
        self.assertEqual(call.metadata["jevlin_upstream_calls"], 2)

    def test_http_failure_is_an_error_call_with_zero_cost(self):
        failed = VendorCall(status="error", latency_ms=5, raw_request={}, raw_response={"error": "spend_ceiling"},
                            error="HTTP 402: {'error': 'spend_ceiling'}", attempts=[{"attempt": 1, "status_code": 402, "latency_ms": 5}], cost_usd=None)
        with _env(), patch("company_research.jevlin_vendor._request", return_value=failed):
            call = search("jevlin_fast", QUERY)
        self.assertEqual(call.status, "error")
        self.assertEqual(call.hits, [])
        self.assertEqual(call.cost_usd, 0.0)
        self.assertIn("402", call.error)
        self.assertEqual(call.metadata, {})


class CliArm(unittest.TestCase):
    CLI_ENV = {"JEVLIN_BIN": "/opt/jevlin/bin/jevlin", "JEVLIN_CLI_CONFIG": "/bench/jevlin.toml",
               "JEVLIN_CONFIG": "/home/operator/.jevlin/jevlin.toml", "JEVLIN_API_KEY": "test-key"}

    def test_runs_the_released_client_and_shows_its_merge(self):
        with patch.dict(os.environ, self.CLI_ENV), patch("jevlin_core.jevlin.subprocess.run", return_value=_completed(json.dumps(CLI_OK))) as run:
            call = search("jevlin_cli", QUERY, max_results=2)
        self.assertEqual(run.call_count, 1)
        self.assertEqual(run.call_args.args[0], ["/opt/jevlin/bin/jevlin", "search", "-config", "/bench/jevlin.toml", "--stdin", "-no-flush"])
        # The operator's own JEVLIN_CONFIG must not reach the arm.
        self.assertEqual(run.call_args.kwargs["env"]["JEVLIN_CONFIG"], "/bench/jevlin.toml")
        self.assertEqual(json.loads(run.call_args.kwargs["input"]), {"version": 1, "query": QUERY, "max_results": 2})
        self.assertEqual(call.status, "ok")
        self.assertEqual([h["url"] for h in call.hits], [VERSIONS_URL, CDM_URL])
        self.assertEqual(set(call.hits[0]), {"url", "title", "snippet", "metadata"})
        self.assertEqual(call.cost_usd, 0.008)
        self.assertEqual(call.raw_request["method"], "CLI")
        self.assertEqual(call.raw_request["body"], {"version": 1, "query": QUERY, "max_results": 2})
        self.assertEqual(call.raw_response, {"exit_code": 0, "body": CLI_OK, "stderr": ""})
        meta = call.metadata
        self.assertTrue(meta["jevlin_client_merge"])
        self.assertEqual(meta["jevlin_request_id"], "req_cli_0001")
        self.assertEqual(meta["jevlin_cost_micros"], 8000)
        self.assertEqual(meta["jevlin_upstream_calls"], 2)
        self.assertEqual(meta["jevlin_merged_total"], 3)
        self.assertEqual(meta["jevlin_hits_shown"], 2)
        self.assertEqual(meta["jevlin_http_status"], 200)

    def test_ok_false_is_an_error_call_with_the_client_code(self):
        envelope = {"ok": False, "http_status": 429, "retryable": True, "retry_after_ms": 5000,
                    "error": {"code": "rate_limited", "message": "slow down"}}
        with patch("jevlin_core.jevlin.subprocess.run", return_value=_completed(json.dumps(envelope), code=3)):
            call = search("jevlin_cli", QUERY)
        self.assertEqual(call.status, "error")
        self.assertIn("rate_limited", call.error)
        self.assertEqual(call.attempts[0]["status_code"], 429)
        self.assertEqual(call.cost_usd, 0.0)
        self.assertEqual(call.hits, [])

    def test_total_failure_behind_ok_true_is_empty_and_flagged(self):
        envelope = {"ok": True, "http_status": 200, "result": {"request_id": "r", "chosen": -1, "merged": [],
                    "candidates": [{"provider": "exa", "status": "timeout"}, {"provider": "brave", "status": "error"}]}}
        with patch("jevlin_core.jevlin.subprocess.run", return_value=_completed(json.dumps(envelope))):
            call = search("jevlin_cli", QUERY)
        self.assertEqual(call.status, "ok")
        self.assertEqual(call.hits, [])
        self.assertTrue(call.metadata["jevlin_no_usable_result"])
        self.assertIsNone(call.cost_usd)  # the envelope carried no usage: unknown, not free

    def test_missing_binary_timeout_and_garbage_are_errors_not_crashes(self):
        with patch("jevlin_core.jevlin.subprocess.run", side_effect=FileNotFoundError("jevlin")):
            call = search("jevlin_cli", QUERY)
        self.assertEqual(call.status, "error")
        self.assertIn("FileNotFoundError", call.error)
        with patch("jevlin_core.jevlin.subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="jevlin", timeout=45)):
            call = search("jevlin_cli", QUERY)
        self.assertIn("TimeoutExpired", call.error)
        with patch("jevlin_core.jevlin.subprocess.run", return_value=_completed("not json", code=1, stderr="boom")):
            call = search("jevlin_cli", QUERY)
        self.assertEqual(call.status, "error")
        self.assertIn("boom", call.error)


class FixturePath(unittest.TestCase):
    def test_fixture_call_runs_the_merged_presentation_without_a_provider(self):
        call = jevlin_vendor.fixture_call(MERGED, QUERY, max_results=10)
        self.assertEqual(call.status, "ok")
        self.assertEqual(call.raw_request["method"], "FIXTURE")
        self.assertEqual(call.cost_usd, 0.01)
        self.assertEqual([h["url"] for h in call.hits][:2], [VERSIONS_URL, SHARED_URL])
        self.assertEqual(call.metadata["jevlin_request_id"], "req_fixture_0001")
        self.assertEqual(call.raw_response, MERGED)
        self.assertEqual(jevlin_vendor.FIXTURE_SPEC.public_dict()["key"], "jevlin_fixture")


if __name__ == "__main__":
    unittest.main()
