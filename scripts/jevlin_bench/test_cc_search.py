"""The wrapper's two arm rules: a forms arm carries the job's previous query, once, and nothing else does;
the split arm shows the model the router's whole merged list, its compact tail as one line each, and
nothing else does."""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(HERE))

import cc_search  # noqa: E402
import jevlin_core.jevlin as core  # noqa: E402
from company_research.vendors import VendorCall  # noqa: E402


def _call() -> VendorCall:
    return VendorCall(status="ok", latency_ms=1, raw_request={}, raw_response={}, error=None, attempts=[], cost_usd=0.0,
                      hits=[{"url": "https://a.example", "title": "A", "snippet": "s", "metadata": {}}])


class WrapperForms(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.hops = Path(self.tmp.name) / "hops"

    def tearDown(self):
        self.tmp.cleanup()

    def _search(self, arm: str, query: str) -> tuple[dict, list]:
        env = {"CC_ARM": arm, "CC_HOP_DIR": str(self.hops), "CC_MAX_SEARCHES": "14", "CC_MAX_RESULTS": "10"}
        with mock.patch.dict(os.environ, env), mock.patch.object(cc_search, "load_environment"), \
                mock.patch.object(cc_search, "run_search", return_value=_call()) as run, contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(cc_search.main(["web_search", query]), 0)
        hop = json.loads(sorted(self.hops.glob("search-*.json"))[-1].read_text())
        return hop, run.call_args.args[3:]

    def test_first_search_sends_no_forms(self):
        hop, passed = self._search("jevlin_balanced_forms", "first query")
        self.assertEqual(hop["search_queries"], [])
        self.assertEqual(list(passed), [[]])

    def test_second_search_with_a_different_query_carries_the_first_as_its_form(self):
        self._search("jevlin_balanced_forms", "first query")
        hop, passed = self._search("jevlin_balanced_forms", "second query")
        self.assertEqual(hop["search_queries"], ["first query"])
        self.assertEqual(list(passed), [["first query"]])
        self.assertEqual(set(hop["shown"]), {"vendor", "query", "hits", "error"}, "the model sees the same shape as every arm")

    def test_only_the_most_recent_query_is_carried(self):
        for q in ("one", "two"):
            self._search("jevlin_balanced_forms", q)
        hop, _ = self._search("jevlin_balanced_forms", "three")
        self.assertEqual(hop["search_queries"], ["two"])

    def test_a_repeated_query_carries_no_form_even_in_another_case(self):
        self._search("jevlin_balanced_forms", "Same Query")
        hop, _ = self._search("jevlin_balanced_forms", "same query")
        self.assertEqual(hop["search_queries"], [])

    def test_other_arms_never_carry_forms(self):
        for arm in ("jevlin_balanced", "jevlin_fast", "exa_instant"):
            self.hops = Path(self.tmp.name) / arm
            self._search(arm, "first query")
            hop, _ = self._search(arm, "second query")
            self.assertEqual(hop["search_queries"], [], arm)

    def test_an_unreadable_hop_is_skipped_when_finding_the_previous_query(self):
        self._search("jevlin_balanced_forms", "first query")
        (self.hops / "search-02.json").write_text("{not json")
        hop, _ = self._search("jevlin_balanced_forms", "third query")
        self.assertEqual(hop["search_queries"], ["first query"])


def no_network() -> contextlib.ExitStack:
    """Any request fails loudly: a past test reached the router and made a real, billed call."""
    stack = contextlib.ExitStack()
    refusal = AssertionError("the test suite must not reach the network")
    for target in ("company_research.vendors._request", "company_research.jevlin_vendor._request", "requests.sessions.Session.request"):
        stack.enter_context(mock.patch(target, side_effect=refusal))
    return stack


def _payload(long: int = 10, compact: int = 30) -> dict:
    merged = [{"url": f"https://long.test/{i}", "title": f"Long page {i}", "excerpt": f"Long excerpt {i}. " + "x" * 300} for i in range(long)]
    merged += [{"url": f"https://compact.test/{i}", "title": f"Compact page {i}", "excerpt": f"short text {i}", "compact": True} for i in range(compact)]
    return {"request_id": "req_split", "merged": merged, "decision": {"tier": "balanced"}, "usage": {"cost_micros": 1000}}


def _router_answer(payload: dict) -> VendorCall:
    return VendorCall(status="ok", latency_ms=1, raw_request={"method": "POST", "url": "https://jevlin.test/v1/search", "headers": {}, "body": {}, "params": None},
                      raw_response=payload, error=None, attempts=[{"attempt": 1, "status_code": 200, "latency_ms": 1}], cost_usd=None)


class WrapperSplit(unittest.TestCase):
    """The real run_search and adapter, with the router's answer staged: no network is reachable."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.hops = Path(self.tmp.name) / "hops"
        guard = no_network()
        guard.__enter__()
        self.addCleanup(guard.__exit__, None, None, None)

    def tearDown(self):
        self.tmp.cleanup()

    def _search(self, arm: str, payload: dict, *, max_results: int = 10, query: str = "colorado insight partners"):
        env = {"CC_ARM": arm, "CC_HOP_DIR": str(self.hops), "CC_MAX_SEARCHES": "14", "CC_MAX_RESULTS": str(max_results),
               "JEVLIN_API_KEY": "test-key", "JEVLIN_API_URL": "https://jevlin.test"}
        with mock.patch.dict(os.environ, env), mock.patch.object(cc_search, "load_environment"), \
                mock.patch("company_research.jevlin_vendor._request", return_value=_router_answer(payload)) as request, \
                contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(cc_search.main(["web_search", query]), 0)
        hop = json.loads(sorted(self.hops.glob("search-*.json"))[-1].read_text())
        return json.loads(out.getvalue()), hop, request.call_args.kwargs["body"]

    def test_the_model_reads_the_whole_list_long_entries_as_objects_and_the_tail_as_lines(self):
        shown, hop, body = self._search("jevlin_balanced_split", _payload())
        self.assertEqual(body["excerpt_mode"], "split")
        self.assertEqual(set(shown), {"vendor", "query", "hits", "error"}, "the tool output keeps its shape")
        hits = shown["hits"]
        self.assertEqual(len(hits), 40)
        self.assertTrue(all(isinstance(h, dict) and set(h) == {"url", "title", "snippet", "metadata"} for h in hits[:10]))
        self.assertTrue(all(isinstance(h, str) and "\n" not in h for h in hits[10:]))
        self.assertEqual(hits[10], "Compact page 0 | https://compact.test/0 | short text 0")
        self.assertEqual([h["url"] for h in hits[:10]], [f"https://long.test/{i}" for i in range(10)])
        self.assertEqual([h.split(" | ")[1] for h in hits[10:]], [f"https://compact.test/{i}" for i in range(30)])

    def test_the_cut_that_applies_to_every_other_arm_does_not_apply_to_the_split_arm(self):
        shown, _, body = self._search("jevlin_balanced_split", _payload(), max_results=3)
        self.assertEqual((len(shown["hits"]), body["max_results"]), (40, 3))
        self.hops = Path(self.tmp.name) / "balanced"
        shown, _, _ = self._search("jevlin_balanced", _payload(), max_results=3)
        self.assertEqual(len(shown["hits"]), 3)

    def test_the_hop_keeps_what_was_shown_and_the_hits_it_was_made_from(self):
        shown, hop, _ = self._search("jevlin_balanced_split", _payload())
        self.assertEqual(hop["shown"], shown)
        recorded = hop["call"]["hits"]
        self.assertEqual(len(recorded), 40)
        self.assertEqual([h["url"] for h in recorded], [f"https://long.test/{i}" for i in range(10)] + [f"https://compact.test/{i}" for i in range(30)])
        self.assertEqual(sum(1 for h in recorded if h.get("compact")), 30)

    def test_hop_metadata_matches_what_the_model_received(self):
        shown, hop, _ = self._search("jevlin_balanced_split", _payload())
        meta = hop["call"]["metadata"]
        self.assertEqual((meta["jevlin_split_entries_shown"], meta["jevlin_split_compact_shown"]), (40, 30))
        self.assertEqual(meta["jevlin_split_chars_shown"], len(json.dumps(shown["hits"], ensure_ascii=False)))
        self.assertEqual(sum(isinstance(h, str) for h in shown["hits"]), meta["jevlin_split_compact_shown"])

    def test_other_arms_are_shown_exactly_what_they_were_shown_before(self):
        for arm in ("jevlin_balanced", "jevlin_balanced_forms", "jevlin_fast"):
            with self.subTest(arm=arm):
                self.hops = Path(self.tmp.name) / arm
                shown, hop, body = self._search(arm, _payload(), max_results=10)
                self.assertEqual(len(shown["hits"]), 10)
                self.assertTrue(all(isinstance(h, dict) and set(h) == {"url", "title", "snippet", "metadata"} for h in shown["hits"]))
                self.assertNotIn("excerpt_mode", body)
                self.assertEqual(hop["shown"], shown)
                self.assertFalse([key for key in hop["call"]["metadata"] if "split" in key])

    def test_a_failed_split_search_shows_an_error_and_no_hits(self):
        env = {"CC_ARM": "jevlin_balanced_split", "CC_HOP_DIR": str(self.hops), "CC_MAX_SEARCHES": "14", "CC_MAX_RESULTS": "10",
               "JEVLIN_API_KEY": "test-key", "JEVLIN_API_URL": "https://jevlin.test"}
        failed = VendorCall(status="error", latency_ms=1, raw_request={}, raw_response={"error": "x", "code": "malformed_json"},
                            error="HTTP 400: malformed_json", attempts=[{"attempt": 1, "status_code": 400}], cost_usd=None)
        with mock.patch.dict(os.environ, env), mock.patch.object(cc_search, "load_environment"), \
                mock.patch("company_research.jevlin_vendor._request", return_value=failed), contextlib.redirect_stdout(io.StringIO()) as out:
            cc_search.main(["web_search", "q"])
        shown = json.loads(out.getvalue())
        self.assertEqual((shown["hits"], shown["error"]), ([], "HTTP 400: malformed_json"))


if __name__ == "__main__":
    unittest.main()
