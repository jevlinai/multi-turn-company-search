"""The wrapper's forms rule: a forms arm carries the job's previous query, once, and nothing else does."""

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


if __name__ == "__main__":
    unittest.main()
