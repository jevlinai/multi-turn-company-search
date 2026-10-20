from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import analyze_company as an  # noqa: E402


class Report(unittest.TestCase):
    def test_per_arm_numbers_from_rows_and_search_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            def row(arm, f1, cost, metas):
                path = Path(tmp) / f"{arm}-{f1}.json"
                path.write_text(json.dumps({"result": {"searches": [{"call": {"metadata": m}} for m in metas]}}))
                return {"arm": arm, "f1": f1, "cost_usd": cost, "search_count": len(metas), "artifact": str(path)}
            rows = [
                row("jevlin_balanced", 0.5, 0.08, [{"jevlin_hits_shown": 10}, {"jevlin_hits_shown": 8}]),
                row("jevlin_balanced_forms", 0.75, 0.12, [
                    {"jevlin_hits_shown": 10, "jevlin_forms_sent": []},
                    {"jevlin_hits_shown": 10, "jevlin_forms_sent": ["q"], "jevlin_coverage": {"reformulation_only": 3}},
                ]),
            ]
            out = an.report(rows)
        self.assertEqual(out["jevlin_balanced"]["hits_shown_per_search"], 9.0)
        self.assertIsNone(out["jevlin_balanced"]["mean_reformulation_only"])
        forms = out["jevlin_balanced_forms"]
        self.assertEqual((forms["f1"], forms["searches_per_trial"], forms["cost_usd_per_trial"]), (0.75, 2.0, 0.12))
        self.assertEqual((forms["mean_reformulation_only"], forms["share_searches_with_form"]), (3.0, 0.5))


if __name__ == "__main__":
    unittest.main()
