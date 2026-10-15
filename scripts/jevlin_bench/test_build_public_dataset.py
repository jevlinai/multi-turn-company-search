"""The public-dataset converter and the dataset freeze, offline.

A synthetic two-row release stands in for the Hugging Face files; when the real files are present
under datasets/company-research/hf/<revision>/ (after build_public_dataset.py ran), one test also
checks that they reproduce the published content hash. No network: `download` is exercised with a
fake opener.
"""

from __future__ import annotations

import copy
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "experiments"))

import build_public_dataset as bpd  # noqa: E402
import catalog_freeze  # noqa: E402
from company_research import dataset as ds  # noqa: E402

PUBLISHED_HASH = "sha256:b987888daa5e0516ac7cf18deaba3830496929fa653d2cb68b494bd2945664ab"


def _row(key: str, order: int, names: list[str]) -> dict:
    return {
        "case_key": key,
        "question": f"Which companies count Insight Partners among their investors and are headquartered in {key}?",
        "family": "investor",
        "constraint_count": 2,
        "constraints": ["count Insight Partners among their investors", f"are headquartered in {key}"],
        "predicates": {"axes": ["investor_backing", "geo_region"], "labels": {"investor_backing": "inv_insight_partners", "geo_region": key}},
        "census_metadata": {"answer_count": len(names), "ok": True},
        "source_metadata": {"combination": "geo_region+investor_backing", "coverage": {"known_product": 0.9}},
        "sort_order": order,
        "gold": [{"entity_key": ds.entity_key(n), "name": n, "domain": f"{n.lower().replace(' ', '')}.example", "aliases": [f"{n} Inc."], "label_notes": None} for n in names],
    }


def _release() -> tuple[dict, list[dict]]:
    rows = [_row("Boulder", 1, ["Automox", "Valimail"]), _row("Denver", 2, ["Quantum Metric"])]
    metadata = {
        "schema_version": "company-research-dataset-v1",
        "dataset": {"slug": "ob-test-v1", "name": "OB test", "region": "Global", "notes": "n"},
        "source": {"type": "clickhouse_fiber_companies", "snapshot": "s", "gold_status": "published_hand_labelled"},
        "gold_release": {"version": 1, "source": "hand_labelled"},
    }
    # the published hash is over the verbatim assembly, as on Hugging Face
    metadata["content_sha256"] = ds.content_sha256(bpd.assemble({**metadata, "content_sha256": ""}, rows))
    return metadata, rows


class Converter(unittest.TestCase):
    def test_reassembly_reproduces_the_published_hash_and_only_the_release_source_moves(self):
        metadata, rows = _release()
        payload, report = bpd.convert(metadata, rows, revision="abc123", file_sha256={"data/test.jsonl": "f" * 64})
        self.assertTrue(report["published_content_sha256_reproduced"])
        self.assertIn("source='generated'", report["verbatim_rejected_by_validator"])
        self.assertEqual([o["path"] for o in report["overrides"]], ["gold_release.source", "dataset.slug", "dataset.name"])
        self.assertEqual(report["derived_fields"], [])
        self.assertEqual(payload["gold_release"], {"version": 1, "source": "generated"})
        self.assertEqual(payload["dataset"]["slug"], "ob-test-v1-converted")
        self.assertEqual(payload["source"]["gold_status"], "published_hand_labelled")  # the published source block is kept
        conversion = payload["source"]["conversion"]
        self.assertEqual(conversion["published_gold_release"], {"version": 1, "source": "hand_labelled"})
        self.assertEqual(conversion["published_content_sha256"], metadata["content_sha256"])
        self.assertEqual(conversion["from"]["revision"], "abc123")
        self.assertEqual(conversion["from"]["files"]["data/test.jsonl"], "f" * 64)
        # rows are carried verbatim, label_notes included, and the artifact validates
        self.assertEqual(payload["cases"], rows)
        self.assertIn("label_notes", payload["cases"][0]["gold"][0])
        summary = ds.validate_artifact(payload)
        self.assertEqual((summary.case_count, summary.company_count, summary.gold_membership_count), (2, 3, 3))
        self.assertEqual(summary.content_sha256, payload["content_sha256"])
        self.assertNotEqual(payload["content_sha256"], metadata["content_sha256"])
        # conversion is deterministic: the artifact hash does not depend on when it was built
        again, _ = bpd.convert(metadata, rows, revision="abc123", file_sha256={"data/test.jsonl": "f" * 64})
        self.assertEqual(again["content_sha256"], payload["content_sha256"])

    def test_a_release_that_already_says_generated_gets_no_gold_override(self):
        metadata, rows = _release()
        metadata["gold_release"] = {"version": 1, "source": "generated"}
        metadata["content_sha256"] = ds.content_sha256(bpd.assemble({**metadata, "content_sha256": ""}, rows))
        payload, report = bpd.convert(metadata, rows)
        self.assertIsNone(report["verbatim_rejected_by_validator"])
        self.assertEqual([o["path"] for o in report["overrides"]], ["dataset.slug", "dataset.name"])
        ds.validate_artifact(payload)

    def test_a_tampered_row_is_reported_as_a_hash_mismatch_not_hidden(self):
        metadata, rows = _release()
        rows = copy.deepcopy(rows)
        rows[0]["question"] = rows[0]["question"].replace("Boulder", "Boulder, Colorado")
        rows[0]["constraints"][1] = "are headquartered in Boulder, Colorado"
        payload, report = bpd.convert(metadata, rows)
        self.assertFalse(report["published_content_sha256_reproduced"])
        self.assertFalse(payload["source"]["conversion"]["published_content_sha256_reproduced"])

    def test_without_a_slug_suffix_the_slug_is_kept(self):
        metadata, rows = _release()
        payload, report = bpd.convert(metadata, rows, slug_suffix="")
        self.assertEqual(payload["dataset"]["slug"], "ob-test-v1")
        self.assertEqual([o["path"] for o in report["overrides"]], ["gold_release.source"])

    def test_download_writes_the_three_release_files_through_the_opener(self):
        served = {"data/test.jsonl": b'{"a":1}\n', "metadata.json": b"{}", "README.md": b"# r"}

        class Resp(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        def opener(url, timeout):
            self.assertTrue(url.startswith("https://huggingface.co/datasets/openbenchmarks/OB-Company-Websearch/resolve/rev1/"))
            return Resp(served[url.rsplit("/resolve/rev1/", 1)[1]])

        with tempfile.TemporaryDirectory() as tmp:
            files = bpd.download("openbenchmarks/OB-Company-Websearch", "rev1", Path(tmp), opener=opener)
            self.assertEqual(set(files), {"test.jsonl", "metadata.json", "README.md"})
            self.assertEqual(files["test.jsonl"].read_bytes(), b'{"a":1}\n')

    def test_main_from_a_local_release_writes_artifact_and_provenance(self):
        metadata, rows = _release()
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "src"
            src.mkdir()
            (src / "metadata.json").write_text(json.dumps(metadata))
            (src / "test.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            (src / "README.md").write_text("# card\n")
            out = Path(tmp) / "out"
            with patch("sys.stdout", new=io.StringIO()):
                code = bpd.main(["--from", str(src), "--out", str(out), "--revision", "rev9"])
            self.assertEqual(code, 0)
            artifact = out / "ob-test-v1-converted.json"
            payload, summary = ds.load_artifact(artifact)
            self.assertEqual(summary.case_count, 2)
            provenance = json.loads((out / "provenance.json").read_text())
            self.assertTrue(provenance["published_content_sha256_reproduced"])
            self.assertEqual(provenance["revision"], "rev9")
            self.assertEqual(set(provenance["files"]), {"data/test.jsonl", "metadata.json", "README.md"})
            self.assertIn("built_at", provenance)
            self.assertTrue((out / "hf" / "rev9" / "test.jsonl").is_file())
            self.assertEqual(payload["source"]["conversion"]["from"]["files"], provenance["files"])

    def test_the_real_release_reproduces_the_published_hash(self):
        raw = ROOT / "datasets" / "company-research" / "hf" / bpd.PINNED_REVISION
        if not (raw / "test.jsonl").is_file():
            self.skipTest("public release not downloaded; run build_public_dataset.py --download")
        metadata, rows, readme = bpd.read_release(raw)
        self.assertEqual(metadata["content_sha256"], PUBLISHED_HASH)
        file_sha256 = {remote: bpd.sha256_file(raw / short) for short, remote in bpd.RELEASE_FILES.items()}
        payload, report = bpd.convert(metadata, rows, file_sha256=file_sha256)
        self.assertTrue(report["published_content_sha256_reproduced"])
        self.assertEqual(report["artifact"]["case_count"], 10)
        self.assertEqual(report["artifact"]["company_count"], 100)
        self.assertEqual(report["artifact"]["gold_membership_count"], 103)
        self.assertEqual(report["artifact"]["families"], ["accelerator", "funding", "investor"])
        self.assertEqual(sorted(report["artifact"]["constraint_counts"]), [3, 3, 3, 4, 4, 4, 4, 4, 4, 4])
        self.assertIn(PUBLISHED_HASH, readme)
        built = ROOT / "datasets" / "company-research" / "ob-company-websearch-v1-converted.json"
        if built.is_file():
            self.assertEqual(json.loads(built.read_text())["content_sha256"], payload["content_sha256"])


class Freeze(unittest.TestCase):
    def test_freeze_then_verify_then_catch_a_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            tree = Path(tmp) / "tree"
            (tree / "hf" / "rev").mkdir(parents=True)
            (tree / "set.json").write_text("{}")
            (tree / "hf" / "rev" / "test.jsonl").write_text("{}\n")
            (tree / "provenance.json").write_text('{"built_at": "now"}')
            frozen = Path(tmp) / "FROZEN.sha256"
            with patch.object(catalog_freeze, "TASKS", tree), patch.object(catalog_freeze, "FROZEN", frozen), \
                 patch("sys.stdout", new=io.StringIO()), patch("sys.stderr", new=io.StringIO()):
                self.assertEqual(catalog_freeze.main(["x", "verify"]), 1)  # not frozen yet
                self.assertEqual(catalog_freeze.main(["x", "freeze"]), 0)
                self.assertEqual(frozen.read_text().splitlines()[1], "files 2")  # provenance.json is skipped
                self.assertEqual(catalog_freeze.main(["x", "verify"]), 0)
                (tree / "provenance.json").write_text('{"built_at": "later"}')
                self.assertEqual(catalog_freeze.main(["x", "verify"]), 0)  # evidence, not an input
                (tree / "set.json").write_text('{"changed": true}')
                self.assertEqual(catalog_freeze.main(["x", "verify"]), 1)


if __name__ == "__main__":
    unittest.main()
