#!/usr/bin/env python3
"""Build the public OB-Company-Websearch set as a `company-research-dataset-v1` artifact.

The Hugging Face release `openbenchmarks/OB-Company-Websearch` is this harness's own artifact
split in two: `metadata.json` holds the envelope (schema_version, dataset, source, gold_release,
content_sha256) and `data/test.jsonl` holds the ten `cases[]` rows, one per line. Reassembled in
the harness's key order, with every row field carried verbatim (including each gold member's
`label_notes`), it reproduces the published content hash byte-for-byte. This script checks that
and records the outcome.

One field has to change. `dataset.validate_artifact` at abe60c8 admits only
`gold_release.source == "generated"`, and the published release says `"hand_labelled"`; without
the override `load_artifact` refuses the file and nothing can run. Nothing else is derived or
invented. The override, the published values it replaces, and the file provenance are written
into `source.conversion` — inside the artifact, so its own content hash covers them — and into a
sidecar `provenance.json` together with the download time, which must not move the hash. The slug
gains a suffix so the derived artifact cannot be mistaken for the published one.

  build_public_dataset.py --download --out datasets/company-research
  build_public_dataset.py --from <dir holding test.jsonl, metadata.json, README.md> --out datasets/company-research

Then freeze the tree: `experiments/catalog_freeze.py freeze` (and `verify` before every pass).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from company_research import dataset as ds  # noqa: E402

DATASET_ID = "openbenchmarks/OB-Company-Websearch"
# The dataset repository's revision on 2026-10-15 (lastModified 2026-08-25T22:53:03Z).
PINNED_REVISION = "e0c66acdd138f77a733ec53d93ca908e339a8295"
RELEASE_FILES = {"test.jsonl": "data/test.jsonl", "metadata.json": "metadata.json", "README.md": "README.md"}
HARNESS_COMMIT = "abe60c8"
DEFAULT_SLUG_SUFFIX = "-converted"
DEFAULT_OUT = ROOT / "datasets" / "company-research"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(dataset_id: str, revision: str, dest: Path, *, opener=urllib.request.urlopen) -> dict[str, Path]:
    """Fetch the release files at one revision into dest. Returns local paths by short name."""
    dest.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    for short, remote in RELEASE_FILES.items():
        url = f"https://huggingface.co/datasets/{dataset_id}/resolve/{revision}/{remote}"
        with opener(url, timeout=120) as response:
            data = response.read()
        (dest / short).write_bytes(data)
        out[short] = dest / short
    return out


def read_release(directory: Path) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (directory / "test.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    readme = (directory / "README.md").read_text(encoding="utf-8") if (directory / "README.md").is_file() else ""
    return metadata, rows, readme


def assemble(metadata: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The published artifact, reassembled verbatim in the harness's key order."""
    return {
        "schema_version": metadata.get("schema_version"),
        "dataset": copy.deepcopy(metadata.get("dataset")),
        "source": copy.deepcopy(metadata.get("source")),
        "gold_release": copy.deepcopy(metadata.get("gold_release")),
        "cases": copy.deepcopy(rows),
        "content_sha256": metadata.get("content_sha256"),
    }


def convert(
    metadata: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    dataset_id: str = DATASET_ID,
    revision: str = PINNED_REVISION,
    file_sha256: dict[str, str] | None = None,
    slug_suffix: str = DEFAULT_SLUG_SUFFIX,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reassemble, apply the one override the validator forces, re-hash, validate.

    Returns the artifact and a report: whether the published hash reproduced, what changed, and
    the validator's summary.
    """
    verbatim = assemble(metadata, rows)
    published_hash = str(metadata.get("content_sha256") or "")
    reproduced = ds.content_sha256(verbatim) == published_hash
    try:
        ds.validate_artifact(verbatim)
        verbatim_accepted: str | None = None
    except ds.DatasetValidationError as exc:
        verbatim_accepted = str(exc)

    payload = copy.deepcopy(verbatim)
    overrides: list[dict[str, Any]] = []
    published_release = copy.deepcopy(payload.get("gold_release")) or {}
    if published_release.get("source") != "generated":
        payload["gold_release"] = {"version": 1, "source": "generated"}
        overrides.append({
            "path": "gold_release.source",
            "from": published_release.get("source"),
            "to": "generated",
            "why": f"dataset.validate_artifact at {HARNESS_COMMIT} admits only source='generated'; the file is refused otherwise",
        })
    published_slug = str(payload["dataset"].get("slug") or "")
    published_name = str(payload["dataset"].get("name") or "")
    if slug_suffix:
        payload["dataset"]["slug"] = published_slug + slug_suffix
        payload["dataset"]["name"] = f"{published_name} (converted for the {HARNESS_COMMIT} contract)"
        overrides.append({"path": "dataset.slug", "from": published_slug, "to": payload["dataset"]["slug"],
                          "why": "a derived artifact must not carry the published slug"})
        overrides.append({"path": "dataset.name", "from": published_name, "to": payload["dataset"]["name"], "why": "same"})
    payload["source"]["conversion"] = {
        "tool": "scripts/jevlin_bench/build_public_dataset.py (jevlinai/multi-turn-company-search, branch jevlin-backend)",
        "harness_commit": HARNESS_COMMIT,
        "from": {"dataset_id": dataset_id, "revision": revision, "files": dict(file_sha256 or {})},
        "published_content_sha256": published_hash,
        "published_content_sha256_reproduced": reproduced,
        "published_gold_release": published_release,
        "published_slug": published_slug,
        "overrides": overrides,
        "derived_fields": [],  # nothing is synthesised: every row field is carried verbatim
    }
    payload.pop("content_sha256", None)
    payload["content_sha256"] = ds.content_sha256(payload)
    summary = ds.validate_artifact(payload)
    report = {
        "published_content_sha256": published_hash,
        "published_content_sha256_reproduced": reproduced,
        "verbatim_rejected_by_validator": verbatim_accepted,
        "overrides": overrides,
        "derived_fields": [],
        "artifact": {
            "slug": summary.dataset_slug,
            "content_sha256": summary.content_sha256,
            "case_count": summary.case_count,
            "company_count": summary.company_count,
            "gold_membership_count": summary.gold_membership_count,
            "families": sorted({str(case.get("family")) for case in payload["cases"]}),
            "constraint_counts": [int(case.get("constraint_count") or 0) for case in payload["cases"]],
        },
    }
    return payload, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--download", action="store_true", help="fetch the release files from Hugging Face at --revision")
    source.add_argument("--from", dest="from_dir", type=Path, help="directory already holding test.jsonl, metadata.json, README.md")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--revision", default=PINNED_REVISION)
    parser.add_argument("--slug-suffix", default=DEFAULT_SLUG_SUFFIX)
    args = parser.parse_args(argv)

    out = args.out.resolve()
    raw_dir = out / "hf" / args.revision
    if args.download:
        files = download(args.dataset_id, args.revision, raw_dir)
    else:
        raw_dir.mkdir(parents=True, exist_ok=True)
        files = {}
        for short in RELEASE_FILES:
            src = (args.from_dir / short).resolve()
            if not src.is_file():
                parser.error(f"missing {src}")
            if src != (raw_dir / short).resolve():
                shutil.copyfile(src, raw_dir / short)
            files[short] = raw_dir / short
    file_sha256 = {RELEASE_FILES[short]: sha256_file(path) for short, path in files.items()}
    metadata, rows, _ = read_release(raw_dir)
    payload, report = convert(metadata, rows, dataset_id=args.dataset_id, revision=args.revision,
                              file_sha256=file_sha256, slug_suffix=args.slug_suffix)
    artifact_path = out / f"{payload['dataset']['slug']}.json"
    ds.write_artifact(payload, artifact_path)
    provenance = {
        "built_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dataset_id": args.dataset_id,
        "revision": args.revision,
        "source": "huggingface download" if args.download else f"local copy of {args.from_dir}",
        "files": file_sha256,
        "artifact_path": str(artifact_path),
        **report,
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(provenance, indent=2, ensure_ascii=False))
    print(f"wrote {artifact_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
