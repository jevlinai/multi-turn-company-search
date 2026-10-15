#!/usr/bin/env python3
"""Freeze or verify the dataset tree.

Vendored from `jevlinai/web-search-for-coding-agents` @ f01b032 (`experiments/catalog_freeze.py`);
only the paths changed. The tree is `datasets/company-research`: the converted
`company-research-dataset-v1` artifact and the raw Hugging Face files it was built from.
`provenance.json` is skipped because it carries the build time (evidence, not an input).

  freeze   hash every file under the tree (sorted relative paths) and write
           experiments/FROZEN.sha256: one tree hash plus the per-file list
  verify   recompute and compare; exit 1 on any difference (cc_smoke.sh / cc_pilot.sh run this first)

The artifact is also content-addressed by the harness itself (`content_sha256`, which every trial
artifact records and the judge refuses to mix); this freeze pins the bytes on disk, so a rebuilt or
edited file is caught before a pass starts rather than at judging time.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "datasets" / "company-research"
FROZEN = ROOT / "experiments" / "FROZEN.sha256"
SKIP = {"provenance.json"}  # build-time evidence, not an input


def file_hashes() -> list[tuple[str, str]]:
    out = []
    for path in sorted(TASKS.rglob("*")):
        if path.is_file() and path.name not in SKIP:
            out.append((str(path.relative_to(TASKS)), hashlib.sha256(path.read_bytes()).hexdigest()))
    return out


def tree_hash(entries: list[tuple[str, str]]) -> str:
    digest = hashlib.sha256()
    for rel, h in entries:
        digest.update(f"{h}  {rel}\n".encode())
    return digest.hexdigest()


def main(argv: list[str]) -> int:
    if not TASKS.is_dir():
        print(f"no dataset tree at {TASKS}; run scripts/jevlin_bench/build_public_dataset.py first", file=sys.stderr)
        return 2
    entries = file_hashes()
    tree = tree_hash(entries)
    cmd = argv[1] if len(argv) > 1 else "verify"
    if cmd == "freeze":
        lines = [f"tree {tree}", f"files {len(entries)}"] + [f"{h}  {rel}" for rel, h in entries]
        FROZEN.write_text("\n".join(lines) + "\n")
        print(f"frozen: tree {tree} files={len(entries)} -> {FROZEN}")
        return 0
    if cmd == "verify":
        if not FROZEN.is_file():
            print("not frozen: experiments/FROZEN.sha256 missing", file=sys.stderr)
            return 1
        expected = FROZEN.read_text().splitlines()[0].split()[1]
        if expected != tree:
            print(f"DATASET MISMATCH: frozen {expected} current {tree}", file=sys.stderr)
            frozen_files = {line.split("  ", 1)[1]: line.split()[0] for line in FROZEN.read_text().splitlines()[2:]}
            current = dict((rel, h) for rel, h in entries)
            for rel in sorted(set(frozen_files) | set(current)):
                if frozen_files.get(rel) != current.get(rel):
                    print(f"  differs: {rel}", file=sys.stderr)
            return 1
        print(f"dataset verified: tree {tree} files={len(entries)}")
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
