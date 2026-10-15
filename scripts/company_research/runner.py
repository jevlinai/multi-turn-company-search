"""File-backed orchestration for the multi-turn company-search benchmark."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .agent import AgentConfig, result_manifest, run_agent
from .dataset import load_artifact
from .model_client import OPENAI_REQUIRED_ENV
from .parser import parse_company_set
from .vendors import VENDORS, capability_inventory, required_env, validate_vendor_keys

RESEARCH_MODES = ("search_and_fetch", "search_only")


@dataclass(frozen=True)
class RunnerConfig:
    dataset_path: Path
    output_dir: Path = Path("runs")
    vendor_keys: tuple[str, ...] = tuple(VENDORS)
    trial_count: int = 3
    vendor_concurrency: int = 11
    trial_concurrency: int = 3
    query_offset: int = 0
    query_limit: int | None = None
    research_mode: str = "search_and_fetch"
    retry_errors: bool = False
    agent: AgentConfig = AgentConfig()

    def validate(self) -> None:
        if not self.dataset_path.is_file():
            raise ValueError(f"dataset does not exist: {self.dataset_path}")
        if not self.vendor_keys:
            raise ValueError("at least one vendor is required")
        if len(self.vendor_keys) != len(set(self.vendor_keys)):
            raise ValueError("vendor keys must be unique")
        validate_vendor_keys(self.vendor_keys)
        if self.trial_count <= 0:
            raise ValueError("trial_count must be positive")
        if self.vendor_concurrency <= 0 or self.trial_concurrency <= 0:
            raise ValueError("concurrency values must be positive")
        if self.query_offset < 0:
            raise ValueError("query_offset cannot be negative")
        if self.query_limit is not None and self.query_limit <= 0:
            raise ValueError("query_limit must be positive when supplied")
        if self.research_mode not in RESEARCH_MODES:
            raise ValueError(f"research_mode must be one of {RESEARCH_MODES}")
        if self.research_mode == "search_only" and self.agent.max_fetches != 0:
            raise ValueError("search_only mode requires max_fetches=0")
        if self.research_mode == "search_and_fetch" and self.agent.max_fetches <= 0:
            raise ValueError("search_and_fetch mode requires max_fetches > 0")
        if self.research_mode == "search_and_fetch":
            # A vendor with no fetch path would silently run search-only under the fetch label.
            search_only = [key for key in self.vendor_keys if not (VENDORS[key].native_fetch or VENDORS[key].custom_fetch)]
            if search_only:
                raise ValueError(f"search_and_fetch mode needs a fetch path; search-only vendors: {search_only}")
        self.agent.validate()


def _selected_cases(config: RunnerConfig) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload, _ = load_artifact(config.dataset_path)
    cases = list(payload["cases"])
    if config.query_offset >= len(cases):
        raise ValueError("query_offset must be smaller than the dataset")
    end = None if config.query_limit is None else config.query_offset + config.query_limit
    selected = cases[config.query_offset:end]
    if not selected:
        raise ValueError("query selection is empty")
    return payload, selected


def runner_plan(config: RunnerConfig) -> dict[str, Any]:
    config.validate()
    payload, cases = _selected_cases(config)
    return {
        "mode": "offline-dry-run",
        "dataset_slug": payload["dataset"]["slug"],
        "dataset_sha256": payload["content_sha256"],
        "dataset_path": str(config.dataset_path),
        "output_dir": str(config.output_dir),
        "vendors": list(config.vendor_keys),
        "research_mode": config.research_mode,
        "question_count": len(cases),
        "trial_count": config.trial_count,
        "planned_agent_runs": len(config.vendor_keys) * len(cases) * config.trial_count,
        "agent": asdict(config.agent),
        "concurrency": {
            "vendor_workers": min(config.vendor_concurrency, len(config.vendor_keys)),
            "trial_workers_per_vendor": min(config.trial_concurrency, config.trial_count),
            "questions_per_vendor": "sequential",
        },
        "native_tools": {
            "openai_web_search": False,
            "openai_web_fetch": False,
            "vendor_search": True,
            "fetch_enabled": config.research_mode == "search_and_fetch",
        },
        "capabilities": capability_inventory(),
        "required_env_when_executed": [
            *OPENAI_REQUIRED_ENV,
            *required_env(config.vendor_keys),
        ],
        "writes": "local JSON only; paid calls require --execute --confirm-paid",
    }


def _artifact_path(
    config: RunnerConfig,
    dataset_slug: str,
    vendor_key: str,
    case_key: str,
    trial_index: int,
) -> Path:
    return (
        config.output_dir
        / dataset_slug
        / config.research_mode
        / vendor_key
        / case_key
        / f"trial-{trial_index}.json"
    )


def _existing_is_complete(path: Path, retry_errors: bool) -> bool:
    if not path.is_file():
        return False
    try:
        status = json.loads(path.read_text())["result"]["status"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        return False
    return status == "ok" or (status == "error" and not retry_errors)


def _run_trial(
    *,
    client: Any,
    config: RunnerConfig,
    dataset: dict[str, Any],
    case: dict[str, Any],
    vendor_key: str,
    trial_index: int,
) -> dict[str, Any]:
    path = _artifact_path(
        config,
        dataset["dataset"]["slug"],
        vendor_key,
        case["case_key"],
        trial_index,
    )
    if _existing_is_complete(path, config.retry_errors):
        return {"status": "skipped", "path": str(path)}
    result = run_agent(
        case["question"], vendor_key=vendor_key, client=client, config=config.agent
    )
    parsed: list[dict[str, Any]] = []
    parse_error: str | None = None
    if result.status == "ok":
        try:
            parsed = [asdict(company) for company in parse_company_set(result.final_response)]
        except ValueError as exc:
            parse_error = str(exc)
    artifact = {
        "schema_version": "multi-turn-company-search-run-v1",
        "dataset": {
            "slug": dataset["dataset"]["slug"],
            "content_sha256": dataset["content_sha256"],
        },
        "research_mode": config.research_mode,
        "vendor_key": vendor_key,
        "vendor": VENDORS[vendor_key].public_dict(),
        "case": {
            "case_key": case["case_key"],
            "question": case["question"],
            "gold": case["gold"],
        },
        "trial_index": trial_index,
        "agent_config": asdict(config.agent),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "result": result_manifest(result),
        "parsed_companies": parsed,
        "parse_error": parse_error,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)
    return {"status": result.status, "path": str(path), "parse_error": parse_error}


def execute(config: RunnerConfig, client: Any) -> dict[str, Any]:
    """Execute configured paid calls and persist only local JSON artifacts."""
    config.validate()
    dataset, cases = _selected_cases(config)

    def run_vendor(vendor_key: str) -> dict[str, Any]:
        results: list[dict[str, Any]] = []
        for case in cases:
            with ThreadPoolExecutor(
                max_workers=min(config.trial_concurrency, config.trial_count)
            ) as pool:
                futures = [
                    pool.submit(
                        _run_trial,
                        client=client,
                        config=config,
                        dataset=dataset,
                        case=case,
                        vendor_key=vendor_key,
                        trial_index=trial_index,
                    )
                    for trial_index in range(1, config.trial_count + 1)
                ]
                question_results = [future.result() for future in as_completed(futures)]
            results.extend(question_results)
            if question_results and all(row["status"] == "error" for row in question_results):
                break
        return {"vendor": vendor_key, "results": results}

    with ThreadPoolExecutor(
        max_workers=min(config.vendor_concurrency, len(config.vendor_keys))
    ) as pool:
        futures = [pool.submit(run_vendor, key) for key in config.vendor_keys]
        vendors = [future.result() for future in as_completed(futures)]
    return {
        "dataset_slug": dataset["dataset"]["slug"],
        "research_mode": config.research_mode,
        "vendors": sorted(vendors, key=lambda row: row["vendor"]),
    }
