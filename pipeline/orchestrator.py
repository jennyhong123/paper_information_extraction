#!/usr/bin/env python3
"""파이프라인 지휘자.

collect → extract → score 순서로 단계를 연결한다.
실제 로직은 article_client / extract / score에 두고,
여기서는 CLI·설정 로드·단계 장벽·resume·run manifest만 담당한다.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .article_client import PmcArticleClient, freeze_uid_manifest, write_metadata_csv
from .common import (
    PACKAGE_ROOT,
    PROJECT_ROOT,
    OllamaClient,
    OllamaConfig,
    atomic_write_json,
    env_versions,
    find_xml_only_dirs,
    git_commit,
    load_toml,
    new_run_id,
    source_hashes,
    utc_now,
)
from .extract import configure_retrieval, run_extract_batch
from .score import run_score_batch

logger = logging.getLogger(__name__)


def load_config(path: Path) -> Dict[str, Any]:
    cfg = load_toml(path)
    collect = cfg.setdefault("collect", {})
    if collect.get("ncbi_api_key_env"):
        key = os.environ.get(collect["ncbi_api_key_env"])
        if key:
            collect["api_key"] = key
    return cfg


def _out_dir(cfg: Dict[str, Any]) -> Path:
    p = Path(cfg["collect"].get("out_dir", "pmc_articles"))
    return p if p.is_absolute() else PROJECT_ROOT / p


def _status_log(out_dir: Path, run_id: str) -> Path:
    return out_dir / "manifests" / f"{run_id}.status.jsonl"


def _latest_uid_manifest(out_dir: Path) -> Optional[Path]:
    manifests = sorted((out_dir / "manifests").glob("*.json"), reverse=True)
    for path in manifests:
        name = path.name
        if name.endswith(".run.json") or name.endswith(".dry-run.json"):
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if "pmc_uids" in data and "query" in data:
            return path
    return None


def _ollama_provenance(cfg: Dict[str, Any]) -> Dict[str, Any]:
    models = {
        "extract": (cfg.get("extract") or {}).get("model", "qwen3.5:9b"),
        "score": (cfg.get("score") or {}).get("model", "qwen3.5:9b"),
    }
    client = OllamaClient(OllamaConfig(model=models["extract"]))
    out: Dict[str, Any] = {"available": client.is_available()}
    if not out["available"]:
        return out
    out["ollama_version"] = client.version()
    out["models"] = {k: client.provenance(v) for k, v in models.items()}
    return out


def _write_run_manifest(out_dir: Path, run_id: str, cfg: Dict[str, Any], extra: Dict[str, Any]) -> Path:
    path = out_dir / "manifests" / f"{run_id}.run.json"
    payload = {
        "run_id": run_id,
        "created_at": utc_now(),
        "git_commit": git_commit(),
        "config": cfg,
        "env": env_versions(),
        "ollama": _ollama_provenance(cfg),
        "source_hashes": source_hashes(
            [
                PACKAGE_ROOT / "article_client.py",
                PACKAGE_ROOT / "article_collector.py",
                PACKAGE_ROOT / "orchestrator.py",
                PACKAGE_ROOT / "common.py",
                PACKAGE_ROOT / "extract.py",
                PACKAGE_ROOT / "score.py",
                PACKAGE_ROOT / "configs" / "queries.json",
                PACKAGE_ROOT / "configs" / "rubrics.md",
                PROJECT_ROOT / "pipeline.toml",
            ]
        ),
        **extra,
    }
    atomic_write_json(path, payload)
    return path


def stage_collect(
    cfg: Dict[str, Any],
    run_id: str,
    force: bool = False,
    dry_run: bool = False,
    resume: bool = False,
) -> Dict[str, Any]:
    collect_cfg = cfg["collect"]
    out_dir = _out_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    client = PmcArticleClient(
        email=collect_cfg.get("ncbi_email"),
        tool=collect_cfg.get("ncbi_tool", "pmc_collect_parse_download"),
        api_key=collect_cfg.get("api_key"),
    )

    uid_path = _latest_uid_manifest(out_dir) if resume else None
    if uid_path and resume:
        frozen = json.loads(uid_path.read_text(encoding="utf-8"))
        uids: List[str] = list(frozen["pmc_uids"])
        search_result = {
            "esearch_total": frozen.get("esearch_total"),
            "pmc_uids": uids,
            "returned": len(uids),
        }
        print(f"resume UIDs from {uid_path} ({len(uids)})")
    else:
        search_result = client.search_all_uids(
            collect_cfg["query"],
            year_from=collect_cfg.get("year_from"),
            page_size=collect_cfg.get("page_size", 200),
            max_retries=collect_cfg.get("max_retries", 3),
        )
        uids = search_result["pmc_uids"]
        if dry_run:
            uid_path = out_dir / "manifests" / f"{run_id}.dry-run.json"
            print(f"dry-run would freeze {len(uids)} UIDs (esearch_total={search_result['esearch_total']})")
        else:
            uid_path = freeze_uid_manifest(
                out_dir,
                run_id,
                collect_cfg["query"],
                collect_cfg.get("year_from"),
                search_result,
            )
            print(f"frozen {len(uids)} UIDs -> {uid_path}")

    if dry_run:
        results = client.collect_uids(uids, out_dir, force=force, dry_run=True)
        needs = sum(1 for r in results if r.get("status") == "needs_work")
        xml_only = find_xml_only_dirs(out_dir)
        return {
            "stage": "collect",
            "uid_manifest": str(uid_path),
            "total": len(uids),
            "needs_work": needs,
            "ok": len(uids) - needs,
            "xml_only_repair": xml_only,
        }

    results = client.collect_uids(
        uids,
        out_dir,
        force=force,
        status_log=_status_log(out_dir, run_id),
    )
    for article_dir in out_dir.glob("PMC_UID_*"):
        uid = article_dir.name.replace("PMC_UID_", "", 1)
        if (article_dir / "article.xml").is_file() and not (article_dir / "parsed_sections.json").is_file():
            if client.repair_parsed(uid, out_dir) is not None:
                print(f"repaired parsed_sections.json for {uid}")
    ok = [r for r in results if r.get("status") == "ok"]
    failed = [r for r in results if r.get("status") == "failed"]
    write_metadata_csv(ok, out_dir / "article_metadata.csv")
    summary = {
        "stage": "collect",
        "uid_manifest": str(uid_path),
        "esearch_total": search_result.get("esearch_total"),
        "total": len(uids),
        "ok": len(ok),
        "failed": len(failed),
        "failed_uids": [r["pmc_uid"] for r in failed],
    }
    if collect_cfg.get("strict", True) and failed:
        raise RuntimeError(f"collect strict failure: {len(failed)} UIDs failed")
    return summary


def _configure_from_cfg(cfg: Dict[str, Any]) -> None:
    r = cfg.get("retrieval", {})
    configure_retrieval(
        embedding_model=r.get("embedding_model"),
        chunk_size=r.get("chunk_size"),
        chunk_overlap=r.get("chunk_overlap"),
    )


def stage_extract(
    cfg: Dict[str, Any],
    run_id: str,
    force: bool = False,
    dry_run: bool = False,
) -> Dict[str, Any]:
    out_dir = _out_dir(cfg)
    meta = out_dir / "article_metadata.csv"
    if not dry_run:
        client = PmcArticleClient(
            email=(cfg.get("collect") or {}).get("ncbi_email"),
            tool=(cfg.get("collect") or {}).get("ncbi_tool", "pmc_collect_parse_download"),
            api_key=(cfg.get("collect") or {}).get("api_key"),
        )
        for uid in find_xml_only_dirs(out_dir):
            client.repair_parsed(uid, out_dir)
        if not meta.is_file():
            raise RuntimeError("collect incomplete: missing article_metadata.csv")
        if not list(out_dir.glob("PMC_UID_*/parsed_sections.json")):
            raise RuntimeError("collect incomplete: no parsed_sections.json files")
        leftover = find_xml_only_dirs(out_dir)
        if leftover:
            raise RuntimeError(f"collect incomplete: XML-only articles remain: {leftover}")
    _configure_from_cfg(cfg)
    e = cfg.get("extract", {})
    r = cfg.get("retrieval", {})
    stats = run_extract_batch(
        root=out_dir,
        metadata_csv=meta,
        model=e.get("model", "qwen3.5:9b"),
        seed=e.get("seed", 42),
        temperature=float(e.get("temperature", 0.0)),
        top_k=int(r.get("top_k", 4)),
        force=force,
        dry_run=dry_run,
        max_retries=int(e.get("max_retries", 2)),
        status_log=_status_log(out_dir, run_id),
    )
    if cfg["collect"].get("strict", True) and stats.get("failed"):
        raise RuntimeError(f"extract strict failure: {stats['failed']} articles failed")
    return {"stage": "extract", **stats}


def stage_score(
    cfg: Dict[str, Any],
    run_id: str,
    force: bool = False,
    dry_run: bool = False,
) -> Dict[str, Any]:
    out_dir = _out_dir(cfg)
    ok_extractions = []
    for path in out_dir.glob("PMC_UID_*/extraction.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if data.get("status") == "ok":
            ok_extractions.append(path)
    if not ok_extractions and not dry_run:
        raise RuntimeError("extract incomplete: no successful extraction.json files")
    s = cfg.get("score", {})
    thinking = s.get("thinking", False)
    stats = run_score_batch(
        root=out_dir,
        model=s.get("model", "qwen3.5:9b"),
        seed=s.get("seed", 42),
        temperature=float(s.get("temperature", 0.0)),
        thinking=False if thinking in (False, "false", None) else thinking,
        force=force,
        dry_run=dry_run,
        max_retries=int(s.get("max_retries", 2)),
        status_log=_status_log(out_dir, run_id),
    )
    if cfg["collect"].get("strict", True) and stats.get("failed"):
        raise RuntimeError(f"score strict failure: {stats['failed']} articles failed")
    return {"stage": "score", **stats}


def run_pipeline(
    cfg: Dict[str, Any],
    stages: List[str],
    force: bool = False,
    dry_run: bool = False,
    resume: bool = False,
) -> Dict[str, Any]:
    run_id = new_run_id()
    out_dir = _out_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Any] = {"run_id": run_id, "stages": {}}
    for stage in stages:
        if stage == "collect":
            results["stages"]["collect"] = stage_collect(
                cfg, run_id, force=force, dry_run=dry_run, resume=resume
            )
        elif stage == "extract":
            results["stages"]["extract"] = stage_extract(
                cfg, run_id, force=force, dry_run=dry_run
            )
        elif stage == "score":
            results["stages"]["score"] = stage_score(
                cfg, run_id, force=force, dry_run=dry_run
            )
        else:
            raise ValueError(f"unknown stage: {stage}")
        print(json.dumps(results["stages"][stage], ensure_ascii=False))
    results["finished_at"] = utc_now()
    if not dry_run:
        _write_run_manifest(out_dir, run_id, cfg, {"results": results})
    return results


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="PMC collect → extract → score pipeline")
    parser.add_argument(
        "command",
        choices=["run", "collect", "extract", "score", "resume"],
        help="pipeline command",
    )
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "pipeline.toml")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    cfg = load_config(args.config)

    if args.command == "run":
        stages = ["collect", "extract", "score"]
        resume = False
    elif args.command == "resume":
        stages = ["collect", "extract", "score"]
        resume = True
    else:
        stages = [args.command]
        resume = False

    try:
        run_pipeline(
            cfg,
            stages=stages,
            force=args.force,
            dry_run=args.dry_run,
            resume=resume,
        )
    except Exception as e:
        logger.error("%s", e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
