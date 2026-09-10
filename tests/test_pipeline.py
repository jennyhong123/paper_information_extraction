"""Unit/integration tests for the paper pipeline (no network / no Ollama)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pipeline.article_client import PmcArticleClient, freeze_uid_manifest, write_metadata_csv
from pipeline.common import (
    ExternalArticle,
    OllamaClient,
    OllamaConfig,
    assign_paper_ids,
    atomic_write_json,
    find_xml_only_dirs,
    normalize_pmcid,
    sha256_text,
    write_csv_atomic,
)
from pipeline.extract import aggregate_information_csv, extract_fingerprint, run_extract_batch
from pipeline.orchestrator import stage_extract, stage_score
from pipeline.score import SCORE_COLUMNS, _parse_scores, aggregate_score_csv, run_score_batch, score_fingerprint


def test_normalize_and_stable_paper_ids():
    assert normalize_pmcid("123/PMC99") == "PMC99"
    mapping = assign_paper_ids(["PMC2", "PMC1"], {"PMC1": "P010"})
    assert mapping["PMC1"] == "P010"
    assert mapping["PMC2"] == "P001"


def test_atomic_json_and_csv(tmp_path: Path):
    p = tmp_path / "a.json"
    atomic_write_json(p, {"ok": True})
    assert json.loads(p.read_text())["ok"] is True
    csv_path = tmp_path / "a.csv"
    write_csv_atomic(csv_path, ["A", "B"], [{"A": 1, "B": 2}])
    assert "A,B" in csv_path.read_text()


def test_search_all_uids_paginates():
    client = PmcArticleClient(email="test@example.com")
    pages = [
        {"esearchresult": {"count": "3", "idlist": ["2", "1"]}},
        {"esearchresult": {"count": "3", "idlist": ["3", "1"]}},
    ]

    class Resp:
        def __init__(self, data):
            self._data = data

        def json(self):
            return self._data

        def raise_for_status(self):
            return None

    with patch.object(client, "_ncbi_get", side_effect=[Resp(pages[0]), Resp(pages[1])]):
        result = client.search_all_uids("q", page_size=2)
    assert result["esearch_total"] == 3
    assert result["pmc_uids"] == ["1", "2", "3"]


def test_collect_dry_run_and_repair(tmp_path: Path):
    client = PmcArticleClient(email="test@example.com")
    uid = "100"
    article_dir = tmp_path / f"PMC_UID_{uid}"
    article_dir.mkdir()
    (article_dir / "article.xml").write_text(
        "<article><article-meta><article-title>T</article-title>"
        '<article-id pub-id-type="pmcid">PMC100</article-id></article-meta>'
        "<body><sec><title>Methods</title><p>plasma proteomics</p></sec></body></article>",
        encoding="utf-8",
    )
    assert find_xml_only_dirs(tmp_path) == [uid]
    assert client.repair_parsed(uid, tmp_path) is not None
    assert find_xml_only_dirs(tmp_path) == []
    dry = client.collect_uids([uid], tmp_path, dry_run=True)
    assert dry[0]["status"] == "ok"


def test_freeze_uid_manifest(tmp_path: Path):
    path = freeze_uid_manifest(
        tmp_path,
        "run1",
        query="q",
        year_from=2020,
        search_result={"esearch_total": 2, "pmc_uids": ["1", "2"], "returned": 2},
    )
    data = json.loads(path.read_text())
    assert data["pmc_uids"] == ["1", "2"]
    assert data["query"] == "q"


def test_write_metadata_preserves_paper_id(tmp_path: Path):
    csv_path = tmp_path / "article_metadata.csv"
    write_csv_atomic(
        csv_path,
        ["Paper_ID", "DB", "year", "DOI", "PMID/PMCID", "title", "journal", "author", "source_url"],
        [{"Paper_ID": "P007", "DB": "pmc", "year": "2020", "DOI": "", "PMID/PMCID": "1/PMC9",
          "title": "old", "journal": "", "author": "", "source_url": ""}],
    )
    article = ExternalArticle(
        source_id="1",
        source="pmc",
        title="new",
        abstract="",
        authors=["A"],
        year="2021",
        link="",
        journal="J",
        pmcid="PMC9",
    )
    write_metadata_csv(
        [{"status": "ok", "pmcid": "PMC9", "article": article}],
        csv_path,
    )
    text = csv_path.read_text()
    assert "P007" in text
    assert "PMC9" in text


def test_parse_scores_strict():
    scores = _parse_scores(
        '{"MCI_score":1,"AD_score":2,"Blood_score":3,"Protein_score":4,"Quality_score":5,"Valid_score":0}'
    )
    assert scores["MCI_score"] == 1
    with pytest.raises(ValueError):
        _parse_scores(
            '{"MCI_score":9,"AD_score":0,"Blood_score":0,"Protein_score":0,"Quality_score":0,"Valid_score":0}'
        )
    with pytest.raises(ValueError):
        _parse_scores('{"MCI_score":1}')


def test_fingerprint_changes_with_input(tmp_path: Path):
    p = tmp_path / "parsed_sections.json"
    p.write_text("{}", encoding="utf-8")
    a = extract_fingerprint(p, "m", 42, 0.0, 4)
    p.write_text('{"x":1}', encoding="utf-8")
    b = extract_fingerprint(p, "m", 42, 0.0, 4)
    assert a != b
    assert len(sha256_text("x")) == 64


def test_extract_fingerprint_skip(tmp_path: Path):
    article = tmp_path / "PMC_UID_1"
    article.mkdir()
    parsed = article / "parsed_sections.json"
    parsed.write_text(
        json.dumps(
            {
                "pmc_uid": "1",
                "metadata": {"pmcid": "PMC1"},
                "abstract": "x",
                "introduction": [],
                "discussion": [],
                "all_sections": [],
                "tables": [],
            }
        ),
        encoding="utf-8",
    )
    meta = tmp_path / "article_metadata.csv"
    write_csv_atomic(
        meta,
        ["Paper_ID", "DB", "year", "DOI", "PMID/PMCID", "title", "journal", "author", "source_url"],
        [{"Paper_ID": "P001", "DB": "pmc", "year": "", "DOI": "", "PMID/PMCID": "PMC1",
          "title": "", "journal": "", "author": "", "source_url": ""}],
    )
    fp = extract_fingerprint(parsed, "m", 42, 0.0, 4)
    atomic_write_json(
        article / "extraction.json",
        {"status": "ok", "fingerprint": fp, "answers": {"Paper_ID": "P001", "PMCID": "PMC1"}},
    )
    stats = run_extract_batch(
        root=tmp_path,
        metadata_csv=meta,
        model="m",
        seed=42,
        temperature=0.0,
        top_k=4,
        dry_run=True,
    )
    assert stats["skipped"] == 1
    assert stats["needs_work"] == 0


def test_score_fingerprint_skip(tmp_path: Path):
    article = tmp_path / "PMC_UID_1"
    article.mkdir()
    ext = article / "extraction.json"
    atomic_write_json(
        ext,
        {
            "status": "ok",
            "answers": {
                "Paper_ID": "P001",
                "PMCID": "PMC1",
                "cohort": "c",
                "sample_size": "1",
                "comparison_group": "g",
                "sample_type": "plasma",
                "experimental_technique": "ms",
                "biomarker": "b",
                "key_result": "r",
            },
        },
    )
    fp = score_fingerprint(ext, "m", 42, 0.0)
    atomic_write_json(
        article / "score.json",
        {"status": "ok", "fingerprint": fp, "scores": {k: 1 for k in SCORE_COLUMNS}},
    )
    stats = run_score_batch(tmp_path, model="m", seed=42, temperature=0.0, dry_run=True)
    assert stats["skipped"] == 1


def test_stage_barriers(tmp_path: Path):
    cfg = {
        "collect": {"out_dir": str(tmp_path), "strict": True},
        "extract": {"model": "m", "seed": 42, "temperature": 0.0},
        "score": {"model": "m", "seed": 42, "temperature": 0.0},
        "retrieval": {"top_k": 4},
    }
    with pytest.raises(RuntimeError, match="collect incomplete"):
        stage_extract(cfg, "run", dry_run=False)
    with pytest.raises(RuntimeError, match="extract incomplete"):
        stage_score(cfg, "run", dry_run=False)


def test_aggregate_csvs(tmp_path: Path):
    d = tmp_path / "PMC_UID_1"
    d.mkdir()
    atomic_write_json(
        d / "extraction.json",
        {
            "status": "ok",
            "answers": {
                "Paper_ID": "P001",
                "PMCID": "PMC1",
                "cohort": "c",
                "sample_size": "1",
                "comparison_group": "g",
                "sample_type": "plasma",
                "experimental_technique": "ms",
                "biomarker": "b",
                "key_result": "r",
            },
        },
    )
    atomic_write_json(
        d / "score.json",
        {
            "status": "ok",
            "paper_id": "P001",
            "pmcid": "PMC1",
            "scores": {k: 1 for k in SCORE_COLUMNS},
        },
    )
    assert aggregate_information_csv(tmp_path, tmp_path / "info.csv") == 1
    assert aggregate_score_csv(tmp_path, tmp_path / "score.csv") == 1
    assert (tmp_path / "info.csv").read_text().count("PMC1") == 1


def test_ollama_payload_seed():
    client = OllamaClient(OllamaConfig(model="m", temperature=0.0, options={"seed": 42}))
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {"response": "ok"}
    with patch.object(client.session, "post", return_value=mock_resp) as post:
        client.generate("hi")
    payload = post.call_args.kwargs["json"]
    assert payload["options"]["seed"] == 42
    assert payload["options"]["temperature"] == 0.0


def test_ollama_provenance_methods():
    client = OllamaClient(OllamaConfig(model="m"))
    ver = MagicMock()
    ver.status_code = 200
    ver.json.return_value = {"version": "0.9.0"}
    show = MagicMock()
    show.status_code = 200
    show.json.return_value = {"digest": "sha256:abc"}
    with patch.object(client.session, "get", return_value=ver), patch.object(
        client.session, "post", return_value=show
    ):
        prov = client.provenance("m")
    assert prov["ollama_version"] == "0.9.0"
    assert prov["model_digest"] == "sha256:abc"
