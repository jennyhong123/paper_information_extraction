# paper_information_extraction

PMC collect → RAG extract → LLM score pipeline.

## Setup

```bash
pip install -r requirements.txt
ollama serve
ollama pull qwen3.5:9b
```

Optional: `export NCBI_API_KEY=...`

## Run

```bash
python -m pipeline run --config pipeline.toml
python -m pipeline resume --config pipeline.toml
python -m pipeline collect --config pipeline.toml
python -m pipeline extract --config pipeline.toml
python -m pipeline score --config pipeline.toml

# inspect without LLM writes
python -m pipeline extract --dry-run
```

Outputs under `pmc_articles/`:
- `manifests/<run_id>.json` — frozen UID list
- `manifests/<run_id>.run.json` — provenance
- `PMC_UID_*/{article.xml,parsed_sections.json,extraction.json,score.json}`
- `article_metadata.csv`, `article_information.csv`, `article_score.csv`

## Package layout

```text
pipeline/
  orchestrator.py
  article_client.py
  article_collector.py
  common.py
  extract.py
  score.py
  configs/
    queries.json
    rubrics.md
```

## Tests

```bash
pytest -q
```
