# Workflow

## Frozen reproduction

1. Install `requirements.txt`.
2. Run `python run_pipeline.py --mode frozen`.
3. Inspect `outputs/frozen/audit_diagnostic.json`.
4. Inspect regenerated Figures 6–9 in `outputs/frozen/figures/`.
5. Confirm the terminal reports `Validation passed: 61 checks.`

## Live update

1. Obtain an OpenAlex API key.
2. Copy `.env.example` to `.env` and paste the key.
3. Run `python run_pipeline.py --mode live`.
4. The OQL query is downloaded with cursor pagination to `work/data/`.
5. Referenced works are batch-fetched.
6. `src/core_analysis_live.py` constructs the map and the core knowledge-development outputs.
7. `src/knowledge_maps.py` produces potential/challenge and co-occurrence analyses.
8. The live validator compares headline results with the frozen submission as a drift check.

Use `--force-download` only when you intentionally want to discard a completed live download and query OpenAlex again.
