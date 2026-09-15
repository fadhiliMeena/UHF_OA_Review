# Health Financing for UHC in Low- and Lower-Middle-Income Countries

## Reproducibility repository

This repository reproduces the systematic mapping review:

**Health Financing for Universal Health Coverage in Low- and Lower-Middle-Income Countries: Potential and Challenges**

Author: **Fadhili Meena**  
Department of Mathematics and Statistics, University of Dodoma, Tanzania  
ORCID: 0009-0001-5483-9499

The primary design is a **systematic mapping review**. Bibliometric and science-mapping methods are supporting analyses. The map describes the development, geography, leadership, thematic structure, and title/abstract potential/challenge signals of the indexed literature. It does **not** estimate intervention effects or provide study-level certainty-of-effect ratings.

---

## Two reproducibility modes

### 1. Frozen mode — exact submitted analysis

Use this mode to reproduce and validate the archived September 2026 analysis without contacting OpenAlex.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_pipeline.py --mode frozen
```

Frozen mode:

- reruns the TF-IDF/logistic-regression screening diagnostic on the archived 350-record audit;
- regenerates Figures 6–9 and their underlying matrices from the archived 5,898-record corpus;
- validates 61 manuscript-critical quantities against `config/expected_results.json`.

The validation includes 14,646 records retrieved, 25 duplicates in the full robust duplicate audit, 14,621 unique records, 5,898 records in the final map, leadership estimates, geographic concentration, all nine financing-theme counts, and all potential/challenge signal percentages.

### 2. Live mode — start from an OpenAlex API key

OpenAlex is a living index. Live mode reruns the frozen OQL query against the current index and therefore may produce counts different from the September 2026 snapshot.

Create a free OpenAlex API key, then:

```bash
cp .env.example .env
```

Edit `.env`:

```text
OPENALEX_API_KEY=your_key_here
```

Run:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_pipeline.py --mode live
```

The live workflow is:

```text
API key
  ↓
frozen OQL query
  ↓
OpenAlex POST + cursor pagination
  ↓
raw work metadata
  ↓
referenced-work metadata
  ↓
high-specificity financing screen
  ↓
deduplication
  ↓
systematic map
  ↓
knowledge-development analyses
  ↓
authorship / geography / themes / citations
  ↓
potential–challenge text mapping
  ↓
co-occurrence and network analyses
  ↓
figures, matrices, tables and validation
```

The API key is read from `.env`, sent in the HTTP Authorization header, and is never written to output files. `.env` is excluded by `.gitignore`.

---

## Repository structure

```text
.
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── .env.example
├── .gitignore
├── run_pipeline.py
│
├── config/
│   ├── final_openalex_search.oql
│   ├── country_classification_FY2027.csv
│   └── expected_results.json
│
├── protocol/
│   ├── LOCKED_SYSTEMATIC_REVIEW_PROTOCOL_v1.txt
│   ├── PROTOCOL_AMENDMENT_2026-09-14.txt
│   └── SCREENING_AUDIT_CALIBRATION_ARCHIVE_v2.txt
│
├── data/
│   ├── frozen/
│   │   ├── systematic_map_corpus_5898.csv
│   │   ├── screening_decisions.csv
│   │   ├── financing_theme_assignments.csv
│   │   ├── abstract_potential_challenge_signals.csv
│   │   ├── analysis_summary.json
│   │   ├── core_analysis_tables/
│   │   └── core_analysis_networks/
│   ├── manual/
│   │   ├── audit_350.csv
│   │   └── robust_duplicates_25.csv
│   └── derived/
│       ├── Figure_7_Theme_Outcome_Signal_Triples.csv
│       ├── Figure_8_Term_Frequencies.csv
│       ├── Figure_8_Term_Cooccurrence_Edges.csv
│       ├── Figure_9_Theme_Outcome_Cooccurrence.csv
│       └── Figure_9_Chord_Matrix.csv
│
├── src/
│   ├── openalex_download.py
│   ├── core_analysis_live.py
│   ├── knowledge_maps.py
│   ├── audit_diagnostic.py
│   ├── validate_results.py
│   └── *_original.py
│
├── figures/
│   ├── main/
│   └── supplementary/
│
└── outputs/
```

The files ending in `_original.py` preserve the production scripts archived with the journal submission. Portable wrappers/versions are provided separately so the repository can run outside the original build environment.

---

## Search and country frame

`config/final_openalex_search.oql` is the exact executable OQL search used for the submitted analysis. The search covers 1990–2026, restricts retrieval to non-retracted articles and reviews, and combines health-financing concepts with names/aliases for the target countries and generic low-/lower-middle-income descriptors.

`config/country_classification_FY2027.csv` fixes the analytical cohort to **72 World Bank FY2027 economies: 25 low-income and 47 lower-middle-income economies**. The fixed contemporary classification provides a common historical frame and does not imply that each country had the same income classification in every year from 1990 to 2026.

---

## Screening and the 25-versus-9 duplicate counts

Two duplicate counts appear in the reproducibility trail and refer to different stages:

- `data/manual/robust_duplicates_25.csv` contains the **25 duplicate records identified across the full 14,646-record retrieval**, leaving 14,621 unique records.
- `data/frozen/screening_decisions.csv` marks **9 duplicates among the 5,907 records that passed the high-specificity financing screen**. Removing those 9 gives the final 5,898-record map.

The remaining 16 duplicate records occurred among records that did not pass the high-specificity financing screen. Thus the two counts are compatible rather than competing estimates.

The final map is deliberately interpreted as **conservative rather than exhaustive** because the manual audit showed that the high-specificity screen prioritized precision over sensitivity.

---

## Screening diagnostic

The archived 350-record manual audit contains:

- Include: 100
- Exclude: 159
- Map only: 59
- Unclear: 32

`src/audit_diagnostic.py` reconstructs the documented diagnostic text model. Titles and abstracts are represented with TF-IDF features and entered into logistic regression. For the diagnostic retention task, `Exclude` is the negative class and the other three audit categories are retained for further consideration. Five-fold stratified cross-validation gives a mean ROC AUC of **0.902** after rounding.

This model is **diagnostic only**. It does not determine final membership in the 5,898-record systematic map.

---

## Main frozen benchmarks

The archived submission analysis contains:

- 14,646 OpenAlex records retrieved
- 25 full-retrieval duplicate records flagged
- 14,621 unique records after the robust duplicate audit
- 5,898 records in the final systematic map
- 14,047 unique authors
- 3,153 institutions
- 93,092 OpenAlex citations
- 5,682 publications with a target study country identified
- country-attention Gini = 0.756
- top-ten economies = 72.3% of target-country mentions
- target-country first authorship: 51.4% in 1990–2004 → 78.2% in 2021–2026
- odds ratio for target-country first authorship = 1.88 per decade

Financing-theme assignments are non-mutually-exclusive. The main counts are:

| Theme | Records | Share |
|---|---:|---:|
| Insurance/prepayment | 3,983 | 67.5% |
| Financial protection/OOP | 1,761 | 29.9% |
| User fees/cost sharing | 491 | 8.3% |
| Revenue/public financing | 435 | 7.4% |
| Performance-based financing | 342 | 5.8% |
| Willingness to pay | 205 | 3.5% |
| Purchasing/provider payment | 155 | 2.6% |
| Pooling | 78 | 1.3% |
| External financing | 61 | 1.0% |

All exact benchmarks used by automated validation are stored in `config/expected_results.json`.

---

## Potential and challenge mapping

Potential and challenge measures are predefined title/abstract text signals. They are not causal effects, quality assessments, or certainty ratings. Exact theme-level denominators come from the frozen financing-theme assignments.

Examples of the archived challenge-signal prevalence are:

- user fees/cost sharing: 82.1%
- pooling: 62.8%
- financial protection/OOP: 58.0%
- purchasing/provider payment: 52.3%

The underlying signal table and category definitions are in `data/frozen/`.

---

## OpenAlex live-download implementation

`src/openalex_download.py` follows the current OpenAlex HTTP interface:

- long OQL is executed by `POST https://api.openalex.org/`;
- deep result sets use cursor pagination with 100 works per page;
- cited works are batch-fetched by OpenAlex ID in groups of up to 100;
- 429/5xx responses use exponential backoff;
- checkpoints allow interrupted downloads to resume;
- download manifests record counts and API cost, but never the API key.

The downloaded raw metadata are placed under `work/`, which is intentionally ignored by Git because the live index can be regenerated from the query.

---

## Reproducibility boundary

The frozen data reproduce the submitted September 2026 analysis. A future live API run may differ because OpenAlex updates records, citations, affiliations, abstracts, indexing coverage and work identities over time. A difference in live counts is therefore not automatically a computational failure. Use **frozen mode** when checking the article's reported results and **live mode** when updating the evidence map.

---

## Data and licensing

Repository code is released under the MIT License. OpenAlex metadata are CC0. The repository contains bibliographic metadata and derived analytical outputs; it does not redistribute full-text journal articles.
