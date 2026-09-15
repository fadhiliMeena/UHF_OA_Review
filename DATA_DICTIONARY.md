# Data dictionary and provenance

## `data/frozen/systematic_map_corpus_5898.csv`
One row per retained OpenAlex bibliographic record. Key variables include OpenAlex work ID, DOI, title, year, publication date/type/language, OpenAlex citation count, FWCI where available, source, open-access status, reconstructed abstract, reference count, country count and institution count.

## `data/frozen/screening_decisions.csv`
One row per retrieved record. `screen_include` is the high-specificity financing rule result; `duplicate_removed` marks duplicates among screen-eligible records; `final_include` identifies the final 5,898-record map.

## `data/manual/robust_duplicates_25.csv`
The 25 duplicates identified in the full 14,646-record robust duplicate audit. This file is kept separately because the production screening log records duplicate removal only among records that first passed the financing screen.

## `data/manual/audit_350.csv`
Fixed 350-record manual audit used to characterize screening behavior and calibrate the diagnostic TF-IDF/logistic-regression model.

## `data/frozen/financing_theme_assignments.csv`
Long-form, non-mutually-exclusive work-to-theme assignments for the nine financing themes.

## `data/frozen/core_analysis_tables/`
Archived analytical tables generated from the production OpenAlex metadata snapshot, including authorship, affiliation, country-attention, leadership, theme, citation, source and reference outputs.

## `data/derived/`
Exact matrices/edge lists underlying Figures 7–9.

## `figures/`
Published/submitted raster versions of the nine main figures and four supplementary figures. Scripts regenerate the analytical content; minor rendering differences can occur across matplotlib/font/library versions.
