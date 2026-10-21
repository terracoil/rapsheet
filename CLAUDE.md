# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A reference implementation of a methodology (calibrated, abstaining classification of charge text from the Cook County SAO dispositions table, Socrata `apwk-dzx8`) — explicitly **not a screening tool**. It is driven from `notebooks/reading_the_rap_sheet.ipynb`, which is the article's narrative; the package holds the logic the notebook calls. See `README.md` for the data/ethics framing.

## Commands

```bash
poetry install --with dev                      # train + test + jupyter groups (Python >=3.13.7,<3.14 per pyproject)
poetry run pytest                              # all tests (cache goes to tmp/.pytest_cache)
poetry run pytest tests/test_dataset.py -k <name>   # single test / subset
poetry run ruff check rapsheet tests && poetry run ruff format rapsheet tests
poetry run rapsheet-pull [--force]             # → data/raw/apwk-dzx8.parquet + .manifest.json (cached unless --force)
poetry run rapsheet-publish dataset|model [--yes] [--folder …]   # push to the Hub; dry run without --yes
poetry run jupyter lab notebooks/reading_the_rap_sheet.ipynb
```

`SODA_APP_TOKEN` lives in `.env` (gitignored) and is loaded automatically by `poetry-dotenv-plugin` under `poetry run` — don't `export` it or hardcode it. It is optional; it only avoids portal throttling. Tests use a synthetic frame in `tests/conftest.py` with the real column names — no network or data files needed.

## Architecture

Data flows one way: **pull → load/refuse → clean → split → baselines → encoder → headline metrics → calibrate → abstain → figures → card**.

Sub-packages form a DAG, enforced by `tests/test_architecture.py` (allowed edges, no self-imports, no
module-file imports): `core → {ingest, dataset, figures} → {baselines, encoder, evaluation, publish} → cli` (cli may import publish).
`core` = `types`, `errors`, `columns`, `manifest` (no internal deps, used by 3+ packages — that's the
membership test; don't let it grow by convenience). `baselines` and `encoder` are separate so that
importing a baseline never loads torch (optional `train` group). `evaluation` must never import
`baselines` or `encoder`; it scores only through the `Classifier` protocol. `figures` depends on `core` only and
draws from plain tables (never a model or a logit), so matplotlib stays out of every other package.

- `SodaClient` → `RunManifest`: paged SoQL pull to a parquet cache plus a manifest (query, counts, `:updated_at`, seeds, git sha, hardware). `RunManifest` is extended via `.with_seeds/.with_config/.with_extra` through the notebook and persisted to `data/processed/manifest.json` — every seed and count should land there.
- `Columns` (`core/columns.py`) is the single column policy: `used` (pulled), `protected` (race, gender, age, judge, … — never pulled), `leakage` (`offense_category`), `forbidden` = protected + leakage, `required`. `DispositionsDataset.load` raises `DatasetError` on any forbidden column. Always reference columns via `Columns`, never string literals; don't add protected columns to `used`.
- `DispositionsDataset` (`dataset/dataset.py`) does clean → split. Cleaning row counts ride along in `frame.attrs['cleaning']`. `excluded_labels` (`PROMIS Conversion`, a legacy-system provenance flag) are set aside via `excluded_rows`, never trained on. `split` = temporal hold-out (`received_date >= 2023-01-01`) first, then `StratifiedGroupKFold` on `case_id` (fold 0 = test, fold 1 = val).
- `DatasetSplits` holds `train/val/test/temporal` and round-trips to `data/processed/`. Each frame carries derived columns: `label_train` (labels with <20 training rows folded to `(rare, folded)`; real label stays in `label`), `rare_title` (title seen <5× in train — the drift alarm), `split`.
- `LabelSpace` fixes the column order of every probability matrix (`names` from `DatasetSplits.label_names`, fold bucket last). **Every model's `predict_proba` must return columns in `LabelSpace` order**; `encode` maps unseen labels to -1 (never correct).
- Baselines (`MajorityBaseline`, `TitleLookupBaseline`, `StatuteRulesBaseline`, `TfidfBaseline`) satisfy the `Classifier` protocol in `core/types.py`: `name`, `fit(train) -> fitted copy` (frozen dataclasses, so fit returns `replace(self, ...)`), `predict_proba(frame) -> Matrix`. `KeyedLabelCounts` is the shared (key × label) count matrix behind the title/statute lookups and the weighted-unique-pair training in TF-IDF.
- `HeadlineMetrics` scores any list of `Classifier`s identically on `test` and `temporal` (accuracy, macro-F1 over labels present, rare-class recall, rare-title accuracy); every model plugs in through that protocol.
- `EncoderClassifier` (`encoder/`): `legal-bert-small` on `title [SEP] citation`, plain-torch loop over weighted draws of distinct inputs, best epoch by validation NLL, `save`/`load` with `rapsheet_encoder.json`. `EncoderConfig.class_temper` (default 1.0; the notebook uses 0.5) tempers class shares via `ClassTemperedDraws`, and the matching logit adjustment is stored with the model and added back in `logits_for_keys`. `LogitStore` saves logits + labels + rare-title flags per split so calibration never retrains.
- `TemperatureCalibrator` (one scalar `T` by golden-section search on validation NLL; ECE over 15 bins) and `SelectiveEvaluator` (risk–coverage, AURC, per-class τ at an error budget using a one-sided Wilson bound on validation errors; the fold bucket never earns a τ; `budget_sweep`) work only on saved logits. Thresholds are earned on `val` and scored on `test`/`temporal`/the PROMIS probe. `ConfusionTable` ranks (true → predicted) mistakes.
- `FigureWriter` (`figures/`) draws F1 reliability, F2 risk–coverage, F3 top confusions, F4 budget sweep as SVG + PNG into `figures/`.
- Publishing: `HubSnapshot` (dataset card + parquet per split, no identifiers), `ModelCard` (model `README.md` filled from the manifest and `rapsheet_encoder.json` — never typed by hand), `HubPublisher` (the only network push; mirrors the folder; refuses a folder without a card). The notebook guards pushes behind `PUBLISH` and `PUBLISH_MODEL`; `rapsheet-publish` is the same push from a shell. **The Hub namespace is never a literal**: `HubRepos` reads `HF_USERNAME` from `.env` and builds `dataset_id`/`model_id`/URLs; the notebook does `HUB = HubRepos.from_env()` in setup and uses `HUB.*` everywhere; tests read `HF_USERNAME` (set by an autouse fixture in `conftest.py`). Auth: `HF_TOKEN` in `.env` (not `HUGGING_FACE_TOKEN`); `HubPublisher` refuses a namespace the token doesn't own.
- Notebook switches: `FORCE_PULL` (0b), `RETRAIN` / `RETRAIN_TEMPERED` (§7/§7b reuse saved models by default), `FEATURED` (§8 picks the encoder by a validation-only rule: lower NLL and higher rare-class recall), `PUBLISH`, `PUBLISH_MODEL`.
- `DispositionMap` + `data/disposition_map.yaml`: the 36 raw disposition strings → 5 outcomes is a human-authored, versioned lookup, deliberately not a model. Bump `version` on any change.

## Conventions (project-specific)

These follow the global rules but with local particulars:

- 2-space indent, single quotes, line length 100 (`ruff.toml`); one class per file; frozen dataclasses; Python `logging` via module-level `log`.
- **Methods have no explicit `return`**: the tail expression is the return value via `@implicit_return` from `modgud.app` (stack `@classmethod`/`@staticmethod`/`@property` *above* it). That's why `B018` is ignored in ruff — don't "fix" bare tail expressions. Guard clauses use `raise` before the tail. **Only decorate a method whose value a caller uses**; a procedure (CLI `main`/`run`, `LogitStore.save`, `FigureWriter.legend`) is a plain `-> None` method with no decorator and no tail value.
- **Re-export per sub-package**: `rapsheet/__init__.py` re-exports nothing. Each sub-package's `__init__.py` re-exports, via `__all__`, the public names defined in its own folder (tests and the notebook count as outside consumers). Cross-package: `from rapsheet.<pkg> import Name` — never `from rapsheet.<pkg>.<module>`. Intra-package: `from .module import Name` — never `from . import x`, never an absolute import of your own package. Always list names explicitly — no `import *`, even past five names (overrides the global >5 rule here).
- Module-level `COLS = Columns()` is the idiom for column access.
- Errors derive from `RapsheetError` (`core/errors.py`).
- Notebook cells are excluded from ruff; cell prose uses the ⚙️ **What** / 🧭 **Why** pattern.

## Gotchas

- `ruff.toml` has `target-version = "py311"` while `pyproject.toml` requires `>=3.13.7,<3.14`; ruff won't flag 3.12+-only syntax as modern. Align them if that starts to matter.
- This file and `.claude/` are excluded via `.git/info/exclude` / `.gitignore` and never committed. Don't reference this file from any tracked file.
- `data/raw/*.parquet`, `data/processed/*.parquet`, `data/processed/logits*/`, `data/snapshot/` and `models/` are gitignored; regenerate via `rapsheet-pull` and the notebook. `figures/` is committed.
- `modgud` compiles methods against their own module globals, so tests must `monkeypatch.setattr(Class, 'method', …)`, not the module-level name; a multi-line string starting at column 0 inside an `@implicit_return` method breaks its re-parse (keep templates at module level); adjacent f-string literals at a tail keep only the last.
- transformers 5: `BertTokenizer(vocab=...)`, not `vocab_file=`; `num_labels` is redundant when `id2label` is given (and triggers a bogus mismatch warning).
