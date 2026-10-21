# rapsheet

Calibrated, abstaining classification of charge text from one county's public dispositions
table. **A reference implementation of a methodology — not a screening tool.**

A small fine-tuned encoder learns to read charge titles; temperature scaling and a per-class
abstention threshold decide which decisions it may make alone. The 36 raw disposition strings are
handled by a versioned, human-authored lookup (`data/disposition_map.yaml`) — *rules are not weights*.

* 📄 [The article](docs/RAPSHEET.md) ([on the blog](https://git-steven.github.io/architecture/ai/ml/reading-the-rap-sheet/)) 
* 📓 [The notebook](notebooks/reading_the_rap_sheet.ipynb) 
* 🗃️ [The data snapshot](https://huggingface.co/datasets/terracoil/rapsheet-charge-titles) 
* 🪪 [The model](https://huggingface.co/terracoil/rapsheet-legal-bert-small)

## Data

Cook County State's Attorney — Dispositions, Socrata
[`apwk-dzx8`](https://datacatalog.cookcountyil.gov/d/apwk-dzx8), public domain, ~1.08M charge-level
rows, de-identified by the publisher. We classify charge text and never attempt identity matching.

The publisher stopped maintaining the table in December 2024, but the portal's row timestamps moved
later, so reproducibility rests on two things: the run manifest (query, counts, timestamps, seeds,
git SHA) and a public snapshot of the splits,
[`terracoil/rapsheet-charge-titles`](https://huggingface.co/datasets/terracoil/rapsheet-charge-titles),
which keeps only the columns the models use and no case, participant or charge identifiers.

`PROMIS Conversion` is a value of the label column that marks records migrated from a legacy case
system, not an offense. Those rows are set aside, never trained on, and published as a label-free
drift probe.

Columns **never pulled and never features**: `race`, `gender`, `age_at_incident`, `judge`,
`incident_city`, `law_enforcement_agency`, `unit`; `offense_category` is dropped as label leakage.
`DispositionsDataset.load` refuses a frame that carries any of them.

## Setup

```bash
poetry env use 3.13            # >=3.13.7, <3.14
poetry install --with dev      # train + test + jupyter groups
poetry self add poetry-dotenv-plugin   # once; loads .env for every `poetry run`
echo 'SODA_APP_TOKEN=...' > .env   # optional; avoids throttling (portal → profile → Developer Settings)
echo 'HF_USERNAME=...' >> .env   # your Hugging Face account; HF_TOKEN=... (write access) too, if you publish
poetry run rapsheet-pull       # → data/raw/apwk-dzx8.parquet + .manifest.json, prints the group-by
poetry run jupyter lab notebooks/reading_the_rap_sheet.ipynb
```

`.env` is gitignored and read automatically by `poetry-dotenv-plugin` for every `poetry run`,
so the token never lands in a notebook or a commit; no `export` needed.

`rapsheet-pull` reuses the cache when parquet + manifest exist; `--force` re-pulls.

## Layout

```
rapsheet/                   sub-packages form a DAG: core → {ingest, dataset, figures} → {baselines, encoder, evaluation, publish} → cli
  core/                     → nothing
    columns.py              Columns               — the column policy (used / protected / leakage)
    manifest.py             RunManifest           — query, counts, :updated_at, seeds, config, git sha, hardware
    types.py · errors.py                          — type aliases + the `Classifier` protocol · exception hierarchy
  ingest/                   → core
    soda_client.py          SodaClient            — paged SoQL pull, app token, parquet cache, manifest
  dataset/                  → core
    disposition_map.py      DispositionMap        — load YAML v1, map, report unmapped, review flags
    dataset.py              DispositionsDataset   — load, refuse, clean, dedupe, grouped + temporal splits
    dataset_splits.py       DatasetSplits         — train/val/test/temporal + summary, rare-class table, I/O
    label_space.py          LabelSpace            — label order, training counts and prior for every probability matrix
    keyed_label_counts.py   KeyedLabelCounts      — (key × label) counts behind the lookups and weighted training
  baselines/                → core, dataset
    majority_baseline.py    MajorityBaseline      — the floor: always the training base rates
    title_lookup_baseline.py TitleLookupBaseline  — title → the label it usually gets in training
    rules_baseline.py       StatuteRulesBaseline  — chapter/act/section → majority category, with back-off
    tfidf_baseline.py       TfidfBaseline         — word + character TF-IDF into logistic regression
  encoder/                  → core, dataset        (needs the optional `train` group: torch, transformers)
    encoder_config.py       EncoderConfig         — fine-tune hyperparameters (manifest-ready)
    encoder_inputs.py       EncoderInputs         — title + `720 ILCS 5/24-1.6` citation as a sentence pair
    encoder_classifier.py   EncoderClassifier     — legal-bert-small: weighted draws, best epoch by val NLL, save/load
    class_tempered_draws.py ClassTemperedDraws    — class shares ∝ rows^temper, and the logit adjustment that undoes the tilt
  evaluation/               → core, dataset
    headline_metrics.py     HeadlineMetrics       — accuracy, macro-F1, rare-class recall, rare-title accuracy
    logit_store.py          LogitStore            — saved logits + labels + rare-title flags per split
    temperature_calibrator.py TemperatureCalibrator — one scalar T fitted on validation NLL; ECE and reliability bins
    selective_evaluator.py  SelectiveEvaluator    — risk–coverage, AURC, per-class τ at an error budget (Wilson bound), budget sweep
    confusion_table.py      ConfusionTable        — the most common (true → predicted) mistakes, by name
  figures/                  → core                (needs matplotlib, in the `train` group)
    figure_writer.py        FigureWriter          — F1 reliability, F2 risk–coverage, F3 top confusions, F4 budget sweep (SVG + PNG)
  publish/                  → core, dataset
    hub_repos.py            HubRepos              — the Hub namespace (`HF_USERNAME`) and both repo ids / URLs, built in one place
    hub_snapshot.py         HubSnapshot           — public copy of the splits (no identifiers) + dataset card
    model_card.py           ModelCard             — the Hub model card, filled from the manifest and the saved model's metadata
    hub_publisher.py        HubPublisher          — the one class that pushes anything to the Hugging Face Hub
  cli/                      → core, ingest, dataset, publish
    pull_command.py         PullCommand           — `rapsheet-pull`
    publish_command.py      PublishCommand        — `rapsheet-publish dataset|model [--yes]`
data/disposition_map.yaml
notebooks/reading_the_rap_sheet.ipynb
tests/                 synthetic fixture with the real column names; `poetry run pytest`
                       `test_architecture.py` keeps the package graph a DAG (allowed edges, no self-imports)
```

## Conventions

Classes only, one class per file, frozen dataclasses, single return point with **no explicit
`return`** — method tails are the return values via [`modgud`](https://pypi.org/project/modgud/)
(`from modgud.app import implicit_return`). Lint: `poetry run ruff check rapsheet tests && poetry run ruff format rapsheet tests`.

## Baselines

Four models that read charge text without a neural network, all satisfying the `Classifier`
protocol (`name`, `fit`, `predict_proba`) and scored identically by `HeadlineMetrics` on the
`test` and `temporal` splits. The encoder has to beat these to justify itself.

## Encoder

`legal-bert-small` reads a sentence pair, `[CLS] title [SEP] 720 ILCS 5/24-1.6 [SEP]`
(`include_statute=False` for title only). The 266k training rows hold only a few thousand distinct
inputs, so the plain-torch training loop encodes each once and draws them in proportion to their
frequency (the same distribution as drawing rows). The epoch with the lowest validation NLL wins;
logits for validation, test, temporal and the PROMIS probe are saved for calibration.

Frequency-proportional draws starve small categories (a 25-row class gets ~14 looks in 150k draws).
`EncoderConfig(class_temper=0.5)` draws each class in proportion to the square root of its size and
adds the matching logit adjustment back at prediction time (Menon et al., 2021), so probabilities
still answer for the real class frequencies. The notebook trains both and picks the one to feature
by a validation-only rule (lower NLL *and* higher rare-class recall).

## Calibration and abstention

`TemperatureCalibrator` fits one scalar `T` on the validation logits (NLL, golden-section search) and
reports ECE before and after. `SelectiveEvaluator` turns calibrated probabilities into a policy: per
class, the lowest confidence at which the upper end of a one-sided Wilson interval on validation
errors stays within the budget (default 0.5%); a class that can't show that earns no threshold and
every one of its calls goes to a person. It also scores a single global τ for comparison, draws the
risk–coverage curve, and sweeps the budget. `FigureWriter` draws F1–F4 from those tables into
`figures/`.

## Cleaning and splits

- keep `primary_charge == true`; latest `charge_version_id` per `charge_id`
- titles: trim, collapse whitespace, upper-case; punctuation kept (`<`, `/`, `$` carry meaning)
- temporal hold-out: `received_date >= 2023-01-01`
- the rest: `StratifiedGroupKFold` on `case_id` (10 folds; fold 0 = test, fold 1 = val)
- `PROMIS Conversion` rows set aside (never trained on; published as a drift probe)
- labels with < 20 training rows → `(rare, folded)` in `label_train` (real label kept in
  `updated_offense_category`); not `Other`, because the taxonomy has its own `Other Offense`
- `rare_title`: title seen < 5 times in training — the drift alarm
- every seed and count lands in `data/processed/manifest.json`

## Publishing

Nothing leaves the machine unless you ask: the notebook's publish cells are guarded by
`PUBLISH = False` (the data snapshot) and `PUBLISH_MODEL = False` (the fine-tuned model), and
`HubPublisher` refuses a folder without a card. Each publish makes the Hub repo mirror the folder in
one commit, so re-publishing replaces rather than accumulates. `ModelCard` writes the model's
`README.md` from the manifest, so the card can't disagree with the notebook. The namespace comes from
`HF_USERNAME` in `.env` (`HubRepos` builds both repo ids from it) and the token from `HF_TOKEN`;
`HubPublisher` refuses to push to a namespace the token doesn't own.

From a shell, after the notebook has written the cards:

```bash
poetry run rapsheet-publish dataset          # dry run: shows folder → repo as <user>
poetry run rapsheet-publish dataset --yes    # pushes data/snapshot/
poetry run rapsheet-publish model --yes      # pushes models/rapsheet-legal-bert-small-tempered/ (or --folder …)
```

## Intended use

A reference implementation of a methodology: calibration, abstention, and keeping policy outside the
model. **Not a screening tool.** Out of scope: any decision about a person. One county's records can't
tell you how a model behaves on another county's format, and this repo doesn't pretend otherwise.

Three disposition codes (`BFW`, `WOWI`, `TRAN-FAI`; about 0.16% of rows) are mapped as *not final*
pending confirmation of their meaning; see `review_notes` in `data/disposition_map.yaml`.

## License

Code: MIT (see `LICENSE`). Data: public domain at the source; the snapshot on the Hub carries the same
terms.
