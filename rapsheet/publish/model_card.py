"""ModelCard — the Hub card for the fine-tuned encoder, written from the manifest, not from memory."""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from string import Template

from modgud.app import implicit_return

from rapsheet.core import Columns, JsonDict, PathLike, RunManifest

from .hub_repos import REPO_URL, HubRepos

COLS = Columns()
META_FILE = 'rapsheet_encoder.json'
HUB_CONFIG = 'config.json'
PORTAL_URL = 'https://datacatalog.cookcountyil.gov/d/apwk-dzx8'
ARTICLE_URL = 'https://git-steven.github.io/architecture/ai/ml/reading-the-rap-sheet/'

CARD = Template(
  """---
license: cc-by-sa-4.0
base_model: $base_model
language:
  - en
pipeline_tag: text-classification
library_name: transformers
datasets:
  - $dataset_repo
tags:
  - legal
  - calibration
  - selective-classification
  - abstention
---

# $title

**A reference implementation of a methodology for calibrated, abstaining classification of charge
text. Not a screening tool, and not for any decision about a person.**

Reads one county's charge titles (`AGG UUW/VEH/PREV CONVICTION`) together with the statute citation
(`720 ILCS 5/24-1.6`) and predicts the prosecutor's own offense category. The point of the model is
not the category; it is the calibrated confidence next to it, and the per-class threshold that
decides which calls it may make alone. The method, the code and every number here live in the
[rapsheet]($repo_url) package and its notebook; the write-up is
[the article]($article_url).

## Task and labels

- **Input:** a sentence pair, `[CLS] title [SEP] chapter ILCS act/section [SEP]`
  (`include_statute=$include_statute`), tokenized to $max_length tokens.
- **Output:** $n_labels training labels: $n_named named offense categories plus `$folded_label`, the
  bucket for categories with fewer than $min_class_rows training rows. Label order is fixed in
  `rapsheet_encoder.json` and in `config.json`'s `id2label`.
- **Confidence:** softmax of the logits$adjustment_note. Temperature scaling
  fitted on validation gives **T = $temperature**; the calibrated probabilities feed the abstention
  thresholds below.

## How the base model is used

The transformer is the reader, not the decider. [`$base_model`](https://huggingface.co/$base_model)
is $architecture, pre-trained with masked-language modelling on legal text. Loading it as
`AutoModelForSequenceClassification` keeps every encoder weight, discards the pre-training heads
(`cls.predictions.*`, `cls.seq_relationship.*`) and adds one new **classification head**: a single
linear layer from the pooled `[CLS]` vector to $n_labels logits ($head_shape), weights drawn from
N(0, 0.02²), biases zero. That head is the only part of the network that starts from nothing.

- **What was extended:** the head, and nothing else. No new vocabulary tokens, no adapters or LoRA,
  no frozen layers, no second head, no distillation. The whole network is fine-tuned end to end, so
  the encoder's attention and embeddings move with the head; `id2label` / `label2id` in
  `config.json` name the outputs.
- **Input construction:** segment A is the charge title exactly as the county spells it
  (`AGG UUW/VEH/PREV CONVICTION`); segment B is `chapter ILCS act/section` (`720 ILCS 5/24-1.6`).
  The uncased WordPiece tokenizer lowercases; punctuation stays, because `<$$300` and `>$$300` are
  different charges. Nothing else goes in: no disposition, no date, never a protected column.
  $input_note
- **Output:** $n_labels logits in `LabelSpace` order. What leaves the weights is then (1) shifted by
  the saved logit adjustment, (2) divided by T, (3) softmaxed; the top class is the prediction, its
  probability the confidence, and the per-class τ below decides whether that call may stand alone.
  Steps 1–3 live in `rapsheet_encoder.json` and the package, not in the checkpoint.
- **Loading without the package:** `AutoModelForSequenceClassification.from_pretrained('$repo_id')`
  works, but returns raw logits: no logit adjustment, no temperature, no thresholds, so the
  confidences are not the ones scored below. `EncoderClassifier.load` applies all three.
- **Model selection:** the kept epoch is the one with the lowest validation negative
  log-likelihood, which is why the fitted temperature lands near 1: the checkpoint was already
  chosen for its probabilities, not just its argmax.$selection_note

## Data

- Cook County State's Attorney, *Dispositions* (`$dataset_id`, $portal_url), public domain.
  Retrieved $retrieved: $rows_fetched charge-level rows, de-identified by the publisher.
  The publisher stopped maintaining the table in December 2024; the portal's latest row timestamp is
  `$source_updated_at`.
- Splits (grouped by case, stratified by label, seed $split_seed; `temporal` = received on or
  after $temporal_cutoff): $split_rows. Published without identifiers as
  [`$dataset_repo`](https://huggingface.co/datasets/$dataset_repo).
- Never pulled, never features: `race`, `gender`, `age_at_incident`, `judge`, `incident_city`,
  `law_enforcement_agency`, `unit`; `offense_category` (the pre-update label) dropped as leakage.
- `PROMIS Conversion` rows ($set_aside_rows, a legacy-system flag, not an offense) are never trained
  on; the $probe_rows whose titles never appear with a real label are a label-free drift probe.

## Training

Base model [`$base_model`](https://huggingface.co/$base_model) with the fresh classification head
described above, every parameter trainable. Plain-torch loop (no `Trainer`): $epochs epochs of
$samples_per_epoch draws of distinct (title, citation) inputs,
batch $batch_size, AdamW (lr $learning_rate, weight decay $weight_decay), linear warm-up
($warmup_ratio) and decay, gradient clipping at $max_grad_norm, seed $seed. Draws are
class-tempered with exponent $class_temper (each class's share of draws is proportional to
`rows ** $class_temper`); the matching logit adjustment is added back at prediction time, so the
probabilities answer for the real class frequencies. The epoch with the lowest validation
negative log-likelihood is kept.

| epoch | train loss | val NLL | val accuracy |
|---|---|---|---|
$history_rows

Package git SHA at processing time: `$git_sha`. Hardware: $hardware.

## Evaluation

Scored by the same function as every baseline, on the stratified test split and the 2023+ split.
*Rare-class recall* averages categories with fewer than 500 training rows; *rare-title accuracy*
covers titles seen fewer than five times in training.

| model | split | accuracy | macro-F1 | rare-class recall | rare-title accuracy |
|---|---|---|---|---|---|
$metric_rows

### Calibration (temperature T = $temperature, $bins bins)

| split | ECE before | ECE after | NLL before | NLL after | mean confidence after |
|---|---|---|---|---|---|
$calibration_rows

### Earned autonomy at a $error_budget error budget

A class earns a threshold only if the upper end of a one-sided Wilson interval (z = $z) on its
validation errors stays within the budget. $earned_count of $n_labels classes earn one:

| class | τ |
|---|---|
$threshold_rows

| split | decided alone (per-class τ) | error among those | decided alone (one global τ = $global_tau) | error |
|---|---|---|---|---|
$abstention_rows

Thresholds are earned on validation and scored on rows they never saw. On the 2023+ split the
per-class policy runs over budget: the thresholds need an owner and a review date, not just a value.

## Known failure modes

- **Label noise.** The category describes the case; the title describes the charge. Around 3% of
  rows carry a label their input cannot predict, which caps every model near 96.7%.
- **Temporal drift.** Thresholds earned before 2023 do not hold on 2023+ cases.
- **Single jurisdiction.** One county's spelling of charges. The legacy `PROMIS` dialect shows what
  happens on text the model has never seen: confidence stays high (mean top-class confidence
  $probe_confidence on the probe).
- **Small classes.** Categories with few validation rows cannot earn a threshold at this budget,
  whatever their recall.

## Intended use and out of scope

Intended: studying calibration, abstention and the separation of policy from weights on public
data; reproducing the article. **Out of scope: any decision about a person**, screening,
re-identification, or use on records from another jurisdiction without re-evaluation.

## Reproduce

```bash
git clone $repo_url && cd rapsheet
poetry install --with dev && poetry run rapsheet-pull
poetry run jupyter lab notebooks/reading_the_rap_sheet.ipynb   # Restart & Run All
```

```python
from rapsheet.encoder import EncoderClassifier
model = EncoderClassifier.load('path/to/this/repo')   # after `hf download $repo_id`
probabilities = model.predict_proba(frame)             # columns in LabelSpace order
```

License: the weights derive from `$base_model` (CC BY-SA 4.0) and carry the same terms; the
rapsheet code is MIT.
"""
)


@dataclass(frozen=True)
class ModelCard:
  """Write ``README.md`` into a saved model folder, filled from the run manifest and the model's metadata.

  Nothing here is typed in by hand: metrics come from ``manifest.extra['baselines']``, calibration
  from ``extra['calibration']``, thresholds from ``extra['abstention']``, and the training config
  and history from the folder's ``rapsheet_encoder.json``. If a number is on the card, it was in
  the manifest first.
  """

  repo_id: str = field(default_factory=lambda: HubRepos().model_id)
  dataset_repo: str = field(default_factory=lambda: HubRepos().dataset_id)
  title: str = 'rapsheet · legal-bert-small, calibrated and abstaining'
  comparison: str = 'statute rules'

  @implicit_return
  def write(self, directory: PathLike, manifest: RunManifest, model_name: str) -> Path:
    """Render the card for the model saved in ``directory`` and write it as ``README.md``.

    The folder's ``config.json`` (transformers' own) supplies the encoder's shape when present.
    """
    target = Path(directory)
    meta = json.loads((target / META_FILE).read_text())
    hub_config = target / HUB_CONFIG
    shape = json.loads(hub_config.read_text()) if hub_config.exists() else {}
    path = target / 'README.md'
    path.write_text(self.render(meta, manifest, model_name, shape))
    path

  @implicit_return
  def render(
    self, meta: JsonDict, manifest: RunManifest, model_name: str, shape: JsonDict | None = None
  ) -> str:
    """Return the card text. ``shape`` is transformers' ``config.json`` (hidden size, layers)."""
    config = meta['config']
    extra = manifest.extra
    calibration = extra.get('calibration', {})
    abstention = extra.get('abstention', {})
    thresholds: dict[str, float] = abstention.get('thresholds', {})
    names = list(meta['label_names'])
    rows = [r for r in extra.get('baselines', []) if r['model'] in (model_name, self.comparison)]
    rows.sort(key=lambda r: (r['model'] != model_name, r['split'] != 'test', r['split']))
    split_rows = manifest.config.get('split_rows', {})
    adjustment = any(abs(x) > 0 for x in meta.get('logit_adjustment', ()))
    report = calibration.get('report', {})
    CARD.substitute(
      title=self.title,
      repo_id=self.repo_id,
      repo_url=REPO_URL,
      article_url=ARTICLE_URL,
      portal_url=PORTAL_URL,
      dataset_repo=self.dataset_repo,
      dataset_id=manifest.dataset_id,
      base_model=config['model_name'],
      include_statute=config['include_statute'],
      max_length=config['max_length'],
      n_labels=len(names),
      n_named=len(names) - 1,
      folded_label=meta['folded_label'],
      min_class_rows=manifest.config.get('min_class_rows', 20),
      adjustment_note=(
        ' after the class-tempering logit adjustment saved alongside the weights'
        if adjustment
        else ''
      ),
      architecture=self.architecture(shape or {}),
      head_shape=self.head_shape(shape or {}, len(names)),
      input_note=self.input_note(manifest, meta),
      selection_note=(
        f' Class tempering (exponent {config.get("class_temper", 1.0)}) was chosen by the same'
        ' validation-only rule the notebook uses to feature a model: lower validation NLL and'
        ' higher validation rare-class recall than the untempered run. The test split played no'
        ' part.'
        if adjustment
        else ''
      ),
      temperature=self.number(calibration.get('temperature'), 3),
      retrieved=manifest.retrieved_at[:10],
      rows_fetched=f'{manifest.row_count_fetched:,}',
      source_updated_at=manifest.source_updated_at,
      split_seed=manifest.seeds.get('split', 'n/a'),
      temporal_cutoff=manifest.config.get('temporal_cutoff', '2023-01-01'),
      split_rows=', '.join(f'{k} {v:,}' for k, v in split_rows.items()),
      set_aside_rows=f'{manifest.config.get("set_aside_rows", 0):,} rows',
      probe_rows=f'{manifest.config.get("set_aside_unseen_title_rows", 0):,} rows',
      epochs=config['epochs'],
      samples_per_epoch=f'{config["samples_per_epoch"]:,}',
      batch_size=config['batch_size'],
      learning_rate=config['learning_rate'],
      weight_decay=config['weight_decay'],
      warmup_ratio=config['warmup_ratio'],
      max_grad_norm=config['max_grad_norm'],
      seed=config['seed'],
      class_temper=config.get('class_temper', 1.0),
      history_rows='\n'.join(
        f'| {h["epoch"]} | {h["train_loss"]:.4f} | {h.get("val_nll", float("nan")):.4f} | '
        f'{h.get("val_accuracy", float("nan")):.4f} |'
        for h in meta.get('history', [])
      )
      or '| – | – | – | – |',
      git_sha=manifest.config.get('processed_git_sha') or manifest.git_sha or 'n/a',
      hardware=manifest.config.get('processed_on', manifest.hardware),
      metric_rows='\n'.join(
        f'| {r["model"]} | {r["split"]} | {r["accuracy"]:.4f} | {r["macro_f1"]:.4f} | '
        f'{r["rare_class_recall"]:.4f} | {r["rare_title_accuracy"]:.4f} |'
        for r in rows
      )
      or '| – | – | – | – | – | – |',
      bins=calibration.get('bins', 15),
      calibration_rows='\n'.join(
        f'| {split} | {self.percent(r.get("ece_before"))} | {self.percent(r.get("ece_after"))} | '
        f'{self.number(r.get("nll_before"), 4)} | {self.number(r.get("nll_after"), 4)} | '
        f'{self.number(r.get("mean_confidence_after"), 4)} |'
        for split, r in sorted(report.items(), key=lambda kv: self.split_rank(kv[0]))
      )
      or '| – | – | – | – | – | – |',
      error_budget=self.percent(abstention.get('error_budget')),
      z=abstention.get('z', 'n/a'),
      earned_count=len(thresholds),
      threshold_rows='\n'.join(
        f'| {name} | {tau:.4f} |' for name, tau in sorted(thresholds.items())
      )
      or '| – | – |',
      global_tau=self.number(abstention.get('global_tau'), 4),
      abstention_rows='\n'.join(
        f'| {split} | {self.percent(s.get("per_class_coverage"))} | {self.percent(s.get("per_class_error"))} | '
        f'{self.percent(s.get("global_coverage"))} | {self.percent(s.get("global_error"))} |'
        for split, s in abstention.get('summary', {}).items()
      )
      or '| – | – | – | – | – |',
      probe_confidence=self.number(extra.get('probe_confidence', {}).get(model_name), 3),
    )

  @staticmethod
  @implicit_return
  def architecture(shape: JsonDict) -> str:
    """Describe the encoder from transformers' config, or generically when it is absent."""
    layers, hidden = shape.get('num_hidden_layers'), shape.get('hidden_size')
    (
      f'a {layers}-layer BERT encoder with hidden size {hidden}'
      if layers and hidden
      else 'a BERT encoder'
    )

  @staticmethod
  @implicit_return
  def head_shape(shape: JsonDict, n_labels: int) -> str:
    """Size the classification head: hidden × labels weights plus one bias per label."""
    hidden = shape.get('hidden_size')
    (
      f'{hidden} × {n_labels} weights + {n_labels} biases, {hidden * n_labels + n_labels:,} parameters'
      if hidden
      else f'hidden size × {n_labels} weights + {n_labels} biases'
    )

  @staticmethod
  @implicit_return
  def input_note(manifest: RunManifest, meta: JsonDict) -> str:
    """Say how many rows collapse onto how many distinct titles, when the manifest knows."""
    titles = manifest.config.get('title_stats', {}).get('titles')
    train_rows = sum(meta.get('train_counts', ()))
    (
      f'The {train_rows:,} training rows carry only {titles:,} distinct titles, so each distinct'
      ' (title, citation) input is encoded once and drawn in proportion to its rows rather than'
      ' fed row by row.'
      if titles and train_rows
      else 'Each distinct (title, citation) input is encoded once and drawn in proportion to its'
      ' rows rather than fed row by row.'
    )

  @staticmethod
  @implicit_return
  def split_rank(split: str) -> tuple[int, str]:
    """Order calibration rows: the held-out splits first, validation (where T was fitted) last."""
    order = {'test': 0, 'temporal': 1, 'val': 2, 'validation': 2}
    (order.get(split, 3), split)

  @staticmethod
  @implicit_return
  def number(value: float | None, digits: int) -> str:
    """Format a number, or ``n/a`` when it is missing or NaN."""
    'n/a' if value is None or (
      isinstance(value, float) and math.isnan(value)
    ) else f'{value:.{digits}f}'

  @staticmethod
  @implicit_return
  def percent(value: float | None) -> str:
    """Format a share as a percentage, or ``n/a``."""
    'n/a' if value is None or (isinstance(value, float) and math.isnan(value)) else f'{value:.2%}'
