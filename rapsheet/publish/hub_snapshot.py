"""HubSnapshot — the public copy of what the models were trained and scored on."""

from dataclasses import dataclass, field
from pathlib import Path
from string import Template

import pandas as pd
from modgud.app import implicit_return

from rapsheet.core import Columns, Frame, JsonDict, PathLike, RunManifest
from rapsheet.dataset import DatasetSplits

from .hub_repos import REPO_URL, HubRepos

COLS = Columns()
PORTAL_URL = 'https://datacatalog.cookcountyil.gov/d/apwk-dzx8'

CARD = Template(
  """---
license: other
license_name: public-domain
license_link: $portal_url
language:
  - en
task_categories:
  - text-classification
pretty_name: Rapsheet — Cook County charge titles (snapshot)
size_categories:
  - $size
configs:
  - config_name: default
    data_files:
$split_yaml
---

# Rapsheet — Cook County charge titles (snapshot)

**A reference dataset for a methodology: calibrated, abstaining classification of charge text. Not a
screening tool, and not for any decision about a person.**

Charge titles (`AGG UUW/VEH/PREV CONVICTION`, `RET THEFT/DISP MERCH/<$$300/PRECONV`) with the
prosecutor's own offense category, split for training and evaluation. Built by the
[rapsheet]($repo_url) package and its notebook; every number below comes from the run manifest.

## Source

- Cook County State's Attorney, *Dispositions* (`apwk-dzx8`), $portal_url, public domain.
- Retrieved $retrieved: $rows_fetched charge-level rows. The publisher stopped
  maintaining the table in December 2024; the portal's latest row timestamp is
  `$source_updated_at`. This snapshot exists so the experiment can be repeated even if
  the source changes.
- De-identified by the publisher. Never pulled: `race`, `gender`, `age_at_incident`, `judge`,
  `incident_city`, `law_enforcement_agency`, `unit`, and `offense_category` (label leakage).
- Pulled but **not published here**: $withheld. Nothing after the
  split needs them, and this project never attempts identity matching.

## Processing

- Primary charges only, latest charge version per charge; titles trimmed and upper-cased with
  punctuation kept (`<`, `/`, `$$` carry meaning).
- `PROMIS Conversion` is a value of the label column that marks records migrated from a legacy case
  system, not an offense. Those rows are never trained on; they are published as
  `$set_aside_name`, a label-free drift probe (`label_train` is empty there).
- `temporal`: received on or after $temporal_cutoff. The other splits
  are grouped by case and stratified by label (seed $seed), so no
  case appears in more than one split.
- `label_train` folds categories with fewer than $min_class_rows training rows into
  `$folded_label`; `updated_offense_category` keeps the real name.
- `rare_title` is true when the title appears fewer than $rare_title_max times in
  training.

## Splits

| split | rows |
|---|---|
$split_table

## Known limitations

- One county, one prosecutor's office. Cross-jurisdiction format drift is the real test, and this
  data can't run it; the legacy-system dialect in `$set_aside_name` is the closest it gets.
- $multi_label_titles of $titles titles carry more than one label, so no title-only
  classifier can be perfect (ceiling: $ceiling of rows).
- Three disposition codes (`BFW`, `WOWI`, `TRAN-FAI`; about 0.16% of source rows) are mapped as
  *not final* in the project's disposition map pending confirmation of their meaning.
"""
)


@dataclass(frozen=True)
class HubSnapshot:
  """Write the splits as a Hugging Face dataset: model-relevant columns only, no identifiers.

  The source table is unmaintained and its row timestamps have moved since it was frozen, so the
  manifest alone can prove what was pulled but can't bring it back. This snapshot can. Case,
  participant and charge identifiers are dropped: nothing downstream of the split needs them, and
  this project never attempts identity matching.
  """

  repo_id: str = field(default_factory=lambda: HubRepos().dataset_id)
  set_aside_name: str = 'promis_set_aside'

  @property
  @implicit_return
  def columns(self) -> tuple[str, ...]:
    """Published columns, in order."""
    (
      COLS.TITLE,
      COLS.CHAPTER,
      COLS.ACT,
      COLS.SECTION,
      COLS.CLASS,
      COLS.DISPOSITION,
      COLS.LABEL,
      COLS.LABEL_TRAIN,
      COLS.RECEIVED_DATE,
      COLS.RARE_TITLE,
      COLS.SPLIT,
    )

  @property
  @implicit_return
  def withheld(self) -> tuple[str, ...]:
    """Columns that were pulled but are not published."""
    tuple(c for c in COLS.used if c not in self.columns)

  @implicit_return
  def frames(self, splits: DatasetSplits, set_aside: Frame) -> dict[str, Frame]:
    """Return the publishable frames, keyed by Hub split name (``val`` becomes ``validation``)."""
    frequent = splits.train[COLS.TITLE].value_counts()
    frequent_titles = set(frequent[frequent >= 5].index)
    aside = set_aside.assign(
      **{
        COLS.LABEL_TRAIN: pd.NA,
        COLS.RARE_TITLE: ~set_aside[COLS.TITLE].isin(frequent_titles),
        COLS.SPLIT: self.set_aside_name,
      }
    )
    named = {
      'train': splits.train,
      'validation': splits.val,
      'test': splits.test,
      'temporal': splits.temporal,
      self.set_aside_name: aside,
    }
    {name: frame[list(self.columns)].reset_index(drop=True) for name, frame in named.items()}

  @implicit_return
  def write(
    self, splits: DatasetSplits, set_aside: Frame, manifest: RunManifest, directory: PathLike
  ) -> JsonDict:
    """Write ``data/<split>.parquet`` files and a dataset card; return a summary for the manifest.

    Parquet files from an earlier write are removed first. ``HubPublisher`` makes the Hub repo
    mirror this folder, so a stale file here would be a stale file there.
    """
    target = Path(directory)
    (target / 'data').mkdir(parents=True, exist_ok=True)
    for stale in (target / 'data').glob('*.parquet'):
      stale.unlink()
    frames = self.frames(splits, set_aside)
    for name, frame in frames.items():
      frame.to_parquet(target / 'data' / f'{name}.parquet', index=False)
    rows = {name: len(frame) for name, frame in frames.items()}
    (target / 'README.md').write_text(self.card(rows, manifest))
    {
      'repo_id': self.repo_id,
      'directory': str(target),
      'rows': rows,
      'columns': list(self.columns),
      'withheld': list(self.withheld),
    }

  @implicit_return
  def card(self, rows: dict[str, int], manifest: RunManifest) -> str:
    """Return the Hugging Face dataset card (YAML front matter + Markdown)."""
    config = manifest.config
    stats = config.get('title_stats', {})
    total = sum(rows.values())
    size = (
      '100K<n<1M'
      if 100_000 <= total < 1_000_000
      else 'n>1M'
      if total >= 1_000_000
      else '10K<n<100K'
    )
    CARD.substitute(
      portal_url=PORTAL_URL,
      repo_url=REPO_URL,
      size=size,
      split_yaml='\n'.join(
        f'      - split: {name}\n        path: data/{name}.parquet' for name in rows
      ),
      split_table='\n'.join(f'| `{name}` | {count:,} |' for name, count in rows.items()),
      retrieved=manifest.retrieved_at[:10],
      rows_fetched=f'{manifest.row_count_fetched:,}',
      source_updated_at=manifest.source_updated_at,
      withheld=', '.join(f'`{c}`' for c in self.withheld),
      set_aside_name=self.set_aside_name,
      temporal_cutoff=config.get('temporal_cutoff', '2023-01-01'),
      seed=manifest.seeds.get('split', 'n/a'),
      min_class_rows=config.get('min_class_rows', 20),
      folded_label=config.get('folded_label', '(rare, folded)'),
      rare_title_max=config.get('rare_title_max', 5),
      multi_label_titles=f'{stats.get("multi_label_titles", 0):,}',
      titles=f'{stats.get("titles", 0):,}',
      ceiling=f'{stats.get("title_majority_ceiling", 0.0):.1%}',
    )
