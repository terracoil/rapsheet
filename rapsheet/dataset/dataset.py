"""DispositionsDataset — load, refuse forbidden columns, clean, dedupe, and split by case."""

import logging
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from modgud.app import implicit_return
from sklearn.model_selection import StratifiedGroupKFold

from rapsheet.core import Columns, DatasetError, Frame, JsonDict, PathLike, Series

from .dataset_splits import DatasetSplits

log = logging.getLogger(__name__)
COLS = Columns()


@dataclass(frozen=True)
class DispositionsDataset:
  """Turns the raw parquet into training-ready, leakage-free splits.

  Pipeline: ``load`` → ``clean`` → ``split``. Each step is pure; the cleaning step records
  its row counts in ``frame.attrs['cleaning']`` so the manifest can carry them.
  """

  min_class_rows: int = 20
  rare_title_max: int = 5
  temporal_cutoff: str = '2023-01-01'
  n_folds: int = 10
  seed: int = 42
  folded_label: str = '(rare, folded)'
  # Values of the label column that are not offense categories. ``PROMIS Conversion`` marks
  # records migrated from the legacy case system — a provenance flag, not an offense.
  excluded_labels: tuple[str, ...] = ('PROMIS Conversion',)

  # ---------------------------------------------------------------- load

  @implicit_return
  def load(self, path: PathLike) -> Frame:
    """Read the cached parquet, refuse forbidden columns, coerce types."""
    frame = pd.read_parquet(path)
    self.validate_columns(frame)
    self.coerce(frame)

  @implicit_return
  def validate_columns(self, frame: Frame) -> Frame:
    """Raise ``DatasetError`` if a required column is missing or a forbidden one is present."""
    present = set(frame.columns)
    missing = [c for c in COLS.required if c not in present]
    forbidden = [c for c in COLS.forbidden if c in present]
    if missing:
      raise DatasetError(f'missing required columns: {missing}')
    if forbidden:
      raise DatasetError(f'forbidden columns present (protected or leakage): {forbidden}')
    frame

  @implicit_return
  def coerce(self, frame: Frame) -> Frame:
    """Typed copy: booleans, numeric version ids, datetimes; everything else stays as text."""
    out = frame.copy()
    out[COLS.PRIMARY_CHARGE] = self.as_bool(out[COLS.PRIMARY_CHARGE])
    out[COLS.CHARGE_VERSION_ID] = pd.to_numeric(out[COLS.CHARGE_VERSION_ID], errors='coerce')
    for col in (COLS.RECEIVED_DATE, COLS.DISPOSITION_DATE):
      if col in out.columns:
        out[col] = pd.to_datetime(out[col], errors='coerce')
    out

  @staticmethod
  @implicit_return
  def as_bool(values: Series) -> Series:
    """SODA returns checkboxes as JSON booleans, but strings survive round-trips; accept both."""
    truthy = {'true', 't', '1', 'yes', 'y'}
    values.map(lambda v: v if isinstance(v, (bool, np.bool_)) else str(v).strip().lower() in truthy)

  # ---------------------------------------------------------------- clean

  @implicit_return
  def clean(self, frame: Frame) -> Frame:
    """Primary charges, latest version, normalised titles, no null or excluded labels."""
    complete = self.complete_rows(frame)
    excluded = complete[COLS.LABEL].isin(self.excluded_labels)
    steps: JsonDict = {
      **complete.attrs['cleaning'],
      'label_excluded': int(excluded.sum()),
      'after_label_exclusion': int((~excluded).sum()),
    }
    cleaned = complete[~excluded].reset_index(drop=True)
    cleaned.attrs['cleaning'] = steps
    cleaned.attrs['excluded_labels'] = list(self.excluded_labels)
    log.info('cleaning: %s', steps)
    cleaned

  @implicit_return
  def excluded_rows(self, frame: Frame) -> Frame:
    """Return cleaned rows whose label is in ``excluded_labels`` — kept aside, never trained on."""
    complete = self.complete_rows(frame)
    complete[complete[COLS.LABEL].isin(self.excluded_labels)].reset_index(drop=True)

  @implicit_return
  def complete_rows(self, frame: Frame) -> Frame:
    """Primary charges, latest version per charge, normalised titles, title and label present."""
    steps: JsonDict = {'loaded': len(frame)}
    primary = frame[frame[COLS.PRIMARY_CHARGE]]
    steps['primary_charge'] = len(primary)
    latest = primary.sort_values(
      [COLS.CHARGE_ID, COLS.CHARGE_VERSION_ID], ascending=[True, False]
    ).drop_duplicates(subset=COLS.CHARGE_ID, keep='first')
    steps['latest_charge_version'] = len(latest)
    titled = latest.assign(**{COLS.TITLE: self.normalize_title(latest[COLS.TITLE])})
    complete = titled.dropna(subset=[COLS.TITLE, COLS.LABEL, COLS.CASE_ID])
    complete = complete[complete[COLS.TITLE].str.len() > 0]
    steps['with_title_and_label'] = len(complete)
    out = complete.reset_index(drop=True)
    out.attrs['cleaning'] = steps
    out

  @staticmethod
  @implicit_return
  def normalize_title(titles: Series) -> Series:
    """Trim, collapse whitespace, upper-case. Punctuation (``<``, ``/``, ``$``) is kept."""
    titles.astype('string').str.strip().str.replace(r'\s+', ' ', regex=True).str.upper()

  @implicit_return
  def category_counts(self, frame: Frame) -> Frame:
    """Group by label — the table that opens the notebook: rows and share per category."""
    counts = frame[COLS.LABEL].value_counts()
    pd.DataFrame({'rows': counts, 'share': (counts / counts.sum()).round(4)})

  @implicit_return
  def title_stats(self, frame: Frame) -> JsonDict:
    """Measure how open the open vocabulary is.

    ``title_majority_ceiling`` is the share of rows whose label equals their title's most common
    label — the best any classifier that sees only the title can score on titles it has seen.
    """
    pair_counts = frame.groupby([COLS.TITLE, COLS.LABEL]).size()
    majority = pair_counts.groupby(level=0).max()
    labels_per_title = pair_counts.groupby(level=0).size()
    multi = labels_per_title[labels_per_title > 1].index
    {
      'rows': len(frame),
      'titles': int(len(labels_per_title)),
      'multi_label_titles': int(len(multi)),
      'rows_on_multi_label_titles': int(frame[COLS.TITLE].isin(multi).sum()),
      'title_majority_ceiling': round(float(majority.sum() / max(len(frame), 1)), 4),
    }

  @staticmethod
  @implicit_return
  def unseen_titles(rows: Frame, reference: Frame) -> Series:
    """Return a mask over ``rows``: True where the title never appears in ``reference``."""
    ~rows[COLS.TITLE].isin(set(reference[COLS.TITLE]))

  # ---------------------------------------------------------------- split

  @implicit_return
  def split(self, frame: Frame) -> DatasetSplits:
    """Temporal hold-out first, then case-grouped stratified train/val/test on the rest."""
    cutoff = pd.Timestamp(self.temporal_cutoff)
    in_time = frame[frame[COLS.RECEIVED_DATE] < cutoff]
    temporal = frame[frame[COLS.RECEIVED_DATE] >= cutoff]
    train_idx, val_idx, test_idx = self.grouped_folds(in_time)
    train = in_time.iloc[train_idx]
    folded = self.folded_classes(train)
    frequent = self.frequent_titles(train)
    finish = lambda part, name: self.annotate(part, folded, frequent, name)  # noqa: E731
    DatasetSplits(
      train=finish(train, 'train'),
      val=finish(in_time.iloc[val_idx], 'val'),
      test=finish(in_time.iloc[test_idx], 'test'),
      temporal=finish(temporal, 'temporal'),
      folded_classes=folded,
      seed=self.seed,
      folded_label=self.folded_label,
    )

  @implicit_return
  def grouped_folds(self, frame: Frame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Positional indices for train / val / test: folds 0 and 1 of a case-grouped stratified split."""
    splitter = StratifiedGroupKFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
    labels = frame[COLS.LABEL].to_numpy()
    groups = frame[COLS.CASE_ID].to_numpy()
    with warnings.catch_warnings():
      warnings.simplefilter('ignore', UserWarning)  # "least populated class has fewer members"
      folds = [held for _, held in splitter.split(np.zeros(len(frame)), labels, groups)]
    test_idx, val_idx = folds[0], folds[1]
    train_idx = np.setdiff1d(np.arange(len(frame)), np.concatenate([test_idx, val_idx]))
    (train_idx, val_idx, test_idx)

  @implicit_return
  def folded_classes(self, train: Frame) -> tuple[str, ...]:
    """Labels with fewer than ``min_class_rows`` training rows — folded for training."""
    counts = train[COLS.LABEL].value_counts()
    tuple(sorted(counts[counts < self.min_class_rows].index))

  @implicit_return
  def frequent_titles(self, train: Frame) -> frozenset[str]:
    """Titles seen at least ``rare_title_max`` times in training. Everything else is rare."""
    counts = train[COLS.TITLE].value_counts()
    frozenset(counts[counts >= self.rare_title_max].index)

  @implicit_return
  def annotate(
    self, part: Frame, folded: tuple[str, ...], frequent: frozenset[str], name: str
  ) -> Frame:
    """Add ``label_train``, ``rare_title`` and ``split`` columns to one split."""
    label_train = part[COLS.LABEL].where(~part[COLS.LABEL].isin(folded), self.folded_label)
    rare = ~part[COLS.TITLE].isin(frequent)
    part.assign(
      **{COLS.LABEL_TRAIN: label_train, COLS.RARE_TITLE: rare.astype(bool), COLS.SPLIT: name}
    ).reset_index(drop=True)

  @implicit_return
  def subsample(self, train: Frame, n_rows: int) -> Frame:
    """Seeded uniform subsample of the training split (for a time-boxed fine-tune)."""
    train.sample(n=min(n_rows, len(train)), random_state=self.seed).reset_index(drop=True)
