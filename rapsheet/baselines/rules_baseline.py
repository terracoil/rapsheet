"""StatuteRulesBaseline — the statute book: chapter/act/section → majority category in training."""

from dataclasses import dataclass, replace

import numpy as np
from modgud.app import implicit_return

from rapsheet.core import Columns, Frame, JsonDict, Matrix, NotFittedError, Series
from rapsheet.dataset import KeyedLabelCounts, LabelSpace

COLS = Columns()
LEVELS = ('chapter', 'act', 'section')  # coarse → fine; the finest match wins


@dataclass(frozen=True)
class StatuteRulesBaseline:
  """A rule a lawyer could write down, learned from training counts.

  The model never sees the statute columns; this baseline sees *only* them. Keys back off from
  ``chapter|act|section`` to ``chapter|act`` to ``chapter`` to the prior, so a citation the
  training data never saw still lands somewhere sensible.
  """

  space: LabelSpace
  alpha: float = 1.0
  name: str = 'statute rules'
  tables: tuple[tuple[str, KeyedLabelCounts], ...] = ()

  @staticmethod
  @implicit_return
  def normalize(values: Series) -> Series:
    """Trim and upper-case a statute field; blanks become missing."""
    cleaned = values.astype('string').str.strip().str.upper()
    cleaned.mask(cleaned == '')

  @implicit_return
  def keys(self, frame: Frame) -> dict[str, Series]:
    """Return the three statute keys per row; any missing part makes that key missing."""
    chapter = self.normalize(frame[COLS.CHAPTER])
    act = self.normalize(frame[COLS.ACT])
    section = self.normalize(frame[COLS.SECTION])
    {'chapter': chapter, 'act': chapter + '|' + act, 'section': chapter + '|' + act + '|' + section}

  @implicit_return
  def fit(self, train: Frame) -> 'StatuteRulesBaseline':
    """Count labels per statute key at every level."""
    labels = self.space.encode(train[COLS.LABEL_TRAIN])
    keys = self.keys(train)
    tables = tuple(
      (level, KeyedLabelCounts.build(keys[level], labels, self.space.size)) for level in LEVELS
    )
    replace(self, tables=tables)

  @implicit_return
  def fitted(self) -> dict[str, KeyedLabelCounts]:
    """Return the per-level tables, or raise if ``fit`` hasn't run."""
    if not self.tables:
      raise NotFittedError(f'{self.name}: call fit() before predicting')
    dict(self.tables)

  @implicit_return
  def predict_proba(self, frame: Frame) -> Matrix:
    """Start from the prior, then overwrite with each finer level that has a match."""
    tables = self.fitted()
    prior = self.space.prior()
    keys = self.keys(frame)
    proba = np.tile(prior, (len(frame), 1))
    for level in LEVELS:
      table = tables[level]
      proba = table.lookup(keys[level], table.distributions(prior, self.alpha), proba)
    proba

  @implicit_return
  def match_levels(self, frame: Frame) -> JsonDict:
    """Return the share of rows whose finest match is each level (or none at all)."""
    tables = self.fitted()
    keys = self.keys(frame)
    finest = np.full(len(frame), 'none', dtype=object)
    for level in LEVELS:
      known = keys[level].isin(set(tables[level].keys)).fillna(False).to_numpy(dtype=bool)
      finest = np.where(known, level, finest)
    values, counts = np.unique(finest, return_counts=True)
    {str(v): round(float(c) / max(len(frame), 1), 4) for v, c in zip(values, counts, strict=True)}
