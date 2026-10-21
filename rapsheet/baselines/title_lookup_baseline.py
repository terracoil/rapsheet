"""TitleLookupBaseline — the dictionary: title → the label it usually gets in training."""

from dataclasses import dataclass, replace

import numpy as np
from modgud.app import implicit_return

from rapsheet.core import Columns, Frame, Matrix, NotFittedError
from rapsheet.dataset import KeyedLabelCounts, LabelSpace

COLS = Columns()


@dataclass(frozen=True)
class TitleLookupBaseline:
  """Memorization, made explicit.

  For a title seen in training, the prediction is that title's label distribution (smoothed toward
  the prior by ``alpha`` pseudo-rows); for an unseen title it is the prior. Its confidence is the
  title's purity, which makes it a fair rival for the encoder on calibration and abstention too.
  On seen titles it approaches the title-majority ceiling; the encoder has to beat it elsewhere.
  """

  space: LabelSpace
  alpha: float = 1.0
  name: str = 'title lookup'
  table: KeyedLabelCounts | None = None

  @implicit_return
  def fit(self, train: Frame) -> 'TitleLookupBaseline':
    """Count labels per training title."""
    labels = self.space.encode(train[COLS.LABEL_TRAIN])
    replace(self, table=KeyedLabelCounts.build(train[COLS.TITLE], labels, self.space.size))

  @implicit_return
  def fitted(self) -> KeyedLabelCounts:
    """Return the count table, or raise if ``fit`` hasn't run."""
    if self.table is None:
      raise NotFittedError(f'{self.name}: call fit() before predicting')
    self.table

  @implicit_return
  def predict_proba(self, frame: Frame) -> Matrix:
    """Return smoothed per-title distributions; the prior for unseen titles."""
    table = self.fitted()
    prior = self.space.prior()
    fallback = prior[None, :].repeat(len(frame), axis=0)
    table.lookup(frame[COLS.TITLE], table.distributions(prior, self.alpha), fallback)

  @implicit_return
  def seen(self, frame: Frame) -> np.ndarray:
    """Return a boolean mask: True where the title occurred in training."""
    frame[COLS.TITLE].isin(set(self.fitted().keys)).to_numpy()
