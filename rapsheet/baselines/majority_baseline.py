"""MajorityBaseline — the floor: always answer with the training base rates."""

from dataclasses import dataclass

import numpy as np
from modgud.app import implicit_return

from rapsheet.core import Frame, Matrix
from rapsheet.dataset import LabelSpace


@dataclass(frozen=True)
class MajorityBaseline:
  """Predicts the most frequent training label for every row (here, *Narcotics*).

  Its probabilities are the training prior, so its confidence never exceeds the majority share.
  Any model that cannot beat this row in the headline table isn't reading the title.
  """

  space: LabelSpace
  name: str = 'majority class (floor)'

  @implicit_return
  def fit(self, train: Frame) -> 'MajorityBaseline':
    """Nothing to learn beyond the prior, which the label space already holds."""
    self

  @implicit_return
  def predict_proba(self, frame: Frame) -> Matrix:
    """Return the prior for every row."""
    np.tile(self.space.prior(), (len(frame), 1))
