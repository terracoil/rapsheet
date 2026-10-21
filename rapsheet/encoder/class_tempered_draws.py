"""ClassTemperedDraws — how often each class gets drawn in training, and how to undo the tilt."""

from dataclasses import dataclass

import numpy as np
from modgud.app import implicit_return

from rapsheet.core import Matrix


@dataclass(frozen=True)
class ClassTemperedDraws:
  """Draw probabilities over (input, label) pairs, with each class's share tempered by an exponent.

  Class ``c`` gets a share of the draws in proportion to ``n_c ** temper``; within a class, inputs
  are still drawn as often as they occur.

  - ``temper=1.0`` draws rows as they occur. A class with 25 training rows gets about 14 of 150k
    draws, which is too few for a fresh head to learn it.
  - ``temper=0.5`` (square root) gives that class about 280, while the largest class keeps a
    double-digit share.
  - ``temper=0.0`` gives every class the same share.

  A tempered model learns a flattened prior, and its probabilities say so. ``logit_adjustment``
  returns ``log(prior) - log(training share)`` per class: added to the logits, it puts the real
  class frequencies back (post-hoc logit adjustment, Menon et al., 2021). Because only the class
  shares move, never the mix of inputs inside a class, the correction is exact.
  """

  labels: np.ndarray
  weights: np.ndarray
  n_classes: int
  temper: float = 1.0

  @implicit_return
  def class_rows(self) -> Matrix:
    """Return the training rows behind each class (pair weights summed by label)."""
    np.bincount(self.labels, weights=self.weights, minlength=self.n_classes).astype(float)

  @implicit_return
  def class_shares(self) -> Matrix:
    """Return each class's share of the draws; a class with no rows gets none."""
    rows = self.class_rows()
    tempered = np.where(rows > 0, np.where(rows > 0, rows, 1.0) ** self.temper, 0.0)
    tempered / tempered.sum()

  @implicit_return
  def probabilities(self) -> Matrix:
    """Return one draw probability per pair, summing to 1."""
    rows = self.class_rows()
    per_row = np.divide(
      self.class_shares(), rows, out=np.zeros(self.n_classes, dtype=float), where=rows > 0
    )
    tilted = self.weights * per_row[self.labels]
    tilted / tilted.sum()

  @implicit_return
  def logit_adjustment(self) -> Matrix:
    """Return ``log(prior) - log(share)`` per class: zeros when ``temper`` is 1."""
    rows = self.class_rows()
    present = rows > 0
    prior = np.where(present, rows / rows.sum(), 1.0)
    shares = np.where(present, self.class_shares(), 1.0)
    np.where(present & (self.temper != 1.0), np.log(prior) - np.log(shares), 0.0)
