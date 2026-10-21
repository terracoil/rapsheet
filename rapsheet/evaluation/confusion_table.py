"""ConfusionTable — the model's most common mistakes, by name."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from modgud.app import implicit_return

from rapsheet.core import Frame, Matrix
from rapsheet.dataset import LabelSpace


@dataclass(frozen=True)
class ConfusionTable:
  """Count (true label → predicted label) mistakes and rank them.

  A 68 × 68 confusion matrix is mostly empty and entirely unreadable. The top few off-diagonal
  cells tell the same story: where the categories overlap, and where the labels themselves
  disagree. Rows with no training label (-1) are left out; they have no "true" class to name.
  """

  space: LabelSpace

  @implicit_return
  def top(self, proba: Matrix, labels: np.ndarray, n: int = 10) -> Frame:
    """Return the ``n`` most frequent mistakes with their share of all mistakes."""
    predicted = proba.argmax(axis=1)
    wrong = (predicted != labels) & (labels >= 0)
    pairs = pd.DataFrame(
      {
        'true': self.space.decode(labels[wrong]),
        'predicted': self.space.decode(predicted[wrong]),
      }
    )
    counts = pairs.value_counts().rename('rows').reset_index()
    counts.assign(share_of_errors=(counts['rows'] / max(int(wrong.sum()), 1)).round(4)).head(n)
