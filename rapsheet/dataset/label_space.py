"""LabelSpace — the training label vocabulary, its order, and its prior."""

from dataclasses import dataclass

import numpy as np
from modgud.app import implicit_return

from rapsheet.core import Columns, Matrix, Series

from .dataset_splits import DatasetSplits

COLS = Columns()


@dataclass(frozen=True)
class LabelSpace:
  """Column order for every probability matrix in the package.

  ``names`` follows ``DatasetSplits.label_names`` (sorted, fold bucket last); ``train_counts``
  are the training rows per label, which give the prior and decide which classes are rare.
  """

  names: tuple[str, ...]
  train_counts: tuple[int, ...]
  folded_label: str

  @classmethod
  @implicit_return
  def from_splits(cls, splits: DatasetSplits) -> 'LabelSpace':
    """Build the space from the training split's ``label_train`` column."""
    counts = splits.train[COLS.LABEL_TRAIN].value_counts()
    cls(
      names=splits.label_names,
      train_counts=tuple(int(counts.get(name, 0)) for name in splits.label_names),
      folded_label=splits.folded_label,
    )

  @property
  @implicit_return
  def size(self) -> int:
    """Number of training labels."""
    len(self.names)

  @implicit_return
  def encode(self, labels: Series) -> np.ndarray:
    """Return label indices; a label outside the training space becomes -1 (never correct)."""
    lookup = {name: i for i, name in enumerate(self.names)}
    labels.astype(object).map(lookup).fillna(-1).astype(int).to_numpy()

  @implicit_return
  def decode(self, indices: np.ndarray) -> np.ndarray:
    """Return label names for an array of indices."""
    np.asarray(self.names, dtype=object)[indices]

  @implicit_return
  def prior(self) -> Matrix:
    """Return training label frequencies — the always-guess-the-base-rate distribution."""
    counts = np.asarray(self.train_counts, dtype=float)
    counts / counts.sum()

  @implicit_return
  def rare_classes(self, max_train_rows: int = 500) -> tuple[int, ...]:
    """Return indices of named classes with fewer than ``max_train_rows`` training rows."""
    tuple(
      i
      for i, (name, count) in enumerate(zip(self.names, self.train_counts, strict=True))
      if count < max_train_rows and name != self.folded_label
    )
