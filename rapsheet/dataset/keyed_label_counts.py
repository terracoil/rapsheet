"""KeyedLabelCounts — a (unique key × label) count matrix, the workhorse behind every lookup."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from modgud.app import implicit_return

from rapsheet.core import Matrix, Series


@dataclass(frozen=True)
class KeyedLabelCounts:
  """How often each label occurred for each distinct key (a title, a statute, …).

  Three uses:
  - **lookups** — smoothed per-key distributions, with a fallback for unseen keys;
  - **weighted training** — ``pairs()`` yields one row per non-zero (key, label) cell with its
    count as the weight. Fitting on those pairs optimizes the same loss as fitting on every row,
    because identical rows contribute identical terms;
  - **diagnostics** — how ambiguous each key is.
  """

  keys: tuple[str, ...]
  counts: Matrix

  @classmethod
  @implicit_return
  def build(cls, keys: Series, labels: np.ndarray, n_labels: int) -> 'KeyedLabelCounts':
    """Count (key, label) pairs; rows with a missing key or an unknown label are skipped."""
    frame = pd.DataFrame({'key': keys.astype(object).to_numpy(), 'label': labels})
    usable = frame[frame['key'].notna() & (frame['label'] >= 0)]
    table = (
      usable.groupby(['key', 'label'])
      .size()
      .unstack(fill_value=0)
      .reindex(columns=range(n_labels), fill_value=0)
    )
    cls(keys=tuple(str(k) for k in table.index), counts=table.to_numpy(dtype=float))

  @implicit_return
  def distributions(self, prior: Matrix, alpha: float) -> Matrix:
    """Return per-key label distributions, shrunk toward ``prior`` by ``alpha`` pseudo-rows."""
    (self.counts + alpha * prior) / (self.counts.sum(axis=1, keepdims=True) + alpha)

  @implicit_return
  def lookup(self, keys: Series, table: Matrix, fallback: Matrix) -> Matrix:
    """Return ``table`` rows for known keys and ``fallback`` rows (per input) for the rest."""
    index = {key: i for i, key in enumerate(self.keys)}
    positions = keys.astype(object).map(index).fillna(-1).astype(int).to_numpy()
    known = positions >= 0
    picked = table[np.clip(positions, 0, None)] if len(self.keys) else fallback
    np.where(known[:, None], picked, fallback)

  @implicit_return
  def pairs(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(keys, label indices, weights)`` for every non-zero cell."""
    rows, cols = np.nonzero(self.counts)
    (np.asarray(self.keys, dtype=object)[rows], cols, self.counts[rows, cols])
