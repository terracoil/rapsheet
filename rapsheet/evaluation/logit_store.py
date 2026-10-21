"""LogitStore — saved model outputs, so calibration and abstention never have to retrain."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from modgud.app import implicit_return

from rapsheet.core import Columns, Frame, Matrix, PathLike
from rapsheet.dataset import LabelSpace

COLS = Columns()


@dataclass(frozen=True)
class LogitStore:
  """One ``<split>.npz`` per evaluated frame: ``logits``, ``labels`` and ``rare_title``.

  Labels are ``LabelSpace`` indices (-1 where a row has no training label, as on the PROMIS probe),
  so a saved file is self-contained: calibration and selective evaluation need nothing else.
  """

  directory: Path

  def save(self, name: str, frame: Frame, logits: Matrix, space: LabelSpace) -> None:
    """Write ``<name>.npz``: logits plus the labels and rare-title flags of ``frame``."""
    self.directory.mkdir(parents=True, exist_ok=True)
    labels = (
      space.encode(frame[COLS.LABEL_TRAIN])
      if COLS.LABEL_TRAIN in frame.columns
      else np.full(len(frame), -1, dtype=int)
    )
    rare_title = (
      frame[COLS.RARE_TITLE].to_numpy(dtype=bool)
      if COLS.RARE_TITLE in frame.columns
      else np.zeros(len(frame), dtype=bool)
    )
    np.savez_compressed(
      self.directory / f'{name}.npz',
      logits=logits.astype(np.float32),
      labels=labels,
      rare_title=rare_title,
    )

  @implicit_return
  def load(self, name: str) -> dict[str, np.ndarray]:
    """Return the arrays saved under ``name``."""
    with np.load(self.directory / f'{name}.npz') as data:
      arrays = {key: data[key] for key in data.files}
    arrays

  @implicit_return
  def names(self) -> tuple[str, ...]:
    """Return the saved split names, sorted."""
    tuple(sorted(p.stem for p in self.directory.glob('*.npz')))

  @classmethod
  @implicit_return
  def at(cls, directory: PathLike) -> 'LogitStore':
    """Return a store rooted at ``directory``."""
    cls(directory=Path(directory))
