"""Type aliases shared across the package."""

from pathlib import Path
from typing import Any, Mapping, Protocol

import numpy as np
import pandas as pd

Frame = pd.DataFrame
Series = pd.Series
PathLike = str | Path
Row = dict[str, Any]
Rows = list[Row]
JsonDict = dict[str, Any]
ClassMap = Mapping[str, tuple[str, ...]]
Matrix = np.ndarray


class Classifier(Protocol):
  """What every baseline and the encoder expose: a name, ``fit``, and ``predict_proba``."""

  name: str

  def fit(self, train: Frame) -> 'Classifier':
    """Return a fitted copy."""
    ...

  def predict_proba(self, frame: Frame) -> Matrix:
    """Return one row per input, one column per training label, rows summing to 1."""
    ...
