"""TemperatureCalibrator — one number that makes the model's confidence mean what it says."""

import math
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from modgud.app import implicit_return

from rapsheet.core import Frame, JsonDict, Matrix

GOLDEN = (math.sqrt(5) - 1) / 2


@dataclass(frozen=True)
class TemperatureCalibrator:
  """Temperature scaling: probabilities become ``softmax(logits / T)`` for one fitted ``T``.

  - **What moves:** confidence only. Dividing every logit by the same positive number never changes
    which class is largest, so accuracy is untouched. ``T > 1`` softens an overconfident model.
  - **Fit:** minimize validation negative log-likelihood over ``log(1/T)`` by golden-section search.
    The NLL is convex in ``1/T``, so the search can't get stuck, and numpy is all it needs.
  - **Labels of -1** (a true label outside the training space) are left out of the NLL, which
    can't score them, and counted as errors everywhere else, because the model can't be right.
  - **ECE** (expected calibration error) is the row-weighted gap between confidence and accuracy
    across ``bins`` equal-width confidence bins.
  """

  temperature: float = 1.0
  bins: int = 15
  log_inverse_bounds: tuple[float, float] = (-3.0, 3.0)
  iterations: int = 80

  @staticmethod
  @implicit_return
  def softmax(logits: Matrix) -> Matrix:
    """Return row-wise softmax, computed stably."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    exp / exp.sum(axis=1, keepdims=True)

  @staticmethod
  @implicit_return
  def nll_at(logits: Matrix, labels: np.ndarray, inverse_temperature: float) -> float:
    """Return mean NLL of the labeled rows at ``1/T = inverse_temperature`` (NaN if none)."""
    known = labels >= 0
    scaled = logits[known].astype(np.float64) * inverse_temperature
    top = scaled.max(axis=1, keepdims=True) if known.any() else np.zeros((0, 1))
    log_norm = top[:, 0] + np.log(np.exp(scaled - top).sum(axis=1))
    losses = log_norm - scaled[np.arange(known.sum()), labels[known]]
    float(losses.mean()) if known.any() else float('nan')

  @implicit_return
  def fit(self, logits: Matrix, labels: np.ndarray) -> 'TemperatureCalibrator':
    """Return a copy whose ``temperature`` minimizes NLL on (validation) ``logits``."""
    low, high = self.log_inverse_bounds
    for _ in range(self.iterations):
      left = high - GOLDEN * (high - low)
      right = low + GOLDEN * (high - low)
      left_nll = self.nll_at(logits, labels, math.exp(left))
      right_nll = self.nll_at(logits, labels, math.exp(right))
      low, high = (low, right) if left_nll < right_nll else (left, high)
    replace(self, temperature=round(1.0 / math.exp((low + high) / 2), 6))

  @implicit_return
  def probabilities(self, logits: Matrix) -> Matrix:
    """Return calibrated probabilities for ``logits``."""
    self.softmax(logits.astype(np.float64) / self.temperature)

  @implicit_return
  def nll(self, logits: Matrix, labels: np.ndarray) -> float:
    """Return mean NLL of the labeled rows at this temperature."""
    self.nll_at(logits, labels, 1.0 / self.temperature)

  @implicit_return
  def reliability(self, proba: Matrix, labels: np.ndarray) -> Frame:
    """Return per-bin mean confidence, accuracy and row count (empty bins dropped)."""
    confidence = proba.max(axis=1)
    correct = proba.argmax(axis=1) == labels
    edges = np.linspace(0.0, 1.0, self.bins + 1)
    which = np.clip(np.digitize(confidence, edges[1:-1], right=True), 0, self.bins - 1)
    frame = pd.DataFrame({'bin': which, 'confidence': confidence, 'accuracy': correct})
    table = frame.groupby('bin').agg(
      confidence=('confidence', 'mean'), accuracy=('accuracy', 'mean'), rows=('accuracy', 'size')
    )
    table.assign(low=edges[table.index], high=edges[table.index + 1])

  @implicit_return
  def ece(self, proba: Matrix, labels: np.ndarray) -> float:
    """Return expected calibration error over ``bins`` equal-width bins."""
    table = self.reliability(proba, labels)
    gaps = (table['accuracy'] - table['confidence']).abs()
    float((gaps * table['rows']).sum() / table['rows'].sum())

  @implicit_return
  def report(self, logits: Matrix, labels: np.ndarray) -> JsonDict:
    """Return NLL and ECE before (T = 1) and after (this T) for one split."""
    raw = self.softmax(logits.astype(np.float64))
    after = self.probabilities(logits)
    {
      'rows': int(len(labels)),
      'nll_before': round(self.nll_at(logits, labels, 1.0), 4),
      'nll_after': round(self.nll(logits, labels), 4),
      'ece_before': round(self.ece(raw, labels), 4),
      'ece_after': round(self.ece(after, labels), 4),
      'mean_confidence_before': round(float(raw.max(axis=1).mean()), 4),
      'mean_confidence_after': round(float(after.max(axis=1).mean()), 4),
      'accuracy': round(float((after.argmax(axis=1) == labels).mean()), 4),
    }
