"""SelectiveEvaluator — which calls the model may make alone, class by class, and at what cost."""

import math
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from modgud.app import implicit_return

from rapsheet.core import Frame, JsonDict, Matrix
from rapsheet.dataset import LabelSpace


@dataclass(frozen=True)
class SelectiveEvaluator:
  """Risk–coverage curves and per-class confidence thresholds under an error budget.

  - **Selective risk** is the error rate among the rows the model decides alone; **coverage** is
    the share of rows it decides. Sorting by confidence and walking down gives the risk–coverage
    curve; **AURC** is the mean risk along it (lower is better).
  - **Earning a threshold:** for each predicted class, ``tau`` is the lowest confidence at which
    the *upper* end of a one-sided Wilson interval on validation errors stays within
    ``error_budget``. Zero errors in 30 rows doesn't prove 0.5%; zero in about 540 does. A class
    that can't show that on validation has no threshold, and every one of its calls goes to a
    person. The fold bucket never gets one: it isn't a category anyone could act on.
  - **Global threshold:** the same rule with one ``tau`` for all classes, for comparison.
  - Thresholds are chosen on validation and scored on test, temporal and the PROMIS probe, which
    never influence them. Labels of -1 count as errors.
  """

  space: LabelSpace
  error_budget: float = 0.005
  z: float = 1.645
  curve_points: int = 200

  @implicit_return
  def wilson_upper(self, errors: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Return the one-sided Wilson upper bound on the error rate (vectorized)."""
    n = np.maximum(rows, 1).astype(float)
    p = errors / n
    z2 = self.z**2
    centre = (p + z2 / (2 * n)) / (1 + z2 / n)
    margin = self.z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    centre + margin

  @implicit_return
  def earned_threshold(self, confidence: np.ndarray, wrong: np.ndarray) -> float:
    """Return the lowest confidence that keeps the error bound within budget (inf if none)."""
    order = np.argsort(-confidence, kind='stable')
    ranked = confidence[order]
    upper = self.wilson_upper(np.cumsum(wrong[order]), np.arange(1, len(ranked) + 1))
    boundary = np.append(ranked[1:] < ranked[:-1], True) if len(ranked) else np.zeros(0, bool)
    passing = np.nonzero((upper <= self.error_budget) & boundary)[0]
    float(ranked[passing[-1]]) if len(passing) else math.inf

  @implicit_return
  def class_thresholds(self, proba: Matrix, labels: np.ndarray) -> tuple[float, ...]:
    """Return one ``tau`` per class from validation probabilities (inf = never autonomous)."""
    predicted = proba.argmax(axis=1)
    confidence = proba.max(axis=1)
    wrong = predicted != labels
    folded = self.space.names.index(self.space.folded_label)
    tuple(
      math.inf
      if c == folded
      else self.earned_threshold(confidence[predicted == c], wrong[predicted == c])
      for c in range(self.space.size)
    )

  @implicit_return
  def global_threshold(self, proba: Matrix, labels: np.ndarray) -> float:
    """Return one ``tau`` for every class, earned the same way."""
    self.earned_threshold(proba.max(axis=1), proba.argmax(axis=1) != labels)

  @implicit_return
  def autonomous(self, proba: Matrix, thresholds: tuple[float, ...] | float) -> np.ndarray:
    """Return a mask of the rows the model decides alone under ``thresholds``."""
    tau = np.broadcast_to(np.asarray(thresholds, dtype=float), (self.space.size,))
    np.greater_equal(proba.max(axis=1), tau[proba.argmax(axis=1)])

  @implicit_return
  def risk_coverage(self, proba: Matrix, labels: np.ndarray) -> Frame:
    """Return the risk–coverage curve, thinned to about ``curve_points`` points."""
    confidence = proba.max(axis=1)
    order = np.argsort(-confidence, kind='stable')
    wrong = (proba.argmax(axis=1) != labels)[order]
    accepted = np.arange(1, len(order) + 1)
    risk = np.cumsum(wrong) / accepted
    keep = np.unique(np.linspace(0, len(order) - 1, self.curve_points).astype(int))
    pd.DataFrame(
      {
        'coverage': accepted[keep] / len(order),
        'risk': risk[keep],
        'threshold': confidence[order][keep],
      }
    )

  @implicit_return
  def aurc(self, proba: Matrix, labels: np.ndarray) -> float:
    """Return the area under the risk–coverage curve (mean selective risk)."""
    order = np.argsort(-proba.max(axis=1), kind='stable')
    wrong = (proba.argmax(axis=1) != labels)[order]
    float((np.cumsum(wrong) / np.arange(1, len(order) + 1)).mean())

  @implicit_return
  def summary(
    self,
    proba: Matrix,
    labels: np.ndarray,
    thresholds: tuple[float, ...],
    global_tau: float,
  ) -> JsonDict:
    """Return coverage and error for the per-class and global policies on one split."""
    wrong = proba.argmax(axis=1) != labels
    labeled = bool((labels >= 0).any())
    per_class = self.autonomous(proba, thresholds)
    single = self.autonomous(proba, global_tau)
    {
      'rows': int(len(labels)),
      'per_class_coverage': round(float(per_class.mean()), 4),
      'per_class_error': round(float(wrong[per_class].mean()), 4)
      if labeled and per_class.any()
      else float('nan'),
      'global_coverage': round(float(single.mean()), 4),
      'global_error': round(float(wrong[single].mean()), 4)
      if labeled and single.any()
      else float('nan'),
      'aurc': round(self.aurc(proba, labels), 4) if labeled else float('nan'),
    }

  @implicit_return
  def autonomy_table(
    self,
    val: tuple[Matrix, np.ndarray],
    thresholds: tuple[float, ...],
    evaluated: dict[str, tuple[Matrix, np.ndarray]],
  ) -> Frame:
    """Return one row per class: its threshold and what it decided alone on each split.

    Unlabeled splits (the PROMIS probe) get ``_predicted`` and ``_alone`` columns but no
    ``_errors``: without labels there is nothing to count.
    """
    val_predicted = val[0].argmax(axis=1)
    columns: dict[str, list] = {
      'label': list(self.space.names),
      'train_rows': list(self.space.train_counts),
      'val_predicted': [int((val_predicted == c).sum()) for c in range(self.space.size)],
      'tau': [round(t, 4) if math.isfinite(t) else float('nan') for t in thresholds],
    }
    for name, (proba, labels) in evaluated.items():
      predicted = proba.argmax(axis=1)
      alone = self.autonomous(proba, thresholds)
      wrong = predicted != labels
      columns[f'{name}_predicted'] = [int((predicted == c).sum()) for c in range(self.space.size)]
      columns[f'{name}_alone'] = [
        int((alone & (predicted == c)).sum()) for c in range(self.space.size)
      ]
      if (labels >= 0).any():  # an unlabeled probe has coverage, not errors
        columns[f'{name}_errors'] = [
          int((alone & wrong & (predicted == c)).sum()) for c in range(self.space.size)
        ]
    table = pd.DataFrame(columns).set_index('label')
    table.sort_values('train_rows', ascending=False)

  @implicit_return
  def budget_sweep(
    self,
    val: tuple[Matrix, np.ndarray],
    evaluated: dict[str, tuple[Matrix, np.ndarray]],
    budgets: tuple[float, ...] = (0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.05),
  ) -> Frame:
    """Return coverage and error for both policies at each error budget, thresholds from ``val``."""
    rows: list[JsonDict] = []
    for budget in budgets:
      evaluator = replace(self, error_budget=budget)
      thresholds = evaluator.class_thresholds(*val)
      global_tau = evaluator.global_threshold(*val)
      earned = sum(math.isfinite(t) for t in thresholds)
      rows.extend(
        {
          'budget': budget,
          'split': name,
          'classes_earned': earned,
          **evaluator.summary(proba, labels, thresholds, global_tau),
        }
        for name, (proba, labels) in evaluated.items()
      )
    pd.DataFrame(rows).set_index(['budget', 'split'])
