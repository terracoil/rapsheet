"""HeadlineMetrics — the numbers in the article's first table, scored the same way for every model."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from modgud.app import implicit_return
from sklearn.metrics import f1_score

from rapsheet.core import Classifier, Columns, Frame, JsonDict, Matrix
from rapsheet.dataset import DatasetSplits, LabelSpace

COLS = Columns()


@dataclass(frozen=True)
class HeadlineMetrics:
  """Accuracy, macro-F1, rare-class recall and rare-title accuracy against ``label_train``.

  - **macro-F1** averages over the labels present in the evaluated split, so a class that never
    appears in the 2023+ set doesn't count as a zero.
  - **rare-class recall** is the mean recall over named classes with fewer than
    ``rare_max_train_rows`` training rows (the fold bucket excluded) that appear in the split.
  - **rare-title accuracy** is accuracy on rows whose title appeared < 5 times in training.
  """

  space: LabelSpace
  rare_max_train_rows: int = 500

  @implicit_return
  def score(self, model: str, split: str, frame: Frame, proba: Matrix) -> JsonDict:
    """Return one headline row for one model on one split."""
    truth = self.space.encode(frame[COLS.LABEL_TRAIN])
    pred = proba.argmax(axis=1)
    present = np.unique(truth[truth >= 0])
    rare = [c for c in self.space.rare_classes(self.rare_max_train_rows) if (truth == c).any()]
    recalls = [float((pred[truth == c] == c).mean()) for c in rare]
    rare_title = frame[COLS.RARE_TITLE].to_numpy(dtype=bool)
    {
      'model': model,
      'split': split,
      'rows': len(frame),
      'accuracy': round(float((pred == truth).mean()), 4),
      'macro_f1': round(
        float(f1_score(truth, pred, labels=present, average='macro', zero_division=0)), 4
      ),
      'rare_class_recall': round(float(np.mean(recalls)), 4) if recalls else float('nan'),
      'rare_classes_scored': len(rare),
      'rare_title_rows': int(rare_title.sum()),
      'rare_title_accuracy': (
        round(float((pred[rare_title] == truth[rare_title]).mean()), 4)
        if rare_title.any()
        else float('nan')
      ),
    }

  @implicit_return
  def table(
    self,
    models: list[Classifier],
    splits: DatasetSplits,
    names: tuple[str, ...] = ('test', 'temporal'),
  ) -> Frame:
    """Score every fitted model on every named split; one row per (model, split)."""
    rows = [
      self.score(model.name, name, splits.frames[name], model.predict_proba(splits.frames[name]))
      for model in models
      for name in names
    ]
    pd.DataFrame(rows).set_index(['model', 'split'])
