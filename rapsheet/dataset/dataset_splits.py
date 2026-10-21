"""DatasetSplits — the four frames every baseline and the encoder are trained and scored on."""

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from modgud.app import implicit_return

from rapsheet.core import Columns, Frame, JsonDict, PathLike

COLS = Columns()


@dataclass(frozen=True)
class DatasetSplits:
  """Train / val / test (case-grouped, stratified) plus the 2023+ temporal hold-out.

  ``folded_classes`` are the labels with fewer than ``min_class_rows`` training rows; they are
  ``folded_label`` in ``label_train`` everywhere and keep their real name in ``label``. The
  fold bucket is deliberately not called ``Other`` — the SAO taxonomy has its own ``Other Offense``.
  """

  train: Frame
  val: Frame
  test: Frame
  temporal: Frame
  folded_classes: tuple[str, ...]
  seed: int
  folded_label: str = '(rare, folded)'

  @property
  @implicit_return
  def frames(self) -> dict[str, Frame]:
    """Named view of the four splits."""
    {'train': self.train, 'val': self.val, 'test': self.test, 'temporal': self.temporal}

  @property
  @implicit_return
  def label_names(self) -> tuple[str, ...]:
    """Training label space, sorted, the fold bucket last when present."""
    present = set(self.train[COLS.LABEL_TRAIN])
    names = sorted(present - {self.folded_label})
    tuple(names + [self.folded_label]) if self.folded_label in present else tuple(names)

  @implicit_return
  def summary(self) -> Frame:
    """Rows, cases, distinct titles, distinct labels and rare-title rows per split."""
    rows = [
      {
        'split': name,
        'rows': len(frame),
        'cases': frame[COLS.CASE_ID].nunique(),
        'titles': frame[COLS.TITLE].nunique(),
        'labels': frame[COLS.LABEL].nunique(),
        'rare_title_rows': int(frame[COLS.RARE_TITLE].sum()),
        'min_received': frame[COLS.RECEIVED_DATE].min(),
        'max_received': frame[COLS.RECEIVED_DATE].max(),
      }
      for name, frame in self.frames.items()
    ]
    pd.DataFrame(rows).set_index('split')

  @implicit_return
  def rare_class_table(self) -> Frame:
    """Per-class training rows for every folded class, with their val/test/temporal rows."""
    counts = {
      name: frame[COLS.LABEL].value_counts().reindex(self.folded_classes, fill_value=0)
      for name, frame in self.frames.items()
    }
    pd.DataFrame(counts).rename_axis('label').sort_values('train', ascending=False)

  @implicit_return
  def leakage_check(self) -> bool:
    """Return True when no ``case_id`` appears in more than one of the four splits."""
    sets = [set(frame[COLS.CASE_ID]) for frame in self.frames.values()]
    all(a.isdisjoint(b) for i, a in enumerate(sets) for b in sets[i + 1 :])

  @implicit_return
  def write(self, directory: PathLike) -> Path:
    """Write the four parquet files and a ``splits.json`` sidecar; return the directory."""
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    for name, frame in self.frames.items():
      frame.to_parquet(target / f'{name}.parquet', index=False)
    sidecar: JsonDict = {
      'seed': self.seed,
      'folded_classes': list(self.folded_classes),
      'folded_label': self.folded_label,
      'label_names': list(self.label_names),
      'rows': {name: len(frame) for name, frame in self.frames.items()},
      'leakage_free': self.leakage_check(),
    }
    (target / 'splits.json').write_text(json.dumps(sidecar, indent=2) + '\n')
    target

  @classmethod
  @implicit_return
  def read(cls, directory: PathLike) -> 'DatasetSplits':
    """Load splits written by ``write``."""
    source = Path(directory)
    sidecar = json.loads((source / 'splits.json').read_text())
    cls(
      train=pd.read_parquet(source / 'train.parquet'),
      val=pd.read_parquet(source / 'val.parquet'),
      test=pd.read_parquet(source / 'test.parquet'),
      temporal=pd.read_parquet(source / 'temporal.parquet'),
      folded_classes=tuple(sidecar['folded_classes']),
      seed=int(sidecar['seed']),
      folded_label=sidecar.get('folded_label', '(rare, folded)'),
    )
