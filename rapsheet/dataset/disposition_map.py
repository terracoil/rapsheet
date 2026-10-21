"""DispositionMap — the versioned lookup from raw ``charge_disposition`` strings to outcome classes."""

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import pandas as pd
import yaml
from modgud.app import implicit_return

from rapsheet.core import ClassMap, DispositionMapError, Frame, PathLike, Series


@dataclass(frozen=True)
class DispositionMap:
  """A closed vocabulary → class lookup, loaded from ``disposition_map.yaml``.

  Validated on load: every raw string belongs to exactly one class. Lookups never guess —
  an unknown string maps to ``None`` and is surfaced by ``unmapped``.
  """

  version: int
  classes: ClassMap
  review_notes: MappingProxyType[str, str]

  @classmethod
  @implicit_return
  def from_yaml(cls, path: PathLike) -> 'DispositionMap':
    """Load and validate a map file; raise ``DispositionMapError`` on any inconsistency."""
    raw = yaml.safe_load(Path(path).read_text())
    classes = {name: tuple(strings) for name, strings in raw.get('classes', {}).items()}
    duplicates = cls._duplicates(classes)
    if 'version' not in raw:
      raise DispositionMapError(f'{path}: missing `version`')
    if not classes:
      raise DispositionMapError(f'{path}: no classes defined')
    if duplicates:
      raise DispositionMapError(f'{path}: strings in more than one class: {sorted(duplicates)}')
    cls(
      version=int(raw['version']),
      classes=MappingProxyType(classes),
      review_notes=MappingProxyType(dict(raw.get('review_notes', {}))),
    )

  @staticmethod
  @implicit_return
  def _duplicates(classes: ClassMap) -> set[str]:
    """Raw strings that appear under more than one class."""
    seen: set[str] = set()
    dupes: set[str] = set()
    for strings in classes.values():
      for raw in strings:
        dupes.add(raw) if raw in seen else seen.add(raw)
    dupes

  @property
  @implicit_return
  def lookup(self) -> MappingProxyType[str, str]:
    """Flat ``raw string → class`` view."""
    MappingProxyType({raw: name for name, strings in self.classes.items() for raw in strings})

  @property
  @implicit_return
  def vocabulary(self) -> tuple[str, ...]:
    """Every raw string the map knows, in file order."""
    tuple(self.lookup)

  @implicit_return
  def map_series(self, dispositions: Series) -> Series:
    """Map a series of raw strings to classes; unknown strings become ``None``."""
    mapped = dispositions.map(self.lookup)
    mapped.astype('object').where(mapped.notna(), None)

  @implicit_return
  def unmapped(self, dispositions: Series) -> Series:
    """Count raw strings the map does not know, most frequent first."""
    unknown = dispositions[~dispositions.isin(self.vocabulary)]
    unknown.value_counts(dropna=False)

  @implicit_return
  def coverage(self, dispositions: Series) -> Frame:
    """Rows and share per class for ``dispositions`` — the table behind the cold open."""
    mapped = self.map_series(dispositions).fillna('UNMAPPED')
    counts = mapped.value_counts()
    pd.DataFrame({'rows': counts, 'share': (counts / counts.sum()).round(4)})

  @implicit_return
  def flagged(self) -> Frame:
    """Strings carrying a review note, with the class they currently sit under."""
    rows = [
      {'raw': raw, 'class': self.lookup.get(raw), 'note': note}
      for raw, note in self.review_notes.items()
    ]
    pd.DataFrame(rows, columns=['raw', 'class', 'note'])
