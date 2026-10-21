"""RunManifest — the provenance record every number in the article traces back to."""

import json
import platform
import subprocess
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from modgud.app import implicit_return

from .types import JsonDict, PathLike


@dataclass(frozen=True)
class RunManifest:
  """What was pulled, when, from where, with which query — plus seeds and config as they accrue.

  Immutable: ``with_extra`` returns a new manifest. Written as JSON next to the parquet it
  describes, and later next to the trained model.
  """

  dataset_id: str
  endpoint: str
  query: JsonDict
  columns: tuple[str, ...]
  source_updated_at: str | None
  row_count_reported: int | None
  row_count_fetched: int
  page_count: int
  retrieved_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
  git_sha: str | None = field(default_factory=lambda: RunManifest.current_git_sha())
  hardware: str = field(default_factory=lambda: RunManifest.current_hardware())
  seeds: JsonDict = field(default_factory=dict)
  config: JsonDict = field(default_factory=dict)
  extra: JsonDict = field(default_factory=dict)

  @staticmethod
  @implicit_return
  def current_git_sha() -> str | None:
    """Short SHA of HEAD if we are inside a git checkout, else ``None``."""
    try:
      sha = subprocess.run(
        ['git', 'rev-parse', '--short', 'HEAD'],
        capture_output=True,
        text=True,
        timeout=5,
        check=True,
      ).stdout.strip()
      sha or None
    except (OSError, subprocess.SubprocessError):
      None

  @staticmethod
  @implicit_return
  def current_hardware() -> str:
    """One line describing the machine, e.g. ``Darwin arm64 (arm) / Python 3.12.4``."""
    processor = platform.processor() or 'n/a'
    f'{platform.system()} {platform.machine()} ({processor}) / Python {platform.python_version()}'

  @implicit_return
  def with_extra(self, **values: Any) -> 'RunManifest':
    """Return a copy with ``values`` merged into ``extra``."""
    replace(self, extra={**self.extra, **values})

  @implicit_return
  def with_seeds(self, **seeds: int) -> 'RunManifest':
    """Return a copy with ``seeds`` merged in."""
    replace(self, seeds={**self.seeds, **seeds})

  @implicit_return
  def with_config(self, **config: Any) -> 'RunManifest':
    """Return a copy with ``config`` merged in."""
    replace(self, config={**self.config, **config})

  @implicit_return
  def to_dict(self) -> JsonDict:
    """Plain dict, lists in place of tuples, ready for ``json.dumps``."""
    {**asdict(self), 'columns': list(self.columns)}

  @implicit_return
  def write(self, path: PathLike) -> Path:
    """Write the manifest as indented JSON and return the path."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + '\n')
    target

  @classmethod
  @implicit_return
  def read(cls, path: PathLike) -> 'RunManifest':
    """Load a manifest written by ``write``."""
    raw = json.loads(Path(path).read_text())
    cls(**{**raw, 'columns': tuple(raw['columns'])})
