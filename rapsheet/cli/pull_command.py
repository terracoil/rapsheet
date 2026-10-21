"""PullCommand — ``rapsheet-pull``: fetch the table, cache parquet + manifest, print the group-by."""

import argparse
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from modgud.app import implicit_return

from rapsheet.core import Columns, RunManifest
from rapsheet.dataset import DispositionMap, DispositionsDataset
from rapsheet.ingest import SodaClient

COLS = Columns()


@dataclass(frozen=True)
class PullCommand:
  """Command-line entry point. ``SODA_APP_TOKEN`` in the environment avoids throttling."""

  cache_dir: Path = Path('data/raw')
  map_path: Path = Path('data/disposition_map.yaml')
  force: bool = False
  app_token: str | None = None

  @staticmethod
  @implicit_return
  def parser() -> argparse.ArgumentParser:
    """Argument parser for the console script."""
    parser = argparse.ArgumentParser(
      prog='rapsheet-pull',
      description='Pull Cook County SAO Dispositions (apwk-dzx8) into data/raw as parquet.',
    )
    parser.add_argument('--cache-dir', default='data/raw', help='where parquet + manifest go')
    parser.add_argument('--map', default='data/disposition_map.yaml', help='disposition map YAML')
    parser.add_argument('--force', action='store_true', help='ignore an existing cache')
    parser.add_argument(
      '--token', default=None, help='Socrata app token (default: $SODA_APP_TOKEN)'
    )
    parser

  @classmethod
  def main(cls, argv: list[str] | None = None) -> None:
    """Console-script entry: parse the arguments and run.

    Also callable from the notebook, e.g. ``PullCommand.main(['--force', '--cache-dir', ...])``.
    Pass the paths there: the defaults are relative to the project root, and a notebook runs from
    ``notebooks/``. Only ``rapsheet``'s own loggers go to INFO; the root logger stays at WARNING,
    so a call inside a kernel doesn't switch on every library's chatter for the rest of the session.
    """
    args = cls.parser().parse_args(argv)
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s')
    logging.getLogger('rapsheet').setLevel(logging.INFO)
    cls(
      cache_dir=Path(args.cache_dir),
      map_path=Path(args.map),
      force=args.force,
      app_token=args.token or os.environ.get('SODA_APP_TOKEN'),
    ).run()

  def run(self) -> None:
    """Pull, then print the two tables the Monday checkpoint needs."""
    client = SodaClient(app_token=self.app_token)
    parquet_path = client.pull(COLS.used, self.cache_dir, force=self.force)
    manifest = RunManifest.read(self.cache_dir / f'{client.dataset_id}.manifest.json')
    dataset = DispositionsDataset()
    raw = dataset.load(parquet_path)
    cleaned = dataset.clean(raw)
    disposition_map = DispositionMap.from_yaml(self.map_path)
    unmapped = disposition_map.unmapped(raw[COLS.DISPOSITION])
    with pd.option_context('display.max_rows', 200, 'display.width', 120):
      print(f'\n== manifest ==\n{manifest.to_dict()}')
      print(f'\n== cleaning ==\n{cleaned.attrs["cleaning"]}')
      print(f'\n== updated_offense_category ({cleaned[COLS.LABEL].nunique()} values) ==')
      print(dataset.category_counts(cleaned))
      print(f'\n== disposition_map v{disposition_map.version} coverage (raw rows) ==')
      print(disposition_map.coverage(raw[COLS.DISPOSITION]))
      print(f'\n== unmapped dispositions ==\n{unmapped if len(unmapped) else "none"}')
