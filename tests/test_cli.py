"""PullCommand: the console script, and the same call from a notebook cell."""

import logging
import shutil
from pathlib import Path

import pytest
from rapsheet.cli import PullCommand
from rapsheet.core import RunManifest
from rapsheet.ingest import SodaClient


@pytest.fixture
def fake_pull(monkeypatch, raw_parquet):
  """Swap the portal for the fixture parquet; record what PullCommand asked for."""
  calls = []

  def pull(self, columns, cache_dir, force=False):
    cache = Path(cache_dir)
    calls.append((self.app_token, cache, force))
    shutil.copy(raw_parquet, cache / f'{self.dataset_id}.parquet')
    RunManifest(
      dataset_id=self.dataset_id,
      endpoint=self.endpoint,
      query={},
      columns=tuple(columns),
      source_updated_at=None,
      row_count_reported=None,
      row_count_fetched=0,
      page_count=1,
    ).write(cache / f'{self.dataset_id}.manifest.json')
    return cache / f'{self.dataset_id}.parquet'

  # Patch the class, not the module global: modgud compiles methods against their own globals.
  monkeypatch.setattr(SodaClient, 'pull', pull)
  monkeypatch.setenv('SODA_APP_TOKEN', 'from-env')
  return calls


@pytest.fixture
def restore_logging():
  root, package = logging.getLogger(), logging.getLogger('rapsheet')
  saved = (root.level, list(root.handlers), package.level)
  yield
  root.setLevel(saved[0])
  root.handlers[:] = saved[1]
  package.setLevel(saved[2])


@pytest.mark.parametrize(('flags', 'force'), [([], False), (['--force'], True)])
def test_main_forwards_force_and_explicit_paths(
  tmp_path, map_path, fake_pull, restore_logging, capsys, flags, force
):
  cache = tmp_path / 'raw'
  cache.mkdir()
  PullCommand.main([*flags, '--cache-dir', str(cache), '--map', str(map_path)])
  assert fake_pull == [('from-env', cache, force)]
  assert '== disposition_map v' in capsys.readouterr().out


def test_main_leaves_the_root_logger_alone(tmp_path, map_path, fake_pull, restore_logging):
  before = logging.getLogger().level
  PullCommand.main(['--cache-dir', str(tmp_path), '--map', str(map_path)])
  assert logging.getLogger().level == before
  assert logging.getLogger('rapsheet').level == logging.INFO
