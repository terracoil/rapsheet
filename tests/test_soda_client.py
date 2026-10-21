import json
from dataclasses import dataclass

import pytest
import requests
from rapsheet.core import Columns, RunManifest, SodaFetchError
from rapsheet.ingest import SodaClient

COLS = Columns()


@dataclass
class FakeResponse:
  payload: object
  status: int = 200

  def raise_for_status(self):
    if self.status >= 400:
      raise requests.HTTPError(f'{self.status}')

  def json(self):
    return self.payload


class FakeSession:
  """Serves count, updated_at and two pages of rows; records every request."""

  def __init__(self, rows, page_size, failures=0):
    self.rows, self.page_size, self.failures = rows, page_size, failures
    self.calls = []

  def get(self, url, params=None, headers=None, timeout=None):
    self.calls.append({'url': url, 'params': params, 'headers': headers})
    if self.failures:
      self.failures -= 1
      return FakeResponse(None, status=503)
    select = params['$select']
    if select.startswith('count('):
      return FakeResponse([{'n': str(len(self.rows))}])
    if select.startswith('max('):
      return FakeResponse([{'updated_at': '2024-12-30T12:00:00.000'}])
    assert params['$order'] == ':id'
    offset = params['$offset']
    return FakeResponse(self.rows[offset : offset + self.page_size])


@pytest.fixture
def rows(raw_frame):
  return json.loads(raw_frame.to_json(orient='records'))


def test_fetch_all_pages_until_short_page(rows):
  session = FakeSession(rows, page_size=300)
  client = SodaClient(page_size=300, session=session, app_token='tok', backoff_seconds=0)
  frame, pages = client.fetch_all(COLS.used)
  assert len(frame) == len(rows)
  assert pages == len(rows) // 300 + 1
  assert list(frame.columns) == list(COLS.used)
  assert all(c['headers']['X-App-Token'] == 'tok' for c in session.calls)


def test_pull_writes_parquet_and_manifest_then_hits_cache(rows, tmp_path):
  session = FakeSession(rows, page_size=500)
  client = SodaClient(page_size=500, session=session, backoff_seconds=0)
  path = client.pull(COLS.used, tmp_path)
  assert path.exists()
  manifest = RunManifest.read(tmp_path / 'apwk-dzx8.manifest.json')
  assert manifest.row_count_fetched == len(rows)
  assert manifest.row_count_reported == len(rows)
  assert manifest.source_updated_at == '2024-12-30T12:00:00.000'
  assert manifest.query['$order'] == ':id'
  assert tuple(manifest.columns) == COLS.used
  n_calls = len(session.calls)
  client.pull(COLS.used, tmp_path)
  assert len(session.calls) == n_calls  # cache hit → no network


def test_query_retries_then_raises(rows):
  session = FakeSession(rows, page_size=10, failures=99)
  client = SodaClient(session=session, max_retries=2, backoff_seconds=0)
  with pytest.raises(SodaFetchError, match='after 2 attempts'):
    client.count()
  assert len(session.calls) == 2


def test_manifest_roundtrip_and_immutability(tmp_path):
  m = RunManifest('ds', 'ep', {'$order': ':id'}, ('a', 'b'), None, 10, 10, 1)
  m2 = m.with_seeds(split=42).with_config(epochs=2).with_extra(note='x')
  assert m.seeds == {} and m2.seeds == {'split': 42}
  path = m2.write(tmp_path / 'm.json')
  back = RunManifest.read(path)
  assert back.columns == ('a', 'b') and back.config == {'epochs': 2} and back.extra == {'note': 'x'}
