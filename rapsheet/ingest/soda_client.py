"""SodaClient — paged SoQL pull of one Socrata resource into a parquet cache with a manifest."""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import requests
from modgud.app import implicit_return

from rapsheet.core import Frame, JsonDict, PathLike, Rows, RunManifest, SodaFetchError

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SodaClient:
  """Pull a SODA 2.1 resource page by page, ordered by ``:id`` so paging is stable.

  ``fetch_all`` returns a frame; ``pull`` additionally caches it as parquet beside a
  ``RunManifest`` and skips the network when the cache already exists.
  """

  domain: str = 'datacatalog.cookcountyil.gov'
  dataset_id: str = 'apwk-dzx8'
  app_token: str | None = None
  page_size: int = 50_000
  timeout: int = 120
  max_retries: int = 5
  backoff_seconds: float = 2.0
  session: requests.Session = field(default_factory=requests.Session, compare=False, repr=False)

  @property
  @implicit_return
  def endpoint(self) -> str:
    """Resource URL for the JSON representation."""
    f'https://{self.domain}/resource/{self.dataset_id}.json'

  @property
  @implicit_return
  def headers(self) -> dict[str, str]:
    """Request headers; the app token is only sent when one was given."""
    base = {'Accept': 'application/json', 'User-Agent': 'rapsheet/0.1 (+research)'}
    {**base, 'X-App-Token': self.app_token} if self.app_token else base

  @implicit_return
  def query(self, params: JsonDict) -> Rows:
    """Run one SoQL query with retries and exponential backoff; return the decoded rows."""
    attempt = 0
    rows: Rows | None = None
    last_error: Exception | None = None
    while rows is None and attempt < self.max_retries:
      attempt += 1
      try:
        response = self.session.get(
          self.endpoint, params=params, headers=self.headers, timeout=self.timeout
        )
        response.raise_for_status()
        rows = response.json()
      except (requests.RequestException, ValueError) as exc:
        last_error = exc
        wait = self.backoff_seconds * (2 ** (attempt - 1))
        log.warning(
          'SODA attempt %d/%d failed (%s); retrying in %.0fs', attempt, self.max_retries, exc, wait
        )
        time.sleep(wait)
    if rows is None:
      raise SodaFetchError(
        f'{self.endpoint} failed after {self.max_retries} attempts: {last_error}'
      )
    rows

  @implicit_return
  def count(self) -> int:
    """Row count the portal reports for the whole resource."""
    int(self.query({'$select': 'count(*) AS n'})[0]['n'])

  @implicit_return
  def updated_at(self) -> str | None:
    """Most recent ``:updated_at`` across rows — the dataset's effective freeze date."""
    rows = self.query({'$select': 'max(:updated_at) AS updated_at'})
    rows[0].get('updated_at') if rows else None

  @implicit_return
  def fetch_page(self, columns: tuple[str, ...], offset: int) -> Rows:
    """One page of ``page_size`` rows starting at ``offset``."""
    self.query(
      {
        '$select': ','.join(columns),
        '$limit': self.page_size,
        '$offset': offset,
        '$order': ':id',
      }
    )

  @implicit_return
  def fetch_all(self, columns: tuple[str, ...]) -> tuple[Frame, int]:
    """Every row of the resource as a frame, plus the number of pages fetched."""
    pages: list[Rows] = []
    offset = 0
    exhausted = False
    while not exhausted:
      page = self.fetch_page(columns, offset)
      pages.append(page)
      log.info('page %d: %d rows (offset %d)', len(pages), len(page), offset)
      offset += self.page_size
      exhausted = len(page) < self.page_size
    frame = pd.DataFrame([row for page in pages for row in page], columns=list(columns))
    (frame, len(pages))

  @implicit_return
  def pull(self, columns: tuple[str, ...], cache_dir: PathLike, force: bool = False) -> Path:
    """Fetch (or reuse) the cached parquet for ``columns``; write a manifest beside it.

    Returns the parquet path. The manifest lives at ``<stem>.manifest.json``.
    """
    cache = Path(cache_dir)
    parquet_path = cache / f'{self.dataset_id}.parquet'
    manifest_path = cache / f'{self.dataset_id}.manifest.json'
    cached = parquet_path.exists() and manifest_path.exists() and not force
    if cached:
      log.info('cache hit: %s', parquet_path)
    else:
      cache.mkdir(parents=True, exist_ok=True)
      reported = self.count()
      updated = self.updated_at()
      frame, page_count = self.fetch_all(columns)
      frame.to_parquet(parquet_path, index=False)
      RunManifest(
        dataset_id=self.dataset_id,
        endpoint=self.endpoint,
        query={'$select': ','.join(columns), '$order': ':id', '$limit': self.page_size},
        columns=columns,
        source_updated_at=updated,
        row_count_reported=reported,
        row_count_fetched=len(frame),
        page_count=page_count,
      ).write(manifest_path)
      log.info('wrote %s (%d rows, %d pages)', parquet_path, len(frame), page_count)
    parquet_path
