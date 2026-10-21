"""Synthetic dispositions frame with the real column names, shapes and quirks of apwk-dzx8."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from rapsheet.core import Columns

COLS = Columns()
HUB_USER = 'someone'  # the fixture namespace; every test reads it from the environment

TITLES_BY_LABEL = {
  'Retail Theft': [
    'RET THEFT/DISP MERCH/<$300/PRECONV',
    'RETAIL THEFT',
    'RET THEFT/DISP MERCH/>$300',
  ],
  'Narcotics': [
    'POSS AMT CON SUB EXCEPT(A)/(D)',
    'MFG/DEL CANNABIS/10-30 GRAMS',
    'PCS/DEL 1<15 GR COCAINE/ANLG',
  ],
  'UUW - Unlawful Use of Weapon': ['AGG UUW/VEH/PREV CONVICTION', 'AGG UUW/PERSON/LOADED/NO FCCA'],
  'Aggravated Battery': ['AGG BATTERY/GREAT BODILY HARM', 'AGG BATTERY/PEACE OFFICER'],
  'Burglary': ['BURGLARY', 'RESIDENTIAL BURGLARY'],
  'DUI': ['DUI/ALCOHOL', 'AGG DUI/NO VALID DL'],
  'Homicide': ['FIRST DEGREE MURDER'],
  'Arson': ['ARSON/REAL PROPERTY'],
}
# Rare labels: fewer than 20 rows → folded into the rare bucket for training.
RARE_LABELS = {'Arson': 6, 'Homicide': 12}
DISPOSITIONS = [
  'Nolle Prosecution',
  'Plea Of Guilty',
  'FNG',
  'Finding Guilty',
  'Finding Not Not Guilty',
  'BFW',
  'Charge Vacated',
]


def build_frame(seed: int = 7, n_cases: int = 400) -> pd.DataFrame:
  rng = np.random.default_rng(seed)
  labels = [lbl for lbl in TITLES_BY_LABEL if lbl not in RARE_LABELS]
  rows = []
  charge_id = 1000
  for case in range(n_cases):
    case_id = f'C{case:05d}'
    received = pd.Timestamp('2012-01-01') + pd.Timedelta(days=int(rng.integers(0, 4400)))
    n_charges = int(rng.integers(1, 4))
    for k in range(n_charges):
      label = rng.choice(labels)
      title = rng.choice(TITLES_BY_LABEL[label])
      charge_id += 1
      n_versions = int(rng.integers(1, 3))
      for version in range(1, n_versions + 1):
        rows.append(
          {
            COLS.CASE_ID: case_id,
            COLS.PARTICIPANT_ID: f'{case_id}-P1',
            COLS.CHARGE_ID: str(charge_id),
            COLS.CHARGE_VERSION_ID: str(version),
            COLS.PRIMARY_CHARGE: k == 0,
            COLS.TITLE: f'  {title.lower()}  ' if version == 1 else title,
            COLS.CHAPTER: '720',
            COLS.ACT: '5',
            COLS.SECTION: f'{rng.integers(1, 40)}-{rng.integers(1, 9)}',
            COLS.CLASS: rng.choice(['X', '1', '2', '3', '4', 'M']),
            COLS.DISPOSITION: rng.choice(DISPOSITIONS),
            COLS.LABEL: label,
            COLS.RECEIVED_DATE: received.strftime('%Y-%m-%dT00:00:00.000'),
            COLS.DISPOSITION_DATE: (received + pd.Timedelta(days=200)).strftime(
              '%Y-%m-%dT00:00:00.000'
            ),
            COLS.COURT_FACILITY: '26TH Street',
          }
        )
  for label, count in RARE_LABELS.items():
    for i in range(count):
      charge_id += 1
      rows.append(
        {
          **rows[-1],
          COLS.CASE_ID: f'R{label[:3]}{i:03d}',
          COLS.CHARGE_ID: str(charge_id),
          COLS.CHARGE_VERSION_ID: '1',
          COLS.PRIMARY_CHARGE: True,
          COLS.TITLE: TITLES_BY_LABEL[label][0],
          COLS.LABEL: label,
          COLS.RECEIVED_DATE: '2015-06-01T00:00:00.000',
        }
      )
  # One title that only ever appears once → rare_title on whichever split it lands in.
  rows.append(
    {
      **rows[0],
      COLS.CASE_ID: 'ONEOFF',
      COLS.CHARGE_ID: '999999',
      COLS.TITLE: 'THEFT/UNAUTH CON/INTENT <$500/PRECONV',
      COLS.LABEL: 'Retail Theft',
    }
  )
  # Legacy-migration rows: real titles, non-offense label → excluded by policy.
  for i in range(8):
    rows.append(
      {
        **rows[2],
        COLS.CASE_ID: f'PROMIS{i:03d}',
        COLS.CHARGE_ID: str(900000 + i),
        COLS.CHARGE_VERSION_ID: '1',
        COLS.PRIMARY_CHARGE: True,
        COLS.LABEL: 'PROMIS Conversion',
      }
    )
  # One row with an unknown disposition string.
  rows.append(
    {
      **rows[1],
      COLS.CASE_ID: 'UNMAPPED',
      COLS.CHARGE_ID: '999998',
      COLS.DISPOSITION: 'Mystery Outcome',
    }
  )
  return pd.DataFrame(rows)


@pytest.fixture(scope='session')
def raw_frame() -> pd.DataFrame:
  return build_frame()


@pytest.fixture(scope='session')
def raw_parquet(tmp_path_factory, raw_frame) -> Path:
  path = tmp_path_factory.mktemp('raw') / 'apwk-dzx8.parquet'
  raw_frame.to_parquet(path, index=False)
  return path


@pytest.fixture(scope='session')
def map_path() -> Path:
  return Path(__file__).resolve().parents[1] / 'data' / 'disposition_map.yaml'


@pytest.fixture(autouse=True)
def hub_username(monkeypatch):
  monkeypatch.setenv('HF_USERNAME', HUB_USER)
