import os

import numpy as np
import pandas as pd
import pytest
from rapsheet.baselines import (
  MajorityBaseline,
  StatuteRulesBaseline,
  TfidfBaseline,
  TitleLookupBaseline,
)
from rapsheet.core import Columns, NotFittedError, PublishError, RunManifest
from rapsheet.dataset import DispositionsDataset, KeyedLabelCounts, LabelSpace
from rapsheet.evaluation import HeadlineMetrics
from rapsheet.publish import HubPublisher, HubSnapshot

COLS = Columns()


@pytest.fixture(scope='module')
def splits(raw_parquet):
  dataset = DispositionsDataset(seed=42)
  return dataset.split(dataset.clean(dataset.load(raw_parquet)))


@pytest.fixture(scope='module')
def space(splits):
  return LabelSpace.from_splits(splits)


@pytest.fixture(scope='module')
def fitted(space, splits):
  models = (
    MajorityBaseline(space),
    TitleLookupBaseline(space),
    StatuteRulesBaseline(space),
    TfidfBaseline(space, seed=42),
  )
  return [m.fit(splits.train) for m in models]


def test_label_space_order_prior_and_rare(space, splits):
  assert space.names == splits.label_names
  assert space.names[-1] == space.folded_label
  assert abs(space.prior().sum() - 1) < 1e-9
  assert sum(space.train_counts) == len(splits.train)
  assert space.encode(pd.Series([space.names[0], 'Not A Label'])).tolist() == [0, -1]
  assert all(space.names[i] != space.folded_label for i in space.rare_classes(10**6))


def test_keyed_counts_pairs_reproduce_row_counts():
  keys = pd.Series(['A', 'A', 'A', 'B', None])
  labels = np.array([0, 0, 1, 2, 0])
  counts = KeyedLabelCounts.build(keys, labels, 3)
  assert counts.keys == ('A', 'B')
  assert counts.counts.tolist() == [[2, 1, 0], [0, 0, 1]]
  k, lab, w = counts.pairs()
  assert list(zip(k, lab, w, strict=True)) == [('A', 0, 2.0), ('A', 1, 1.0), ('B', 2, 1.0)]
  prior = np.array([0.5, 0.25, 0.25])
  fallback = np.tile(prior, (2, 1))
  out = counts.lookup(pd.Series(['B', 'Z']), counts.distributions(prior, 0.0), fallback)
  assert out[0].tolist() == [0, 0, 1] and out[1].tolist() == prior.tolist()


def test_every_model_returns_proper_distributions(fitted, splits, space):
  for model in fitted:
    for name, frame in splits.frames.items():
      proba = model.predict_proba(frame)
      assert proba.shape == (len(frame), space.size), (model.name, name)
      assert np.allclose(proba.sum(axis=1), 1.0), (model.name, name)
      assert (proba >= 0).all()


def test_majority_predicts_the_biggest_class(fitted, splits, space):
  majority = fitted[0].predict_proba(splits.test).argmax(axis=1)
  assert set(majority) == {int(np.argmax(space.train_counts))}


def test_title_lookup_memorizes_and_falls_back(fitted, space):
  lookup = fitted[1]
  frame = pd.DataFrame({COLS.TITLE: ['BURGLARY', 'A TITLE NOBODY HAS SEEN']})
  proba = lookup.predict_proba(frame)
  assert space.names[proba[0].argmax()] == 'Burglary'
  assert np.allclose(proba[1], space.prior())
  assert lookup.seen(frame).tolist() == [True, False]


def test_rules_back_off_to_coarser_levels(fitted, splits):
  rules = fitted[2]
  levels = rules.match_levels(splits.test)
  assert abs(sum(levels.values()) - 1) < 1e-3
  unseen_section = splits.test.head(1).assign(**{COLS.SECTION: 'NO-SUCH-SECTION'})
  assert rules.match_levels(unseen_section) == {'act': 1.0}
  missing = splits.test.head(1).assign(**{COLS.CHAPTER: None})
  assert rules.match_levels(missing) == {'none': 1.0}


def test_tfidf_reads_titles(fitted, splits, space):
  tfidf = fitted[3]
  pred = tfidf.predict_proba(splits.test).argmax(axis=1)
  truth = space.encode(splits.test[COLS.LABEL_TRAIN])
  assert (pred == truth).mean() > 0.95  # fixture titles are unambiguous


def test_unfitted_models_refuse_to_predict(space, splits):
  for model in (TitleLookupBaseline(space), StatuteRulesBaseline(space), TfidfBaseline(space)):
    with pytest.raises(NotFittedError):
      model.predict_proba(splits.test)


def test_headline_table_shape_and_floor(fitted, splits, space):
  table = HeadlineMetrics(space).table(fitted, splits)
  assert list(table.index.get_level_values('split').unique()) == ['test', 'temporal']
  assert len(table) == 2 * len(fitted)
  floor = table.loc[('majority class (floor)', 'test'), 'accuracy']
  lookup = table.loc[('title lookup', 'test'), 'accuracy']
  assert lookup > floor
  assert {'accuracy', 'macro_f1', 'rare_class_recall', 'rare_title_accuracy'} <= set(table.columns)


def test_every_model_accepts_an_empty_frame(fitted, splits, space):
  empty = splits.test.head(0)
  for model in fitted:
    assert model.predict_proba(empty).shape == (0, space.size), model.name


def test_hub_snapshot_drops_identifiers_and_writes_card(splits, raw_parquet, tmp_path):
  dataset = DispositionsDataset(seed=42)
  set_aside = dataset.excluded_rows(dataset.load(raw_parquet))
  manifest = RunManifest(
    'apwk-dzx8',
    'ep',
    {},
    COLS.used,
    '2026-04-02T00:00:00.000Z',
    10,
    10,
    1,
    seeds={'split': 42},
    config={
      'temporal_cutoff': '2023-01-01',
      'title_stats': {'titles': 17, 'multi_label_titles': 0, 'title_majority_ceiling': 0.9537},
    },
  )
  snapshot = HubSnapshot()
  stale = tmp_path / 'snap' / 'data' / 'renamed_last_week.parquet'
  stale.parent.mkdir(parents=True)
  stale.write_bytes(b'stale')
  summary = snapshot.write(splits, set_aside, manifest, tmp_path / 'snap')
  assert not stale.exists()
  assert set(summary['rows']) == {'train', 'validation', 'test', 'temporal', 'promis_set_aside'}
  assert summary['rows']['promis_set_aside'] == len(set_aside) == 8
  for name in summary['rows']:
    frame = pd.read_parquet(tmp_path / 'snap' / 'data' / f'{name}.parquet')
    assert list(frame.columns) == list(snapshot.columns)
    for identifier in (COLS.CASE_ID, COLS.PARTICIPANT_ID, COLS.CHARGE_ID, COLS.CHARGE_VERSION_ID):
      assert identifier not in frame.columns
    assert not set(COLS.forbidden) & set(frame.columns)
  card = (tmp_path / 'snap' / 'README.md').read_text()
  assert card.startswith('---\nlicense: other')
  assert 'Not a\nscreening tool' in card and 'path: data/promis_set_aside.parquet' in card
  assert '0 of 17 titles' in card and '(ceiling: 95.4% of rows)' in card


class FakeHubApi:
  def __init__(self):
    self.calls = []

  def whoami(self):
    return {'name': os.environ['HF_USERNAME']}

  def create_repo(self, repo_id, **kwargs):
    self.calls.append(('create_repo', repo_id, kwargs))

  def upload_folder(self, **kwargs):
    self.calls.append(('upload_folder', kwargs['repo_id'], kwargs))


def test_hub_publisher_creates_then_uploads(tmp_path):
  (tmp_path / 'README.md').write_text('---\nlicense: other\n---\n')
  api = FakeHubApi()
  publisher = HubPublisher(api=api)
  assert publisher.identity() == os.environ['HF_USERNAME']
  url = publisher.publish(tmp_path, f'{os.environ["HF_USERNAME"]}/x', message='snapshot')
  assert url == f'https://huggingface.co/datasets/{os.environ["HF_USERNAME"]}/x'
  assert [c[0] for c in api.calls] == ['create_repo', 'upload_folder']
  assert api.calls[0][2] == {'repo_type': 'dataset', 'exist_ok': True, 'private': False}
  assert api.calls[1][2]['commit_message'] == 'snapshot'


def test_hub_publisher_mirrors_the_folder_instead_of_appending(tmp_path):
  (tmp_path / 'README.md').write_text('---\nlicense: other\n---\n')
  api = FakeHubApi()
  HubPublisher(api=api).publish(tmp_path, f'{os.environ["HF_USERNAME"]}/x')
  upload = api.calls[1][2]
  assert upload['delete_patterns'] == '*'
  assert '*.DS_Store' in upload['ignore_patterns']


def test_hub_publisher_refuses_a_folder_without_a_card(tmp_path):
  api = FakeHubApi()
  with pytest.raises(PublishError, match='no README.md'):
    HubPublisher(api=api).publish(tmp_path, f'{os.environ["HF_USERNAME"]}/x')
  assert api.calls == []
