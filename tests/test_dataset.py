import pandas as pd
import pytest
from rapsheet.core import Columns, DatasetError
from rapsheet.dataset import DatasetSplits, DispositionsDataset

COLS = Columns()


@pytest.fixture(scope='module')
def dataset() -> DispositionsDataset:
  return DispositionsDataset(min_class_rows=20, rare_title_max=5, seed=42)


@pytest.fixture(scope='module')
def cleaned(dataset, raw_parquet) -> pd.DataFrame:
  return dataset.clean(dataset.load(raw_parquet))


@pytest.fixture(scope='module')
def splits(dataset, cleaned) -> DatasetSplits:
  return dataset.split(cleaned)


def test_load_refuses_protected_and_leakage_columns(dataset, raw_frame, tmp_path):
  for bad in ('race', 'offense_category'):
    path = tmp_path / f'{bad}.parquet'
    raw_frame.assign(**{bad: 'x'}).to_parquet(path, index=False)
    with pytest.raises(DatasetError, match='forbidden'):
      dataset.load(path)


def test_load_refuses_missing_required_column(dataset, raw_frame, tmp_path):
  path = tmp_path / 'missing.parquet'
  raw_frame.drop(columns=[COLS.LABEL]).to_parquet(path, index=False)
  with pytest.raises(DatasetError, match='missing'):
    dataset.load(path)


def test_load_coerces_types(dataset, raw_parquet):
  frame = dataset.load(raw_parquet)
  assert frame[COLS.PRIMARY_CHARGE].dtype == bool
  assert pd.api.types.is_numeric_dtype(frame[COLS.CHARGE_VERSION_ID])
  assert pd.api.types.is_datetime64_any_dtype(frame[COLS.RECEIVED_DATE])


def test_as_bool_accepts_strings_and_bools():
  s = pd.Series([True, False, 'true', 'False', '1', 'no'])
  assert DispositionsDataset.as_bool(s).tolist() == [True, False, True, False, True, False]


def test_clean_keeps_primary_latest_version_normalised(dataset, raw_parquet, cleaned):
  raw = dataset.load(raw_parquet)
  steps = cleaned.attrs['cleaning']
  assert steps['loaded'] == len(raw)
  assert cleaned[COLS.PRIMARY_CHARGE].all()
  assert cleaned[COLS.CHARGE_ID].is_unique
  # Latest version wins: every charge keeps its max version id.
  latest = raw[raw[COLS.PRIMARY_CHARGE]].groupby(COLS.CHARGE_ID)[COLS.CHARGE_VERSION_ID].max()
  kept = cleaned.set_index(COLS.CHARGE_ID)[COLS.CHARGE_VERSION_ID]
  assert (kept == latest.loc[kept.index]).all()
  # Titles are upper-cased, trimmed, punctuation intact.
  titles = cleaned[COLS.TITLE]
  assert (titles == titles.str.upper()).all()
  assert not titles.str.startswith(' ').any()
  assert titles.str.contains('<\\$300', regex=True).any()


def test_category_counts_sum_to_one(dataset, cleaned):
  counts = dataset.category_counts(cleaned)
  assert counts['rows'].sum() == len(cleaned)
  assert abs(counts['share'].sum() - 1.0) < 0.01


def test_splits_are_grouped_by_case(splits):
  assert splits.leakage_check()


def test_temporal_holdout_is_strictly_2023_plus(splits):
  cutoff = pd.Timestamp('2023-01-01')
  assert (splits.temporal[COLS.RECEIVED_DATE] >= cutoff).all()
  for part in (splits.train, splits.val, splits.test):
    assert (part[COLS.RECEIVED_DATE] < cutoff).all()


def test_split_sizes_follow_the_fold_count(splits):
  total = sum(len(f) for f in (splits.train, splits.val, splits.test))
  assert 0.05 < len(splits.test) / total < 0.15
  assert 0.05 < len(splits.val) / total < 0.15


def test_rare_classes_fold_to_other_for_training_only(splits):
  assert set(splits.folded_classes) == {'Arson', 'Homicide'}
  assert splits.folded_label == '(rare, folded)'
  assert splits.label_names[-1] == splits.folded_label
  assert 'Other' not in splits.label_names
  train = splits.train
  assert (
    train.loc[train[COLS.LABEL].isin(splits.folded_classes), COLS.LABEL_TRAIN]
    == splits.folded_label
  ).all()
  assert 'Arson' in set(train[COLS.LABEL])  # real label preserved for the rare-class table
  table = splits.rare_class_table()
  assert list(table.columns) == ['train', 'val', 'test', 'temporal']
  assert table.loc['Homicide'].sum() == 12


def test_rare_title_flag_marks_titles_seen_under_five_times_in_train(splits):
  counts = splits.train[COLS.TITLE].value_counts()
  for name, part in splits.frames.items():
    expected = part[COLS.TITLE].map(lambda t: counts.get(t, 0) < 5)
    assert (part[COLS.RARE_TITLE] == expected).all(), name
  # The one-off title is rare wherever it landed.
  everything = pd.concat(splits.frames.values())
  oneoff = everything[everything[COLS.CASE_ID] == 'ONEOFF']
  assert len(oneoff) == 1 and bool(oneoff[COLS.RARE_TITLE].iloc[0])


def test_summary_and_roundtrip(splits, tmp_path):
  summary = splits.summary()
  assert list(summary.index) == ['train', 'val', 'test', 'temporal']
  out = splits.write(tmp_path / 'processed')
  back = DatasetSplits.read(out)
  assert back.folded_classes == splits.folded_classes
  assert len(back.train) == len(splits.train)
  assert back.leakage_check()


def test_subsample_is_seeded(dataset, splits):
  a = dataset.subsample(splits.train, 50)
  b = dataset.subsample(splits.train, 50)
  assert a[COLS.CHARGE_ID].tolist() == b[COLS.CHARGE_ID].tolist()
  assert len(dataset.subsample(splits.train, 10**9)) == len(splits.train)


def test_excluded_labels_are_dropped_and_kept_aside(dataset, raw_parquet, cleaned):
  assert 'PROMIS Conversion' not in set(cleaned[COLS.LABEL])
  steps = cleaned.attrs['cleaning']
  assert steps['label_excluded'] == 8
  assert steps['after_label_exclusion'] == len(cleaned)
  assert steps['with_title_and_label'] == len(cleaned) + 8
  aside = dataset.excluded_rows(dataset.load(raw_parquet))
  assert len(aside) == 8 and set(aside[COLS.LABEL]) == {'PROMIS Conversion'}


def test_exclusion_can_be_switched_off(raw_parquet):
  keep_all = DispositionsDataset(excluded_labels=())
  cleaned = keep_all.clean(keep_all.load(raw_parquet))
  assert (cleaned[COLS.LABEL] == 'PROMIS Conversion').sum() == 8


def test_title_stats_measures_ambiguity_and_ceiling(dataset):
  frame = pd.DataFrame(
    {
      COLS.TITLE: ['A', 'A', 'A', 'B'],
      COLS.LABEL: ['x', 'x', 'y', 'z'],
    }
  )
  stats = dataset.title_stats(frame)
  assert stats == {
    'rows': 4,
    'titles': 2,
    'multi_label_titles': 1,
    'rows_on_multi_label_titles': 3,
    'title_majority_ceiling': 0.75,
  }


def test_title_stats_on_fixture_is_unambiguous(dataset, cleaned):
  stats = dataset.title_stats(cleaned)
  assert stats['rows'] == len(cleaned)
  assert stats['multi_label_titles'] == 0
  assert stats['title_majority_ceiling'] == 1.0


def test_unseen_titles_mask():
  rows = pd.DataFrame({COLS.TITLE: ['A', 'Q', 'B']})
  reference = pd.DataFrame({COLS.TITLE: ['A', 'B', 'C']})
  assert DispositionsDataset.unseen_titles(rows, reference).tolist() == [False, True, False]
