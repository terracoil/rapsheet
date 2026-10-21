import math

import numpy as np
import pytest
from rapsheet.dataset import LabelSpace
from rapsheet.evaluation import SelectiveEvaluator, TemperatureCalibrator


def overconfident(rows=4000, classes=4, seed=0, sharpen=3.0):
  """Logits whose softmax is `sharpen` times too confident, with labels drawn from the truth."""
  rng = np.random.default_rng(seed)
  true_logits = rng.normal(0, 1.5, size=(rows, classes))
  truth = TemperatureCalibrator.softmax(true_logits)
  labels = np.array([rng.choice(classes, p=p) for p in truth])
  return true_logits * sharpen, labels


def test_temperature_recovers_the_overconfidence_and_keeps_accuracy():
  logits, labels = overconfident()
  calibrator = TemperatureCalibrator().fit(logits, labels)
  assert 2.5 < calibrator.temperature < 3.5
  report = calibrator.report(logits, labels)
  assert report['nll_after'] < report['nll_before']
  assert report['ece_after'] < report['ece_before']
  assert np.array_equal(calibrator.probabilities(logits).argmax(1), logits.argmax(1))


def test_unlabeled_rows_are_left_out_of_the_nll():
  logits, labels = overconfident(rows=200)
  assert math.isnan(TemperatureCalibrator().nll(logits, np.full(200, -1)))
  mixed = labels.copy()
  mixed[:50] = -1
  assert np.isclose(
    TemperatureCalibrator().nll(logits, mixed),
    TemperatureCalibrator().nll(logits[50:], labels[50:]),
  )


def test_reliability_bins_cover_every_row():
  logits, labels = overconfident(rows=500)
  calibrator = TemperatureCalibrator(bins=10)
  table = calibrator.reliability(calibrator.probabilities(logits), labels)
  assert table['rows'].sum() == 500
  assert (table['low'] < table['high']).all() and len(table) <= 10


@pytest.fixture
def space():
  return LabelSpace(
    names=('A', 'B', '(rare, folded)'), train_counts=(900, 90, 10), folded_label='(rare, folded)'
  )


def one_hot_confident(label, confidence, size=3):
  row = np.full(size, (1 - confidence) / (size - 1))
  row[label] = confidence
  return row


def test_a_class_earns_a_threshold_only_with_enough_clean_rows(space):
  # Class A: 1,000 clean rows at 0.99; class B: 40 clean rows at 0.99 (too few to prove 0.5%).
  proba = np.array([one_hot_confident(0, 0.99)] * 1000 + [one_hot_confident(1, 0.99)] * 40)
  labels = np.array([0] * 1000 + [1] * 40)
  evaluator = SelectiveEvaluator(space)
  tau = evaluator.class_thresholds(proba, labels)
  assert tau[0] == pytest.approx(0.99) and math.isinf(tau[1]) and math.isinf(tau[2])
  alone = evaluator.autonomous(proba, tau)
  assert alone[:1000].all() and not alone[1000:].any()


def test_errors_above_budget_push_the_threshold_up(space):
  # 600 rows at 0.999 (clean) and 400 rows at 0.95 with 5% errors: only the top block qualifies.
  proba = np.array([one_hot_confident(0, 0.999)] * 600 + [one_hot_confident(0, 0.95)] * 400)
  labels = np.array([0] * 600 + [0] * 380 + [1] * 20)
  assert SelectiveEvaluator(space).class_thresholds(proba, labels)[0] == pytest.approx(0.999)


def test_the_fold_bucket_never_acts_alone(space):
  proba = np.array([one_hot_confident(2, 0.999)] * 2000)
  assert math.isinf(SelectiveEvaluator(space).class_thresholds(proba, np.full(2000, 2))[2])


def test_risk_coverage_and_summary(space):
  proba = np.array([one_hot_confident(0, 0.999)] * 900 + [one_hot_confident(1, 0.6)] * 100)
  labels = np.array([0] * 900 + [0] * 100)  # every class-B call is wrong
  evaluator = SelectiveEvaluator(space, curve_points=50)
  curve = evaluator.risk_coverage(proba, labels)
  assert curve['coverage'].iloc[-1] == 1.0 and curve['risk'].iloc[-1] == pytest.approx(0.1)
  assert curve['risk'].iloc[0] == 0.0
  tau = evaluator.class_thresholds(proba, labels)
  summary = evaluator.summary(proba, labels, tau, evaluator.global_threshold(proba, labels))
  assert summary['per_class_coverage'] == 0.9 and summary['per_class_error'] == 0.0
  assert 0 < summary['aurc'] < 0.1
  unlabeled = evaluator.summary(proba, np.full(1000, -1), tau, 0.99)
  assert unlabeled['per_class_coverage'] == 0.9 and math.isnan(unlabeled['per_class_error'])


def test_autonomy_table_counts_per_class(space):
  proba = np.array([one_hot_confident(0, 0.999)] * 900 + [one_hot_confident(1, 0.6)] * 100)
  labels = np.array([0] * 1000)
  evaluator = SelectiveEvaluator(space)
  tau = evaluator.class_thresholds(proba, labels)
  table = evaluator.autonomy_table((proba, labels), tau, {'test': (proba, labels)})
  assert table.loc['A', 'test_alone'] == 900 and table.loc['B', 'test_alone'] == 0
  assert table.loc['B', 'test_predicted'] == 100 and np.isnan(table.loc['B', 'tau'])
  assert list(table.index) == ['A', 'B', '(rare, folded)']


def test_autonomy_table_has_no_error_column_for_an_unlabeled_probe(space):
  proba = np.array([one_hot_confident(0, 0.999)] * 900)
  evaluator = SelectiveEvaluator(space)
  tau = evaluator.class_thresholds(proba, np.zeros(900, dtype=int))
  table = evaluator.autonomy_table(
    (proba, np.zeros(900, dtype=int)), tau, {'probe': (proba, np.full(900, -1))}
  )
  assert 'probe_alone' in table.columns and 'probe_errors' not in table.columns
