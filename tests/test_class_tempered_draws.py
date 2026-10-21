import numpy as np
import pytest
from rapsheet.encoder import ClassTemperedDraws

# Three classes: 90 rows over two inputs, 9 rows over two inputs, 1 row; class 3 has none.
LABELS = np.array([0, 0, 1, 1, 2])
WEIGHTS = np.array([60.0, 30.0, 6.0, 3.0, 1.0])


def plan(temper):
  return ClassTemperedDraws(LABELS, WEIGHTS, n_classes=4, temper=temper)


def test_temper_one_draws_rows_as_they_occur_and_adjusts_nothing():
  assert np.allclose(plan(1.0).probabilities(), WEIGHTS / WEIGHTS.sum())
  assert np.array_equal(plan(1.0).logit_adjustment(), np.zeros(4))


@pytest.mark.parametrize('temper', [0.0, 0.5])
def test_tempering_moves_class_shares_but_not_the_mix_inside_a_class(temper):
  p = plan(temper).probabilities()
  assert np.isclose(p.sum(), 1.0)
  shares = np.bincount(LABELS, weights=p, minlength=4)
  expected = np.array([90.0, 9.0, 1.0, 0.0]) ** temper * np.array([1, 1, 1, 0])
  assert np.allclose(shares, expected / expected.sum())
  assert np.isclose(p[0] / p[1], 2.0) and np.isclose(p[2] / p[3], 2.0)


def test_square_root_gives_the_smallest_class_more_draws():
  assert plan(0.5).class_shares()[2] > 5 * plan(1.0).class_shares()[2]


def test_adjustment_puts_the_real_prior_back():
  draws = plan(0.5)
  adjusted = np.log(draws.class_shares()[:3]) + draws.logit_adjustment()[:3]
  prior = np.array([90.0, 9.0, 1.0]) / 100.0
  assert np.allclose(np.exp(adjusted), prior)
  assert draws.logit_adjustment()[3] == 0.0 and draws.class_shares()[3] == 0.0
