import numpy as np
import pandas as pd
from rapsheet.figures import FigureWriter


def curve(scale):
  coverage = np.linspace(0.01, 1, 50)
  return pd.DataFrame(
    {'coverage': coverage, 'risk': scale * coverage**3, 'threshold': 1 - coverage}
  )


def reliability(shift):
  low = np.linspace(0, 14 / 15, 15)
  return pd.DataFrame(
    {
      'confidence': low + 1 / 30,
      'accuracy': np.clip(low + 1 / 30 + shift, 0, 1),
      'rows': np.arange(1, 16) ** 3,
      'low': low,
      'high': low + 1 / 15,
    }
  )


def test_every_figure_writes_svg_and_png(tmp_path):
  figures = FigureWriter.at(tmp_path / 'figures')
  sweep = pd.DataFrame(
    [
      {
        'budget': b,
        'split': s,
        'per_class_coverage': 0.5 * k,
        'per_class_error': b / 2,
        'global_coverage': 0.6 * k,
        'global_error': b,
      }
      for k, b in enumerate((0.005, 0.01, 0.02), start=1)
      for s in ('test', 'temporal')
    ]
  ).set_index(['budget', 'split'])
  confusions = pd.DataFrame({'true': ['A', 'B'], 'predicted': ['B', 'C'], 'rows': [9, 4]})
  written = [
    figures.reliability({'test': reliability(0.0), 'temporal': reliability(-0.05)}, 'T = 1.00'),
    figures.risk_coverage(
      {'test': curve(0.04), 'temporal': curve(0.05)},
      {'test': (0.2, 0.003)},
      0.005,
      'subtitle',
    ),
    figures.top_confusions(confusions, 'subtitle'),
    figures.budget_sweep(sweep, ('test', 'temporal'), 'note'),
  ]
  for png in written:
    assert png.exists() and png.with_suffix('.svg').exists()
    assert png.read_bytes()[:4] == b'\x89PNG'
