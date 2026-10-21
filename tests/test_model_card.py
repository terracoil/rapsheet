import json

from rapsheet.core import Columns, RunManifest
from rapsheet.publish import ModelCard

COLS = Columns()
NAME = 'legal-bert-small (title + statute, class-tempered 0.5)'


def manifest():
  return RunManifest(
    'apwk-dzx8',
    'ep',
    {},
    COLS.used,
    '2026-04-02T14:58:19.622Z',
    1_080_014,
    1_080_014,
    22,
    retrieved_at='2026-10-20T19:13:38+00:00',
    seeds={'split': 42, 'encoder': 42},
    config={
      'split_rows': {'train': 266439, 'val': 33305, 'test': 33305, 'temporal': 19973},
      'set_aside_rows': 9824,
      'set_aside_unseen_title_rows': 4833,
      'processed_git_sha': 'abc1234',
      'processed_on': 'Darwin arm64',
      'title_stats': {'titles': 1532, 'multi_label_titles': 40, 'title_majority_ceiling': 0.967},
    },
    extra={
      'baselines': [
        {
          'model': NAME,
          'split': 'test',
          'accuracy': 0.9641,
          'macro_f1': 0.8802,
          'rare_class_recall': 0.8522,
          'rare_title_accuracy': 0.6474,
        },
        {
          'model': 'statute rules',
          'split': 'test',
          'accuracy': 0.9643,
          'macro_f1': 0.8607,
          'rare_class_recall': 0.7799,
          'rare_title_accuracy': 0.6368,
        },
        {
          'model': 'majority class (floor)',
          'split': 'test',
          'accuracy': 0.3748,
          'macro_f1': 0.008,
          'rare_class_recall': 0.0,
          'rare_title_accuracy': 0.0579,
        },
      ],
      'calibration': {
        'temperature': 1.001557,
        'bins': 15,
        'report': {
          'val': {
            'ece_before': 0.0051,
            'ece_after': 0.0050,
            'nll_before': 0.1630,
            'nll_after': 0.1630,
            'mean_confidence_after': 0.9670,
          },
          'temporal': {
            'ece_before': 0.0090,
            'ece_after': 0.0089,
            'nll_before': 0.1900,
            'nll_after': 0.1899,
            'mean_confidence_after': 0.9600,
          },
          'test': {
            'ece_before': 0.0049,
            'ece_after': 0.0048,
            'nll_before': 0.1641,
            'nll_after': 0.1641,
            'mean_confidence_after': 0.9676,
          },
        },
      },
      'abstention': {
        'error_budget': 0.005,
        'z': 1.645,
        'thresholds': {'Narcotics': 0.9968, 'Retail Theft': 0.9931},
        'global_tau': 0.9936,
        'summary': {
          'test': {
            'per_class_coverage': 0.1415,
            'per_class_error': 0.0032,
            'global_coverage': 0.5372,
            'global_error': 0.0045,
          },
          'promis_unseen': {
            'per_class_coverage': 0.0387,
            'per_class_error': float('nan'),
            'global_coverage': 0.0844,
            'global_error': float('nan'),
          },
        },
      },
      'probe_confidence': {NAME: 0.849},
    },
  )


def test_card_is_written_from_metadata_and_manifest(tmp_path):
  (tmp_path / 'rapsheet_encoder.json').write_text(
    json.dumps(
      {
        'config': {
          'model_name': 'nlpaueb/legal-bert-small-uncased',
          'include_statute': True,
          'max_length': 64,
          'batch_size': 64,
          'learning_rate': 5e-5,
          'weight_decay': 0.01,
          'warmup_ratio': 0.06,
          'epochs': 3,
          'samples_per_epoch': 50000,
          'max_grad_norm': 1.0,
          'seed': 42,
          'class_temper': 0.5,
        },
        'label_names': ['Narcotics', 'Retail Theft', '(rare, folded)'],
        'train_counts': [99880, 15454, 89],
        'folded_label': '(rare, folded)',
        'history': [{'epoch': 1, 'train_loss': 1.19, 'val_nll': 0.186, 'val_accuracy': 0.9608}],
        'logit_adjustment': [0.1, -0.2, 0.3],
      }
    )
  )
  (tmp_path / 'config.json').write_text(json.dumps({'hidden_size': 512, 'num_hidden_layers': 6}))
  path = ModelCard().write(tmp_path, manifest(), NAME)
  card = path.read_text()
  assert path.name == 'README.md'
  assert card.startswith('---\nlicense: cc-by-sa-4.0\nbase_model: nlpaueb/legal-bert-small-uncased')
  assert 'Not a screening tool' in card and 'Out of scope: any decision about a person' in card
  assert '| legal-bert-small (title + statute, class-tempered 0.5) | test | 0.9641 |' in card
  assert '| statute rules | test | 0.9643 |' in card and 'majority class' not in card
  assert 'T = 1.002' in card and '| test | 0.49% | 0.48% |' in card
  assert '2 of 3 classes earn one' in card and '| Narcotics | 0.9968 |' in card
  assert '| promis_unseen | 3.87% | n/a | 8.44% | n/a |' in card
  assert 'class-tempering logit adjustment' in card and '0.849 on the probe' in card
  assert 'exponent 0.5' in card and '`abc1234`' in card and 'train 266,439' in card
  assert 'a 6-layer BERT encoder with hidden size 512' in card
  assert '512 × 3 weights + 3 biases, 1,539 parameters' in card
  assert '115,423 training rows carry only 1,532 distinct titles' in card
  assert 'chosen by the same validation-only rule' in card
  assert (
    card.index('| test | 0.49%') < card.index('| temporal | 0.90%') < card.index('| val | 0.51%')
  )
  assert '`<$300` and `>$300`' in card
  assert '$' not in card.replace('$300', '')  # every placeholder was filled


def test_card_survives_a_manifest_without_evaluation(tmp_path):
  (tmp_path / 'rapsheet_encoder.json').write_text(
    json.dumps(
      {
        'config': {
          'model_name': 'm',
          'include_statute': False,
          'max_length': 48,
          'batch_size': 8,
          'learning_rate': 1e-4,
          'weight_decay': 0.0,
          'warmup_ratio': 0.1,
          'epochs': 1,
          'samples_per_epoch': 10,
          'max_grad_norm': 1.0,
          'seed': 1,
        },
        'label_names': ['A', '(rare, folded)'],
        'train_counts': [5, 1],
        'folded_label': '(rare, folded)',
        'history': [],
        'logit_adjustment': [],
      }
    )
  )
  bare = RunManifest('x', 'ep', {}, COLS.used, None, None, 0, 0)
  card = ModelCard().write(tmp_path, bare, 'm').read_text()
  assert 'T = n/a' in card and '0 of 2 classes earn one' in card and '| – | – |' in card
  assert 'is a BERT encoder,' in card and 'hidden size × 2 weights + 2 biases' in card
  assert 'validation-only rule' not in card and 'Each distinct (title, citation) input' in card
