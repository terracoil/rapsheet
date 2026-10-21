import re

import numpy as np
import pandas as pd
import pytest
from rapsheet.core import Columns, NotFittedError
from rapsheet.dataset import DispositionsDataset, LabelSpace
from rapsheet.encoder import EncoderClassifier, EncoderConfig, EncoderInputs
from rapsheet.evaluation import HeadlineMetrics, LogitStore
from transformers import BertConfig, BertModel, BertTokenizer

COLS = Columns()


@pytest.fixture(scope='module')
def splits(raw_parquet):
  dataset = DispositionsDataset(seed=42)
  return dataset.split(dataset.clean(dataset.load(raw_parquet)))


@pytest.fixture(scope='module')
def space(splits):
  return LabelSpace.from_splits(splits)


@pytest.fixture(scope='module')
def tiny_bert(tmp_path_factory, splits):
  """Build a 1-layer, 32-wide BERT with a vocabulary from the fixture's titles and citations."""
  directory = tmp_path_factory.mktemp('tiny-bert')
  frames = pd.concat(splits.frames.values())
  text = ' '.join(frames[COLS.TITLE].astype(str).str.lower()) + ' ilcs 720 5 / - ( ) < > $'
  words = sorted(set(re.findall(r'\w+|[^\w\s]', text)))
  digits = [str(d) for d in range(10)] + [f'##{d}' for d in range(10)]  # WordPiece continuations
  vocab = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'] + words + digits
  vocab_file = directory / 'vocab.txt'
  vocab_file.write_text('\n'.join(dict.fromkeys(vocab)) + '\n')
  # transformers 5: `vocab=` (path or dict). The 4.x `vocab_file=` keyword is silently ignored.
  tokenizer = BertTokenizer(vocab=str(vocab_file), do_lower_case=True)
  config = BertConfig(
    vocab_size=len(dict.fromkeys(vocab)),
    hidden_size=32,
    num_hidden_layers=1,
    num_attention_heads=2,
    intermediate_size=64,
    max_position_embeddings=64,
  )
  BertModel(config).save_pretrained(directory)
  tokenizer.save_pretrained(directory)
  return directory


@pytest.fixture(scope='module')
def config(tiny_bert):
  return EncoderConfig(
    model_name=str(tiny_bert),
    max_length=32,
    batch_size=32,
    eval_batch_size=64,
    learning_rate=5e-3,
    epochs=2,
    samples_per_epoch=2048,
    device='cpu',
  )


@pytest.fixture(scope='module')
def fitted(space, config, splits):
  return EncoderClassifier(space, config).fit(splits.train, splits.val)


def test_citation_is_written_the_illinois_way():
  frame = pd.DataFrame(
    {
      COLS.TITLE: ['A', 'B', 'C', 'D'],
      COLS.CHAPTER: ['720', '720', '720', None],
      COLS.ACT: ['5', '5', None, '5'],
      COLS.SECTION: ['24-1.6', None, '12', '1'],
    }
  )
  with_statute = EncoderInputs(include_statute=True).citation(frame).tolist()
  assert with_statute == ['720 ILCS 5/24-1.6', '720 ILCS 5', '720', '']
  assert EncoderInputs(include_statute=False).citation(frame).tolist() == ['', '', '', '']


def test_tiny_tokenizer_reads_the_pair(tiny_bert):
  tokenizer = BertTokenizer.from_pretrained(tiny_bert)
  ids = tokenizer('DUI/ALCOHOL', '720 ILCS 5/20-5')['input_ids']
  assert '[UNK]' not in tokenizer.convert_ids_to_tokens(ids)


def test_keys_round_trip_to_title_and_citation():
  frame = pd.DataFrame(
    {COLS.TITLE: ['AGG UUW/VEH'], COLS.CHAPTER: ['720'], COLS.ACT: ['5'], COLS.SECTION: ['24-1.6']}
  )
  keys = EncoderInputs().keys(frame).to_numpy()
  assert EncoderInputs.unpack(keys) == (['AGG UUW/VEH'], ['720 ILCS 5/24-1.6'])


def test_unfitted_encoder_refuses(space, config, splits):
  with pytest.raises(NotFittedError):
    EncoderClassifier(space, config).predict_proba(splits.test)


def test_fit_records_history_and_learns(fitted, splits, space):
  assert [h['epoch'] for h in fitted.history] == [1, 2]
  assert all({'train_loss', 'val_nll', 'val_accuracy'} <= set(h) for h in fitted.history)
  proba = fitted.predict_proba(splits.test)
  assert proba.shape == (len(splits.test), space.size)
  assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
  table = HeadlineMetrics(space).table([fitted], splits)
  assert table.loc[(fitted.name, 'test'), 'accuracy'] > 0.6
  assert fitted.predict_logits(splits.test.head(0)).shape == (0, space.size)


def test_names_say_what_the_model_reads(space, config):
  assert EncoderClassifier(space, config).name == 'legal-bert-small (title + statute)'
  title_only = EncoderConfig(**{**config.to_dict(), 'include_statute': False})
  assert EncoderClassifier(space, title_only).name == 'legal-bert-small (title)'


def test_save_and_load_reproduce_the_logits(fitted, splits, tmp_path):
  fitted.save(tmp_path / 'model')
  loaded = EncoderClassifier.load(tmp_path / 'model', device='cpu')
  assert loaded.space == fitted.space
  assert loaded.history == fitted.history
  assert np.allclose(
    loaded.predict_logits(splits.test), fitted.predict_logits(splits.test), atol=1e-5
  )


def test_title_only_mode_trains(space, config, splits):
  title_only = EncoderConfig(**{**config.to_dict(), 'include_statute': False, 'epochs': 1})
  model = EncoderClassifier(space, title_only).fit(splits.train)
  assert len(model.history) == 1 and 'val_nll' not in model.history[0]
  assert model.predict_proba(splits.test).shape == (len(splits.test), space.size)


def test_logit_store_round_trip(fitted, splits, space, tmp_path):
  store = LogitStore.at(tmp_path / 'logits')
  logits = fitted.predict_logits(splits.test)
  store.save('test', splits.test, logits, space)
  probe = splits.test.drop(columns=[COLS.LABEL_TRAIN, COLS.RARE_TITLE])
  store.save('probe', probe, fitted.predict_logits(probe), space)
  assert store.names() == ('probe', 'test')
  saved = store.load('test')
  assert np.allclose(saved['logits'], logits)
  assert saved['labels'].tolist() == space.encode(splits.test[COLS.LABEL_TRAIN]).tolist()
  assert saved['rare_title'].tolist() == splits.test[COLS.RARE_TITLE].tolist()
  assert set(store.load('probe')['labels']) == {-1}


def test_class_tempered_fit_adjusts_logits_and_survives_save_load(space, config, splits, tmp_path):
  tempered = EncoderConfig(**{**config.to_dict(), 'class_temper': 0.5, 'epochs': 1})
  model = EncoderClassifier(space, tempered).fit(splits.train, splits.val)
  assert model.name == 'legal-bert-small (title + statute, class-tempered 0.5)'
  assert len(model.logit_adjustment) == space.size
  assert np.allclose(np.asarray(model.logit_adjustment) != 0, np.asarray(space.train_counts) > 0)
  loaded = EncoderClassifier.load(model.save(tmp_path / 'tempered'), device='cpu')
  assert loaded.logit_adjustment == model.logit_adjustment
  assert np.allclose(
    loaded.predict_logits(splits.test), model.predict_logits(splits.test), atol=1e-5
  )


def test_untempered_fit_has_no_adjustment(fitted, space):
  assert fitted.logit_adjustment == tuple([0.0] * space.size)
