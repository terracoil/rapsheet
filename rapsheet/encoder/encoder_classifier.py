"""EncoderClassifier — legal-bert-small, fine-tuned on charge text, scored like every baseline."""

import json
import logging
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from modgud.app import implicit_return
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from rapsheet.core import Columns, Frame, JsonDict, Matrix, NotFittedError, PathLike
from rapsheet.dataset import KeyedLabelCounts, LabelSpace

from .class_tempered_draws import ClassTemperedDraws
from .encoder_config import EncoderConfig
from .encoder_inputs import EncoderInputs

log = logging.getLogger(__name__)
COLS = Columns()
META_FILE = 'rapsheet_encoder.json'


@dataclass(frozen=True)
class EncoderClassifier:
  """A small pretrained encoder with a fresh classification head, fine-tuned on (title, citation).

  - **Training data:** each distinct input is encoded once; each step draws inputs in proportion
    to how often they occur in training, so the loss is the row-level loss. With
    ``config.class_temper`` below 1, small classes get a bigger share of the draws, and
    ``logit_adjustment`` takes the tilt back out of every logit this class returns.
  - **Model selection:** the epoch with the lowest validation negative log-likelihood wins. NLL,
    not accuracy, because calibration comes next and rewards honest probabilities.
  - **Output:** ``predict_logits`` returns logits in ``LabelSpace`` order, prior-adjusted when the
    draws were tempered (the calibrator divides them by a temperature); ``predict_proba`` is their
    softmax.
  """

  space: LabelSpace
  config: EncoderConfig = field(default_factory=EncoderConfig)
  model: Any = field(default=None, compare=False, repr=False)
  tokenizer: Any = field(default=None, compare=False, repr=False)
  history: tuple[JsonDict, ...] = ()
  logit_adjustment: tuple[float, ...] = ()

  @property
  @implicit_return
  def name(self) -> str:
    """Name used in the headline table."""
    reads = 'title + statute' if self.config.include_statute else 'title'
    tempered = (
      '' if self.config.class_temper == 1.0 else f', class-tempered {self.config.class_temper:g}'
    )
    f'legal-bert-small ({reads}{tempered})'

  @property
  @implicit_return
  def inputs(self) -> EncoderInputs:
    """The input builder for this config."""
    EncoderInputs(include_statute=self.config.include_statute)

  # ------------------------------------------------------------------ fit

  @implicit_return
  def fit(self, train: Frame, val: Frame | None = None) -> 'EncoderClassifier':
    """Fine-tune on ``train``; keep the epoch with the best NLL on ``val`` (the last if no ``val``)."""
    cfg = self.config
    torch.manual_seed(cfg.seed)
    device = cfg.resolved_device()
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    # The head's size comes from id2label. Passing num_labels as well makes transformers check it
    # against the checkpoint's 2-label default first and warn about a mismatch that isn't one.
    model = AutoModelForSequenceClassification.from_pretrained(
      cfg.model_name,
      id2label=dict(enumerate(self.space.names)),
      label2id={name: i for i, name in enumerate(self.space.names)},
    ).to(device)
    counts = KeyedLabelCounts.build(
      self.inputs.keys(train), self.space.encode(train[COLS.LABEL_TRAIN]), self.space.size
    )
    keys, labels, weights = counts.pairs()
    encoded = self.encode(tokenizer, keys, device)
    targets = torch.as_tensor(labels, dtype=torch.long, device=device)
    plan = ClassTemperedDraws(labels, weights, self.space.size, cfg.class_temper)
    probabilities = torch.as_tensor(plan.probabilities(), dtype=torch.float)
    adjustment = tuple(float(x) for x in plan.logit_adjustment())
    steps_per_epoch = math.ceil(cfg.samples_per_epoch / cfg.batch_size)
    optimizer = torch.optim.AdamW(
      model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    scheduler = self.linear_schedule(optimizer, steps_per_epoch * cfg.epochs)
    generator = torch.Generator().manual_seed(cfg.seed)
    history: list[JsonDict] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_nll = math.inf
    for epoch in range(1, cfg.epochs + 1):
      draws = torch.multinomial(
        probabilities, steps_per_epoch * cfg.batch_size, replacement=True, generator=generator
      )
      loss = self.train_epoch(model, encoded, targets, draws, optimizer, scheduler)
      fitted = replace(self, model=model.eval(), tokenizer=tokenizer, logit_adjustment=adjustment)
      scores = fitted.evaluate(val) if val is not None else {}
      history.append({'epoch': epoch, 'train_loss': round(loss, 5), **scores})
      log.info('epoch %d: %s', epoch, history[-1])
      improved = val is None or scores['val_nll'] < best_nll
      best_nll = scores.get('val_nll', best_nll) if improved else best_nll
      best_state = (
        {k: v.detach().to('cpu', copy=True) for k, v in model.state_dict().items()}
        if improved
        else best_state
      )
    model.load_state_dict(best_state)
    replace(
      self,
      model=model.eval(),
      tokenizer=tokenizer,
      history=tuple(history),
      logit_adjustment=adjustment,
    )

  @implicit_return
  def linear_schedule(self, optimizer: Any, total_steps: int) -> Any:
    """Return a linear warm-up then linear decay schedule (plain torch, no Trainer)."""
    warmup = max(1, int(total_steps * self.config.warmup_ratio))
    torch.optim.lr_scheduler.LambdaLR(
      optimizer,
      lambda step: min(
        (step + 1) / warmup, max(0.0, (total_steps - step) / max(1, total_steps - warmup))
      ),
    )

  @implicit_return
  def train_epoch(
    self,
    model: Any,
    encoded: dict[str, torch.Tensor],
    targets: torch.Tensor,
    draws: torch.Tensor,
    optimizer: Any,
    scheduler: Any,
  ) -> float:
    """Run one epoch of weighted draws; return the mean training loss."""
    model.train()
    total = 0.0
    batches = draws.split(self.config.batch_size)
    for batch in batches:
      index = batch.to(targets.device)
      logits = model(**{name: tensor[index] for name, tensor in encoded.items()}).logits
      loss = F.cross_entropy(logits, targets[index])
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.max_grad_norm)
      optimizer.step()
      scheduler.step()
      optimizer.zero_grad(set_to_none=True)
      total += float(loss.detach())
    total / max(len(batches), 1)

  @implicit_return
  def evaluate(self, frame: Frame) -> JsonDict:
    """Return row-weighted validation NLL and accuracy, scoring each distinct input once."""
    counts = KeyedLabelCounts.build(
      self.inputs.keys(frame), self.space.encode(frame[COLS.LABEL_TRAIN]), self.space.size
    )
    keys, labels, weights = counts.pairs()
    unique_keys, positions = np.unique(np.asarray(keys), return_inverse=True)
    log_probs = torch.log_softmax(torch.as_tensor(self.logits_for_keys(unique_keys)), dim=1)
    picked = log_probs[torch.as_tensor(positions), torch.as_tensor(labels)].numpy()
    hits = log_probs.argmax(dim=1).numpy()[positions] == labels
    {
      'val_nll': round(float(-(picked * weights).sum() / weights.sum()), 5),
      'val_accuracy': round(float((hits * weights).sum() / weights.sum()), 5),
      'val_inputs': int(len(unique_keys)),
    }

  # ------------------------------------------------------------------ predict

  @implicit_return
  def encode(self, tokenizer: Any, keys: np.ndarray, device: str) -> dict[str, torch.Tensor]:
    """Tokenize (title, citation) pairs into padded tensors on ``device``."""
    titles, citations = EncoderInputs.unpack(keys)
    pairs = citations if self.config.include_statute else None
    batch = tokenizer(
      titles,
      pairs,
      truncation=True,
      max_length=self.config.max_length,
      padding='max_length',
      return_tensors='pt',
    )
    {name: tensor.to(device) for name, tensor in batch.items()}

  @implicit_return
  def fitted(self) -> tuple[Any, Any]:
    """Return the model and tokenizer, or raise if ``fit`` hasn't run."""
    if self.model is None or self.tokenizer is None:
      raise NotFittedError(f'{self.name}: call fit() or load() before predicting')
    (self.model, self.tokenizer)

  @implicit_return
  def logits_for_keys(self, keys: np.ndarray) -> Matrix:
    """Return (prior-adjusted) logits for distinct input keys, in ``eval_batch_size`` chunks."""
    model, tokenizer = self.fitted()
    device = next(model.parameters()).device.type
    chunks = [
      keys[start : start + self.config.eval_batch_size]
      for start in range(0, len(keys), self.config.eval_batch_size)
    ]
    with torch.no_grad():
      outputs = [
        model(**self.encode(tokenizer, chunk, device)).logits.float().cpu().numpy()
        for chunk in chunks
      ]
    raw = np.concatenate(outputs) if outputs else np.zeros((0, self.space.size), dtype=np.float32)
    shift = np.asarray(self.logit_adjustment or np.zeros(self.space.size), dtype=np.float32)
    raw + shift

  @implicit_return
  def predict_logits(self, frame: Frame) -> Matrix:
    """Return logits, one row per input row, columns in ``LabelSpace`` order."""
    keys = self.inputs.keys(frame).to_numpy()
    unique, inverse = np.unique(keys, return_inverse=True)
    self.logits_for_keys(unique)[inverse]

  @implicit_return
  def predict_proba(self, frame: Frame) -> Matrix:
    """Return the softmax of the logits (uncalibrated; see the temperature calibrator)."""
    torch.softmax(torch.as_tensor(self.predict_logits(frame)), dim=1).numpy()

  # ------------------------------------------------------------------ persist

  @implicit_return
  def save(self, directory: PathLike) -> Path:
    """Write weights, tokenizer and a metadata file that lets ``load`` rebuild this object."""
    model, tokenizer = self.fitted()
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(target)
    tokenizer.save_pretrained(target)
    meta = {
      'config': self.config.to_dict(),
      'label_names': list(self.space.names),
      'train_counts': list(self.space.train_counts),
      'folded_label': self.space.folded_label,
      'history': list(self.history),
      'logit_adjustment': list(self.logit_adjustment),
    }
    (target / META_FILE).write_text(json.dumps(meta, indent=2) + '\n')
    target

  @classmethod
  @implicit_return
  def load(cls, directory: PathLike, device: str = 'auto') -> 'EncoderClassifier':
    """Rebuild a fitted classifier from a folder written by ``save``."""
    source = Path(directory)
    meta = json.loads((source / META_FILE).read_text())
    # Keep the base model's name: a re-saved model should still say where it came from.
    config = EncoderConfig(**{**meta['config'], 'device': device})
    space = LabelSpace(
      names=tuple(meta['label_names']),
      train_counts=tuple(meta['train_counts']),
      folded_label=meta['folded_label'],
    )
    model = AutoModelForSequenceClassification.from_pretrained(source)
    cls(
      space=space,
      config=config,
      model=model.to(config.resolved_device()).eval(),
      tokenizer=AutoTokenizer.from_pretrained(source),
      history=tuple(meta['history']),
      logit_adjustment=tuple(meta.get('logit_adjustment', ())),
    )
