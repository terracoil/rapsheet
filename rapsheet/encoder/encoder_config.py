"""EncoderConfig — every knob the fine-tune turns, in one immutable, manifest-ready place."""

from dataclasses import asdict, dataclass

import torch
from modgud.app import implicit_return

from rapsheet.core import JsonDict


@dataclass(frozen=True)
class EncoderConfig:
  """Hyperparameters for the ``legal-bert-small`` fine-tune.

  ``samples_per_epoch`` counts weighted draws of (title, citation) pairs. Drawing a pair in
  proportion to how often it occurs is the same distribution as drawing a training row, so three
  epochs of 50k draws see the data the way ~150k rows would, without re-encoding duplicates.

  ``class_temper`` tilts those draws toward small classes: each class gets a share in proportion to
  ``rows ** class_temper`` (1.0 = as the rows occur, 0.5 = square root, 0.0 = equal shares). The
  classifier adds the matching logit adjustment back, so its probabilities still answer for the
  real class frequencies. See ``ClassTemperedDraws``.
  """

  model_name: str = 'nlpaueb/legal-bert-small-uncased'
  include_statute: bool = True
  max_length: int = 64
  batch_size: int = 64
  eval_batch_size: int = 512
  learning_rate: float = 5e-5
  weight_decay: float = 0.01
  warmup_ratio: float = 0.06
  epochs: int = 3
  samples_per_epoch: int = 50_000
  class_temper: float = 1.0
  max_grad_norm: float = 1.0
  seed: int = 42
  device: str = 'auto'

  @implicit_return
  def resolved_device(self) -> str:
    """Return the device to train on: Apple GPU, then CUDA, then CPU, unless one is named."""
    auto = (
      'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    )
    auto if self.device == 'auto' else self.device

  @implicit_return
  def to_dict(self) -> JsonDict:
    """Return the config as a plain dict for the manifest and the saved model folder."""
    asdict(self)
