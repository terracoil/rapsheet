"""HubRepos — the Hugging Face namespace and repo ids, read from the environment, never typed in."""

import os
from dataclasses import dataclass, field

from modgud.app import implicit_return

from rapsheet.core import PublishError

HUB = 'https://huggingface.co'
USERNAME_VAR = 'HF_USERNAME'
# The code lives on GitHub under a fixed name; only the Hub namespace comes from the environment.
REPO_URL = 'https://github.com/terracoil/rapsheet'


@dataclass(frozen=True)
class HubRepos:
  """Where the snapshot and the model live on the Hub.

  The username comes from ``HF_USERNAME`` (put it in ``.env`` next to ``HF_TOKEN``); the repo
  names are fixed. Every repo id and URL in the package and the notebook is built here, so the
  namespace is set in exactly one place.
  """

  username: str = field(default_factory=lambda: os.environ.get(USERNAME_VAR, ''))
  dataset_name: str = 'rapsheet-charge-titles'
  model_name: str = 'rapsheet-legal-bert-small'

  @classmethod
  @implicit_return
  def from_env(cls) -> 'HubRepos':
    """Return the repos for ``HF_USERNAME``, or raise if it isn't set."""
    username = os.environ.get(USERNAME_VAR, '').strip()
    if not username:
      raise PublishError(f'{USERNAME_VAR} is not set; add it to .env next to HF_TOKEN')
    cls(username=username)

  @implicit_return
  def require(self) -> str:
    """Return the username, or raise if it is empty."""
    if not self.username:
      raise PublishError(f'{USERNAME_VAR} is not set; add it to .env next to HF_TOKEN')
    self.username

  @property
  @implicit_return
  def dataset_id(self) -> str:
    """``<username>/rapsheet-charge-titles``."""
    f'{self.require()}/{self.dataset_name}'

  @property
  @implicit_return
  def model_id(self) -> str:
    """``<username>/rapsheet-legal-bert-small``."""
    f'{self.require()}/{self.model_name}'

  @property
  @implicit_return
  def dataset_url(self) -> str:
    """Hub page of the dataset."""
    f'{HUB}/datasets/{self.dataset_id}'

  @property
  @implicit_return
  def model_url(self) -> str:
    """Hub page of the model."""
    f'{HUB}/{self.model_id}'
