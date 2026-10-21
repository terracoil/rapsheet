"""HubPublisher — the one place anything leaves this machine for the Hugging Face Hub."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi
from modgud.app import implicit_return

from rapsheet.core import PathLike, PublishError

# What an OS or an editor drops into any folder. Never worth publishing.
CLUTTER = ('*.DS_Store', '*.ipynb_checkpoints/*')


@dataclass(frozen=True)
class HubPublisher:
  """Create a Hub repo if needed and make it mirror a prepared folder.

  Publishing is public and deliberate: nothing in the package calls this on its own, the notebook
  guards it behind ``PUBLISH = True``, and a folder without a ``README.md`` card is refused.
  Authentication comes from ``hf auth login`` or an ``HF_TOKEN`` entry in ``.env``.

  Re-publishing is a single commit that replaces files in place, so it is safe to repeat:

  - **Mirror, not append:** remote files that are no longer in the folder are deleted in the same
    commit (``delete_patterns='*'``; the Hub always keeps ``.gitattributes``), so a renamed split
    can't linger next to its replacement.
  - **Nothing changed, nothing committed:** unchanged files are deduplicated, and if every file
    matches, ``huggingface_hub`` skips the commit.
  - **History stays:** earlier versions remain in the repo's git history, which is the point: each
    publish is a revision a reader can pin.
  """

  api: Any = field(default_factory=HfApi, compare=False, repr=False)

  @implicit_return
  def identity(self) -> str:
    """Return the Hub username the current token belongs to (raises if not logged in)."""
    self.api.whoami()['name']

  @implicit_return
  def publish(
    self, folder: PathLike, repo_id: str, repo_type: str = 'dataset', message: str = 'Update'
  ) -> str:
    """Make ``repo_id`` mirror ``folder`` in one commit and return the repo's URL."""
    source = Path(folder)
    if not (source / 'README.md').exists():
      raise PublishError(f'{source}: no README.md card; refusing to publish an undocumented repo')
    namespace = repo_id.split('/')[0]
    if namespace != self.identity():
      raise PublishError(
        f'{repo_id}: the token belongs to {self.identity()!r}, not {namespace!r}; check HF_USERNAME'
      )
    self.api.create_repo(repo_id, repo_type=repo_type, exist_ok=True, private=False)
    self.api.upload_folder(
      repo_id=repo_id,
      folder_path=str(source),
      repo_type=repo_type,
      commit_message=message,
      ignore_patterns=list(CLUTTER),
      delete_patterns='*',
    )
    prefix = 'datasets/' if repo_type == 'dataset' else ''
    f'https://huggingface.co/{prefix}{repo_id}'
