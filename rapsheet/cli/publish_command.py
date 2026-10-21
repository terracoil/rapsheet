"""PublishCommand — ``rapsheet-publish``: push the snapshot or the model to the Hub from a shell."""

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from modgud.app import implicit_return

from rapsheet.publish import HubPublisher, HubRepos

FOLDERS = {
  'dataset': Path('data/snapshot'),
  'model': Path('models/rapsheet-legal-bert-small-tempered'),
}


@dataclass(frozen=True)
class PublishCommand:
  """The notebook's publish cells, as a command.

  ``rapsheet-publish dataset`` pushes ``data/snapshot/`` to ``$HF_USERNAME/rapsheet-charge-titles``;
  ``rapsheet-publish model`` pushes a saved model folder (default: the class-tempered one) to
  ``$HF_USERNAME/rapsheet-legal-bert-small``. Both need the folder's card to exist already (the
  notebook writes them), ``HF_USERNAME`` and ``HF_TOKEN`` in ``.env``, and ``--yes``: nothing is
  pushed by accident.
  """

  what: str
  folder: Path
  message: str
  confirmed: bool = False
  repos: HubRepos = field(default_factory=HubRepos.from_env)
  publisher: HubPublisher = field(default_factory=HubPublisher)

  @staticmethod
  @implicit_return
  def parser() -> argparse.ArgumentParser:
    """Argument parser for the console script."""
    parser = argparse.ArgumentParser(
      prog='rapsheet-publish', description='Publish the data snapshot or the model to the Hub.'
    )
    parser.add_argument('what', choices=sorted(FOLDERS), help='which repo to publish')
    parser.add_argument('--folder', default=None, help='folder to upload (default per repo)')
    parser.add_argument('--message', default='Update from rapsheet-publish', help='commit message')
    parser.add_argument('--yes', action='store_true', help='actually push (otherwise a dry run)')
    parser

  @classmethod
  def main(cls, argv: list[str] | None = None, **overrides: Any) -> None:
    """Console-script entry: parse the arguments and run."""
    args = cls.parser().parse_args(argv)
    cls(
      what=args.what,
      folder=Path(args.folder) if args.folder else FOLDERS[args.what],
      message=args.message,
      confirmed=args.yes,
      **overrides,
    ).run()

  @property
  @implicit_return
  def repo_id(self) -> str:
    """The target repo for ``what``."""
    self.repos.dataset_id if self.what == 'dataset' else self.repos.model_id

  def run(self) -> None:
    """Print the plan; push only with ``--yes``."""
    print(f'{self.what}: {self.folder}/ → {self.repo_id} as {self.publisher.identity()}')
    if not self.confirmed:
      print('dry run; add --yes to publish')
    else:
      print(
        self.publisher.publish(self.folder, self.repo_id, repo_type=self.what, message=self.message)
      )
