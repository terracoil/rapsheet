import os

import pytest
from rapsheet.cli import PublishCommand
from rapsheet.core import PublishError
from rapsheet.publish import HubPublisher, HubRepos


class FakeHubApi:
  def __init__(self, name=None):
    self.name, self.calls = name or os.environ['HF_USERNAME'], []

  def whoami(self):
    return {'name': self.name}

  def create_repo(self, repo_id, **kwargs):
    self.calls.append(('create_repo', repo_id, kwargs['repo_type']))

  def upload_folder(self, **kwargs):
    self.calls.append(('upload_folder', kwargs['repo_id'], kwargs['repo_type']))


def test_hub_repos_build_every_id_from_the_environment(monkeypatch):
  monkeypatch.setenv('HF_USERNAME', 'someone-else')
  repos = HubRepos.from_env()
  assert repos.dataset_id == 'someone-else/rapsheet-charge-titles'
  assert repos.model_url == 'https://huggingface.co/someone-else/rapsheet-legal-bert-small'
  assert repos.dataset_url == 'https://huggingface.co/datasets/someone-else/rapsheet-charge-titles'


def test_hub_repos_refuse_a_missing_username(monkeypatch):
  monkeypatch.delenv('HF_USERNAME')
  with pytest.raises(PublishError, match='HF_USERNAME'):
    HubRepos.from_env()
  with pytest.raises(PublishError, match='HF_USERNAME'):
    HubRepos().dataset_id  # noqa: B018


def test_publisher_refuses_a_namespace_the_token_does_not_own(tmp_path):
  (tmp_path / 'README.md').write_text('---\n---\n')
  api = FakeHubApi(name='somebody')
  with pytest.raises(PublishError, match='check HF_USERNAME'):
    HubPublisher(api=api).publish(tmp_path, 'someone/x')
  assert api.calls == []


def test_command_dry_runs_unless_confirmed(tmp_path, capsys):
  (tmp_path / 'README.md').write_text('---\n---\n')
  api = FakeHubApi()
  publisher = HubPublisher(api=api)
  PublishCommand.main(['model', '--folder', str(tmp_path)], publisher=publisher)
  assert 'dry run' in capsys.readouterr().out and api.calls == []
  PublishCommand.main(['model', '--folder', str(tmp_path), '--yes'], publisher=publisher)
  assert api.calls == [
    ('create_repo', 'someone/rapsheet-legal-bert-small', 'model'),
    ('upload_folder', 'someone/rapsheet-legal-bert-small', 'model'),
  ]
  assert 'huggingface.co/someone/rapsheet-legal-bert-small' in capsys.readouterr().out


def test_command_defaults_the_dataset_folder(tmp_path, monkeypatch, capsys):
  monkeypatch.chdir(tmp_path)
  (tmp_path / 'data' / 'snapshot').mkdir(parents=True)
  (tmp_path / 'data' / 'snapshot' / 'README.md').write_text('---\n---\n')
  api = FakeHubApi()
  PublishCommand.main(['dataset', '--yes'], publisher=HubPublisher(api=api))
  assert api.calls[0] == ('create_repo', 'someone/rapsheet-charge-titles', 'dataset')
