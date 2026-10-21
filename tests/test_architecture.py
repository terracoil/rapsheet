"""Package-graph fitness test: the sub-package layering, enforced."""

import ast
import graphlib
from pathlib import Path

import pytest

ROOT: Path = Path(__file__).resolve().parent.parent
PACKAGE: Path = ROOT / 'rapsheet'

# Who may import whom. Anything not listed is a layering violation.
ALLOWED: dict[str, frozenset[str]] = {
  'core': frozenset(),
  'ingest': frozenset({'core'}),
  'dataset': frozenset({'core'}),
  'baselines': frozenset({'core', 'dataset'}),
  'encoder': frozenset({'core', 'dataset'}),
  'evaluation': frozenset({'core', 'dataset'}),
  'publish': frozenset({'core', 'dataset'}),
  'cli': frozenset({'core', 'ingest', 'dataset', 'publish'}),
  'figures': frozenset({'core'}),
}
SOURCES: list[Path] = sorted(PACKAGE.glob('*/*.py'))
TESTS: list[Path] = sorted((ROOT / 'tests').glob('*.py'))


def test_allowed_edges_form_a_dag() -> None:
  assert list(graphlib.TopologicalSorter(ALLOWED).static_order())[0] == 'core'


def test_every_sub_package_is_declared() -> None:
  assert {p.parent.name for p in SOURCES} == set(ALLOWED)


@pytest.mark.parametrize('path', SOURCES, ids=lambda p: f'{p.parent.name}/{p.name}')
def test_package_imports_follow_the_layering(path: Path) -> None:
  own: str = path.parent.name
  nodes: list[ast.ImportFrom] = [
    n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n, ast.ImportFrom)
  ]
  for node in nodes:
    assert node.level in (0, 1), f'{path.name}: multi-dot relative import'
    assert not (node.level == 1 and node.module is None), f'{path.name}: `from . import` banned'
    if node.level == 0 and node.module and node.module.startswith('rapsheet.'):
      parts: list[str] = node.module.split('.')
      assert len(parts) == 2, f'{path.name}: import from a module file: {node.module}'
      assert parts[1] != own, f'{path.name}: absolute self-import; use `from .module import`'
      assert parts[1] in ALLOWED[own], f'{own} -> {parts[1]} is not an allowed edge'
    if path.name == '__init__.py':
      assert node.level == 1, f'{own}/__init__.py may only re-export from its own folder'


@pytest.mark.parametrize('path', TESTS, ids=lambda p: p.name)
def test_tests_import_from_sub_package_inits(path: Path) -> None:
  nodes: list[ast.ImportFrom] = [
    n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n, ast.ImportFrom)
  ]
  for node in nodes:
    if node.module and node.module.startswith('rapsheet'):
      assert len(node.module.split('.')) == 2, f'{path.name}: {node.module}'
