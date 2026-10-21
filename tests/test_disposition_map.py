import pandas as pd
import pytest
from rapsheet.core import DispositionMapError
from rapsheet.dataset import DispositionMap

HANDOFF_STRINGS = [
  'Nolle Prosecution',
  'Plea Of Guilty',
  'FNG',
  'Finding Guilty',
  'Death Suggested-Cause Abated',
  'Verdict Guilty',
  'Verdict-Not Guilty',
  'Finding Not Not Guilty',
  'Superseded by Indictment',
  'BFW',
  'Case Dismissed',
  'FNG Reason Insanity',
  'Finding Guilty - Lesser Included',
  'Plea of Guilty - Amended Charge',
  'Mistrial Declared',
  'SOL',
  'Transferred - Misd Crt',
  'Charge Vacated',
  'Nolle On Remand',
  'Plea of Guilty - Lesser Included',
  'FNPC',
  'Verdict Guilty - Lesser Included',
  'WOWI',
  'Finding Guilty - Amended Charge',
  'Withdrawn',
  'Finding Guilty But Mentally Ill',
  'Plea of Guilty But Mentally Ill',
  'Charge Rejected',
  'Sexually Dangerous Person',
  'Hold Pending Interlocutory',
  'SOLW',
  'Charge Reversed',
  'Verdict Guilty - Amended Charge',
  'Nolle Pros - Aonic',
  'Verdict Guilty But Mentally Ill',
  'TRAN-FAI',
]


def test_v1_covers_all_36_strings_exactly_once(map_path):
  m = DispositionMap.from_yaml(map_path)
  assert m.version == 1
  assert len(HANDOFF_STRINGS) == 36
  assert set(m.vocabulary) == set(HANDOFF_STRINGS)
  assert len(m.vocabulary) == 36


def test_unmapped_strings_are_reported_not_guessed(map_path):
  m = DispositionMap.from_yaml(map_path)
  s = pd.Series(['FNG', 'Mystery Outcome', 'Mystery Outcome', 'Plea Of Guilty'])
  assert m.map_series(s).tolist() == ['acquitted', None, None, 'convicted']
  assert m.unmapped(s).to_dict() == {'Mystery Outcome': 2}
  assert m.coverage(s).loc['UNMAPPED', 'rows'] == 2


def test_flagged_strings_carry_their_current_class(map_path):
  flagged = DispositionMap.from_yaml(map_path).flagged().set_index('raw')
  assert flagged.loc['BFW', 'class'] == 'non_final'
  assert flagged.loc['FNG Reason Insanity', 'class'] == 'acquitted'


def test_duplicate_string_across_classes_is_rejected(tmp_path):
  bad = tmp_path / 'bad.yaml'
  bad.write_text('version: 9\nclasses:\n  a: [X]\n  b: [X]\n')
  with pytest.raises(DispositionMapError, match='more than one class'):
    DispositionMap.from_yaml(bad)


def test_missing_version_is_rejected(tmp_path):
  bad = tmp_path / 'bad.yaml'
  bad.write_text('classes:\n  a: [X]\n')
  with pytest.raises(DispositionMapError, match='version'):
    DispositionMap.from_yaml(bad)
