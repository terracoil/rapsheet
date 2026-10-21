"""Column policy for the Cook County SAO Dispositions table (Socrata 4x4 ``apwk-dzx8``).

Everything the package touches is named here, once. The protected and leakage columns are
listed so they can be refused at load time rather than quietly ignored.
"""

from dataclasses import dataclass

from modgud.app import implicit_return


@dataclass(frozen=True)
class Columns:
  """Named columns, grouped by the role they play."""

  DATASET_ID: str = 'apwk-dzx8'

  # Identity and record structure
  CASE_ID: str = 'case_id'
  PARTICIPANT_ID: str = 'case_participant_id'
  CHARGE_ID: str = 'charge_id'
  CHARGE_VERSION_ID: str = 'charge_version_id'
  PRIMARY_CHARGE: str = 'primary_charge'

  # Model input
  TITLE: str = 'disposition_charged_offense_title'

  # Rules baseline + optional grade head (withheld from the model)
  CHAPTER: str = 'disposition_charged_chapter'
  ACT: str = 'disposition_charged_act'
  SECTION: str = 'disposition_charged_section'
  CLASS: str = 'disposition_charged_class'

  # Lookup input (36 raw strings → disposition_map.yaml)
  DISPOSITION: str = 'charge_disposition'

  # Label
  LABEL: str = 'updated_offense_category'

  # Time and venue
  RECEIVED_DATE: str = 'received_date'
  DISPOSITION_DATE: str = 'disposition_date'
  COURT_FACILITY: str = 'court_facility'

  # Derived in DispositionsDataset — never pulled
  LABEL_TRAIN: str = 'label_train'
  RARE_TITLE: str = 'rare_title'
  SPLIT: str = 'split'

  @property
  @implicit_return
  def used(self) -> tuple[str, ...]:
    """Columns selected from the portal, in ``$select`` order."""
    (
      self.CASE_ID,
      self.PARTICIPANT_ID,
      self.CHARGE_ID,
      self.CHARGE_VERSION_ID,
      self.PRIMARY_CHARGE,
      self.TITLE,
      self.CHAPTER,
      self.ACT,
      self.SECTION,
      self.CLASS,
      self.DISPOSITION,
      self.LABEL,
      self.RECEIVED_DATE,
      self.DISPOSITION_DATE,
      self.COURT_FACILITY,
    )

  @property
  @implicit_return
  def protected(self) -> tuple[str, ...]:
    """Never pulled, never features. Named as dropped in the model card."""
    (
      'race',
      'gender',
      'age_at_incident',
      'judge',
      'incident_city',
      'law_enforcement_agency',
      'unit',
    )

  @property
  @implicit_return
  def leakage(self) -> tuple[str, ...]:
    """Pre-update label; would leak the target."""
    ('offense_category',)

  @property
  @implicit_return
  def forbidden(self) -> tuple[str, ...]:
    """Union of protected and leakage columns — refused at load time."""
    self.protected + self.leakage

  @property
  @implicit_return
  def required(self) -> tuple[str, ...]:
    """Columns a frame must carry before cleaning can start."""
    (
      self.CASE_ID,
      self.CHARGE_ID,
      self.CHARGE_VERSION_ID,
      self.PRIMARY_CHARGE,
      self.TITLE,
      self.DISPOSITION,
      self.LABEL,
      self.RECEIVED_DATE,
    )
