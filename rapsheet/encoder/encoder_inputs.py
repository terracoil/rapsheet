"""EncoderInputs — what the encoder reads: the charge title, and optionally its statute citation."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from modgud.app import implicit_return

from rapsheet.core import Columns, Frame, Series

COLS = Columns()
SEPARATOR = '␟'  # ␟ never appears in a charge title or a citation


@dataclass(frozen=True)
class EncoderInputs:
  """Turn rows into (title, citation) text pairs, and pairs into dedup keys.

  The citation is written the way Illinois writes it, ``720 ILCS 570/402(c)``, and goes in as the
  second segment of a BERT sentence pair: ``[CLS] title [SEP] citation [SEP]``. With
  ``include_statute=False`` the second segment is empty and the model reads the title alone.
  """

  include_statute: bool = True

  @staticmethod
  @implicit_return
  def part(values: Series) -> Series:
    """Trim and upper-case one statute field; missing becomes the empty string."""
    values.astype('string').str.strip().str.upper().fillna('')

  @implicit_return
  def citation(self, frame: Frame) -> Series:
    """Return ``chapter ILCS act/section`` per row, as much of it as is present."""
    chapter = self.part(frame[COLS.CHAPTER])
    act = self.part(frame[COLS.ACT])
    section = self.part(frame[COLS.SECTION])
    head = (chapter + ' ILCS ' + act).where(act != '', chapter)
    full = (head + '/' + section).where((section != '') & (act != ''), head)
    built = full.where(chapter != '', '').str.strip()
    built if self.include_statute else pd.Series('', index=frame.index, dtype='string')

  @implicit_return
  def keys(self, frame: Frame) -> Series:
    """Return one string per row that identifies its full encoder input."""
    titles = frame[COLS.TITLE].astype('string').fillna('')
    (titles + SEPARATOR + self.citation(frame)).astype(object)

  @staticmethod
  @implicit_return
  def unpack(keys: np.ndarray) -> tuple[list[str], list[str]]:
    """Split dedup keys back into titles and citations."""
    parts = pd.Series(keys, dtype=object).str.split(SEPARATOR, n=1, expand=True)
    (parts[0].tolist(), parts[1].fillna('').tolist() if parts.shape[1] > 1 else [''] * len(keys))
