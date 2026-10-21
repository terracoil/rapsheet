"""Exception hierarchy for rapsheet."""


class RapsheetError(Exception):
  """Base class for every error raised by this package."""


class SodaFetchError(RapsheetError):
  """A SODA request failed after retries, or a page came back malformed."""


class DispositionMapError(RapsheetError):
  """The disposition map YAML is malformed or internally inconsistent."""


class DatasetError(RapsheetError):
  """The dispositions frame is missing required columns or carries a forbidden one."""


class NotFittedError(RapsheetError):
  """A baseline or model was asked to predict before ``fit`` was called."""


class PublishError(RapsheetError):
  """A push to the Hugging Face Hub was refused before anything left the machine."""
