"""TfidfBaseline — the bag of n-grams: word + character TF-IDF into logistic regression."""

from dataclasses import dataclass, replace

import numpy as np
from modgud.app import implicit_return
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion

from rapsheet.core import Columns, Frame, Matrix, NotFittedError
from rapsheet.dataset import KeyedLabelCounts, LabelSpace

COLS = Columns()


@dataclass(frozen=True)
class TfidfBaseline:
  """The classical model that usually embarrasses a transformer on short, stylized text.

  - **Words** split on whitespace and ``/`` only, so ``<$300`` and ``PRECONV`` survive as tokens.
  - **Characters** (2–5, word-bounded) catch clerk abbreviations: ``AGG``, ``UUW``, ``MFG/DEL``.
  - **Weighted unique pairs**: the classifier trains on each distinct (title, label) pair with its
    count as the weight — the same loss as every row, in a fraction of the time. The vectorizer's
    IDF is fitted on distinct titles, so a title's frequency doesn't distort its own features.
  """

  space: LabelSpace
  C: float = 10.0
  max_iter: int = 3000
  seed: int = 42
  name: str = 'TF-IDF + logistic regression'
  vectorizer: FeatureUnion | None = None
  classifier: LogisticRegression | None = None

  @staticmethod
  @implicit_return
  def make_vectorizer() -> FeatureUnion:
    """Return the unfitted word + character feature union."""
    FeatureUnion(
      [
        (
          'words',
          TfidfVectorizer(
            token_pattern=r'[^\s/]+', ngram_range=(1, 2), sublinear_tf=True, lowercase=False
          ),
        ),
        (
          'chars',
          TfidfVectorizer(
            analyzer='char_wb', ngram_range=(2, 5), sublinear_tf=True, lowercase=False
          ),
        ),
      ]
    )

  @implicit_return
  def fit(self, train: Frame) -> 'TfidfBaseline':
    """Fit the vectorizer on distinct titles and the classifier on weighted (title, label) pairs."""
    labels = self.space.encode(train[COLS.LABEL_TRAIN])
    counts = KeyedLabelCounts.build(train[COLS.TITLE], labels, self.space.size)
    titles, label_idx, weights = counts.pairs()
    vectorizer = self.make_vectorizer().fit(list(counts.keys))
    classifier = LogisticRegression(C=self.C, max_iter=self.max_iter, random_state=self.seed)
    classifier.fit(vectorizer.transform(list(titles)), label_idx, sample_weight=weights)
    replace(self, vectorizer=vectorizer, classifier=classifier)

  @implicit_return
  def fitted(self) -> tuple[FeatureUnion, LogisticRegression]:
    """Return the fitted vectorizer and classifier, or raise if ``fit`` hasn't run."""
    if self.vectorizer is None or self.classifier is None:
      raise NotFittedError(f'{self.name}: call fit() before predicting')
    (self.vectorizer, self.classifier)

  @implicit_return
  def predict_proba(self, frame: Frame) -> Matrix:
    """Score each distinct title once, then spread the rows back out."""
    vectorizer, classifier = self.fitted()
    titles = frame[COLS.TITLE].astype(str).to_numpy()
    unique, inverse = np.unique(titles, return_inverse=True)
    scored = (
      classifier.predict_proba(vectorizer.transform(list(unique)))
      if len(unique)
      else np.zeros((0, len(classifier.classes_)))
    )
    full = np.zeros((len(unique), self.space.size))
    full[:, classifier.classes_] = scored
    full[inverse]
