"""Baselines: the four non-neural classifiers the encoder has to beat."""

from .majority_baseline import MajorityBaseline
from .rules_baseline import StatuteRulesBaseline
from .tfidf_baseline import TfidfBaseline
from .title_lookup_baseline import TitleLookupBaseline

__all__ = ['MajorityBaseline', 'StatuteRulesBaseline', 'TfidfBaseline', 'TitleLookupBaseline']
