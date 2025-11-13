"""Agent framework for design editing."""

from .base import Agent
from .zeroshot import ZeroShotAgent
from .singleshot import SingleShotAgent
from .multishot import MultiShotAgent

__all__ = ['Agent', 'ZeroShotAgent', 'SingleShotAgent', 'MultiShotAgent']
