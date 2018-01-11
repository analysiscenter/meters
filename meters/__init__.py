"""Meters package"""
import sys

from .batch import MeterBatch
from . import dataset # pylint: disable=wildcard-import

__version__ = '0.1.0'


if sys.version_info < (3, 5):
    raise ImportError("Meters module requires Python 3.5 or higher")
