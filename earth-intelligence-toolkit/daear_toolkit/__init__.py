from __future__ import annotations

"""
daear_toolkit
Shared Earth observation, indicator, cube-building, governance, and
visualization utilities used across the Daear Consulting demo repo portfolio:

    wildfire-landscape-intelligence
    soil-watershed-intelligence
    tribal-wildfire-intelligence
    climate-resilience-indicators

See README.md for the status of live-data vs. synthetic-demo mode.
"""

from . import data_access
from . import indicators
from . import cube
from . import governance
from . import viz

from .regions import POUDRE_CAMERON_PEAK

__all__ = [
    "data_access",
    "indicators",
    "cube",
    "governance",
    "viz",
    "POUDRE_CAMERON_PEAK",
]

__version__ = "0.1.0"
