"""
Shared demo region definitions.

All four downstream repos import POUDRE_CAMERON_PEAK so the portfolio
demonstrates one continuous story about one real place, rather than four
disconnected toy examples.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Region:
    name: str
    description: str
    bbox: tuple  # (min_lon, min_lat, max_lon, max_lat), WGS84
    fire_name: str
    fire_year: int
    fire_acres: int  # verified: en.wikipedia.org/wiki/Cameron_Peak_fire (accessed 2026-07)
    fire_start: str
    fire_contained: str
    watershed: str
    downstream_communities: tuple


POUDRE_CAMERON_PEAK = Region(
    name="Cache la Poudre Watershed \u2014 Cameron Peak Burn Scar",
    description=(
        "Upper Cache la Poudre River watershed, Larimer County, Colorado, "
        "spanning the area burned by the 2020 Cameron Peak Fire down to the "
        "river's exit onto the Front Range near Fort Collins."
    ),
    bbox=(-105.85, 40.55, -105.45, 40.85),
    fire_name="Cameron Peak Fire",
    fire_year=2020,
    fire_acres=208_663,  # corrected from an earlier unverified figure of 208,913
    fire_start="2020-08-13",
    fire_contained="2020-12-02",
    watershed="Cache la Poudre River",
    downstream_communities=("Fort Collins", "Greeley", "Poudre Canyon communities"),
)
