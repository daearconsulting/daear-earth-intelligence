from __future__ import annotations

"""
daear_toolkit.climate_access
Live climate and hydrology data access for `climate-resilience-indicators`.

Design notes
Everything here hits a real service. There is no bundled sample data, on
purpose: a demo repo that ships a cached CSV proves nothing about whether the
pipeline works, and reviewers on an accelerator panel have seen enough of those
to discount them.

The tradeoff is that these functions break when upstream services change, and
they do change. gridMET and MACAv2 both live on the Northwest Knowledge
Network THREDDS server, which has moved ports and paths before; the NRCS AWDB
REST API replaced a SOAP service in 2023. Each function below states its
endpoint explicitly so it can be repaired rather than debugged.

Cache aggressively. gridMET over OPeNDAP is a slow lazy read and re-running a
notebook should not re-download 40 years of daily grids.

Services used
    gridMET   4 km daily CONUS observed, 1979-present    (THREDDS/OPeNDAP)
    MACAv2    4 km daily CONUS downscaled CMIP5          (THREDDS/OPeNDAP)
    SNOTEL    point SWE/precip/temp                      (NRCS AWDB REST)
    NWIS      daily streamflow                           (USGS water services)
"""

import functools
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import requests
import xarray as xr

CACHE_DIR = os.environ.get("DAEAR_CACHE", "./data/climate_cache")


# Region registry
@dataclass(frozen=True)
class Region:
    """A study region: bbox in (min_lon, min_lat, max_lon, max_lat), WGS84."""

    key: str
    name: str
    bbox: tuple[float, float, float, float]
    state: str
    notes: str = ""
    fire_year: int | None = None
    snotel_state: str = "CO"
    tags: tuple[str, ...] = field(default_factory=tuple)

    @property
    def centroid(self) -> tuple[float, float]:
        min_lon, min_lat, max_lon, max_lat = self.bbox
        return ((min_lon + max_lon) / 2, (min_lat + max_lat) / 2)


POUDRE = Region(
    key="poudre",
    name="Cache la Poudre/Cameron Peak",
    bbox=(-106.00, 40.40, -105.00, 40.90),
    state="CO",
    fire_year=2020,
    notes=(
        "Anchor region for the whole demo portfolio. The 2020 Cameron Peak Fire "
        "(208,913 acres, largest in Colorado history) burned the upper watershed "
        "that supplies Fort Collins and Greeley, which is why post-fire water "
        "quality here has a dollar figure attached and a motivated audience."
    ),
    tags=("burned", "municipal-supply"),
)

BLACK_HILLS = Region(
    key="black_hills",
    name="Black Hills (He Sapa)",
    bbox=(-104.50, 43.40, -103.20, 44.70),
    state="SD",
    snotel_state="SD",
    notes=(
        "Ponderosa-dominated island range in the northern Great Plains. Included "
        "because it breaks the composite index in a useful way: lower elevation "
        "and a very different snow regime from the Colorado sites, so an index "
        "that ranks it identically to the Front Range is an index that is not "
        "actually discriminating.\n\n"
        "He Sapa is treaty territory under the 1868 Fort Laramie Treaty and "
        "sacred to the Oceti Sakowin. All data used here is federally sourced and "
        "public, and nothing in this repo publishes Tribal-held data or "
        "community-specific outputs: see Module 4 for how that separation is "
        "structured rather than assumed."
    ),
    tags=("treaty-territory", "island-range"),
)

FRONT_RANGE = Region(
    key="front_range",
    name="Upper South Platte / Front Range foothills",
    bbox=(-105.70, 39.30, -105.00, 40.20),
    state="CO",
    notes=(
        "Denver's municipal supply watershed, south of and adjacent to the Poudre. "
        "Adjacency is deliberate: two regions in the same physiographic setting "
        "with different fire and management histories is the cleanest test of "
        "whether the composite index is picking up real differences or just "
        "latitude. Note the shared airshed and partly shared SNOTEL network -- "
        "these are not independent samples and should not be treated as such in "
        "any statistical claim."
    ),
    tags=("municipal-supply", "adjacent-to-poudre"),
)

REGIONS = {r.key: r for r in (POUDRE, BLACK_HILLS, FRONT_RANGE)}


# gridMET observed daily surface meteorology, 4 km, 1979-present
_GRIDMET_BASE = "http://thredds.northwestknowledge.net:8080/thredds/dodsC"

# gridMET file-name variable codes: the variable name inside the NetCDF.
# These differ, which is the single most common thing to trip over.
GRIDMET_VARS = {
    "pr": "precipitation_amount",
    "tmmx": "air_temperature",              # daily max, K
    "tmmn": "air_temperature",              # daily min, K
    "rmax": "relative_humidity",
    "rmin": "relative_humidity",
    "vpd": "mean_vapor_pressure_deficit",   # kPa
    "pet": "potential_evapotranspiration",  # mm, reference ET (alfalfa)
    "erc": "energy_release_component-g",    # NFDRS, fire danger
    "bi": "burning_index-g",
    "fm100": "dead_fuel_moisture_100hr",
    "fm1000": "dead_fuel_moisture_1000hr",
    "vs": "wind_speed",
    "srad": "surface_downwelling_shortwave_flux_in_air",
    "pdsi": "palmer_drought_severity_index",
}


def _cache_path(name: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, name)


def get_gridmet(
    region: Region | tuple,
    variables=("pr", "tmmx", "tmmn", "vpd"),
    start: str = "1991-01-01",
    end: str = "2024-12-31",
    use_cache: bool = True,
) -> xr.Dataset:
    """
    Daily gridMET surface meteorology for a region, as an xarray Dataset.

    Parameters
    region : Region or bbox tuple
    variables : sequence of gridMET codes
        See GRIDMET_VARS. 'erc' is the fire-danger workhorse; 'vpd' is the
        single best drought-stress variable for the Western US and the one I
        would keep if forced to keep only one.
    start, end : ISO date strings

    Returns a Dataset with dims (day, lat, lon), temperatures converted to C.

    Caveat that matters for trend work: gridMET is a *reanalysis-blended*
    product, not station observations. Its early record (pre-1990) is less
    constrained by station density in mountain terrain, and apparent trends
    that start in the 1980s can partly reflect changing input density rather
    than climate. Starting at 1991 is the conservative choice and is why that
    is the default here.
    """
    bbox = region.bbox if isinstance(region, Region) else tuple(region)
    key = region.key if isinstance(region, Region) else "bbox"
    min_lon, min_lat, max_lon, max_lat = bbox

    cache = _cache_path(f"gridmet_{key}_{start}_{end}_{'-'.join(variables)}.nc")
    if use_cache and os.path.exists(cache):
        return xr.open_dataset(cache)

    arrays = {}
    for code in variables:
        url = f"{_GRIDMET_BASE}/agg_met_{code}_1979_CurrentYear_CONUS.nc"
        ds = xr.open_dataset(url)

        # Internal variable name is not the file code; fall back to the sole
        # data variable if the mapping is stale after an upstream change.
        varname = GRIDMET_VARS.get(code)
        if varname not in ds:
            candidates = [v for v in ds.data_vars if ds[v].ndim == 3]
            if not candidates:
                raise KeyError(f"Cannot locate a 3-D variable for gridMET '{code}' in {url}")
            varname = candidates[0]

        da = ds[varname].sel(
            lon=slice(min_lon, max_lon),
            lat=slice(max_lat, min_lat),   # gridMET latitude is descending
            day=slice(start, end),
        )
        if code in ("tmmx", "tmmn"):
            da = da - 273.15
            da.attrs["units"] = "degC"

        arrays[code] = da.load()
        ds.close()

    out = xr.Dataset(arrays)
    out.attrs.update(source="gridMET (Abatzoglou 2013)", resolution_m=4000, region=key)
    if use_cache:
        out.to_netcdf(cache)
    return out


# MACAv2 — statistically downscaled CMIP5 projections, 4 km daily
_MACA_BASE = "http://thredds.northwestknowledge.net:8080/thredds/dodsC/MACAV2"

MACA_VARS = {"tasmax": "air_temperature", "tasmin": "air_temperature", "pr": "precipitation", "rhsmax": "relative_humidity", "rhsmin": "relative_humidity", "was": "wind_speed"}

# The five-model subset below is not arbitrary. Running all 20 MACA models is
# the right thing to do for a publication; for a demo it is prohibitively slow.
# These five span the CMIP5 range on both climate sensitivity and the
# wet/dry direction of projected precipitation change over the Southern
# Rockies, so the envelope they produce is honest about spread rather than
# quietly collapsing it. Say which models were used in any figure caption 
# a projection plot without its model list is not interpretable.
MACA_MODEL_SUBSET = ("HadGEM2-ES365", "CNRM-CM5", "MRI-CGCM3", "NorESM1-M", "IPSL-CM5A-MR")

_MACA_TIME_BLOCKS = {
    "historical": [(1950, 1954), (1955, 1959), (1960, 1964), (1965, 1969), (1970, 1974), (1975, 1979), (1980, 1984), (1985, 1989), (1990, 1994), (1995, 1999), (2000, 2005)],
    "future": [(2006, 2010), (2011, 2015), (2016, 2020), (2021, 2025), (2026, 2030), (2031, 2035), (2036, 2040), (2041, 2045), (2046, 2050), (2051, 2055), (2056, 2060), (2061, 2065), (2066, 2070), (2071, 2075), (2076, 2080), (2081, 2085), (2086, 2090), (2091, 2095), (2096, 2099)],
}


def _maca_blocks(scenario: str, start_year: int, end_year: int):
    blocks = _MACA_TIME_BLOCKS["historical" if scenario == "historical" else "future"]
    return [(a, b) for a, b in blocks if not (b < start_year or a > end_year)]


def get_macav2(
    region: Region | tuple,
    variable: str = "tasmax",
    model: str = "HadGEM2-ES365",
    scenario: str = "rcp85",
    start_year: int = 2040,
    end_year: int = 2069,
    ensemble: str = "r1i1p1",
    use_cache: bool = True,
) -> xr.DataArray:
    """
    MACAv2-METDATA downscaled projections for one model/scenario/variable.

    MACA files are stored in 5-year blocks, so a 30-year window means opening
    six URLs and concatenating. That is handled here.

    `scenario` is 'historical', 'rcp45', or 'rcp85'. Use rcp45 alongside rcp85
    in anything shown to a partner: presenting only the high-forcing pathway
    invites the objection that the analysis is picking a worst case, and the
    spread between pathways is usually smaller than the spread across models
    anyway; which is itself the more interesting finding.

    Note MACAv2 is CMIP5. LOCA2 (CMIP6) is the more current choice and is worth
    migrating to; MACA is used here because its THREDDS endpoint is stable and
    well documented, which matters more for a demo than being on the newest
    generation. Flag this in any proposal so it reads as a considered choice.
    """
    bbox = region.bbox if isinstance(region, Region) else tuple(region)
    key = region.key if isinstance(region, Region) else "bbox"
    min_lon, min_lat, max_lon, max_lat = bbox

    cache = _cache_path(f"maca_{key}_{variable}_{model}_{scenario}_{start_year}_{end_year}.nc")
    if use_cache and os.path.exists(cache):
        return xr.open_dataarray(cache)

    pieces = []
    for a, b in _maca_blocks(scenario, start_year, end_year):
        url = f"{_MACA_BASE}/{model}/macav2metdata_{variable}_{model}_{ensemble}_{scenario}_{a}_{b}_CONUS_daily.nc"
        ds = xr.open_dataset(url)
        varname = MACA_VARS.get(variable, list(ds.data_vars)[0])
        if varname not in ds:
            varname = list(ds.data_vars)[0]
        da = ds[varname]

        # MACA longitudes are 0-360; the bbox is -180/180.
        lon_name = "lon" if "lon" in da.dims else "longitude"
        lat_name = "lat" if "lat" in da.dims else "latitude"
        da = da.sel({lon_name: slice(min_lon % 360, max_lon % 360), lat_name: slice(min_lat, max_lat)})
        pieces.append(da.load())
        ds.close()

    if not pieces:
        raise RuntimeError(f"No MACA time blocks overlap {start_year}-{end_year} for scenario '{scenario}'")

    out = xr.concat(pieces, dim="time").sel(time=slice(f"{start_year}-01-01", f"{end_year}-12-31"))
    if variable in ("tasmax", "tasmin"):
        out = out - 273.15
        out.attrs["units"] = "degC"
    out.attrs.update(model=model, scenario=scenario, source="MACAv2-METDATA (CMIP5)")

    if use_cache:
        out.to_netcdf(cache)
    return out


def maca_ensemble(region, variable, scenario, start_year, end_year, models=MACA_MODEL_SUBSET, reduce="mean"):
    """
    Multi-model ensemble for one scenario, returned with a `model` dimension.

    Keeping the model dimension rather than collapsing it immediately is the
    point: the across-model spread is the uncertainty estimate, and averaging it
    away at read time makes it impossible to report.
    """
    das = []
    for m in models:
        try:
            das.append(get_macav2(region, variable, m, scenario, start_year, end_year).assign_coords(model=m))
        except Exception as exc:  # a single model failing should not kill the ensemble
            print(f"  ! {m} failed: {type(exc).__name__}: {exc}")
    if not das:
        raise RuntimeError("All models failed")
    # Grids are identical across MACA models, but guard against off-by-one extents.
    stacked = xr.concat(das, dim="model", join="override")
    return stacked if reduce is None else stacked


# SNOTEL NRCS Air and Water Database REST service
_AWDB = "https://wcc.sc.egov.usda.gov/awdbRestApi/services/v1"


def get_snotel_stations(state: str = "CO", bbox=None) -> pd.DataFrame:
    """
    SNOTEL station metadata for a state, optionally filtered to a bbox.

    Station triplets look like '551:CO:SNTL' and are the identifier every other
    AWDB call wants.
    """
    resp = requests.get(f"{_AWDB}/stations", params={"stateCds": state, "networkCds": "SNTL", "returnForecastPointMetadata": "false"}, timeout=60)
    resp.raise_for_status()
    df = pd.DataFrame(resp.json())
    if df.empty:
        return df

    df = df.rename(columns={"stationTriplet": "triplet", "latitude": "lat", "longitude": "lon", "elevation": "elev_ft"})
    if bbox is not None:
        min_lon, min_lat, max_lon, max_lat = bbox
        df = df[(df["lon"].between(min_lon, max_lon)) & (df["lat"].between(min_lat, max_lat))]
    return df.reset_index(drop=True)


def get_snotel_data(triplets, element: str = "WTEQ", start: str = "1991-10-01", end: str = "2024-09-30", duration: str = "DAILY") -> pd.DataFrame:
    """
    Daily SNOTEL time series for one or more stations.

    `element` codes: WTEQ (snow water equivalent, in), SNWD (snow depth),
    PREC (accumulated precipitation), TAVG/TMAX/TMIN.

    SWE is the variable that matters for water supply, and peak SWE plus
    melt-out date carry most of the signal. Beware the elevation-sampling
    problem: SNOTEL sites were sited to be *representative of snow*, which
    means they cluster in mid-to-upper elevation forest openings and
    systematically under-sample both low elevation and alpine. Trends from
    SNOTEL are trends at SNOTEL elevations, not basin-wide trends.

    Returns a long DataFrame: triplet, date, value.
    """
    if isinstance(triplets, str):
        triplets = [triplets]

    rows = []
    # The API accepts multiple triplets per call but rejects very long lists;
    # chunking at 20 keeps requests reliable.
    for i in range(0, len(triplets), 20):
        chunk = triplets[i : i + 20]
        resp = requests.get(
            f"{_AWDB}/data",
            params={
                "stationTriplets": ",".join(chunk),
                "elements": element,
                "duration": duration,
                "beginDate": start,
                "endDate": end,
                "periodRef": "END",
                "returnFlags": "false",
            },
            timeout=180,
        )
        resp.raise_for_status()
        for station in resp.json():
            triplet = station.get("stationTriplet")
            for series in station.get("data", []):
                for point in series.get("values", []):
                    rows.append({"triplet": triplet, "date": point.get("date"), "value": point.get("value")})

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df.attrs["element"] = element
    return df


# USGS NWIS daily streamflow
def get_streamflow(bbox=None, sites=None, start: str = "1991-01-01", end: str = "2024-12-31", parameter: str = "00060") -> pd.DataFrame:
    """
    USGS NWIS daily values. `parameter` 00060 is discharge in cfs.

    Either `bbox` or an explicit `sites` list. The bbox form is convenient but
    returns everything including short-record and regulated gauges: filter on
    record length before computing any trend, since a 12-year gauge will
    happily produce a significant-looking slope that is really just decadal
    variability.

    Regulation is the bigger trap. Many Front Range gauges sit below
    transbasin diversions and reservoirs, so their trends measure operations,
    not climate. Check the NWIS site description before interpreting anything.
    """
    params = {"format": "json", "parameterCd": parameter, "startDT": start, "endDT": end, "siteStatus": "all"}
    if sites:
        params["sites"] = ",".join(sites) if not isinstance(sites, str) else sites
    elif bbox is not None:
        params["bBox"] = ",".join(f"{v:.6f}" for v in bbox)
    else:
        raise ValueError("supply bbox or sites")

    resp = requests.get("https://waterservices.usgs.gov/nwis/dv/", params=params, timeout=300)
    resp.raise_for_status()
    series = resp.json().get("value", {}).get("timeSeries", [])

    rows = []
    for ts in series:
        info = ts["sourceInfo"]
        site_no = info["siteCode"][0]["value"]
        name = info["siteName"]
        lat = float(info["geoLocation"]["geogLocation"]["latitude"])
        lon = float(info["geoLocation"]["geogLocation"]["longitude"])
        for v in ts["values"][0]["value"]:
            rows.append({"site_no": site_no, "site_name": name, "lat": lat, "lon": lon,
                         "date": v["dateTime"][:10], "value": pd.to_numeric(v["value"], errors="coerce")})

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    df.loc[df["value"] < 0, "value"] = np.nan   # NWIS uses -999999 for missing
    return df


def filter_by_record_length(df: pd.DataFrame, min_years: int = 25, max_gap_frac: float = 0.1) -> pd.DataFrame:
    """
    Keep only gauges with a long enough, complete enough record for trend work.

    Two filters because they catch different failures: `min_years` rejects
    short records, `max_gap_frac` rejects long records that are mostly holes.
    A gauge that operated 1990-1995 and 2015-2024 spans 34 years and is still
    useless for a continuous trend.
    """
    keep = []
    for site, g in df.groupby("site_no"):
        span = (g["date"].max() - g["date"].min()).days / 365.25
        expected = span * 365.25
        gap_frac = 1 - (len(g) / expected) if expected > 0 else 1
        if span >= min_years and gap_frac <= max_gap_frac:
            keep.append(site)
    return df[df["site_no"].isin(keep)].copy()
