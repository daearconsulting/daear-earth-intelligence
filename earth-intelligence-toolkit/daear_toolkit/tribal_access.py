from __future__ import annotations

"""
daear_toolkit.tribal_access
Data access for `tribal-wildfire-intelligence`, with governance gating built in.

Every function here takes a `GovernanceContext` and calls `ctx.check(...)`
before returning anything. That is the difference between this module and
`fire_access` not the data, which is the same public federal data, but the
requirement that the caller has stated which tier they are operating at.

**All sources in this module are public federal datasets.** Reservation
boundaries, BIA land area representations, MTBS perimeters, and the Fire
Occurrence Database are published by federal agencies. That makes them legally
available; it does not make an *analysis* of them public by default. See
`sovereignty.GovernanceContext.check_publication`.

Sources
    Census TIGERweb AIANNH   reservation and off-reservation trust land boundaries
    BIA LAR                  land area representations (trust/fee/allotment structure)
    MTBS                     burn severity and perimeters, 1984-present
    Short FOD                Fire Program Analysis fire-occurrence records, 1992-2020
"""

import io
import os
import sqlite3
import zipfile

import numpy as np
import pandas as pd
import requests

from .sovereignty import GovernanceContext, Tier, CulturalResourceGuard

try:
    import geopandas as gpd
except ImportError:  # pragma: no cover
    gpd = None

CACHE_DIR = os.environ.get("DAEAR_CACHE", "./data/tribal_cache")


def _cache(name: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, name)


def _arcgis_query(service: str, layer: int, bbox=None, where: str = "1=1", timeout: int = 120) -> dict:
    params = {"where": where, "outFields": "*", "returnGeometry": "true",
              "outSR": 4326, "f": "geojson"}
    if bbox is not None:
        params.update({"geometry": ",".join(str(v) for v in bbox),
                       "geometryType": "esriGeometryEnvelope", "inSR": 4326,
                       "spatialRel": "esriSpatialRelIntersects"})
    resp = requests.get(f"{service}/{layer}/query", params=params, timeout=timeout)
    resp.raise_for_status()
    payload = resp.json()
    if "error" in payload:
        raise RuntimeError(f"ArcGIS service error: {payload['error']}")
    return payload


# _AIANNH_RESERVATION_LAYER = 0 # Federal AIR/off-reservation trust land; verify against the directory

# Jurisdictional boundaries
_AIANNH = "https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/AIANNHA/MapServer"
_AIANNH_RESERVATION_LAYER = 2  # Federal American Indian Reservations (current)


_BIA_LAR = "https://biamaps.geoplatform.gov/server/rest/services/DivLTR/BIA_Land_Area_Representations/MapServer"
_BIA_LAR_LAYER = 0


def get_tribal_boundary(ctx: GovernanceContext, name: str | None = None, bbox=None):
    """
    Census AIANNH boundary for a reservation or off-reservation trust land.

    `name` matches against the Census NAME field (ex. "Pine Ridge"). Either
    `name` or `bbox` is required.

    A caveat that matters analytically, not just legally: the Census AIANNH
    polygon is an administrative boundary and is not the same thing as the
    Nation's territory, its treaty territory, or the extent of its interest. For
    the Oceti Sakowin, the 1868 Fort Laramie Treaty territory is vastly larger
    than the present reservation polygons. An analysis that silently equates
    "the reservation boundary" with "the area of Tribal concern" has made a
    substantive claim it did not intend to make so say which one you mean.
    """
    if gpd is None:
        raise ImportError("geopandas is required for get_tribal_boundary()")
    ctx.check(Tier.PUBLIC, dataset="census_aiannh", action="read boundary")

    where = f"UPPER(NAME) LIKE '%{name.upper()}%'" if name else "1=1"
    geojson = _arcgis_query(_AIANNH, _AIANNH_RESERVATION_LAYER, bbox=bbox, where=where)
    gdf = gpd.GeoDataFrame.from_features(geojson["features"], crs="EPSG:4326")
    if gdf.empty:
        raise RuntimeError(f"No AIANNH polygon matched name={name!r} bbox={bbox}. "
                           f"Check the layer index against the service directory Census renumbers these.")
    gdf.columns = [c.lower() for c in gdf.columns]
    return gdf.reset_index(drop=True)


def get_land_status(ctx: GovernanceContext, bbox):
    """
    BIA Land Area Representations: the trust/allotted/fee ownership mosaic.

    This is the layer that makes Tribal wildfire different from any other
    wildfire analysis, and it is the one most often omitted.

    Allotment under the Dawes Act (1887) and subsequent fee patenting produced a
    checkerboard of trust land, individually allotted trust land, and fee land
    inside reservation boundaries. The practical consequence for fire: response
    authority, funding eligibility, and treatment permitting can change parcel
    to parcel. BIA fire crews, a Tribal fire department, county districts, and
    federal agencies may all hold authority within one drainage.

    An exposure analysis that treats the reservation as one jurisdiction gets
    the fire-response picture wrong in a way that matters operationally. That is
    an analytical failure, not a political nicety.
    """
    if gpd is None:
        raise ImportError("geopandas is required for get_land_status()")
    ctx.check(Tier.PUBLIC, dataset="bia_lar", action="read land status")

    geojson = _arcgis_query(_BIA_LAR, _BIA_LAR_LAYER, bbox=bbox)
    gdf = gpd.GeoDataFrame.from_features(geojson["features"], crs="EPSG:4326")
    if not gdf.empty:
        gdf.columns = [c.lower() for c in gdf.columns]
    return gdf.reset_index(drop=True)


# Fire history
_MTBS_SERVICE = "https://apps.fs.usda.gov/arcx/rest/services/EDW/EDW_MTBS_01/MapServer"


def get_fire_history(ctx: GovernanceContext, bbox, start_year: int = 1984, end_year: int = 2024, min_acres: float = 500):
    """
    MTBS fire perimeters across a multi-decade window.

    **The record starts in 1984 and that start date is not neutral.** MTBS
    begins with Landsat 5. What it shows is the satellite era, which is also the
    era of maximum fire suppression and of the near-total interruption of
    Indigenous cultural burning.

    So a fire-history analysis built only on MTBS will describe a landscape with
    an unusually low fire frequency and conclude that recent fires are
    anomalous. Some are. But the baseline it is comparing against is itself the
    anomaly is the product of policy, not of ecology. Module 1 handles this
    explicitly rather than treating 1984-present as "the historical record".
    """
    if gpd is None:
        raise ImportError("geopandas is required for get_fire_history()")
    ctx.check(Tier.PUBLIC, dataset="mtbs", action="read fire history")

    geojson = _arcgis_query(_MTBS_SERVICE, 0, bbox=bbox)
    gdf = gpd.GeoDataFrame.from_features(geojson["features"], crs="EPSG:4326")
    if gdf.empty:
        return gdf
    gdf.columns = [c.lower() for c in gdf.columns]

    for col in ("year", "fire_year", "ig_year", "startyear"):
        if col in gdf.columns:
            years = pd.to_numeric(gdf[col], errors="coerce")
            gdf = gdf[(years >= start_year) & (years <= end_year)]
            gdf["fire_year"] = years[gdf.index]
            break
    for col in ("acres", "burnbndac", "fire_acres"):
        if col in gdf.columns:
            gdf["acres"] = pd.to_numeric(gdf[col], errors="coerce")
            gdf = gdf[gdf["acres"] >= min_acres]
            break

    return gdf.reset_index(drop=True)


_FOD_URL = "https://www.fs.usda.gov/rds/archive/products/RDS-2013-0009.6/RDS-2013-0009.6_SQLITE.zip"


def get_ignitions(ctx: GovernanceContext, bbox, start_year: int = 1992, end_year: int = 2020, fod_path: str | None = None):
    """
    Fire Program Analysis Fire-Occurrence Database (Short 2022), RDS-2013-0009.6.

    Ignition points with cause and discovery date, 1992-2020. Unlike MTBS this
    captures small fires, which is where the fire-response story mostly lives 
    most ignitions never become large, and the ones that do not are evidence of
    successful initial attack.

    Not an API: a ~700 MB SQLite archive from the Forest Service Research Data
    Archive, downloaded once and cached. Pass `fod_path` if you already have it.

    **Read the cause codes carefully and skeptically.** NWCG statistical cause
    is assigned by the reporting agency, "Missing data/not specified/undetermined"
    is a large category, and reporting practice varies systematically across
    jurisdictions, including between BIA, Tribal, county, and federal reporters
    within a single reservation. Cause-attribution differences across a
    jurisdictional boundary may be reporting differences rather than behavioural
    ones, and reading them as behavioural is how a deficit narrative gets
    manufactured out of an artifact.
    """
    ctx.check(Tier.PUBLIC, dataset="fod", action="read ignition records")

    path = fod_path or _cache("fod.sqlite")
    if not os.path.exists(path):
        print(f"Downloading FOD (~700 MB, one time) from the Forest Service Research Data Archive ...")
        resp = requests.get(_FOD_URL, timeout=3600, stream=True)
        resp.raise_for_status()
        blob = io.BytesIO(resp.content)
        with zipfile.ZipFile(blob) as z:
            db = next(n for n in z.namelist() if n.lower().endswith((".sqlite", ".db")))
            with z.open(db) as src, open(path, "wb") as dst:
                dst.write(src.read())
        print(f"  cached at {path}")

    min_lon, min_lat, max_lon, max_lat = bbox
    query = """
        SELECT FOD_ID, FIRE_YEAR, DISCOVERY_DOY, NWCG_GENERAL_CAUSE, FIRE_SIZE,
               FIRE_SIZE_CLASS, LATITUDE, LONGITUDE, STATE, OWNER_DESCR
        FROM Fires
        WHERE LATITUDE BETWEEN ? AND ? AND LONGITUDE BETWEEN ? AND ?
          AND FIRE_YEAR BETWEEN ? AND ?
    """
    with sqlite3.connect(path) as conn:
        try:
            df = pd.read_sql_query(query, conn, params=(min_lat, max_lat, min_lon, max_lon, start_year, end_year))
        except pd.errors.DatabaseError:
            # Column names differ between FOD releases; fall back to SELECT *.
            df = pd.read_sql_query(
                "SELECT * FROM Fires WHERE LATITUDE BETWEEN ? AND ? AND LONGITUDE BETWEEN ? AND ?",
                conn, params=(min_lat, max_lat, min_lon, max_lon))
            df.columns = [c.upper() for c in df.columns]
            df = df[(df["FIRE_YEAR"] >= start_year) & (df["FIRE_YEAR"] <= end_year)]

    df.columns = [c.lower() for c in df.columns]
    if gpd is not None and not df.empty:
        return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["longitude"], df["latitude"]), crs="EPSG:4326")
    return df


# Safe-by-default loader for community-provided data
def load_community_dataset(ctx: GovernanceContext, path: str, dataset_name: str, tier: Tier = Tier.PARTNER):
    """
    Load a dataset provided by a Nation, with the guard rails applied first.

    Order of operations is the whole point: check the agreement, then scan for
    cultural resource content, then load. Scanning after loading means the data
    is already in memory, already in the notebook's output, and possibly already
    in a checkpoint file.
    """
    if gpd is None:
        raise ImportError("geopandas is required for load_community_dataset()")

    ctx.check(tier, dataset=dataset_name, action="load community dataset")
    gdf = gpd.read_file(path)
    CulturalResourceGuard.scan(gdf, raise_on_match=True)

    print(f"Loaded '{dataset_name}' ({len(gdf)} features) at {tier.name} tier under "
          f"{ctx.agreement.summary() if ctx.agreement else 'no agreement'}")
    return gdf
