from __future__ import annotations

"""
daear_toolkit.fire_access

Fire-specific data sources for `wildfire-landscape-intelligence`.

The core toolkit already provides Sentinel-2 (`get_optical_scene`), terrain
(`get_terrain`), soils (`get_soil_properties`), and MTBS severity rasters
(`get_burn_severity`). This module adds what the four wildfire modules need on
top of that:

    get_landfire            fuel model/canopy structure (Module 1)
    get_mtbs_perimeters     fire boundary polygons (Modules 2-4)
    get_active_fire         VIIRS/MODIS detections for progression (Module 2)
    get_buildings           OSM structure footprints (Module 4)
    get_places              Census incorporated places (Module 4)
    get_water_intakes       manual registry of Poudre municipal intakes (Module 4)

Access patterns differ a lot across these. LANDFIRE is an asynchronous GP
service job (submit, poll, download). FIRMS is a simple CSV endpoint but needs
a free key. OSM Overpass is open but rate-limited and will refuse large bboxes.
Each function documents its own failure mode, because they fail differently and
generic error handling would hide that.
"""

import io
import os
import time
import zipfile

import numpy as np
import pandas as pd
import requests

try:
    import geopandas as gpd
    from shapely.geometry import shape
except ImportError:  # pragma: no cover
    gpd = None

try:
    import xarray as xr
except ImportError:  # pragma: no cover
    xr = None

CACHE_DIR = os.environ.get("DAEAR_CACHE", "./data/fire_cache")


def _cache(name: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, name)


def _arcgis_query(service: str, layer: int, bbox, where: str = "1=1", timeout: int = 90) -> dict:
    """
    Query an ArcGIS REST FeatureLayer by bbox, returning a GeoJSON-style dict.

    Some ArcGIS services return null geometry when queried with f=geojson.
    This function tries f=geojson first; if features come back with null
    geometry it retries with f=json and converts to GeoJSON manually.
    """
    params = {
        "geometry":     ",".join(str(v) for v in bbox),
        "geometryType": "esriGeometryEnvelope",
        "inSR":         4326,
        "outSR":        4326,
        "spatialRel":   "esriSpatialRelIntersects",
        "where":        where,
        "outFields":    "*",
        "returnGeometry": "true",
        "f":            "geojson",
    }
    url = f"{service}/{layer}/query"

    resp = requests.get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    payload = resp.json()

    if "error" in payload:
        raise RuntimeError(f"ArcGIS service error: {payload['error']}")

    # Check if features came back with null geometry, if so, retry with f=json
    features = payload.get("features", [])
    if features and features[0].get("geometry") is None:
        params["f"] = "json"
        resp2 = requests.get(url, params=params, timeout=timeout)
        resp2.raise_for_status()
        esri_json = resp2.json()

        if "error" in esri_json:
            raise RuntimeError(f"ArcGIS service error (json retry): {esri_json['error']}")

        # Convert esri JSON features to GeoJSON-style
        geojson_features = []
        for feat in esri_json.get("features", []):
            geom = feat.get("geometry")
            if geom is None:
                continue
            # convert esri polygon to GeoJSON polygon
            if "rings" in geom:
                geo = {"type": "Polygon", "coordinates": geom["rings"]}
            elif "paths" in geom:
                geo = {"type": "MultiLineString", "coordinates": geom["paths"]}
            elif "x" in geom:
                geo = {"type": "Point", "coordinates": [geom["x"], geom["y"]]}
            else:
                continue
            geojson_features.append({
                "type":       "Feature",
                "geometry":   geo,
                "properties": feat.get("attributes", {}),
            })

        return {"type": "FeatureCollection", "features": geojson_features}

    return payload

# Pre-fire vintage for Cameron Peak (burned 2020): use LF 2020 (200*).
# Using a post-fire LANDFIRE release to describe pre-fire fuels is circular 
# the release already incorporates the burn as a disturbance.
LANDFIRE_PREFIRE_2020 = {k: v.replace("230", "200") for k, v in LANDFIRE_LAYERS.items()}

def get_landfire(
    bbox,
    layers=("fbfm40", "cc", "ch"),
    prefire: bool = False,
    poll_seconds: int = 10,
    max_wait: int = 900,
    ):
    """
    Fetch LANDFIRE layers for a bbox via the LANDFIRE Product Service GP job API.

    LFPS is an asynchronous job service: submit a POST request, poll for
    completion, then retrieve a zip of GeoTIFFs. Small bbox jobs typically
    finish in 1-3 minutes; large ones can take longer.

    Parameters
    prefire : bool
        Use the LF 2020 (200*) vintage. Set True for any pre-fire conditions
        analysis of Cameron Peak: see note on LANDFIRE_PREFIRE_2020.

    Failure mode
    Job status 'Failed' usually means a layer code no longer exists in the
    current release. Check LANDFIRE_LAYERS codes against the current LFPS
    service directory when this happens.
    """
    if xr is None:
        raise ImportError("xarray is required for get_landfire()")

    table = LANDFIRE_PREFIRE_2020 if prefire else LANDFIRE_LAYERS
    codes = [table[l] for l in layers]
    min_lon, min_lat, max_lon, max_lat = bbox

    tag = (f"landfire_{'pre' if prefire else 'cur'}"
        f"_{'-'.join(layers)}"
        f"_{min_lon:.3f}_{min_lat:.3f}.zip")
    path = _cache(tag)

    if not os.path.exists(path):
        # Submit job
        submit = requests.get(
            _LFPS_SUBMIT,
            params={
                "Layer_List":        ";".join(codes),
                "Area_of_Interest":  f"{min_lon} {min_lat} {max_lon} {max_lat}",
                "Output_Projection": "4326",
                "f":                 "json",
            },
            timeout=90,
        )
        submit.raise_for_status()

        if not submit.text.strip():
            raise RuntimeError(
                f"LFPS returned an empty body (status {submit.status_code}).\n"
                f"Content-Type: {submit.headers.get('Content-Type')}\n"
                f"URL: {submit.url}\n"
                f"This usually means either (a) one of the requested layer "
                f"codes is invalid for this vintage, or (b) something between "
                f"you and lfps.usgs.gov (proxy/AV) stripped the response body.\n"
                f"Requested codes: {codes}\n"
                f"Try hitting the URL above directly in a browser to see the raw response."
            )

        try:
            result = submit.json()
        except ValueError:
            raise RuntimeError(
                f"LFPS did not return valid JSON (status {submit.status_code}).\n"
                f"Content-Type: {submit.headers.get('Content-Type')}\n"
                f"First 500 chars of response:\n{submit.text[:500]}"
            )

        job_id = (result.get("jobId")
                or result.get("JobId")
                or result.get("jobid"))
        if not job_id:
            raise RuntimeError(
                f"LFPS did not return a job id.\n"
                f"Response: {submit.text[:400]}\n"
                f"Check that layer codes are valid: {codes}"
            )
        print(f"  LANDFIRE job {job_id} submitted; polling every {poll_seconds}s ...")

        # Poll for completion
        waited = 0
        while waited < max_wait:
            time.sleep(poll_seconds)
            waited += poll_seconds

            poll = requests.get(
                f"{_LFPS_JOBS}/{job_id}",
                params={"f": "json"},
                timeout=60,
            )
            poll.raise_for_status()
            status_json = poll.json()
            state = str(
                status_json.get("jobStatus")
                or status_json.get("status")
                or ""
            ).lower()

            print(f"  ... {waited}s elapsed, status: {state}")

            if "succeeded" in state or "success" in state:
                # Retrieve output file URL
                out_resp = requests.get(
                    f"{_LFPS_JOBS}/{job_id}/results/Output_File",
                    params={"f": "json"},
                    timeout=60,
                )
                out_resp.raise_for_status()
                out_json = out_resp.json()
                url = (out_json.get("value")
                    or out_json.get("outputFile")
                    or out_json.get("url"))
                if not url:
                    raise RuntimeError(
                        f"Job succeeded but no output URL found.\n"
                        f"Result response: {out_json}"
                    )
                blob = requests.get(url, timeout=600)
                blob.raise_for_status()
                with open(path, "wb") as fh:
                    fh.write(blob.content)
                print(f"  LANDFIRE download complete → {path}")
                break

            if "fail" in state or "error" in state:
                raise RuntimeError(
                    f"LANDFIRE job {job_id} failed.\n"
                    f"Status response: {status_json}\n"
                    f"Check layer codes against current LFPS release: {codes}"
                )
        else:
            raise TimeoutError(
                f"LANDFIRE job {job_id} did not finish within {max_wait}s. "
                f"Try increasing max_wait or reducing the bbox."
            )


# LANDFIRE fuels and canopy structure
# LFPS is an ArcGIS GP service, not a REST API.
# Submit: POST to submitJob
# Poll:   GET  jobs/{jobId}
# Output: GET  jobs/{jobId}/results/Output_File

_LFPS_BASE = (
    "https://lfps.usgs.gov/arcgis/rest/services/"
    "LandfireProductService/GPServer/LandfireProductService"
)
_LFPS_SUBMIT = f"{_LFPS_BASE}/submitJob"
_LFPS_JOBS   = f"{_LFPS_BASE}/jobs"

# Layer codes change with each LANDFIRE release. The prefix encodes the
# vintage: 200* = LF 2020, 220* = LF 2022, 230* = LF 2023, 240* = LF 2024.
# Check https://lfps.usgs.gov/arcgis/rest/services for current release.
LANDFIRE_LAYERS = {
    "fbfm40": "230FBFM40",   # Scott & Burgan 40 fire behavior fuel models
    "cc":     "230CC",       # canopy cover (%)
    "ch":     "230CH",       # canopy height (m*10)
    "cbh":    "230CBH",      # canopy base height (m*10)
    "cbd":    "230CBD",      # canopy bulk density (kg/m3*100)
    "evt":    "230EVT",      # existing vegetation type
    "evc":    "230EVC",      # existing vegetation cover
    "slpd":   "230SLPD",     # slope degrees
}

# Pre-fire vintage for Cameron Peak (burned 2020): use LF 2020 (200*).
# Using a post-fire LANDFIRE release to describe pre-fire fuels is circular 
# the release already incorporates the burn as a disturbance.
LANDFIRE_PREFIRE_2020 = {k: v.replace("230", "200") for k, v in LANDFIRE_LAYERS.items()}

def get_landfire(
    bbox,
    layers=("fbfm40", "cc", "ch"),
    prefire: bool = False,
    poll_seconds: int = 10,
    max_wait: int = 900,
    ):
    """
    Fetch LANDFIRE layers for a bbox via the LANDFIRE Product Service GP job API.

    LFPS is an asynchronous job service: submit a POST request, poll for
    completion, then retrieve a zip of GeoTIFFs. Small bbox jobs typically
    finish in 1-3 minutes; large ones can take longer.

    Parameters
    prefire : bool
        Use the LF 2020 (200*) vintage. Set True for any pre-fire conditions
        analysis of Cameron Peak — see note on LANDFIRE_PREFIRE_2020.

    Failure mode
    Job status 'Failed' usually means a layer code no longer exists in the
    current release. Check LANDFIRE_LAYERS codes against the current LFPS
    service directory when this happens.
    """
    if xr is None:
        raise ImportError("xarray is required for get_landfire()")

    table = LANDFIRE_PREFIRE_2020 if prefire else LANDFIRE_LAYERS
    codes = [table[l] for l in layers]
    min_lon, min_lat, max_lon, max_lat = bbox

    tag = (f"landfire_{'pre' if prefire else 'cur'}"
        f"_{'-'.join(layers)}"
        f"_{min_lon:.3f}_{min_lat:.3f}.zip")
    path = _cache(tag)

    if not os.path.exists(path):
        # Submit job
        submit = requests.get(
            _LFPS_SUBMIT,
            params={
                "Layer_List":        ";".join(codes),
                "Area_of_Interest":  f"{min_lon} {min_lat} {max_lon} {max_lat}",
                "Output_Projection": "4326",
                "f":                 "json",
            },
            timeout=90,
        )
        submit.raise_for_status()
        result = submit.json()

        job_id = (result.get("jobId")
                or result.get("JobId")
                or result.get("jobid"))
        if not job_id:
            raise RuntimeError(
                f"LFPS did not return a job id.\n"
                f"Response: {submit.text[:400]}\n"
                f"Check that layer codes are valid: {codes}"
            )
        print(f"  LANDFIRE job {job_id} submitted; polling every {poll_seconds}s ...")

        # Poll for completion
        waited = 0
        while waited < max_wait:
            time.sleep(poll_seconds)
            waited += poll_seconds

            poll = requests.get(
                f"{_LFPS_JOBS}/{job_id}",
                params={"f": "json"},
                timeout=60,
            )
            poll.raise_for_status()
            status_json = poll.json()
            state = str(
                status_json.get("jobStatus")
                or status_json.get("status")
                or ""
            ).lower()

            print(f"  ... {waited}s elapsed, status: {state}")

            if "succeeded" in state or "success" in state:
                # Retrieve output file URL
                out_resp = requests.get(
                    f"{_LFPS_JOBS}/{job_id}/results/Output_File",
                    params={"f": "json"},
                    timeout=60,
                )
                out_resp.raise_for_status()
                out_json = out_resp.json()
                url = (out_json.get("value")
                    or out_json.get("outputFile")
                    or out_json.get("url"))
                if not url:
                    raise RuntimeError(
                        f"Job succeeded but no output URL found.\n"
                        f"Result response: {out_json}"
                    )
                blob = requests.get(url, timeout=600)
                blob.raise_for_status()
                with open(path, "wb") as fh:
                    fh.write(blob.content)
                print(f"  LANDFIRE download complete → {path}")
                break

            if "fail" in state or "error" in state:
                raise RuntimeError(
                    f"LANDFIRE job {job_id} failed.\n"
                    f"Status response: {status_json}\n"
                    f"Check layer codes against current LFPS release: {codes}"
                )
        else:
            raise TimeoutError(
                f"LANDFIRE job {job_id} did not finish within {max_wait}s. "
                f"Try increasing max_wait or reducing the bbox."
            )
    # Load GeoTIFFs from zip
    import rioxarray

    arrays = {}
    with zipfile.ZipFile(path) as z:
        tifs = [n for n in z.namelist() if n.lower().endswith(".tif")]
        for name, code in zip(layers, codes):
            match = next(
                (t for t in tifs if code.lower() in t.lower()), None
            )
            if match is None:
                print(f"  ! No raster in zip matching {code}. Got: {tifs}")
                continue
            with z.open(match) as fh:
                da = (rioxarray.open_rasterio(io.BytesIO(fh.read()))
                      .squeeze(drop=True))
            # LANDFIRE ships canopy structure layers scaled by 10 or 100 to
            # keep them integer; convert to physical units.
            if name in ("ch", "cbh"):
                da = da.where(da > 0) / 10.0
            elif name == "cbd":
                da = da.where(da > 0) / 100.0
            arrays[name] = da

    ds = xr.Dataset(arrays)
    ds.attrs.update(
        source="LANDFIRE",
        vintage="LF2020" if prefire else "LF2023",
        layer_codes=";".join(codes),
    )
    return ds


# MTBS perimeters
_MTBS_SERVICE    = "https://apps.fs.usda.gov/arcx/rest/services/EDW/EDW_MTBS_01/MapServer"
_MTBS_PERIM_LAYER = 0


def get_mtbs_perimeters(bbox, year=None, min_acres: float = 1000):
    """
    MTBS burned-area boundaries intersecting a bbox.

    MTBS maps fires above 1,000 acres in the West and is the standard severity
    reference, but perimeters and severity for a given fire year typically
    appear 1-2 years later. For recent fires, fall back to NIFC or GeoMAC
    perimeters and be explicit that severity is not yet available.
    """
    if gpd is None:
        raise ImportError("geopandas is required for get_mtbs_perimeters()")

    geojson = _arcgis_query(_MTBS_SERVICE, _MTBS_PERIM_LAYER, bbox)
    gdf = gpd.GeoDataFrame.from_features(geojson["features"], crs="EPSG:4326")
    if gdf.empty:
        return gdf

    gdf.columns = [c.lower() for c in gdf.columns]
    if year is not None:
        for col in ("year", "fire_year", "ig_year", "startyear"):
            if col in gdf.columns:
                gdf = gdf[pd.to_numeric(gdf[col], errors="coerce") == year]
                break
    for col in ("acres", "burnbndac", "fire_acres"):
        if col in gdf.columns:
            gdf = gdf[pd.to_numeric(gdf[col], errors="coerce") >= min_acres]
            break

    return gdf.reset_index(drop=True)


# FIRMS active fire detections
def get_active_fire(
    bbox,
    start: str,
    days: int = 10,
    sensor: str = "VIIRS_SNPP_SP",
    map_key: str | None = None,
) -> pd.DataFrame:
    """
    NASA FIRMS active-fire detections, used to reconstruct fire progression.

    Requires a free MAP_KEY from https://firms.modaps.eosdis.nasa.gov/api/
    Pass it explicitly or set FIRMS_MAP_KEY in the environment.

    Two things detections are not:
    - A perimeter. They are thermal anomalies at overpass times (~twice daily).
    - A severity measure. Use dNBR for severity; use detections for timing only.
    """
    key = map_key or os.environ.get("FIRMS_MAP_KEY")
    if not key:
        raise ValueError(
            "FIRMS needs a MAP_KEY. Get one free at "
            "https://firms.modaps.eosdis.nasa.gov/api/"
        )

    min_lon, min_lat, max_lon, max_lat = bbox
    area = f"{min_lon},{min_lat},{max_lon},{max_lat}"
    url  = (f"https://firms.modaps.eosdis.nasa.gov/api/area/csv"
            f"/{key}/{sensor}/{area}/{min(days, 10)}/{start}")

    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    if resp.text.strip().lower().startswith("invalid"):
        raise RuntimeError(f"FIRMS rejected the request: {resp.text[:200]}")

    df = pd.read_csv(io.StringIO(resp.text))
    if df.empty:
        return df
    df["acq_date"] = pd.to_datetime(df["acq_date"])
    return df


def active_fire_series(
    bbox,
    start: str,
    end: str,
    sensor: str = "VIIRS_SNPP_SP",
    map_key: str | None = None,
) -> pd.DataFrame:
    """Chunked FIRMS retrieval across an arbitrary date range (10-day API cap)."""
    frames  = []
    cursor  = pd.Timestamp(start)
    stop    = pd.Timestamp(end)
    while cursor <= stop:
        span = min(10, (stop - cursor).days + 1)
        try:
            frames.append(
                get_active_fire(bbox, cursor.strftime("%Y-%m-%d"),
                                days=span, sensor=sensor, map_key=map_key)
            )
        except Exception as exc:
            print(f"  ! FIRMS chunk {cursor.date()} failed: {exc}")
        cursor += pd.Timedelta(days=span)

    frames = [f for f in frames if f is not None and not f.empty]
    return (pd.concat(frames, ignore_index=True).drop_duplicates()
            if frames else pd.DataFrame())


# OSM buildings
_OVERPASS = "https://overpass-api.de/api/interpreter"


def get_buildings(bbox, timeout: int = 300):
    """
    OSM building footprints (returned as centroids) for exposure counting.

    OSM completeness is the dominant uncertainty. Coverage in rural mountain
    Colorado is decent for permanent homes and poor for outbuildings and
    seasonal cabins. A structure count from OSM is a lower bound. Microsoft
    Building Footprints is more complete but is a bulk download rather than
    an API.
    """
    if gpd is None:
        raise ImportError("geopandas is required for get_buildings()")

    min_lon, min_lat, max_lon, max_lat = bbox
    query = f"""
    [out:json][timeout:{timeout}];
    (
      way["building"]({min_lat},{min_lon},{max_lat},{max_lon});
      relation["building"]({min_lat},{min_lon},{max_lat},{max_lon});
    );
    out center tags;
    """
    resp = requests.post(_OVERPASS, data={"data": query}, timeout=timeout + 60)
    resp.raise_for_status()

    rows = []
    for el in resp.json().get("elements", []):
        centre = el.get("center") or (
            {"lat": el.get("lat"), "lon": el.get("lon")}
            if el.get("lat") else None
        )
        if not centre:
            continue
        tags = el.get("tags", {})
        rows.append({
            "osm_id":   el.get("id"),
            "lat":      centre["lat"],
            "lon":      centre["lon"],
            "building": tags.get("building", "yes"),
            "name":     tags.get("name"),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return gpd.GeoDataFrame(df, geometry=[], crs="EPSG:4326")
    return gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["lon"], df["lat"]),
        crs="EPSG:4326",
    )


# Census places
_TIGERWEB    = "https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/Places_CouSub_ConCity_SubMCD/MapServer"
_PLACES_LAYER = 22   # Incorporated Places: verify against service directory


def get_places(bbox):
    """
    Census incorporated places for aggregating exposure to named communities.

    Places are a reporting frame, not a population denominator. Unincorporated
    population (a large share in Larimer County WUI) is excluded. Use block
    groups if you need a population denominator.
    """
    if gpd is None:
        raise ImportError("geopandas is required for get_places()")

    geojson = _arcgis_query(_TIGERWEB, _PLACES_LAYER, bbox)
    gdf = gpd.GeoDataFrame.from_features(geojson["features"], crs="EPSG:4326")
    if not gdf.empty:
        gdf.columns = [c.lower() for c in gdf.columns]
    return gdf.reset_index(drop=True)


# Water infrastructure
# There is no national API for municipal surface-water intake locations 
# partly deliberate, as they are sensitive infrastructure. These are
# approximate public locations for the Poudre municipal systems, adequate
# for demonstrating watershed-to-intake exposure but not for operational
# planning. Confirm with utilities before any deliverable.
POUDRE_INTAKES = pd.DataFrame([
    {
        "name":        "Fort Collins Water Treatment Facility (Poudre intake)",
        "lat":         40.6653,
        "lon":         -105.2178,
        "system":      "City of Fort Collins",
        "approximate": True,
    },
    {
        "name":        "Soldier Canyon Filter Plant (Tri-Districts)",
        "lat":         40.6017,
        "lon":         -105.1794,
        "system":      "Tri-Districts (ELCO / FCLWD / NWCWD)",
        "approximate": True,
    },
    {
        "name":        "Greeley Bellvue Water Treatment Plant",
        "lat":         40.6389,
        "lon":         -105.1900,
        "system":      "City of Greeley",
        "approximate": True,
    },
])


def get_water_intakes():
    """
    Approximate municipal intake locations on the Poudre.

    Every row is flagged `approximate=True` a demo should not imply
    survey-grade infrastructure locations it does not have.
    """
    df = POUDRE_INTAKES.copy()
    if gpd is None:
        return df
    return gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["lon"], df["lat"]),
        crs="EPSG:4326",
    )