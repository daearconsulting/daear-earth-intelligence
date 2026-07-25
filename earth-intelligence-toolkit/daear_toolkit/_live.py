"""
Live (network-backed) implementations of the data_access.get_* functions.

Kept in a separate module so `daear_toolkit` imports cleanly even when the
`live` extras (earthaccess, pystac-client, planetary-computer, stackstac,
rioxarray, requests) aren't installed — data_access.py only imports from
here inside the `synthetic=False` branch.

CONFIDENCE NOTES (read before relying on this in a deadline-critical run):

- Sentinel-2 / terrain via Microsoft Planetary Computer's STAC API: high
  confidence. This is a stable, well-documented public API and the
  collection/asset names below (sentinel-2-l2a, cop-dem-glo30, B04/B08/B11)
  are standard as of this build.
- FIRMS active-fire API: high confidence. The URL pattern is stable and
  documented at https://firms.modaps.eosdis.nasa.gov/api/ -- you need a
  free MAP_KEY from https://firms.modaps.eosdis.nasa.gov/api/map_key/.
- MTBS direct download: MEDIUM confidence. MTBS does not have a clean
  queryable REST API for individual fire perimeters/severity the way FIRMS
  and Planetary Computer do -- their site has changed structure over the
  years. `mtbs_burn_severity_from_direct_download()` below takes a fire-
  specific download URL as an argument rather than guessing one, because I
  can't verify the current exact URL from this sandbox. Get the right URL
  by finding "Cameron Peak" at https://mtbs.gov/direct-download and copying
  the dNBR/severity GeoTIFF link. `burn_severity_from_sentinel2()` is the
  no-MTBS-dependency alternative: it computes dNBR directly from two real
  Sentinel-2 scenes using daear_toolkit.indicators.dnbr, which only depends
  on the high-confidence Sentinel-2 path above.
- SSURGO via USDA Soil Data Access (SDA): MEDIUM confidence. The Tabular
  service endpoint and JSON request shape below match SDA's documented
  usage, but the exact spatial-intersection SQL function name has changed
  between SDA versions -- verify `SDA_Get_Mukey_from_intersection_with_WktWgs84`
  against current docs at https://sdmdataaccess.nrcs.usda.gov if the query
  fails, and adjust the function name/signature as needed.
"""

from __future__ import annotations

import os
import numpy as np
import xarray as xr


def _require(*packages):
    missing = []
    for p in packages:
        try:
            __import__(p)
        except ImportError:
            missing.append(p)
    if missing:
        raise ImportError(
            f"Live retrieval needs: {', '.join(missing)}. "
            f"Install with: pip install 'daear-toolkit[live]' "
            f"(or: pip install {' '.join(missing)})"
        )


# ---------------------------------------------------------------------------
# Sentinel-2 (Microsoft Planetary Computer) -- HIGH confidence
# ---------------------------------------------------------------------------

def sentinel2_scene(bbox: tuple, date: str, cloud_cover_lt: int = 20, days_window: int = 15) -> xr.Dataset:
    """
    Real Sentinel-2 L2A surface reflectance scene (red/nir/swir16) nearest
    `date` over `bbox`, via Planetary Computer's public STAC catalog.

    No credentials required for search or read -- Planetary Computer signs
    asset URLs with a short-lived SAS token automatically via
    `planetary_computer.sign_inplace`.
    """
    _require("pystac_client", "planetary_computer", "stackstac", "rioxarray")
    import pystac_client
    import planetary_computer
    import stackstac
    import pandas as pd

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    center = pd.Timestamp(date)
    start = (center - pd.Timedelta(days=days_window)).strftime("%Y-%m-%d")
    end = (center + pd.Timedelta(days=days_window)).strftime("%Y-%m-%d")

    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=f"{start}/{end}",
        query={"eo:cloud_cover": {"lt": cloud_cover_lt}},
    )
    items = list(search.items())
    if not items:
        raise RuntimeError(
            f"No Sentinel-2 scenes found for bbox={bbox} in {start}/{end} "
            f"with cloud_cover<{cloud_cover_lt}. Try widening days_window "
            f"or raising cloud_cover_lt."
        )
    # closest-in-time scene
    items.sort(key=lambda it: abs(pd.Timestamp(it.properties["datetime"]) - center))
    item = items[0]

    stack = stackstac.stack(
        [item],
        assets=["B04", "B08", "B11"],  # red, nir, swir16
        bounds_latlon=bbox,
        epsg=4326,
        resolution=0.0002,  # ~20m in degrees at this latitude
    ).squeeze("time", drop=True) / 10000.0  # Sentinel-2 L2A reflectance scale factor

    ds = xr.Dataset(
        {
            "red": stack.sel(band="B04").drop_vars("band"),
            "nir": stack.sel(band="B08").drop_vars("band"),
            "swir16": stack.sel(band="B11").drop_vars("band"),
        }
    ).rename({"x": "lon", "y": "lat"})
    ds.attrs.update(
        source="sentinel-2-l2a (Planetary Computer)",
        scene_id=item.id,
        scene_datetime=item.properties["datetime"],
        cloud_cover=item.properties.get("eo:cloud_cover"),
        synthetic=False,
    )
    return ds


# ---------------------------------------------------------------------------
# Burn severity -- MEDIUM confidence (MTBS) + HIGH confidence fallback
# ---------------------------------------------------------------------------

def mtbs_burn_severity_from_direct_download(geotiff_url: str, bbox: tuple) -> xr.DataArray:
    """
    Load an MTBS dNBR/burn-severity GeoTIFF you've already located via
    https://mtbs.gov/direct-download (search "Cameron Peak"), clipped to
    bbox. Pass the direct GeoTIFF URL you copied from that page.

    MTBS severity classes are typically coded 1 (unburned) - 5 (high) or
    higher; this rescales to 0-1 for consistency with the rest of the
    toolkit. Check the specific product's legend before trusting the
    rescaling blindly -- MTBS product schemas have varied by release.
    """
    _require("rioxarray")
    import rioxarray

    da = rioxarray.open_rasterio(geotiff_url).squeeze("band", drop=True)
    da = da.rio.clip_box(minx=bbox[0], miny=bbox[1], maxx=bbox[2], maxy=bbox[3])
    da = da.rename({"x": "lon", "y": "lat"})

    valid = da.where(da > 0)  # MTBS commonly uses 0 for "outside fire perimeter / no data"
    rescaled = (valid - valid.min()) / (valid.max() - valid.min() + 1e-9)
    rescaled.name = "burn_severity"
    rescaled.attrs.update(source="MTBS direct download", url=geotiff_url, synthetic=False)
    return rescaled


def burn_severity_from_sentinel2(bbox: tuple, pre_fire_date: str, post_fire_date: str) -> xr.DataArray:
    """
    No-MTBS-dependency alternative: real dNBR computed directly from two
    real Sentinel-2 scenes via daear_toolkit.indicators.dnbr. Only depends
    on the high-confidence sentinel2_scene() path above -- use this if the
    MTBS URL in mtbs_burn_severity_from_direct_download() doesn't resolve.
    """
    from . import indicators  # local import to avoid a package-level cycle

    pre = sentinel2_scene(bbox, pre_fire_date)
    post = sentinel2_scene(bbox, post_fire_date)
    da = indicators.dnbr(pre, post)
    da.attrs.update(synthetic=False, method="Sentinel-2 dNBR (no MTBS)")
    return da


# ---------------------------------------------------------------------------
# FIRMS active fire detections -- HIGH confidence
# ---------------------------------------------------------------------------

def firms_fire_detections(bbox: tuple, start_date: str, end_date: str, source: str = "VIIRS_SNPP_NRT"):
    """
    Real active-fire point detections from NASA FIRMS.

    Requires a free MAP_KEY: https://firms.modaps.eosdis.nasa.gov/api/map_key/
    Set it as the FIRMS_MAP_KEY environment variable, or pass map_key= directly.

    FIRMS' area-CSV endpoint takes a single day-range length (max 10 days
    per request as of this API version), so multi-week windows are fetched
    in chunks and concatenated.
    """
    _require("requests")
    import requests
    import pandas as pd

    map_key = os.environ.get("FIRMS_MAP_KEY")
    if not map_key:
        raise RuntimeError(
            "Set FIRMS_MAP_KEY (get one free at "
            "https://firms.modaps.eosdis.nasa.gov/api/map_key/)"
        )

    west, south, east, north = bbox
    area = f"{west},{south},{east},{north}"

    frames = []
    dates = pd.date_range(start_date, end_date, freq="10D")
    if len(dates) == 0 or dates[-1] < pd.Timestamp(end_date):
        dates = dates.append(pd.DatetimeIndex([pd.Timestamp(end_date)]))

    for chunk_start in dates:
        url = (
            f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
            f"{map_key}/{source}/{area}/10/{chunk_start.date()}"
        )
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        from io import StringIO
        chunk = pd.read_csv(StringIO(resp.text))
        if len(chunk):
            frames.append(chunk)

    if not frames:
        return pd.DataFrame(columns=["latitude", "longitude", "confidence", "acq_date"])

    df = pd.concat(frames, ignore_index=True)
    df = df.rename(columns={"latitude": "lat", "longitude": "lon"})
    mask = (
        (df["acq_date"] >= start_date) & (df["acq_date"] <= end_date)
    )
    return df.loc[mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Terrain (Copernicus DEM via Planetary Computer) -- HIGH confidence
# ---------------------------------------------------------------------------

def copernicus_dem(bbox: tuple) -> xr.Dataset:
    """Real 30m Copernicus DEM elevation + derived slope, via Planetary Computer."""
    _require("pystac_client", "planetary_computer", "stackstac", "rioxarray")
    import pystac_client
    import planetary_computer
    import stackstac

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    items = list(catalog.search(collections=["cop-dem-glo30"], bbox=bbox).items())
    if not items:
        raise RuntimeError(f"No Copernicus DEM tiles found for bbox={bbox}")

    stack = stackstac.stack(
        items, assets=["data"], bounds_latlon=bbox, epsg=4326, resolution=0.0003,
    ).squeeze("band", drop=True)
    elevation = stack.mean("time") if "time" in stack.dims else stack
    elevation = elevation.rename({"x": "lon", "y": "lat"})
    elevation.name = "elevation_m"

    dz_dy, dz_dx = np.gradient(elevation.values)
    deg_to_m = 111_000
    px_deg = float(abs(elevation["lon"][1] - elevation["lon"][0]))
    slope = np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2) / (deg_to_m * px_deg)))

    ds = xr.Dataset(
        {
            "elevation_m": elevation,
            "slope_deg": (("lat", "lon"), slope.astype("float32")),
        }
    )
    ds.attrs.update(source="cop-dem-glo30 (Planetary Computer)", synthetic=False)
    return ds


# ---------------------------------------------------------------------------
# Soil (USDA Soil Data Access) -- MEDIUM confidence
# ---------------------------------------------------------------------------

def ssurgo_soil_properties(bbox: tuple) -> "pandas.DataFrame":  # noqa: F821
    """
    Real SSURGO soil properties (organic matter %, sand %, drainage class)
    for map units intersecting bbox, via USDA's Soil Data Access (SDA)
    Tabular REST service.

    Returns a per-map-unit DataFrame (SSURGO is polygon/map-unit data, not
    a regular raster grid) -- rasterizing onto the toolkit's lat/lon grid
    for direct use in indicators.soil_vulnerability() is a follow-up step
    (e.g. geopandas + rasterio rasterize), not included here since it needs
    the actual map unit geometries alongside the attribute table this
    function returns.

    Verify the spatial-intersection SQL function name against current SDA
    docs (https://sdmdataaccess.nrcs.usda.gov) if this returns an error --
    SDA's exact function names have changed between versions.
    """
    _require("requests")
    import requests
    import pandas as pd

    west, south, east, north = bbox
    wkt_polygon = (
        f"POLYGON(({west} {south}, {east} {south}, {east} {north}, "
        f"{west} {north}, {west} {south}))"
    )

    query = f"""
    SELECT mu.mukey, mu.muname, c.comppct_r, c.compname,
           ch.hzdept_r, ch.hzdepb_r, ch.om_r AS organic_matter_pct,
           ch.sandtotal_r AS sand_pct, c.hydgrp
    FROM mapunit mu
    INNER JOIN component c ON c.mukey = mu.mukey
    INNER JOIN chorizon ch ON ch.cokey = c.cokey
    WHERE mu.mukey IN (
        SELECT DISTINCT mukey FROM SDA_Get_Mukey_from_intersection_with_WktWgs84('{wkt_polygon}')
    )
    AND ch.hzdept_r = 0
    """

    resp = requests.post(
        "https://sdmdataaccess.sc.egov.usda.gov/Tabular/post.rest",
        json={"format": "JSON+COLUMNNAME", "query": query},
        timeout=60,
    )
    resp.raise_for_status()
    payload = resp.json()
    table = payload.get("Table")
    if not table:
        raise RuntimeError(
            "SDA query returned no rows -- verify the SQL function name "
            "against current Soil Data Access docs and retry."
        )
    columns, *rows = table
    df = pd.DataFrame(rows, columns=columns)
    df.attrs["source"] = "USDA Soil Data Access (SSURGO)"
    df.attrs["synthetic"] = False
    return df
