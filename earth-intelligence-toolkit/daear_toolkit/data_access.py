from __future__ import annotations

"""
Uniform data access layer.

Each `get_*` function has the real (network-backed) implementation and a
`synthetic=True` fallback. The synthetic path is what runs in this build 
see the top-level README for why but the function signatures, coordinate
handling, and output shapes match what the live path returns, so downstream
code (indicators, cube-building, dashboards) never has to know which mode
produced the array.

Live-data dependencies (install with `pip install 'daear-toolkit[live]'`):
pystac-client, planetary-computer, stackstac, rioxarray, requests. See
`daear_toolkit._live` for the implementations and a per-source confidence
note (some endpoints, like MTBS, don't have a clean stable API and need a
URL you supply; others, like Planetary Computer and FIRMS, are solid).
"""

import numpy as np
import xarray as xr

from . import _live


# Shared grid helper
def _grid(bbox: tuple, resolution_deg: float = 0.002):
    """Build a lon/lat coordinate grid for a bbox at ~200m resolution by default."""
    min_lon, min_lat, max_lon, max_lat = bbox
    lons = np.arange(min_lon, max_lon, resolution_deg)
    lats = np.arange(min_lat, max_lat, resolution_deg)
    return lons, lats


def _smooth_random_field(shape: tuple, seed: int, smoothness: int = 9) -> np.ndarray:
    """
    Spatially-autocorrelated pseudo-random field in [0, 1], used to stand in
    for realistic-looking (but synthetic) raster values. Uses a simple
    moving-average smooth of white noise rather than a real geostatistical
    simulator, which is sufficient for demonstrating the analysis pipeline.
    """
    rng = np.random.default_rng(seed)
    field = rng.random(shape)
    k = smoothness
    kernel = np.ones((k, k)) / (k * k)
    # naive 2D convolution via FFT-free padding (small arrays, fine for demo)
    pad = k // 2
    padded = np.pad(field, pad, mode="reflect")
    out = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            out[i, j] = padded[i:i + k, j:j + k].mean()
    out -= out.min()
    out /= out.max() + 1e-9
    return out


# Sentinel-2 / Landsat surface reflectance
def get_optical_scene(
    bbox: tuple,
    date: str,
    source: str = "sentinel-2",
    synthetic: bool = True,
    seed: int = 0,
) -> xr.Dataset:
    """
    Return a surface-reflectance scene with red, nir, and swir16 bands
    (enough to compute NDVI and NBR) over `bbox` near `date`.

    Live path (synthetic=False): real Sentinel-2 L2A via Microsoft Planetary
    Computer's STAC API `daear_toolkit._live.sentinel2_scene`. No
    credentials needed for read access. HIGH confidence (see _live.py
    module docstring).

    Demo path (synthetic=True): generates structurally realistic bands with
    the same dims/coords a real scene would have.
    """
    if not synthetic:
        return _live.sentinel2_scene(bbox, date)

    lons, lats = _grid(bbox)
    shape = (len(lats), len(lons))
    date_seed = seed + abs(hash(date)) % 10_000

    # Vegetation-like base pattern: forested canyon slopes vs. open valley floor
    veg_base = _smooth_random_field(shape, seed=date_seed, smoothness=11)

    red = 0.05 + 0.10 * (1 - veg_base) + 0.01 * np.random.default_rng(date_seed).random(shape)
    nir = 0.20 + 0.45 * veg_base + 0.02 * np.random.default_rng(date_seed + 1).random(shape)
    swir16 = 0.10 + 0.20 * (1 - veg_base) + 0.02 * np.random.default_rng(date_seed + 2).random(shape)

    ds = xr.Dataset(
        {
            "red": (("lat", "lon"), red.astype("float32")),
            "nir": (("lat", "lon"), nir.astype("float32")),
            "swir16": (("lat", "lon"), swir16.astype("float32")),
        },
        coords={"lat": lats, "lon": lons},
        attrs={"source": source, "date": date, "synthetic": True, "bbox": bbox},
    )
    return ds


# Fire products: MTBS burn severity, FIRMS active fire detections
def get_burn_severity(bbox: tuple, fire_year: int, synthetic: bool = True, seed: int = 1) -> xr.DataArray:
    """
    Return a dNBR-like burn severity raster (0 = unburned, 1 = high severity)
    over `bbox` for the fire in `fire_year`.

    Live path: MEDIUM confidence MTBS doesn't have a clean queryable API,
    so this raises with instructions rather than guessing a URL. Two ways
    to get real data:

    1. `daear_toolkit._live.mtbs_burn_severity_from_direct_download(url, bbox)`
       -- find "Cameron Peak" at https://mtbs.gov/direct-download, copy the
       dNBR/severity GeoTIFF link, pass it as `url`.
    2. `daear_toolkit._live.burn_severity_from_sentinel2(bbox, pre_date, post_date)`
       no MTBS dependency at all; computes real dNBR from two real
       Sentinel-2 scenes. HIGH confidence, since it only depends on the
       Planetary Computer path.
    """
    if not synthetic:
        raise NotImplementedError(
            "get_burn_severity has no single live call MTBS doesn't expose "
            "a stable queryable API. Use one of:\n"
            "  daear_toolkit._live.mtbs_burn_severity_from_direct_download(url, bbox)\n"
            "  daear_toolkit._live.burn_severity_from_sentinel2(bbox, pre_date, post_date)\n"
            "See the docstring above and _live.py's module docstring for details."
        )

    lons, lats = _grid(bbox)
    shape = (len(lats), len(lons))
    # Burn scar concentrated in the upper (northwest) part of the bbox,
    # tapering toward the valley floor mimics the real Cameron Peak
    # perimeter's relationship to the Poudre canyon.
    field = _smooth_random_field(shape, seed=seed, smoothness=15)
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
    lat_norm = (lat_grid - lat_grid.min()) / (lat_grid.max() - lat_grid.min())
    severity = np.clip(field * (0.4 + 0.6 * lat_norm) - 0.15, 0, 1)

    da = xr.DataArray(
        severity.astype("float32"),
        coords={"lat": lats, "lon": lons},
        dims=("lat", "lon"),
        name="burn_severity",
        attrs={"source": "MTBS (synthetic demo)", "fire_year": fire_year, "synthetic": True},
    )
    return da


def get_fire_detections(bbox: tuple, date_range: tuple, synthetic: bool = True, seed: int = 2):
    """
    Return FIRMS-style active-fire point detections (lon, lat, confidence,
    acq_date) within bbox/date_range.

    Live path: real NASA FIRMS active-fire detections. HIGH confidence.
    Requires a free MAP_KEY (https://firms.modaps.eosdis.nasa.gov/api/map_key/)
    set as the FIRMS_MAP_KEY environment variable.
    """
    if not synthetic:
        return _live.firms_fire_detections(bbox, date_range[0], date_range[1])

    rng = np.random.default_rng(seed)
    min_lon, min_lat, max_lon, max_lat = bbox
    n = 40
    lons = rng.uniform(min_lon, max_lon, n)
    lats = rng.uniform(min_lat, max_lat, n)
    confidence = rng.integers(50, 100, n)
    import pandas as pd

    dates = pd.date_range(date_range[0], date_range[1], periods=n)
    return pd.DataFrame(
        {"lon": lons, "lat": lats, "confidence": confidence, "acq_date": dates}
    )


# Terrain
def get_terrain(bbox: tuple, synthetic: bool = True, seed: int = 3) -> xr.Dataset:
    """
    Return an elevation raster plus derived slope over bbox.

    Live path: real Copernicus 30m DEM via Planetary Computer's
    `cop-dem-glo30` collection. HIGH confidence, no credentials needed.
    """
    if not synthetic:
        return _live.copernicus_dem(bbox)

    lons, lats = _grid(bbox)
    shape = (len(lats), len(lons))
    base = _smooth_random_field(shape, seed=seed, smoothness=21)
    # Canyon-like relief: higher/steeper toward the west (mountain) edge
    lon_grid, _ = np.meshgrid(lons, lats)
    lon_norm = 1 - (lon_grid - lon_grid.min()) / (lon_grid.max() - lon_grid.min())
    elevation = 1600 + 1400 * base + 900 * lon_norm  # meters, roughly plausible for the Poudre canyon

    dz_dy, dz_dx = np.gradient(elevation)
    slope = np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2) / 111_000 / 0.002))

    ds = xr.Dataset(
        {
            "elevation_m": (("lat", "lon"), elevation.astype("float32")),
            "slope_deg": (("lat", "lon"), slope.astype("float32")),
        },
        coords={"lat": lats, "lon": lons},
        attrs={"source": "USGS 3DEP (synthetic demo)", "synthetic": True},
    )
    return ds

# Soil

def get_soil_properties(bbox: tuple, synthetic: bool = True, seed: int = 4) -> xr.Dataset:
    """
    Return soil organic matter (%), sand fraction (%), and hydrologic soil
    group proxy (0-1, higher = more runoff-prone) over bbox.

    Live path: MEDIUM confidence real SSURGO data via USDA's Soil Data
    Access (SDA). Note the shape difference: SSURGO is polygon/map-unit
    data, not a regular grid, so the live path returns a per-map-unit
    DataFrame (`daear_toolkit._live.ssurgo_soil_properties`), not an
    xr.Dataset on this function's synthetic grid. Rasterizing that table
    onto the toolkit's lat/lon grid (ex. with geopandas and rasterio) is a
    follow-up step.
    """
    if not synthetic:
        raise NotImplementedError(
            "get_soil_properties' live path returns a different shape than "
            "its synthetic path (SSURGO is map-unit polygons, not a raster "
            "grid). Call daear_toolkit._live.ssurgo_soil_properties(bbox) "
            "directly and rasterize the result onto your grid of choice."
        )

    lons, lats = _grid(bbox)
    shape = (len(lats), len(lons))
    organic_matter = 2 + 6 * _smooth_random_field(shape, seed=seed, smoothness=13)
    sand_fraction = 30 + 40 * _smooth_random_field(shape, seed=seed + 1, smoothness=13)
    runoff_proxy = _smooth_random_field(shape, seed=seed + 2, smoothness=17)

    ds = xr.Dataset(
        {
            "organic_matter_pct": (("lat", "lon"), organic_matter.astype("float32")),
            "sand_fraction_pct": (("lat", "lon"), sand_fraction.astype("float32")),
            "runoff_potential": (("lat", "lon"), runoff_proxy.astype("float32")),
        },
        coords={"lat": lats, "lon": lons},
        attrs={"source": "SSURGO/SoilGrids (synthetic demo)", "synthetic": True},
    )
    return ds

def get_soil_moisture(bbox: tuple, start: str, end: str) -> "xr.Dataset":
    """
    Load SMAP L3 enhanced soil moisture time series for a bounding box
    and date range. Used by soil-watershed-intelligence Module 4.

    Requires NASA Earthdata credentials set as environment variables:
        EARTHDATA_USERNAME, EARTHDATA_PASSWORD

    Falls back to synthetic data if credentials are not available,
    suitable for demo and teaching contexts.
    """
    if xr is None:
        raise ImportError("xarray is required for get_soil_moisture")

    username = os.environ.get("EARTHDATA_USERNAME")
    password  = os.environ.get("EARTHDATA_PASSWORD")

    # Synthetic fallback
    if not username or not password:
        warnings.warn(
            "NASA Earthdata credentials not found. Returning synthetic SMAP "
            "soil moisture data for demo/teaching purposes.\n"
            "To use real data, set environment variables:\n"
            "  EARTHDATA_USERNAME=your_username\n"
            "  EARTHDATA_PASSWORD=your_password\n"
            "Register at: https://urs.earthdata.nasa.gov/",
            UserWarning, stacklevel=2,
        )
        start_dt = _dt.date.fromisoformat(start)
        end_dt   = _dt.date.fromisoformat(end)
        n_days   = (end_dt - start_dt).days + 1
        rng      = np.random.default_rng(seed=42)

        # Synthetic soil moisture: ~0.15-0.35 m³/m³ with seasonal signal
        days      = np.arange(n_days)
        base_sm   = 0.22 + 0.08 * np.sin(2 * np.pi * days / 365)
        noise     = rng.normal(0, 0.02, size=(n_days, 8, 10))
        sm_data   = (base_sm[:, None, None] + noise).clip(0.05, 0.50)

        times = [start_dt + _dt.timedelta(days=d) for d in range(n_days)]
        lats  = np.linspace(bbox[1], bbox[3], 8)
        lons  = np.linspace(bbox[0], bbox[2], 10)

        da = xr.DataArray(
            sm_data,
            dims=["time", "y", "x"],
            coords={
                "time":      [_dt.datetime.combine(t, _dt.time()) for t in times],
                "latitude":  (["y"], lats),
                "longitude": (["x"], lons),
            },
            attrs={"units": "m3/m3", "source": "synthetic — for demo only"},
        )
        return da.to_dataset(name="soil_moisture")

    # Real SMAP fetch
    base_url = "https://n5eil01u.ecs.nsidc.org/opendap/SMAP/SPL3SMP_E.006"

    start_dt = _dt.date.fromisoformat(start)
    end_dt   = _dt.date.fromisoformat(end)

    datasets = []
    current  = start_dt
    while current <= end_dt:
        date_str = current.strftime("%Y.%m.%d")
        fname    = f"SMAP_L3_SM_P_E_{current.strftime('%Y%m%d')}_R19240_001.h5"
        url      = f"{base_url}/{date_str}/{fname}"
        try:
            r = requests.get(url, auth=(username, password), timeout=60)
            if r.status_code == 200:
                import tempfile, h5py
                with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
                    f.write(r.content)
                    tmp_path = f.name
                with h5py.File(tmp_path, "r") as h5:
                    sm  = h5["Soil_Moisture_Retrieval_Data_AM/soil_moisture"][:]
                    lat = h5["Soil_Moisture_Retrieval_Data_AM/latitude"][:]
                    lon = h5["Soil_Moisture_Retrieval_Data_AM/longitude"][:]
                da = xr.DataArray(
                    sm, dims=["y", "x"],
                    coords={"latitude": (["y", "x"], lat),
                            "longitude": (["y", "x"], lon)},
                )
                da = da.where(da > 0).assign_coords(
                    time=_dt.datetime.combine(current, _dt.time())
                )
                datasets.append(da)
        except Exception as e:
            warnings.warn(f"SMAP fetch failed for {current}: {e}")
        current += _dt.timedelta(days=1)

    if not datasets:
        raise ValueError(f"No SMAP data retrieved for {start} to {end}. "
                         "Check credentials and date range.")

    ds = xr.concat(datasets, dim="time").to_dataset(name="soil_moisture")
    return ds

def get_watershed_boundaries(bbox: tuple) -> "gpd.GeoDataFrame":
    """
    Load HUC8 watershed boundaries from USGS WBD for a bounding box.
    Replaces the illustrative grid-based sub-watershed split in
    wildfire-landscape-intelligence Module 4.
    """
    if gpd is None:
        raise ImportError("geopandas is required for get_watershed_boundaries")

    layer_url = (
        "https://hydro.nationalmap.gov/arcgis/rest/services/"
        "wbd/MapServer/4/query"
    )
    r = requests.get(layer_url, params={
        "where":          "1=1",
        "outFields":      "huc8,name,areasqkm",
        "f":              "geojson",
        "returnGeometry": "true",
        "outSR":          "4326",
        "geometry":       f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
        "geometryType":   "esriGeometryEnvelope",
        "inSR":           "4326",
    }, timeout=60)
    r.raise_for_status()
    gdf = gpd.read_file(r.content)
    return gdf.set_crs("EPSG:4326", allow_override=True)


def get_flowlines(bbox: tuple, min_stream_order: int = 2) -> "gpd.GeoDataFrame":
    """
    Load NHDPlus HR flowlines from USGS NHD for a bounding box.
    """
    if gpd is None:
        raise ImportError("geopandas is required for get_flowlines")

    r = requests.get(
        "https://hydro.nationalmap.gov/arcgis/rest/services/"
        "NHDPlus_HR/MapServer/3/query",
        params={
            "where":          f"streamorde >= {min_stream_order}",
            "outFields":      "reachcode,gnis_name,streamorde,lengthkm",
            "f":              "geojson",
            "returnGeometry": "true",
            "outSR":          "4326",
            "geometry":       f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
            "geometryType":   "esriGeometryEnvelope",
            "inSR":           "4326",
        }, timeout=120)
    r.raise_for_status()
    gdf = gpd.read_file(r.content)
    return gdf.set_crs("EPSG:4326", allow_override=True)


def get_soil_moisture(bbox: tuple, start: str, end: str) -> "xr.Dataset":
    """
    Load SMAP L3 enhanced soil moisture time series for a bounding box
    and date range. Used by soil-watershed-intelligence Module 4.

    Requires NASA Earthdata credentials set as environment variables:
        EARTHDATA_USERNAME, EARTHDATA_PASSWORD
    """
    if xr is None:
        raise ImportError("xarray is required for get_soil_moisture")

    username = os.environ.get("EARTHDATA_USERNAME")
    password  = os.environ.get("EARTHDATA_PASSWORD")
    if not username or not password:
        raise EnvironmentError(
            "NASA Earthdata credentials required. Set environment variables:\n"
            "  EARTHDATA_USERNAME=your_username\n"
            "  EARTHDATA_PASSWORD=your_password\n"
            "Register at: https://urs.earthdata.nasa.gov/"
        )

    # SMAP L3 Enhanced via NSIDC OPeNDAP
    # SPL3SMP_E: daily global, ~9km
    base_url = "https://n5eil01u.ecs.nsidc.org/opendap/SMAP/SPL3SMP_E.006"

    start_dt = _dt.date.fromisoformat(start)
    end_dt   = _dt.date.fromisoformat(end)

    datasets = []
    current  = start_dt
    while current <= end_dt:
        date_str = current.strftime("%Y.%m.%d")
        fname    = f"SMAP_L3_SM_P_E_{current.strftime('%Y%m%d')}_R19240_001.h5"
        url      = f"{base_url}/{date_str}/{fname}"
        try:
            r = requests.get(url, auth=(username, password), timeout=60)
            if r.status_code == 200:
                # Parse HDF5 via xarray — requires h5netcdf or similar
                import tempfile, h5py
                with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
                    f.write(r.content)
                    tmp_path = f.name
                with h5py.File(tmp_path, "r") as h5:
                    sm   = h5["Soil_Moisture_Retrieval_Data_AM/soil_moisture"][:]
                    lat  = h5["Soil_Moisture_Retrieval_Data_AM/latitude"][:]
                    lon  = h5["Soil_Moisture_Retrieval_Data_AM/longitude"][:]
                da = xr.DataArray(sm, dims=["y","x"],
                                  coords={"latitude": (["y","x"], lat),
                                          "longitude": (["y","x"], lon)})
                da = da.where(da > 0)   # mask fill values
                da = da.assign_coords(time=_dt.datetime.combine(current, _dt.time()))
                datasets.append(da)
        except Exception as e:
            warnings.warn(f"SMAP fetch failed for {current}: {e}")
        current += _dt.timedelta(days=1)

    if not datasets:
        raise ValueError(f"No SMAP data retrieved for {start} to {end}")

    ds = xr.concat(datasets, dim="time").to_dataset(name="soil_moisture")
    return ds

# NHD / WBD hydrography
# USGS serves the Watershed Boundary Dataset through the National Map's ArcGIS
# REST endpoint. Layer IDs are stable in practice but are worth re-checking
# against the service directory if a call starts returning empty as USGS has
# renumbered these before.
_WBD_SERVICE = "https://hydro.nationalmap.gov/arcgis/rest/services/wbd/MapServer"
_WBD_LAYERS = {
    2: 1,   # HU2  : region
    4: 2,   # HU4  : subregion
    6: 3,   # HU6  : basin
    8: 4,   # HU8  : subbasin
    10: 5,  # HU10 : watershed
    12: 6,  # HU12 : subwatershed  <- the useful one for burn-scar work
}
_NHD_SERVICE = "https://hydro.nationalmap.gov/arcgis/rest/services/nhd/MapServer"
_NHD_FLOWLINE_LAYER = 6


def _arcgis_query(service: str, layer: int, bbox, where: str = "1=1", timeout: int = 60) -> dict:
    """
    Query an ArcGIS REST FeatureLayer by bounding box, returning GeoJSON.

    `bbox` is (min_lon, min_lat, max_lon, max_lat) to match the rest of the
    toolkit. inSR/outSR 4326 keeps everything in WGS84 lat/lon so the results
    line up with the Sentinel-2 and 3DEP rasters without reprojection.
    """
    params = {
        "geometry": ",".join(str(v) for v in bbox),
        "geometryType": "esriGeometryEnvelope",
        "inSR": 4326,
        "outSR": 4326,
        "spatialRel": "esriSpatialRelIntersects",
        "where": where,
        "outFields": "*",
        "returnGeometry": "true",
        "f": "geojson",
    }
    resp = requests.get(f"{service}/{layer}/query", params=params, timeout=timeout)
    resp.raise_for_status()
    payload = resp.json()
    if "error" in payload:
        raise RuntimeError(f"ArcGIS service error: {payload['error']}")
    return payload


def get_watershed_boundaries(bbox, level: int = 12, clip: bool = True):
    """
    Fetch WBD sub-watershed polygons intersecting `bbox`.

    Parameters
    bbox : tuple
        (min_lon, min_lat, max_lon, max_lat), e.g. `REGION.bbox`.
    level : int
        Hydrologic unit level. 12 (HU12, ~40-160 km^2) is the right grain for
        post-fire watershed work — small enough that a burn scar meaningfully
        dominates some units and not others, which is exactly the contrast
        Module 2 and Module 4 are built around. HU10 if you want fewer, larger
        units for a summary table.
    clip : bool
        Clip polygons to the bbox. Leave False if you want whole units, which
        matters if you plan to report per-unit areas.

    Returns a GeoDataFrame with `huc12` (or `huc10`, etc.) and `name` columns.
    """
    if gpd is None:
        raise ImportError("geopandas is required for get_watershed_boundaries()")
    if level not in _WBD_LAYERS:
        raise ValueError(f"level must be one of {sorted(_WBD_LAYERS)}")

    geojson = _arcgis_query(_WBD_SERVICE, _WBD_LAYERS[level], bbox)
    gdf = gpd.GeoDataFrame.from_features(geojson["features"], crs="EPSG:4326")
    if gdf.empty:
        warnings.warn(f"No HU{level} polygons returned for bbox {bbox}", stacklevel=2)
        return gdf

    # The service returns column names that vary in case across layers.
    gdf.columns = [c.lower() for c in gdf.columns]
    huc_col = next((c for c in gdf.columns if c.startswith("huc")), None)
    if huc_col:
        gdf = gdf.rename(columns={huc_col: f"huc{level}"})

    if clip:
        from shapely.geometry import box

        gdf = gpd.clip(gdf, box(*bbox))

    return gdf.reset_index(drop=True)


def get_flowlines(bbox, min_order: int = 1):
    """
    Fetch NHD flowlines intersecting `bbox`.

    Use these to validate the `hydrology.extract_streams` threshold: if the
    derived network is much denser or sparser than NHD, adjust the threshold
    rather than trusting the default. `min_order` filters by Strahler order if
    the attribute is present — order >= 3 gives you the mainstem and major
    tributaries without the ephemeral drainage clutter.
    """
    if gpd is None:
        raise ImportError("geopandas is required for get_flowlines()")

    geojson = _arcgis_query(_NHD_SERVICE, _NHD_FLOWLINE_LAYER, bbox)
    gdf = gpd.GeoDataFrame.from_features(geojson["features"], crs="EPSG:4326")
    if gdf.empty:
        return gdf
    gdf.columns = [c.lower() for c in gdf.columns]
    if min_order > 1 and "streamorde" in gdf.columns:
        gdf = gdf[gdf["streamorde"] >= min_order]
    return gdf.reset_index(drop=True)


def rasterize_watersheds(gdf, template, id_column: str | None = None):
    """
    Burn watershed polygons onto a raster grid so they can be used as a
    zonal-statistics mask against the toolkit's DataArrays.

    `template` is any DataArray on the target grid (ex. `terrain["slope_deg"]`).
    Returns an integer-coded DataArray plus the lookup mapping codes to HUC IDs.
    """
    from rasterio.features import rasterize
    from rasterio.transform import from_bounds

    if id_column is None:
        id_column = next((c for c in gdf.columns if c.startswith("huc")), gdf.columns[0])

    ydim, xdim = template.dims
    ys = template.coords[ydim].values
    xs = template.coords[xdim].values
    transform = from_bounds(xs.min(), ys.min(), xs.max(), ys.max(), len(xs), len(ys))

    lookup = {i + 1: str(v) for i, v in enumerate(gdf[id_column])}
    shapes = [(geom, i + 1) for i, geom in enumerate(gdf.geometry)]
    arr = rasterize(shapes, out_shape=template.shape, transform=transform, fill=0, dtype="int32")

    out = xr.DataArray(arr.astype("float64"), coords=template.coords, dims=template.dims)
    out = out.where(out > 0)
    out.name = "watershed_id"
    return out, lookup


# SMAP soil moisture
def get_soil_moisture(
    bbox,
    start: str,
    end: str,
    product: str = "SPL3SMP_E",
    version: str = "006",
    overpass: str = "AM",
    cache_dir: str | None = None,
):
    """
    SMAP L3 enhanced (9 km) surface soil moisture time series for `bbox`.

    Parameters
    start, end : str
        ISO dates, 'YYYY-MM-DD'.
    product : str
        'SPL3SMP_E' (9 km enhanced, daily) is the default and the right choice
        for watershed-scale recovery tracking. 'SPL4SMGP' gives you root-zone
        moisture at 9 km / 3-hourly if surface moisture is too noisy for the
        trend you are after worth considering for Module 4, since post-fire
        recovery signals live below the top 5 cm.
    overpass : str
        'AM' (06:00 descending) or 'PM'. AM is the standard choice: the soil
        and canopy are closest to thermal equilibrium, so the retrieval is
        better constrained.

    Returns a DataArray (time, y, x) of volumetric soil moisture, cm^3/cm^3.

    Requires NASA Earthdata credentials. Either run `earthaccess.login()`
    interactively once (it writes a ~/.netrc entry) or set EARTHDATA_USERNAME
    and EARTHDATA_PASSWORD in the environment.

    Caveat worth stating in any deliverable that uses this: 9 km pixels are
    coarse relative to the Cameron Peak burn scar's internal variability. A
    single SMAP pixel spans both treated and untreated ground in most
    restoration layouts, so use this for basin-scale trend context, not for
    treated-vs-untreated contrast. The paired comparison in Module 4 leans on
    optical vegetation indices for that reason.
    """
    try:
        import earthaccess
        import h5py
    except ImportError as exc:  # pragma: no cover
        raise ImportError("get_soil_moisture() needs `earthaccess` and `h5py`: pip install earthaccess h5py") from exc

    if os.environ.get("EARTHDATA_USERNAME") and os.environ.get("EARTHDATA_PASSWORD"):
        earthaccess.login(strategy="environment")
    else:
        earthaccess.login(strategy="netrc")

    results = earthaccess.search_data(
        short_name=product,
        version=version,
        bounding_box=tuple(bbox),
        temporal=(start, end),
    )
    if not results:
        raise RuntimeError(f"No {product} granules found for {bbox} between {start} and {end}")

    files = earthaccess.download(results, cache_dir or "./smap_cache")

    group = f"Soil_Moisture_Retrieval_Data_{overpass}"
    suffix = "" if overpass == "AM" else "_pm"

    stack, times = [], []
    min_lon, min_lat, max_lon, max_lat = bbox

    for path in sorted(files):
        with h5py.File(path, "r") as f:
            if group not in f:
                continue
            g = f[group]
            sm = np.asarray(g[f"soil_moisture{suffix}"][:], dtype="float64")
            lat = np.asarray(g[f"latitude{suffix}"][:], dtype="float64")
            lon = np.asarray(g[f"longitude{suffix}"][:], dtype="float64")

            # SMAP uses -9999 for fill across all these fields.
            sm[sm <= -9990] = np.nan
            lat[lat <= -9990] = np.nan
            lon[lon <= -9990] = np.nan

            # The EASE-Grid 2.0 arrays are global; subset to the AOI by mask.
            in_box = (lon >= min_lon) & (lon <= max_lon) & (lat >= min_lat) & (lat <= max_lat)
            if not in_box.any():
                continue
            rows = np.where(in_box.any(axis=1))[0]
            cols = np.where(in_box.any(axis=0))[0]
            sub = sm[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]

            # Granule date comes from the filename: <product>_<version>_YYYYMMDD_...
            stamp = os.path.basename(path).split("_")[4][:8]
            times.append(_dt.datetime.strptime(stamp, "%Y%m%d"))
            stack.append(sub)

    if not stack:
        raise RuntimeError("Granules downloaded but none contained valid data inside the bbox")

    # Guard against granules with off-by-one subset extents.
    shape = min((a.shape for a in stack), key=lambda s: (s[0], s[1]))
    stack = [a[: shape[0], : shape[1]] for a in stack]

    order = np.argsort(times)
    da = xr.DataArray(
        np.stack([stack[i] for i in order]),
        dims=("time", "y", "x"),
        coords={"time": [times[i] for i in order]},
        name="soil_moisture",
    )
    da.attrs.update(units="cm3/cm3", product=product, version=version, overpass=overpass, resolution_m=9000)
    return da
