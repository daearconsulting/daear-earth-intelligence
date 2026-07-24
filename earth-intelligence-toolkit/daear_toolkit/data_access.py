"""
Uniform data access layer.

Each `get_*` function has the real (network-backed) implementation and a
`synthetic=True` fallback. The synthetic path is what runs in this build 
see the top-level README for why but the function signatures, coordinate
handling, and output shapes match what the live path returns, so downstream
code (indicators, cube-building, dashboards) never has to know which mode
produced the array.

Real-data dependencies (not installed in this sandbox): earthaccess,
pystac-client, rioxarray, requests (for SSURGO/SoilGrids REST endpoints).
"""

from __future__ import annotations

import numpy as np
import xarray as xr


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


# Sentinel-2/Landsat surface reflectance

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

    Live path (synthetic=False): queries Microsoft Planetary Computer's STAC
    API for `source` ("sentinel-2" or "landsat-c2-l2"), reads the scene with
    rioxarray, clips to bbox. Requires network access to
    planetarycomputer.microsoft.com and a SAS token.

    Demo path (synthetic=True): generates structurally realistic bands with
    the same dims/coords a real scene would have.
    """
    if not synthetic:
        raise NotImplementedError(
            "Live retrieval requires pystac-client + rioxarray and network "
            "access to Planetary Computer, not available in this build. "
            "Call with synthetic=True, or see README 'Going live'."
        )

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

    Live path: MTBS (Monitoring Trends in Burn Severity) direct download
    (mtbs.gov), reprojected/clipped. No auth required, but domain not
    reachable from this sandbox.
    """
    if not synthetic:
        raise NotImplementedError(
            "Live retrieval requires network access to mtbs.gov. "
            "Call with synthetic=True, or see README 'Going live'."
        )

    lons, lats = _grid(bbox)
    shape = (len(lats), len(lons))
    # Burn scar concentrated in the upper (northwest) part of the bbox,
    # tapering toward the valley floor which mimics the real Cameron Peak
    # perimeter's relationship to the Poudre canyon.
    field = _smooth_random_field(shape, seed=seed, smoothness=15)
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
    lat_norm = (lat_grid - lat_grid.min())/(lat_grid.max() - lat_grid.min())
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

    Live path: NASA FIRMS API (firms.modaps.eosdis.nasa.gov), requires a
    MAP_KEY. Domain not reachable from this sandbox.
    """
    if not synthetic:
        raise NotImplementedError(
            "Live retrieval requires a FIRMS MAP_KEY and network access to "
            "firms.modaps.eosdis.nasa.gov. Call with synthetic=True."
        )

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

    Live path: USGS 3DEP via py3dep/The National Map, or Planetary
    Computer's `cop-dem-glo30` collection. Domain not reachable here.
    """
    if not synthetic:
        raise NotImplementedError(
            "Live retrieval requires network access to USGS 3DEP or "
            "Planetary Computer. Call with synthetic=True."
        )

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

    Live path: USDA SSURGO via Soil Data Access REST API, or SoilGrids WCS.
    Domains not reachable here.
    """
    if not synthetic:
        raise NotImplementedError(
            "Live retrieval requires network access to SSURGO Soil Data "
            "Access or SoilGrids. Call with synthetic=True."
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
