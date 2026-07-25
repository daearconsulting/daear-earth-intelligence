"""
Data cube utilities -- the same xarray/zarr cube pattern used in the
NIFA-funded Integrated Data Cube contract with Oglala Lakota College,
generalized for reuse across the demo portfolio.
"""

from __future__ import annotations

import xarray as xr


def stack_time_series(scenes: dict, var_builder) -> xr.DataArray:
    """
    Given {date_str: xr.Dataset} scenes and a function that derives one
    indicator DataArray from a scene (e.g. indicators.ndvi), stack the
    results into a single (time, lat, lon) cube.

    Example:
        scenes = {"2020-06-01": scene1, "2020-09-15": scene2}
        ndvi_cube = stack_time_series(scenes, indicators.ndvi)
    """
    dates = sorted(scenes.keys())
    arrays = [var_builder(scenes[d]) for d in dates]
    cube = xr.concat(arrays, dim=xr.DataArray(dates, dims="time", name="time"))
    return cube


def merge_layers(**layers: xr.DataArray) -> xr.Dataset:
    """
    Merge same-grid indicator layers (e.g. ndvi, dnbr, slope, erosion_risk)
    into one xr.Dataset cube for a region -- the object every dashboard and
    composite-index step in this portfolio consumes.
    """
    return xr.Dataset(layers)


def to_zarr(ds: xr.Dataset, path: str):
    """
    Persist a cube to Zarr for reuse across repos/notebooks without
    recomputing. Real deployments would point `path` at cloud object storage
    (e.g. s3://daear-data-cubes/...); this demo writes locally.
    """
    ds.to_zarr(path, mode="w")


def from_zarr(path: str) -> xr.Dataset:
    return xr.open_zarr(path)
