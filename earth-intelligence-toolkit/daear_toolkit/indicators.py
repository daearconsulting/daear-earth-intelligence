"""
Indicator calculations shared across all downstream repos.

These are real formulas (NDVI, NBR/dNBR, a simple erosion-susceptibility
composite) operating on whatever DataArrays data_access.py hands back --
synthetic in this build, real imagery once network access is available.
Nothing about this module changes when the input source changes.
"""

from __future__ import annotations

import numpy as np
import xarray as xr


def ndvi(scene: xr.Dataset) -> xr.DataArray:
    """Normalized Difference Vegetation Index from nir/red bands."""
    out = (scene["nir"] - scene["red"]) / (scene["nir"] + scene["red"] + 1e-9)
    out.name = "ndvi"
    out.attrs["description"] = "Normalized Difference Vegetation Index"
    return out


def nbr(scene: xr.Dataset) -> xr.DataArray:
    """Normalized Burn Ratio from nir/swir16 bands."""
    out = (scene["nir"] - scene["swir16"]) / (scene["nir"] + scene["swir16"] + 1e-9)
    out.name = "nbr"
    out.attrs["description"] = "Normalized Burn Ratio"
    return out


def dnbr(pre_fire_scene: xr.Dataset, post_fire_scene: xr.Dataset) -> xr.DataArray:
    """
    Differenced Normalized Burn Ratio: pre-fire NBR minus post-fire NBR.
    Higher values indicate greater burn severity. This is the standard
    metric MTBS itself is built on.
    """
    out = nbr(pre_fire_scene) - nbr(post_fire_scene)
    out.name = "dnbr"
    out.attrs["description"] = "Differenced Normalized Burn Ratio (pre - post)"
    return out


def burn_severity_class(dnbr_array: xr.DataArray) -> xr.DataArray:
    """
    Classify dNBR into standard USGS severity bins:
    0 unburned, 1 low, 2 moderate-low, 3 moderate-high, 4 high.
    """
    bins = [-np.inf, 0.1, 0.27, 0.44, 0.66, np.inf]
    labels = [0, 1, 2, 3, 4]
    classed = xr.apply_ufunc(
        lambda arr: np.digitize(arr, bins) - 1,
        dnbr_array,
    )
    classed.name = "burn_severity_class"
    classed.attrs["labels"] = {
        0: "unburned", 1: "low", 2: "moderate-low", 3: "moderate-high", 4: "high"
    }
    return classed.clip(0, 4)


def fuel_proxy(ndvi_array: xr.DataArray, drought_stress: xr.DataArray | None = None) -> xr.DataArray:
    """
    Simple fuel-accumulation proxy: high, stable-to-increasing NDVI combined
    with drought stress signals elevated fuel + dryness risk. Without a time
    series, this demo version treats current NDVI as a fuel-load proxy and
    (optionally) discounts it by drought stress.
    """
    if drought_stress is None:
        out = ndvi_array.copy()
    else:
        out = ndvi_array * (1 - 0.5 * drought_stress)
    out.name = "fuel_proxy"
    out.attrs["description"] = "Relative fuel-load proxy (0-1, higher = more fuel)"
    return out


def erosion_susceptibility(
    slope_deg: xr.DataArray,
    burn_severity: xr.DataArray,
    runoff_potential: xr.DataArray,
) -> xr.DataArray:
    """
    Composite post-fire erosion susceptibility index (0-1).

    This is a simplified, transparent stand-in for the logic behind
    USFS BAER post-fire erosion modeling (e.g. ERMiT/Disturbed WEPP):
    steeper slopes + higher burn severity + higher inherent runoff
    potential -> higher erosion susceptibility. Weights are illustrative,
    not calibrated -- flagged clearly for anyone using this beyond a demo.
    """
    slope_norm = (slope_deg - slope_deg.min()) / (slope_deg.max() - slope_deg.min() + 1e-9)
    out = 0.4 * slope_norm + 0.4 * burn_severity + 0.2 * runoff_potential
    out.name = "erosion_susceptibility"
    out.attrs["description"] = (
        "Illustrative composite (0.4*slope + 0.4*burn_severity + 0.2*runoff_potential), "
        "not a calibrated erosion model"
    )
    return out.clip(0, 1)


def soil_vulnerability(
    organic_matter_pct: xr.DataArray,
    sand_fraction_pct: xr.DataArray,
    slope_deg: xr.DataArray,
) -> xr.DataArray:
    """
    Composite soil vulnerability index (0-1, higher = more vulnerable to
    degradation): low organic matter + high sand fraction (low water/nutrient
    retention) + steep slope.
    """
    om_norm = 1 - (organic_matter_pct - organic_matter_pct.min()) / (
        organic_matter_pct.max() - organic_matter_pct.min() + 1e-9
    )
    sand_norm = (sand_fraction_pct - sand_fraction_pct.min()) / (
        sand_fraction_pct.max() - sand_fraction_pct.min() + 1e-9
    )
    slope_norm = (slope_deg - slope_deg.min()) / (slope_deg.max() - slope_deg.min() + 1e-9)
    out = 0.4 * om_norm + 0.3 * sand_norm + 0.3 * slope_norm
    out.name = "soil_vulnerability"
    out.attrs["description"] = "Illustrative composite soil vulnerability index"
    return out.clip(0, 1)


def watershed_resilience_index(
    erosion_susceptibility_da: xr.DataArray,
    soil_vulnerability_da: xr.DataArray,
    vegetation_recovery_da: xr.DataArray | None = None,
) -> xr.DataArray:
    """
    Composite index (0-1, higher = more resilient / lower risk) combining
    erosion susceptibility and soil vulnerability, optionally offset by
    vegetation recovery. This is the indicator climate-resilience-indicators
    aggregates across regions.
    """
    risk = 0.5 * erosion_susceptibility_da + 0.5 * soil_vulnerability_da
    if vegetation_recovery_da is not None:
        risk = risk * (1 - 0.3 * vegetation_recovery_da)
    resilience = 1 - risk.clip(0, 1)
    resilience.name = "watershed_resilience_index"
    resilience.attrs["description"] = "1 - blended risk composite; higher = more resilient"
    return resilience
