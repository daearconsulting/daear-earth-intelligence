from __future__ import annotations

"""
Indicator calculations shared across all downstream repos.

These are real formulas (NDVI, NBR/dNBR, a simple erosion-susceptibility
composite) operating on whatever DataArrays data_access.py hands back
synthetic in this build, real imagery once network access is available.
Nothing about this module changes when the input source changes.
"""

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
    USFS BAER post-fire erosion modeling (ex. ERMiT/Disturbed WEPP):
    steeper slopes and higher burn severity and higher inherent runoff
    potential to higher erosion susceptibility. Weights are illustrative,
    not calibrated but flagged clearly for anyone using this beyond a demo.
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
    Composite index (0-1, higher = more resilient/lower risk) combining
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

"""
take DataArrays, return a named DataArray, keep every layer 0-1 where it makes sense so composites stay
interpretable. Grouped by which module needs them.
"""

import numpy as np
import xarray as xr


def _normalize(da, lo=None, hi=None, invert=False):
    """
    Rescale to 0-1 using percentile clipping.

    Percentiles rather than min/max because a single bad pixel i.e. a cloud edge,
    a DEM void artefact, otherwise compresses the entire rest of the layer
    into a narrow band. 2nd/98th is the toolkit default elsewhere, kept here.
    """
    lo = float(da.quantile(0.02).values) if lo is None else lo
    hi = float(da.quantile(0.98).values) if hi is None else hi
    if hi - lo < 1e-12:
        out = xr.zeros_like(da)
    else:
        out = ((da - lo) / (hi - lo)).clip(0, 1)
    return (1 - out) if invert else out


# Module 1 soil landscape indicators
def bare_soil_frequency(ndvi_stack, threshold: float = 0.20, time_dim: str = "time"):
    """
    Fraction of observations in which a cell was below the NDVI bare-ground
    threshold i.e. how *often* soil is exposed across a growing season, not
    just whether it happened once.

    Frequency rather than a seasonal minimum because the minimum is a
    single-date measurement and inherits every bit of that date's cloud,
    shadow, and BRDF noise. A cell bare in 8 of 10 clear scenes is a different
    management story from one bare in 1 of 10, and the minimum cannot tell
    them apart.

    `threshold` of 0.20 is a reasonable semi-arid default; on the Front Range
    foothills, sparse grass and litter sit right around 0.20-0.30, so check the
    NDVI histogram before trusting it. Higher threshold = more conservative
    (more cells called bare).
    """
    valid = ndvi_stack.notnull().sum(dim=time_dim)
    bare = (ndvi_stack < threshold).where(ndvi_stack.notnull()).sum(dim=time_dim)
    out = (bare / valid.where(valid > 0)).clip(0, 1)
    out.name = "bare_soil_frequency"
    out.attrs["description"] = f"fraction of clear observations with NDVI < {threshold}"
    return out


def vegetation_soil_coupling(ndvi, organic_matter, window: int = 5):
    """
    Local (moving-window) correlation between vegetation vigour and soil
    organic matter.

    The point is diagnostic, not predictive. Where the correlation is strong
    and positive, vegetation is tracking soil condition the landscape is
    behaving the way the soil map says it should. Where it is weak or
    negative, something else is driving vegetation: irrigation, aspect,
    grazing pressure, a burn scar, or an SSURGO map unit that no longer
    reflects the ground. Those decoupled areas are where field visits pay off.

    Implemented as a windowed Pearson correlation via rolling sums, which
    xarray gives us directly.
    """
    a = ndvi
    b = organic_matter
    kw = {d: window for d in a.dims}
    roll = lambda x: x.rolling(**kw, center=True, min_periods=max(3, window * window // 3))

    mean_a = roll(a).mean()
    mean_b = roll(b).mean()
    cov = roll(a * b).mean() - mean_a * mean_b
    std_a = np.sqrt((roll(a * a).mean() - mean_a**2).clip(0, None))
    std_b = np.sqrt((roll(b * b).mean() - mean_b**2).clip(0, None))

    out = (cov / (std_a * std_b).where((std_a * std_b) > 1e-9)).clip(-1, 1)
    out.name = "vegetation_soil_coupling"
    out.attrs["description"] = f"local Pearson r between NDVI and soil organic matter, {window}x{window} window"
    return out


def degradation_indicator(bare_frequency, ndvi_trend, organic_matter):
    """
    Composite degradation flag: ground that is frequently bare, losing
    vegetation over time, and low in organic matter to begin with.

    All three conditions weighted equally. Any one of them alone has an
    innocent explanation i.e. a bare field is a fallow field, a negative NDVI
    trend is a wet year followed by a dry one, low organic matter is just a
    sandy map unit. The conjunction is what is worth flagging.

    This is a screening layer for where to look, not a determination that
    degradation is occurring. Say so wherever it appears in a deliverable.
    """
    trend_component = _normalize(ndvi_trend, invert=True)  # more negative trend = worse
    om_component = _normalize(organic_matter, invert=True)  # less OM = worse
    out = (bare_frequency + trend_component + om_component) / 3.0
    out = out.clip(0, 1)
    out.name = "degradation_indicator"
    out.attrs["description"] = "0-1 screening composite: bare exposure + declining NDVI + low organic matter"
    return out


def linear_trend(stack, time_dim: str = "time", per_year: bool = True):
    """
    Per-pixel ordinary-least-squares slope through a time stack.

    Returns slope in units-per-year by default. `xr.polyfit` handles the NaN
    masking and datetime conversion; the nanosecond-to-year rescale below is
    the part that is easy to get silently wrong.
    """
    fit = stack.polyfit(dim=time_dim, deg=1, skipna=True)
    slope = fit.polyfit_coefficients.sel(degree=1)
    if per_year and np.issubdtype(stack[time_dim].dtype, np.datetime64):
        slope = slope * (365.25 * 24 * 3600 * 1e9)  # per-ns -> per-year
    slope.name = "trend"
    slope.attrs["units"] = "units/year" if per_year else "units/step"
    return slope


# Module 2 hydrologic connectivity
def transfer_weight(burn_severity=None, ndvi=None, w_min: float = 0.05, w_max: float = 1.0):
    """
    Build the W factor consumed by `hydrology.sediment_connectivity_index`.

    W represents how *readily* a cell passes runoff and sediment downslope.
    High W = bare, hydrophobic, severely burned ground with nothing to
    intercept overland flow. Low W = intact canopy, litter, and roughness.

    In the original Borselli formulation W comes from the USLE C factor; the
    Cavalli variant uses terrain roughness. Deriving it from burn severity and
    NDVI is the post-fire adaptation and is what makes the Cameron Peak scar
    show up as a connectivity anomaly rather than just a slope artefact.

    Pass either or both inputs; with both, they are averaged.
    """
    parts = []
    if burn_severity is not None:
        parts.append(_normalize(burn_severity))  # higher severity -> higher W
    if ndvi is not None:
        parts.append(_normalize(ndvi, invert=True))  # denser vegetation -> lower W
    if not parts:
        raise ValueError("supply burn_severity, ndvi, or both")

    combined = sum(parts) / len(parts)
    out = (w_min + combined * (w_max - w_min)).clip(w_min, w_max)
    out.name = "transfer_weight"
    out.attrs["description"] = "Borselli W factor from burn severity / vegetation cover; higher = less impedance"
    return out


def delivery_risk(connectivity, erosion_susceptibility):
    """
    Where erodible ground is also hydrologically wired to the channel network.

    This is the layer that actually answers the water-provider question. High
    erosion susceptibility on a disconnected bench is a soil problem; the same
    susceptibility on a directly-coupled slope is a reservoir-turbidity
    problem. Multiplying rather than averaging is deliberate: either factor
    being near zero should collapse the risk, because sediment that cannot
    move or cannot arrive does not reach the intake.
    """
    ic_norm = _normalize(connectivity)
    out = (ic_norm * erosion_susceptibility).clip(0, 1)
    out.name = "sediment_delivery_risk"
    out.attrs["description"] = "erosion susceptibility gated by hydrologic connectivity to the channel network"
    return out


# Module 4 restoration monitoring
def difference_in_differences(pre, post, treated_mask, control_mask=None):
    """
    Paired difference-in-differences effect estimate for a treatment.

    DiD = (treated_post - treated_pre) - (control_post - control_pre)

    The control term is what makes this worth doing. Treated areas almost
    always improve after treatment, because treatment happens after
    disturbance and disturbance is followed by recovery whether or not anyone
    intervenes. Subtracting the untreated trajectory removes that shared
    recovery signal and the shared weather signal along with it.

    Returns a dict of the components plus the estimate, so the intermediate
    numbers can be reported a DiD without its four constituent means is not
    reviewable.

    The identifying assumption is parallel trends: treated and untreated areas
    would have followed the same trajectory absent treatment. That assumption
    is usually false in restoration siting, because crews treat the worst
    ground first. Check pre-period trends before believing the estimate, and
    report the check.
    """
    if control_mask is None:
        control_mask = ~treated_mask

    def _mean(da, mask):
        return float(da.where(mask).mean().values)

    t_pre, t_post = _mean(pre, treated_mask), _mean(post, treated_mask)
    c_pre, c_post = _mean(pre, control_mask), _mean(post, control_mask)

    return {
        "treated_pre": t_pre,
        "treated_post": t_post,
        "treated_change": t_post - t_pre,
        "control_pre": c_pre,
        "control_post": c_post,
        "control_change": c_post - c_pre,
        "did_estimate": (t_post - t_pre) - (c_post - c_pre),
        "n_treated_cells": int(treated_mask.sum().values),
        "n_control_cells": int(control_mask.sum().values),
    }


def recovery_ratio(current, pre_disturbance, floor: float = 0.05):
    """
    Fraction of pre-disturbance condition recovered, clipped to 0-1.5.

    Allowed to exceed 1 on purpose: post-fire flushes of herbaceous cover
    genuinely exceed pre-fire NDVI in the first few seasons, and clipping that
    to 1.0 would hide a real and diagnostically useful signal (grass, not
    conifer). `floor` guards against dividing by near-zero baselines.
    """
    out = (current / pre_disturbance.where(pre_disturbance > floor)).clip(0, 1.5)
    out.name = "recovery_ratio"
    return out


def landscape_function_index(vegetation_recovery, soil_moisture_recovery, watershed_resilience, weights=(0.4, 0.2, 0.4)):
    """
    Composite of the three recovery dimensions Module 4 tracks.

    Default weights lean on vegetation and watershed resilience over soil
    moisture, for a defensible reason: at 9 km, SMAP contributes basin-scale
    context rather than treatment-scale signal, so it should not drive the
    composite. Adjust if you wire in a finer moisture product.

    Any component may be None, in which case remaining weights are
    renormalized that keeps the index computable in the common case where
    the moisture record has gaps.
    """
    components = [
        (vegetation_recovery, weights[0]),
        (soil_moisture_recovery, weights[1]),
        (watershed_resilience, weights[2]),
    ]
    present = [(c, w) for c, w in components if c is not None]
    if not present:
        raise ValueError("at least one component is required")

    total_w = sum(w for _, w in present)
    out = sum(_normalize(c) * (w / total_w) for c, w in present).clip(0, 1)
    out.name = "landscape_function_index"
    out.attrs["components"] = ", ".join(
        n for n, c in zip(("vegetation", "soil_moisture", "watershed_resilience"),
                          (vegetation_recovery, soil_moisture_recovery, watershed_resilience)) if c is not None
    )
    return out

