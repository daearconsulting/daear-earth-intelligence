from __future__ import annotations

"""
daear_toolkit.fire_indicators

Fire-specific indicator computations for `wildfire-landscape-intelligence`.

    Module 1  fuel_hazard, moisture_deficit, prefire_condition_index
    Module 2  nbr, dnbr, rdnbr, classify_severity, progression_from_detections
    Module 3  recovery_ratio_series, fit_recovery_curve, recovery_half_life
    Module 4  exposure_by_zone, wui_classify, distance_weighted_exposure

Severity thresholds follow Key & Benson (2006) for dNBR and Miller & Thode
(2007) for RdNBR. Both are stated as constants rather than buried in code
because they are calibration choices, not physical constants, and a reader
needs to be able to see and challenge them.
"""

import numpy as np
import pandas as pd
from scipy import optimize, stats

try:
    import xarray as xr
except ImportError:  # pragma: no cover
    xr = None


# Spectral indices
def nbr(scene, nir: str = "B08", swir: str = "B12"):
    """
    Normalized Burn Ratio: (NIR - SWIR2) / (NIR + SWIR2).

    Sentinel-2 B08 (842 nm) and B12 (2190 nm). Healthy vegetation is bright in
    NIR and dark in SWIR2, so NBR is high; burned ground reverses both, so NBR
    drops sharply. That contrast is why NBR outperforms NDVI for severity 
    NDVI saturates in dense canopy and cannot distinguish a scorched-but-intact
    crown from a consumed one.

    B8A (865 nm, 20 m) is the more correct narrow NIR band to pair with B12 at
    native 20 m resolution. B08 is 10 m and broader; using it means resampling
    one band or the other. Either is defensible if stated.
    """
    n, s = scene[nir], scene[swir]
    out = (n - s) / (n + s)
    out.name = "nbr"
    return out


def dnbr(prefire_nbr, postfire_nbr, offset: float | None = None):
    """
    Differenced NBR: prefire - postfire, scaled by 1000 as is conventional.

    `offset` subtracts a control-area dNBR to correct for phenological and
    illumination differences between the two dates. Without it, a pre-fire scene
    from June and a post-fire scene from September carry a seasonal signal that
    is indistinguishable from low-severity burn. The standard approach is to
    take the mean dNBR over unburned ground just outside the perimeter and
    subtract it `compute_dnbr_offset` below does this.

    Skipping the offset is the most common reason a dNBR map shows widespread
    low severity that is really just phenology.
    """
    out = (prefire_nbr - postfire_nbr) * 1000
    if offset is not None:
        out = out - offset
    out.name = "dnbr"
    return out


def compute_dnbr_offset(dnbr_raw, unburned_mask) -> float:
    """
    Mean dNBR over unburned control area, for phenological offset correction.

    `unburned_mask` should cover ground just outside the perimeter with similar
    vegetation and aspect — not distant terrain, which introduces its own
    differences.
    """
    return float(dnbr_raw.where(unburned_mask).median())


def rdnbr(prefire_nbr, postfire_nbr, offset: float | None = None):
    """
    Relativized dNBR (Miller & Thode 2007): dNBR/sqrt(|prefire NBR|).

    dNBR measures absolute change, which means the same fire effect registers
    as higher severity in dense pre-fire canopy than in sparse. Across a
    heterogeneous landscape and the Cameron Peak footprint spans dense
    lodgepole through open ponderosa into grassland that makes dNBR partly a
    map of pre-fire biomass rather than of fire effects.

    RdNBR normalizes by pre-fire condition and is the better choice when
    comparing severity across vegetation types. It is noisier where pre-fire NBR
    is near zero, which is why the denominator is floored below.

    Report which one was used. The two produce visibly different severity maps
    of the same fire and reviewers will ask.
    """
    d = dnbr(prefire_nbr, postfire_nbr, offset=offset)
    denom = np.sqrt(np.abs(prefire_nbr).clip(0.001, None))
    out = d / denom
    out.name = "rdnbr"
    return out


def ndmi(scene, nir: str = "B08", swir: str = "B11"):
    """
    Normalized Difference Moisture Index: (NIR - SWIR1)/(NIR + SWIR1).

    Sensitive to vegetation water content, so it is the pre-fire fuel-dryness
    proxy in Module 1. Falls before a fire season in a way NDVI does not,
    because NDVI tracks greenness and vegetation can stay green while its
    moisture content drops well below the threshold that governs ignition.
    """
    n, s = scene[nir], scene[swir]
    out = (n - s) / (n + s)
    out.name = "ndmi"
    return out


# Severity classification
# Key & Benson (2006) dNBR breakpoints. These are the widely used defaults and
# they are also known to be imperfect outside the ecosystems they were
# calibrated in. For Southern Rockies conifer they are reasonable; for
# shrubland or grassland they are not, and field-calibrated CBI thresholds
# should replace them wherever plot data exists.
DNBR_THRESHOLDS = {"unburned": 100, "low": 270, "moderate": 660, "high": 1300}

# Miller & Thode (2007) RdNBR breakpoints.
RDNBR_THRESHOLDS = {"unburned": 69, "low": 316, "moderate": 641, "high": 1500}

SEVERITY_LABELS = ["unburned/regrowth", "low", "moderate", "high"]


def classify_severity(index_raster, thresholds: dict | None = None, metric: str = "dnbr"):
    """
    Classify a dNBR or RdNBR raster into four severity classes.

    Returns an integer raster: 0 unburned/regrowth, 1 low, 2 moderate, 3 high.

    Note class 0 is 'unburned OR regrowth'. Strongly negative dNBR means the
    post-fire scene is *greener* than the pre-fire one, which inside a perimeter
    usually means the pre-fire scene caught the area before green-up rather than
    meaning anything about the fire. Another argument for date selection
    mattering as much as the algorithm.
    """
    t = thresholds or (DNBR_THRESHOLDS if metric == "dnbr" else RDNBR_THRESHOLDS)
    edges = [t["unburned"], t["low"], t["moderate"]]

    if xr is not None and hasattr(index_raster, "dims"):
        out = xr.apply_ufunc(np.digitize, index_raster, kwargs={"bins": edges})
        out.name = "severity_class"
        out.attrs.update(metric=metric, thresholds=str(t), labels=str(SEVERITY_LABELS))
        return out
    return np.digitize(np.asarray(index_raster), edges)


def severity_summary(severity_class, pixel_area_ha: float = 0.09) -> pd.DataFrame:
    """
    Area by severity class. 0.09 ha = one 30 m pixel; use 0.04 for 20 m Sentinel-2.
    """
    vals = np.asarray(getattr(severity_class, "values", severity_class))
    vals = vals[np.isfinite(vals)]
    rows = []
    for code, label in enumerate(SEVERITY_LABELS):
        n = int((vals == code).sum())
        rows.append({"class": code, "label": label, "pixels": n,
                     "hectares": round(n * pixel_area_ha, 1),
                     "acres": round(n * pixel_area_ha * 2.471, 1),
                     "pct": round(100 * n / len(vals), 1) if len(vals) else np.nan})
    return pd.DataFrame(rows)


# Module 1 pre-fire condition
# Scott & Burgan 40 fuel models, grouped by expected rate of spread and
# intensity. This collapse to an ordinal hazard scale is a simplification 
# fuel models are categorical and not truly ordered but it is what lets fuels
# enter a composite index. State it as a simplification wherever it is used.
FUEL_HAZARD_GROUPS = {
    "non_burnable": (91, 92, 93, 98, 99),
    "low": (101, 102, 121, 141, 142, 161, 181, 182, 183, 184, 185, 186, 187, 188, 189),
    "moderate": (103, 104, 105, 122, 123, 143, 144, 145, 162, 163),
    "high": (106, 107, 108, 109, 124, 146, 147, 148, 149, 164, 165),
    "extreme": (110, 201, 202, 203, 204),
}
_FUEL_SCORE = {"non_burnable": 0.0, "low": 0.25, "moderate": 0.5, "high": 0.75, "extreme": 1.0}


def fuel_hazard(fbfm40):
    """Map Scott & Burgan fuel model codes to a 0-1 ordinal hazard score."""
    arr = np.asarray(getattr(fbfm40, "values", fbfm40), dtype="float64")
    out = np.full(arr.shape, np.nan)
    for group, codes in FUEL_HAZARD_GROUPS.items():
        out[np.isin(arr, codes)] = _FUEL_SCORE[group]

    if xr is not None and hasattr(fbfm40, "dims"):
        res = xr.DataArray(out, coords=fbfm40.coords, dims=fbfm40.dims)
        res.name = "fuel_hazard"
        return res
    return out


def crown_fire_potential(canopy_cover, canopy_base_height, canopy_bulk_density):
    """
    Simple crown-fire susceptibility from LANDFIRE canopy structure.

    Crown fire needs three things together: enough canopy cover for horizontal
    continuity, a low enough canopy base height for surface flames to reach the
    crown, and enough bulk density to sustain propagation once there.

    This is a screening index, not Van Wagner's model so no wind, no surface
    intensity, no foliar moisture. Where it is high, run a real fire behavior
    model. Presenting this as a crown-fire prediction would be overclaiming.
    """
    cc = (canopy_cover / 100.0).clip(0, 1)
    # CBH under ~2 m is a ladder to the crown; above ~8 m surface fire rarely reaches it.
    cbh_component = (1 - ((canopy_base_height - 2.0) / 6.0)).clip(0, 1)
    # ~0.10 kg/m3 is the conventional minimum bulk density for active crown fire.
    cbd_component = (canopy_bulk_density / 0.15).clip(0, 1)

    out = (cc * cbh_component * cbd_component) ** (1 / 3)   # geometric mean: any factor near zero collapses it
    out.name = "crown_fire_potential"
    return out


def prefire_condition_index(fuel_hazard_layer, moisture_deficit, crown_potential, slope_deg, weights=(0.3, 0.3, 0.25, 0.15)):
    """
    Composite pre-fire condition: fuels, dryness, crown structure, and terrain.

    Weights are stated and adjustable. As with the climate composite, the honest
    move is to test how much the ranking depends on them rather than defending a
    single set — the weight-sensitivity machinery in `climate_indicators` works
    on any indicator table and applies directly here.
    """
    def _norm(x):
        lo, hi = float(x.quantile(0.02)), float(x.quantile(0.98))
        return ((x - lo) / (hi - lo)).clip(0, 1) if hi > lo else x * 0

    slope_component = _norm(slope_deg)
    w = np.array(weights, dtype="float64")
    w = w / w.sum()

    out = (w[0] * fuel_hazard_layer + w[1] * _norm(moisture_deficit) +
           w[2] * crown_potential + w[3] * slope_component).clip(0, 1)
    out.name = "prefire_condition_index"
    return out


# Module 2 progression
def progression_from_detections(detections: pd.DataFrame, freq: str = "D") -> pd.DataFrame:
    """
    Daily fire progression statistics from FIRMS detections.

    Returns detection counts, mean/max fire radiative power, and the centroid
    per period; enough to identify major run days and the direction of spread.

    FRP is a decent proxy for fire intensity at the time of overpass and a poor
    proxy for total energy release, since it samples twice daily at best. Big
    night runs between overpasses are systematically under-represented. Use this
    for timing, not for energy budgets.
    """
    if detections.empty:
        return pd.DataFrame()

    df = detections.copy()
    df["acq_date"] = pd.to_datetime(df["acq_date"])
    frp_col = "frp" if "frp" in df.columns else None

    grouped = df.groupby(pd.Grouper(key="acq_date", freq=freq))
    out = pd.DataFrame({
        "detections": grouped.size(),
        "mean_lat": grouped["latitude"].mean(),
        "mean_lon": grouped["longitude"].mean(),
    })
    if frp_col:
        out["mean_frp"] = grouped[frp_col].mean()
        out["max_frp"] = grouped[frp_col].max()
        out["total_frp"] = grouped[frp_col].sum()

    out = out[out["detections"] > 0]
    out["cumulative_detections"] = out["detections"].cumsum()
    # Daily centroid displacement: a proxy for spread direction and run distance.
    out["centroid_shift_km"] = np.sqrt(
        (out["mean_lat"].diff() * 111.0) ** 2 +
        (out["mean_lon"].diff() * 111.0 * np.cos(np.deg2rad(out["mean_lat"]))) ** 2
    )
    return out


def severity_vs_predictor(severity_class, predictor, n_bins: int = 10, labels=None) -> pd.DataFrame:
    """
    Cross-tabulate observed severity against a pre-fire predictor.

    Did pre-fire fuel hazard, crown potential, or
    terrain actually predict where the fire burned severely?

    The answer for large wind-driven fires is often "barely", and that is a
    finding rather than a failure. Cameron Peak's major runs were driven by
    extreme wind and drought; under those conditions fuel treatment effects get
    overwhelmed. An analysis that only reports strong fuel-severity
    relationships is selecting for the fires where fuels mattered.
    """
    sev = np.asarray(getattr(severity_class, "values", severity_class)).ravel()
    pred = np.asarray(getattr(predictor, "values", predictor)).ravel()
    ok = np.isfinite(sev) & np.isfinite(pred)
    sev, pred = sev[ok], pred[ok]
    if len(sev) == 0:
        return pd.DataFrame()

    bins = np.nanpercentile(pred, np.linspace(0, 100, n_bins + 1))
    bins = np.unique(bins)
    idx = np.digitize(pred, bins[1:-1])

    rows = []
    for b in range(len(bins) - 1):
        m = idx == b
        if m.sum() < 50:
            continue
        rows.append({
            "bin": b,
            "predictor_lo": round(float(bins[b]), 3),
            "predictor_hi": round(float(bins[b + 1]), 3),
            "n_pixels": int(m.sum()),
            "mean_severity_class": round(float(sev[m].mean()), 3),
            "pct_high_severity": round(100 * float((sev[m] == 3).mean()), 1),
        })

    out = pd.DataFrame(rows)
    if len(out) > 2:
        # Spearman, not Pearson: the relationship is monotonic at best and
        # severity class is ordinal, not interval.
        rho, p = stats.spearmanr(out["predictor_lo"], out["pct_high_severity"])
        out.attrs["spearman_rho"] = round(float(rho), 3)
        out.attrs["spearman_p"] = round(float(p), 4)
    return out


# Module 3 recovery
def _recovery_model(t, asymptote, rate, initial):
    """Asymptotic exponential: y = A - (A - y0) * exp(-rate * t)."""
    return asymptote - (asymptote - initial) * np.exp(-rate * t)


def fit_recovery_curve(years, values, p0=None) -> dict:
    """
    Fit an asymptotic exponential recovery curve to a post-fire index series.

    Returns the asymptote (recovery ceiling), rate, and derived half-life.

    The model choice encodes an ecological assumption worth making explicit: it
    says recovery approaches a ceiling that may be *below* the pre-fire value,
    which is the right shape for high-severity conifer where the pre-fire forest
    is not coming back on a decadal timescale. A linear fit would imply
    unbounded recovery and a logistic would imply a lag phase this data rarely
    resolves.

    The ceiling is the interesting parameter. A fast rate to a low asymptote is
    grass and shrub establishment, not forest recovery, and the two look similar
    in NDVI for the first several years.
    """
    t = np.asarray(years, dtype="float64")
    y = np.asarray(values, dtype="float64")
    ok = np.isfinite(t) & np.isfinite(y)
    t, y = t[ok] - np.min(t[ok]), y[ok]

    if len(t) < 4:
        return {"asymptote": np.nan, "rate": np.nan, "half_life_years": np.nan, "n": len(t), "converged": False}

    if p0 is None:
        p0 = [float(np.nanmax(y)), 0.3, float(y[0])]

    try:
        popt, pcov = optimize.curve_fit(_recovery_model, t, y, p0=p0, maxfev=20000,
                                        bounds=([0, 1e-4, -np.inf], [np.inf, 5.0, np.inf]))
        asymptote, rate, initial = popt
        resid = y - _recovery_model(t, *popt)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - np.sum(resid**2) / ss_tot if ss_tot > 0 else np.nan
        return {
            "asymptote": float(asymptote), "rate": float(rate), "initial": float(initial),
            "half_life_years": float(np.log(2) / rate),
            "r_squared": float(r2), "n": int(len(t)), "converged": True,
            "param_se": [float(x) for x in np.sqrt(np.diag(pcov))],
        }
    except (RuntimeError, ValueError) as exc:
        return {"asymptote": np.nan, "rate": np.nan, "half_life_years": np.nan,
                "n": int(len(t)), "converged": False, "error": str(exc)}


def recovery_by_severity(ndvi_series: pd.DataFrame, severity_class, prefire_ndvi) -> pd.DataFrame:
    """
    Recovery trajectory stratified by severity class.

    Stratification is essential. A whole-fire mean recovery curve averages
    ground that barely burned with ground that lost its entire canopy, and
    produces a number describing neither. The severity-stratified curves are
    what a land manager can act on.
    """
    rows = []
    for code, label in enumerate(SEVERITY_LABELS):
        mask = severity_class == code
        n = float(mask.sum())
        if n < 100:
            continue
        pre = float(prefire_ndvi.where(mask).mean())
        for year, layer in ndvi_series.items():
            post = float(layer.where(mask).mean())
            rows.append({"severity": label, "severity_class": code, "year": year,
                         "mean_ndvi": round(post, 4), "prefire_ndvi": round(pre, 4),
                         "recovery_ratio": round(post / pre, 4) if pre > 0.05 else np.nan,
                         "n_pixels": int(n)})
    return pd.DataFrame(rows)


# Module 4 exposure
def exposure_by_zone(points_gdf, zone_raster, zone_lookup: dict | None = None, value_col: str | None = None) -> pd.DataFrame:
    """
    Count point assets (structures, intakes) falling in each zone of a raster.

    Sampling a raster at point locations, which is the right operation for
    structures: a building either is or is not in a severity class, and area
    weighting would be meaningless for a point asset.
    """
    if points_gdf.empty:
        return pd.DataFrame()

    ydim, xdim = zone_raster.dims
    sampled = zone_raster.sel(
        {xdim: xr.DataArray(points_gdf.geometry.x.values, dims="pt"),
         ydim: xr.DataArray(points_gdf.geometry.y.values, dims="pt")},
        method="nearest",
    ).values

    df = points_gdf.copy()
    df["zone"] = sampled
    grouped = df.groupby("zone", dropna=True)

    out = pd.DataFrame({"n_assets": grouped.size()})
    if value_col and value_col in df.columns:
        out["total_value"] = grouped[value_col].sum()
    out.index.name = "zone"
    out = out.reset_index()
    if zone_lookup:
        out["zone_label"] = out["zone"].map(zone_lookup)
    return out


def wui_classify(building_density, vegetation_cover, density_threshold: float = 6.17, cover_threshold: float = 0.5):
    """
    Wildland-Urban Interface classification, after the Radeloff et al. definition.

    Two classes:
      - **Interface**: dense housing adjacent to wildland vegetation
      - **Intermix**: housing dispersed *within* wildland vegetation

    Default thresholds are the federal register values: 6.17 housing units per
    km^2, and 50% wildland vegetation cover.

    The distinction matters operationally. Intermix communities have structures
    embedded in continuous fuels and are defended structure-by-structure;
    interface communities have a defensible edge. Collapsing them into one "WUI"
    number loses the thing an emergency manager needs.
    """
    dense = building_density >= density_threshold
    vegetated = vegetation_cover >= cover_threshold

    out = xr.zeros_like(building_density)
    out = out.where(~(dense & vegetated), 2)     # intermix
    out = out.where(~(dense & ~vegetated), 1)    # interface (dense, low veg edge condition)
    out.name = "wui_class"
    out.attrs["labels"] = "0=non-WUI, 1=interface, 2=intermix"
    return out


def distance_weighted_exposure(asset_points, hazard_raster, decay_km: float = 2.0) -> pd.DataFrame:
    """
    Hazard exposure for each asset, weighted by distance-decayed surrounding hazard.

    A structure's risk is not only the hazard of the pixel it sits on it is
    also what is burning nearby. This applies an exponential distance decay so
    that adjacent high-severity ground contributes and distant ground does not.

    `decay_km` of 2 km is a reasonable default for ember and radiant exposure in
    mountain terrain. It is a parameter, not a constant; anything presented to a
    partner should show the result under at least two values.
    """
    if asset_points.empty:
        return pd.DataFrame()

    ydim, xdim = hazard_raster.dims
    ys = hazard_raster.coords[ydim].values
    xs = hazard_raster.coords[xdim].values
    hz = np.asarray(hazard_raster.values, dtype="float64")

    lat_mean = float(np.mean(ys))
    km_per_deg_lat = 111.0
    km_per_deg_lon = 111.0 * np.cos(np.deg2rad(lat_mean))

    rows = []
    for _, asset in asset_points.iterrows():
        dy = (ys[:, None] - asset.geometry.y) * km_per_deg_lat
        dx = (xs[None, :] - asset.geometry.x) * km_per_deg_lon
        dist = np.sqrt(dy**2 + dx**2)

        w = np.exp(-dist / decay_km)
        w[dist > decay_km * 3] = 0.0     # truncate a negligible tail for speed
        valid = np.isfinite(hz) & (w > 0)
        score = float((hz[valid] * w[valid]).sum() / w[valid].sum()) if w[valid].sum() > 0 else np.nan

        rows.append({
            "name": asset.get("name"),
            "lat": asset.geometry.y, "lon": asset.geometry.x,
            "weighted_hazard": round(score, 4) if np.isfinite(score) else np.nan,
            "decay_km": decay_km,
        })
    return pd.DataFrame(rows)
