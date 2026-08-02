"""
daear_toolkit.climate_indicators

Indicator computations for `climate-resilience-indicators`.

Grouped by module:

    Module 1  theil_sen_trend, mann_kendall, seasonal_aggregate
    Module 2  spi, spei, fire_weather_days, snow_metrics, low_flow_metrics
    Module 3  normalize_indicator, composite_index, weight_sensitivity,
              rank_stability
    Module 4  delta_change, emergence_year

Two conventions used throughout:

1. **Robust statistics by default.** Ordinary least squares and Pearson
   correlation are both badly behaved on hydroclimate series, which are
   skewed, autocorrelated, and full of legitimate extremes that are not
   outliers. Theil-Sen and Mann-Kendall cost almost nothing and do not fall
   over on a single flood year.

2. **Higher = worse.** Every indicator is oriented so that a larger value
   means more stress, before it reaches the composite. Getting this wrong is
   the most common way a composite index silently inverts, and it is very hard
   to spot after the fact.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats


# Module 1 Trends
def theil_sen_trend(values, times=None, per_decade: bool = True) -> dict:
    """
    Theil-Sen slope: the median of all pairwise slopes.

    Chosen over OLS because it tolerates up to ~29% contaminated data before
    breaking down, where a single anomalous year can visibly tilt an OLS line
    through a 30-year record. For climate series with real extremes in them,
    that robustness is the difference between a trend estimate and a
    description of the worst year.

    Returns slope, the 95% confidence interval, and the Mann-Kendall test that
    goes with it.
    """
    y = np.asarray(values, dtype="float64")
    x = np.arange(len(y), dtype="float64") if times is None else np.asarray(times, dtype="float64")

    ok = np.isfinite(y) & np.isfinite(x)
    if ok.sum() < 10:
        return {"slope": np.nan, "lo": np.nan, "hi": np.nan, "p_value": np.nan, "n": int(ok.sum())}

    slope, intercept, lo, hi = stats.theilslopes(y[ok], x[ok], alpha=0.95)
    mk = mann_kendall(y[ok])

    scale = 10.0 if per_decade else 1.0
    return {
        "slope": slope * scale,
        "lo": lo * scale,
        "hi": hi * scale,
        "intercept": intercept,
        "p_value": mk["p_value"],
        "tau": mk["tau"],
        "significant": mk["p_value"] < 0.05,
        "n": int(ok.sum()),
        "units": "per decade" if per_decade else "per step",
    }


def mann_kendall(values, autocorr_correction: bool = False) -> dict:
    """
    Mann-Kendall trend test nonparametric, tests for monotonic trend.

    Serial correlation badly inflates Mann-Kendall's false-positive rate: on an
    AR(1) series with rho=0.7 the nominal 5% test rejects the null roughly 35%
    of the time. Annual hydroclimate series especially streamflow and
    groundwater are routinely that autocorrelated, so this is not a corner
    case. Set `autocorr_correction=True` for anything with persistence.

    The correction implemented is Hamed & Rao (1998): inflate Var(S) by an
    effective-sample-size factor computed from the autocorrelation of the
    detrended ranks, retaining only individually significant lags.

    Yue-Pilon trend-free prewhitening is the other standard option and it is
    deliberately *not* used here. Tested against AR(1) noise it made things
    worse, not better (~52% false positives against ~36% uncorrected), because
    it removes the AR structure and then adds the Theil-Sen slope back in 
    under the null that slope is itself the spurious trend, and reinserting it
    into a now-variance-reduced series makes it look more significant rather
    than less. Hamed-Rao touches only the variance and does not have this
    failure mode. `test_climate_indicators.py` checks that it actually reduces
    the false-positive rate rather than assuming it.

    Be aware the correction reduces the problem without solving it: on AR(0.7)
    noise at n=50 it takes the false-positive rate from ~32% to ~18%, against a
    nominal 5%. Statistical power is unaffected (a real trend of comparable
    magnitude is still detected essentially every time). So a corrected p-value
    just above 0.05 on a strongly autocorrelated series should be read as
    "no trend detected", and a corrected p just below it deserves a block
    bootstrap before it goes in a report.

    Ties are handled in the variance term, which matters for indicators with
    repeated integer values like counts of threshold-exceedance days.
    """
    y = np.asarray(values, dtype="float64")
    y = y[np.isfinite(y)]
    n = len(y)
    if n < 10:
        return {"tau": np.nan, "p_value": np.nan, "s": np.nan, "n": n}

    s = int(np.sum(np.sign(y[None, :] - y[:, None])[np.triu_indices(n, 1)]))

    # Variance with the standard tie correction.
    _, counts = np.unique(y, return_counts=True)
    tie_term = np.sum(counts * (counts - 1) * (2 * counts + 5))
    var_s = (n * (n - 1) * (2 * n + 5) - tie_term) / 18.0

    correction = 1.0
    if autocorr_correction and n >= 15:
        slope = stats.theilslopes(y, np.arange(n))[0]
        ranks = stats.rankdata(y - slope * np.arange(n))
        ranks = ranks - ranks.mean()
        denom = float(np.sum(ranks**2))

        acc = 0.0
        if denom > 0:
            for lag in range(1, n - 2):
                rho = float(np.sum(ranks[: n - lag] * ranks[lag:]) / denom)
                # Retain only lags individually significant at 5% (Hamed & Rao's
                # screening step); including every lag adds noise to the
                # correction and can make it worse than no correction at all.
                m = n - lag
                bound = 1.96 * np.sqrt(m - 1) / m
                if abs(rho) > bound:
                    acc += (n - lag) * (n - lag - 1) * (n - lag - 2) * rho
        correction = 1.0 + (2.0 / (n * (n - 1) * (n - 2))) * acc
        correction = max(correction, 1.0)  # never deflate the variance
        var_s *= correction

    if var_s <= 0:
        return {"tau": 0.0, "p_value": 1.0, "s": s, "n": n}

    z = (s - np.sign(s)) / np.sqrt(var_s) if s != 0 else 0.0
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    tau = s / (0.5 * n * (n - 1))

    return {"tau": tau, "p_value": p, "s": s, "z": z, "n": n,
            "autocorr_correction": autocorr_correction, "variance_inflation": correction}


def trend_map(stack, dim: str = "year", per_decade: bool = True):
    """
    Per-pixel Theil-Sen slope and Mann-Kendall p-value across a stack.

    Returns a Dataset with `slope` and `p_value`. Vectorized via apply_ufunc,
    but still slow on large grids this is the right place to coarsen first if
    the point is a regional summary rather than a pixel-level map.

    On the p-value map: with thousands of pixels tested, roughly 5% will show
    p<0.05 by chance alone. Apply a false-discovery-rate correction before
    claiming any spatial pattern of significance, or report the field
    significance rather than the pixel counts.
    """
    def _both(series):
        ok = np.isfinite(series)
        if ok.sum() < 10:
            return np.array([np.nan, np.nan])
        sl = stats.theilslopes(series[ok], np.arange(len(series))[ok])[0]
        p = mann_kendall(series[ok])["p_value"]
        return np.array([sl * (10.0 if per_decade else 1.0), p])

    result = xr.apply_ufunc(
        _both, stack,
        input_core_dims=[[dim]], output_core_dims=[["stat"]],
        vectorize=True, dask="parallelized", output_dtypes=[float],
        dask_gufunc_kwargs={"output_sizes": {"stat": 2}},
    )
    return xr.Dataset({"slope": result.isel(stat=0), "p_value": result.isel(stat=1)})


def water_year(dates):
    """
    Water year (Oct 1 - Sep 30), the correct accounting unit for anything
    snow-driven. Using calendar years splits a single snowpack across two
    buckets and manufactures variance that is purely an artefact of the
    calendar.
    """
    d = pd.DatetimeIndex(dates)
    return np.where(d.month >= 10, d.year + 1, d.year)


# Module 2 drought, fire weather, snow, low flow
def spi(precip_monthly, scale: int = 12, calibration=None) -> pd.Series:
    """
    Standardized Precipitation Index at `scale` months.

    Method: accumulate precipitation over a rolling `scale`-month window, fit a
    gamma distribution *separately for each calendar month* across years, then
    map to standard normal quantiles. The per-month fit is what removes the
    seasonal cycle; fitting one distribution to all months would make every
    July look like a drought in a winter-precipitation regime.

    Zero-precipitation months are handled with the standard mixed distribution
    — gamma cannot accommodate exact zeros, so the zero probability is carried
    separately. This matters more the shorter the scale and the drier the site.

    `scale` choice is not cosmetic: 3-month SPI tracks soil moisture and
    agricultural drought, 12-month tracks reservoir storage and groundwater.
    Pick the one matching the decision being supported and say which.

    `calibration` optionally restricts the fitting period to a fixed baseline
    (ex. slice('1991','2020')) so that later values are expressed relative to a
    stable reference rather than shifting as the record grows.
    """
    s = pd.Series(precip_monthly).astype("float64")
    rolled = s.rolling(scale, min_periods=scale).sum()
    cal = rolled if calibration is None else rolled.loc[calibration]

    out = pd.Series(np.nan, index=s.index, dtype="float64")
    for month in range(1, 13):
        idx = rolled.index[rolled.index.month == month]
        cal_idx = cal.index[cal.index.month == month]
        vals = rolled.loc[idx]
        cal_vals = cal.loc[cal_idx].dropna()
        if len(cal_vals) < 15:
            continue

        zeros = (cal_vals == 0).sum()
        q = zeros / len(cal_vals)               # probability of zero
        positive = cal_vals[cal_vals > 0]
        if len(positive) < 10:
            continue

        shape, loc, scale_p = stats.gamma.fit(positive, floc=0)
        cdf = q + (1 - q) * stats.gamma.cdf(vals.where(vals > 0), shape, loc=loc, scale=scale_p)
        cdf = np.where(vals == 0, q / 2, cdf)   # midpoint convention for zeros
        cdf = np.clip(cdf, 1e-6, 1 - 1e-6)      # keep ppf finite at the tails
        out.loc[idx] = stats.norm.ppf(cdf)

    out.name = f"spi{scale}"
    return out


def spei(precip_monthly, pet_monthly, scale: int = 12, calibration=None) -> pd.Series:
    """
    Standardized Precipitation-Evapotranspiration Index.

    Same construction as SPI but on the climatic water balance (P - PET), fitted
    with a log-logistic distribution because the balance can be negative and
    gamma cannot represent that.

    SPEI rather than SPI is the right call for warming-climate work, and the
    difference is not subtle: in the interior West, rising evaporative demand
    has driven substantial drying that precipitation-only indices miss
    entirely. If a partner asks why drought looks worse in your numbers than in
    their SPI-based reporting, this is the answer.
    """
    p = pd.Series(precip_monthly).astype("float64")
    e = pd.Series(pet_monthly).astype("float64").reindex(p.index)
    balance = (p - e).rolling(scale, min_periods=scale).sum()
    cal = balance if calibration is None else balance.loc[calibration]

    out = pd.Series(np.nan, index=p.index, dtype="float64")
    for month in range(1, 13):
        idx = balance.index[balance.index.month == month]
        vals = balance.loc[idx]
        cal_vals = cal.loc[cal.index[cal.index.month == month]].dropna()
        if len(cal_vals) < 15:
            continue
        # fisk == log-logistic, the SPEI standard (Vicente-Serrano et al. 2010)
        c, loc, sc = stats.fisk.fit(cal_vals - cal_vals.min() + 1e-6)
        cdf = stats.fisk.cdf(vals - cal_vals.min() + 1e-6, c, loc=loc, scale=sc)
        out.loc[idx] = stats.norm.ppf(np.clip(cdf, 1e-6, 1 - 1e-6))

    out.name = f"spei{scale}"
    return out


def fire_weather_days(erc=None, vpd=None, erc_percentile: float = 90, vpd_threshold: float = 2.0, baseline=None) -> pd.Series:
    """
    Annual count of critical fire-weather days.

    Two definitions, because they answer different questions:

    - ERC above its historical 90th percentile. ERC (Energy Release Component)
      integrates fuel moisture across size classes and is the operational fire
      danger metric agencies actually use, so this version is legible to a fire
      manager.
    - VPD above ~2 kPa, the atmospheric-demand threshold above which fine fuels
      dry rapidly. Cleaner physically and better for trend attribution, since
      it does not depend on a fuel model that itself changed after the fire.

    The percentile is taken against `baseline` if given, otherwise the full
    record. Use a fixed baseline for trend work; a percentile computed on the
    full record moves as the record warms, which mechanically suppresses the
    trend you are trying to measure. That mistake is common and it always
    understates the change.
    """
    counts = {}
    if erc is not None:
        s = pd.Series(erc).astype("float64")
        ref = s if baseline is None else s.loc[baseline]
        threshold = np.nanpercentile(ref, erc_percentile)
        counts[f"erc_p{int(erc_percentile)}_days"] = (s > threshold).groupby(s.index.year).sum()
    if vpd is not None:
        v = pd.Series(vpd).astype("float64")
        counts[f"vpd_gt{vpd_threshold}_days"] = (v > vpd_threshold).groupby(v.index.year).sum()

    if not counts:
        raise ValueError("supply erc, vpd, or both")
    return pd.DataFrame(counts)


def snow_metrics(swe_daily: pd.DataFrame, value_col: str = "value") -> pd.DataFrame:
    """
    Per-water-year snowpack metrics from daily SWE.

    Returns peak SWE, the date it occurred (as day-of-water-year), melt-out
    date, and snow-season length.

    Melt-out timing carries more water-supply information than peak SWE alone.
    Two years with identical peaks but a three-week difference in melt-out
    produce very different summer streamflow and very different fire seasons —
    peak SWE is a stock, melt-out is what governs the flow.

    Melt-out is defined as the last day with SWE > 0.1 in, not the first day at
    zero, since late-season storms produce brief re-accumulations that would
    otherwise reset the date.
    """
    df = swe_daily.copy()
    df["wy"] = water_year(df["date"])

    rows = []
    for (triplet, wy), g in df.groupby(["triplet", "wy"], dropna=True):
        g = g.dropna(subset=[value_col]).sort_values("date")
        if len(g) < 200:   # need most of a water year for these to mean anything
            continue
        wy_start = pd.Timestamp(year=int(wy) - 1, month=10, day=1)
        peak_row = g.loc[g[value_col].idxmax()]
        snow_days = g[g[value_col] > 0.1]
        rows.append({
            "triplet": triplet,
            "water_year": int(wy),
            "peak_swe_in": float(peak_row[value_col]),
            "peak_doy_wy": int((peak_row["date"] - wy_start).days),
            "meltout_doy_wy": int((snow_days["date"].max() - wy_start).days) if len(snow_days) else np.nan,
            "onset_doy_wy": int((snow_days["date"].min() - wy_start).days) if len(snow_days) else np.nan,
            "snow_season_days": int(len(snow_days)),
        })

    return pd.DataFrame(rows)


def low_flow_metrics(flow_daily: pd.DataFrame, value_col: str = "value") -> pd.DataFrame:
    """
    Per-water-year low-flow metrics from daily discharge.

    Annual 7-day minimum (the 7Q basis), the date it occurs, and the count of
    days below the long-term 10th percentile.

    Low flow rather than mean flow because that is where the ecological and
    water-supply pain lives, and because mean annual flow is dominated by peak
    snowmelt and can stay flat while the summer minimum collapses.
    """
    df = flow_daily.copy()
    df["wy"] = water_year(df["date"])

    rows = []
    for (site, wy), g in df.groupby(["site_no", "wy"], dropna=True):
        g = g.dropna(subset=[value_col]).sort_values("date")
        if len(g) < 330:
            continue
        rolling7 = g[value_col].rolling(7, min_periods=7).mean()
        if rolling7.notna().sum() == 0:
            continue
        site_p10 = np.nanpercentile(df.loc[df["site_no"] == site, value_col], 10)
        rows.append({
            "site_no": site,
            "water_year": int(wy),
            "min_7day_cfs": float(rolling7.min()),
            "min_7day_date": g.loc[rolling7.idxmin(), "date"],
            "days_below_p10": int((g[value_col] < site_p10).sum()),
            "annual_mean_cfs": float(g[value_col].mean()),
        })

    return pd.DataFrame(rows)


# Module 3 composite index
def normalize_indicator(values, method: str = "minmax", invert: bool = False, reference=None):
    """
    Put an indicator on a 0-1 scale so it can be combined with others.

    Method matters more than people expect, and the choice should be stated
    wherever the index is:

    - 'minmax'  bounded 0-1, but every value depends on the extremes of the
                sample, so adding a fourth region can change all three existing
                scores. Fine for a fixed set, dangerous for a growing one.
    - 'zscore'  unbounded and interpretable as standard deviations, but assumes
                roughly symmetric distributions.
    - 'rank'    fully robust to outliers and to distribution shape, discards
                magnitude information. Often the honest choice when the
                underlying indicators are not commensurable anyway.

    `invert=True` for indicators where higher is *better* (ex. snow season
    length), so that after normalization higher always means more stress.
    """
    v = np.asarray(values, dtype="float64")
    ref = v if reference is None else np.asarray(reference, dtype="float64")

    if method == "minmax":
        lo, hi = np.nanmin(ref), np.nanmax(ref)
        out = np.full_like(v, 0.5) if hi - lo < 1e-12 else (v - lo) / (hi - lo)
    elif method == "zscore":
        mu, sd = np.nanmean(ref), np.nanstd(ref)
        out = np.zeros_like(v) if sd < 1e-12 else (v - mu) / sd
    elif method == "rank":
        finite = np.isfinite(v)
        out = np.full_like(v, np.nan)
        out[finite] = stats.rankdata(v[finite]) / finite.sum()
    else:
        raise ValueError("method must be minmax, zscore, or rank")

    return 1 - out if invert else out


def composite_index(indicator_df: pd.DataFrame, weights: dict | None = None, method: str = "minmax", invert: dict | None = None) -> pd.DataFrame:
    """
    Combine normalized indicators into one score per region.

    `indicator_df` is regions (rows) x indicators (columns), raw values.
    `weights` maps column name -> weight; equal weights if omitted.
    `invert` maps column name -> bool for higher-is-better indicators.

    Returns the normalized components alongside the composite, never the
    composite alone. A single number with no visible components is not
    reviewable and invites exactly the objection that the weights were chosen
    to produce it.
    """
    invert = invert or {}
    weights = weights or {c: 1.0 for c in indicator_df.columns}
    missing = set(indicator_df.columns) - set(weights)
    if missing:
        raise ValueError(f"no weight supplied for: {sorted(missing)}")

    norm = pd.DataFrame(
        {c: normalize_indicator(indicator_df[c].values, method=method, invert=invert.get(c, False)) for c in indicator_df.columns},
        index=indicator_df.index,
    )

    w = np.array([weights[c] for c in indicator_df.columns], dtype="float64")
    w = w / w.sum()
    out = norm.copy()
    out["composite"] = (norm.values * w).sum(axis=1)
    out["rank"] = out["composite"].rank(ascending=False).astype(int)
    return out


def weight_sensitivity(indicator_df: pd.DataFrame, n_draws: int = 5000, method: str = "minmax", invert: dict | None = None, concentration: float = 1.0, seed: int = 0) -> pd.DataFrame:
    """
    Monte Carlo over the weight simplex: how much does the ranking depend on
    the weights we happened to choose?

    Weights are drawn from a Dirichlet distribution, `concentration=1.0` is
    uniform over all valid weight vectors, so this asks "across every defensible
    weighting, how often does each region land in each rank?"

    This is the analysis that determines whether the index says anything. If a
    region is top-ranked under 95% of weightings, the conclusion is robust and
    the specific weights barely matter. If it is top-ranked under 40%, the index
    is reporting the analyst's priorities rather than the landscape's condition,
    and it should be presented as a set of separate indicators instead of a
    composite. Run this before publishing any index, and publish the result
    alongside it.
    """
    rng = np.random.default_rng(seed)
    invert = invert or {}
    cols = list(indicator_df.columns)

    norm = np.column_stack([
        normalize_indicator(indicator_df[c].values, method=method, invert=invert.get(c, False)) for c in cols
    ])

    n_regions = len(indicator_df)
    rank_counts = np.zeros((n_regions, n_regions), dtype="int64")
    scores = np.zeros((n_draws, n_regions))

    for i, w in enumerate(rng.dirichlet(np.full(len(cols), concentration), size=n_draws)):
        s = norm @ w
        scores[i] = s
        order = stats.rankdata(-s, method="ordinal").astype(int) - 1
        rank_counts[np.arange(n_regions), order] += 1

    out = pd.DataFrame(rank_counts / n_draws, index=indicator_df.index, columns=[f"P(rank {i+1})" for i in range(n_regions)])
    out["mean_score"] = scores.mean(axis=0)
    out["score_p05"] = np.percentile(scores, 5, axis=0)
    out["score_p95"] = np.percentile(scores, 95, axis=0)
    return out.round(3)


def rank_stability(sensitivity_df: pd.DataFrame) -> float:
    """
    One number for how trustworthy the ranking is: the mean probability that a
    region occupies its own modal rank.

    Above ~0.8, the ranking is a finding. Below ~0.5, it is an artefact of the
    weights and should not be presented as a ranking at all.
    """
    prob_cols = [c for c in sensitivity_df.columns if c.startswith("P(rank")]
    return float(sensitivity_df[prob_cols].max(axis=1).mean())


# Module 4 projections
def delta_change(historical, future, relative: bool = False) -> dict:
    """
    Change between a historical and a future period, with across-model spread.

    Both inputs carry a `model` dimension. The spread across models is reported
    because it is usually larger than the difference between emissions
    scenarios, and a projection presented without it is not a projection.

    `relative=True` returns percent change appropriate for precipitation,
    inappropriate for temperature.
    """
    h = historical.mean(dim=[d for d in historical.dims if d != "model"])
    f = future.mean(dim=[d for d in future.dims if d != "model"])
    delta = (f - h) / h * 100 if relative else f - h

    vals = np.asarray(delta.values, dtype="float64")
    return {
        "ensemble_mean": float(np.nanmean(vals)),
        "ensemble_median": float(np.nanmedian(vals)),
        "model_min": float(np.nanmin(vals)),
        "model_max": float(np.nanmax(vals)),
        "model_std": float(np.nanstd(vals)),
        "n_models": int(np.isfinite(vals).sum()),
        "models_agree_on_sign": bool(np.all(vals > 0) or np.all(vals < 0)),
        "units": "%" if relative else "absolute",
    }


def emergence_year(annual_series, baseline_slice, n_sigma: float = 2.0, persist: int = 5):
    """
    First year a signal emerges permanently from baseline variability.

    Defined as the first year after which the series stays more than `n_sigma`
    baseline standard deviations from the baseline mean for at least `persist`
    consecutive years. The persistence requirement is what stops a single
    extreme year from being reported as emergence.

    This framing is far more useful to a planner than a mid-century average.
    "Conditions leave the historical envelope around 2038 and do not return" is
    a planning horizon; "+2.4C by 2050" is a number.
    """
    s = pd.Series(annual_series).astype("float64").dropna()
    base = s.loc[baseline_slice]
    mu, sd = base.mean(), base.std()
    if sd == 0 or len(base) < 10:
        return None

    beyond = (s - mu).abs() > n_sigma * sd
    run = 0
    for year, flag in beyond.items():
        run = run + 1 if flag else 0
        if run >= persist:
            return int(year) - persist + 1
    return None
