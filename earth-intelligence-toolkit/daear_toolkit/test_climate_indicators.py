"""Sanity checks for daear_toolkit.climate_indicators on synthetic data."""
import numpy as np
import pandas as pd
import climate_indicators as ci

rng = np.random.default_rng(42)

# Theil-Sen: recover a known slope, and survive a contaminated series 
years = np.arange(1991, 2025)
true_slope = 0.04  # degC/yr = 0.4/decade
clean = 10 + true_slope * (years - 1991) + rng.normal(0, 0.3, len(years))
res = ci.theil_sen_trend(clean, years)
print(f"Theil-Sen clean:  {res['slope']:+.3f}/decade (true {true_slope*10:+.2f}), p={res['p_value']:.4f}")
assert abs(res["slope"] - true_slope * 10) < 0.15
assert res["significant"]

contaminated = clean.copy()
contaminated[5] += 12.0   # one absurd year
contaminated[20] -= 9.0
res_c = ci.theil_sen_trend(contaminated, years)
ols = np.polyfit(years, contaminated, 1)[0] * 10
print(f"Theil-Sen dirty:  {res_c['slope']:+.3f}/decade   vs OLS {ols:+.3f}/decade")
assert abs(res_c["slope"] - true_slope * 10) < abs(ols - true_slope * 10), "robustness advantage not demonstrated"

# Mann-Kendall: no trend on white noise, trend when present 
noise = rng.normal(0, 1, 60)
mk_null = ci.mann_kendall(noise)
mk_trend = ci.mann_kendall(np.arange(60) * 0.1 + rng.normal(0, 1, 60))
print(f"MK white noise:   tau={mk_null['tau']:+.3f} p={mk_null['p_value']:.3f}")
print(f"MK with trend:    tau={mk_trend['tau']:+.3f} p={mk_trend['p_value']:.5f}")
assert mk_null["p_value"] > 0.05 and mk_trend["p_value"] < 0.01

# Hamed-Rao correction must reduce false positives on autocorrelated noise
false_pos_raw = false_pos_hr = 0
N = 400
for _ in range(N):
    ar = np.zeros(50)
    for t in range(1, 50):
        ar[t] = 0.7 * ar[t - 1] + rng.normal()
    false_pos_raw += ci.mann_kendall(ar)["p_value"] < 0.05
    false_pos_hr += ci.mann_kendall(ar, autocorr_correction=True)["p_value"] < 0.05
print(f"AR(0.7) false positives: uncorrected {false_pos_raw/N:.1%}, Hamed-Rao {false_pos_hr/N:.1%} (nominal 5%)")
assert false_pos_hr < false_pos_raw * 0.6, "Hamed-Rao did not meaningfully reduce false positives"

# and it must not destroy power to detect a real trend
detected = 0
for _ in range(200):
    ar = np.zeros(50)
    for t in range(1, 50):
        ar[t] = 0.7 * ar[t - 1] + rng.normal()
    detected += ci.mann_kendall(ar + 0.15 * np.arange(50), autocorr_correction=True)["p_value"] < 0.05
print(f"Power to detect a real trend under AR(0.7), corrected: {detected/200:.1%}")
assert detected / 200 > 0.5, "correction destroyed statistical power"

# SPI: standard normal by construction over the calibration period
idx = pd.date_range("1981-01-01", "2024-12-01", freq="MS")
seasonal = 40 + 25 * np.cos(2 * np.pi * (idx.month - 2) / 12)
precip = pd.Series(np.maximum(rng.gamma(2.0, seasonal / 2.0), 0), index=idx)
s = ci.spi(precip, scale=12).dropna()
print(f"SPI-12: mean={s.mean():+.3f} sd={s.std():.3f} (expect ~0, ~1)")
assert abs(s.mean()) < 0.25 and 0.75 < s.std() < 1.3
# A dry period should register as negative SPI
dry = precip.copy()
dry.loc["2015":"2017"] *= 0.35
s_dry = ci.spi(dry, scale=12)
print(f"SPI-12 during imposed drought: {s_dry.loc['2016'].mean():+.2f}")
assert s_dry.loc["2016"].mean() < -0.8, "imposed drought not detected"

# SPEI: responds to rising PET even with flat precipitation
pet_flat = pd.Series(np.full(len(idx), 60.0) + 30 * np.cos(2 * np.pi * (idx.month - 7) / 12), index=idx)
pet_rising = pet_flat + np.linspace(0, 25, len(idx))
sp_flat = ci.spei(precip, pet_flat, scale=12).dropna()
sp_rise = ci.spei(precip, pet_rising, scale=12).dropna()
print(f"SPEI last decade: flat PET {sp_flat.loc['2015':].mean():+.2f}, rising PET {sp_rise.loc['2015':].mean():+.2f}")
assert sp_rise.loc["2015":].mean() < sp_flat.loc["2015":].mean(), "SPEI insensitive to evaporative demand"

# fire weather days
daily = pd.date_range("1991-01-01", "2024-12-31", freq="D")
vpd = pd.Series(np.maximum(rng.gamma(2, 0.6, len(daily)) + np.linspace(0, 0.6, len(daily)), 0), index=daily)
fw = ci.fire_weather_days(vpd=vpd, vpd_threshold=2.0)
tr = ci.theil_sen_trend(fw.iloc[:, 0].values, fw.index.values)
print(f"Fire-weather days trend: {tr['slope']:+.1f} days/decade, p={tr['p_value']:.4f}")
assert tr["slope"] > 0 and tr["significant"]

# baseline-vs-full-record percentile: full record must understate the trend
erc = pd.Series(50 + np.linspace(0, 20, len(daily)) + rng.normal(0, 10, len(daily)), index=daily)
full = ci.fire_weather_days(erc=erc, erc_percentile=90)
fixed = ci.fire_weather_days(erc=erc, erc_percentile=90, baseline=slice("1991", "2010"))
t_full = ci.theil_sen_trend(full.iloc[:, 0].values)["slope"]
t_fixed = ci.theil_sen_trend(fixed.iloc[:, 0].values)["slope"]
print(f"ERC>p90 trend: full-record baseline {t_full:+.1f}/dec, fixed baseline {t_fixed:+.1f}/dec")
assert t_fixed > t_full, "fixed baseline should show the larger (correct) trend"

# composite index and weight sensitivity
ind = pd.DataFrame({
    "warming":    [0.42, 0.28, 0.38],
    "drying":     [0.55, 0.31, 0.49],
    "fire_days":  [14.0, 6.0, 11.0],
    "snow_loss":  [0.30, 0.12, 0.26],
}, index=["poudre", "black_hills", "front_range"])

comp = ci.composite_index(ind)
print("\nComposite:\n", comp[["composite", "rank"]].round(3))
assert comp.loc["poudre", "rank"] == 1

sens = ci.weight_sensitivity(ind, n_draws=3000)
stab = ci.rank_stability(sens)
print("\nWeight sensitivity:\n", sens)
print(f"Rank stability: {stab:.2f}")
assert 0 <= stab <= 1
# Poudre dominates on every indicator, so its top rank must be near-certain.
assert sens.loc["poudre", "P(rank 1)"] > 0.95, "dominated ranking should be stable"

# A genuinely ambiguous case must come out unstable.
amb = pd.DataFrame({"a": [0.9, 0.1], "b": [0.1, 0.9]}, index=["x", "y"])
sens_amb = ci.weight_sensitivity(amb, n_draws=3000)
print(f"Ambiguous-case stability: {ci.rank_stability(sens_amb):.2f} (should be near 0.5)")
assert ci.rank_stability(sens_amb) < 0.75, "ambiguous ranking wrongly reported as stable"

# normalization: invert flag orients higher = worse
v = np.array([1.0, 2.0, 3.0])
assert np.allclose(ci.normalize_indicator(v, invert=True), [1.0, 0.5, 0.0])

# emergence year
yrs = pd.Series(index=range(1991, 2091), dtype=float)
yrs[:] = 10 + np.concatenate([rng.normal(0, 0.4, 40), np.linspace(0, 5, 60) + rng.normal(0, 0.4, 60)])
em = ci.emergence_year(yrs, baseline_slice=slice(1991, 2020))
print(f"Emergence year: {em}")
assert em is not None and 2020 < em < 2075

print("\nAll climate indicator checks passed.")
