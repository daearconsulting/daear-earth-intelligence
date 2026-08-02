"""Sanity checks for daear_toolkit.fire_indicators on synthetic data."""
import numpy as np
import pandas as pd
import xarray as xr
import fire_indicators as fi

rng = np.random.default_rng(7)

# helper to build a small DataArray on a lat/lon grid
def grid(values, name="x"):
    ny, nx = values.shape
    return xr.DataArray(
        values,
        coords={"lat": np.linspace(40.9, 40.4, ny), "lon": np.linspace(-106.0, -105.0, nx)},
        dims=("lat", "lon"), name=name,
    )

# NBR/dNBR/RdNBR
n = 40
pre = xr.Dataset({"B08": grid(np.full((n, n), 0.35)), "B12": grid(np.full((n, n), 0.10)),
                  "B11": grid(np.full((n, n), 0.15))})
post_vals = np.full((n, n), 0.35)
post_swir = np.full((n, n), 0.10)
post_swir[:20, :] = 0.30          # top half burned: SWIR2 rises sharply
post_vals[:20, :] = 0.20          # and NIR falls
post = xr.Dataset({"B08": grid(post_vals), "B12": grid(post_swir), "B11": grid(np.full((n, n), 0.25))})

pre_nbr = fi.nbr(pre)
post_nbr = fi.nbr(post)
d = fi.dnbr(pre_nbr, post_nbr)
print(f"NBR pre={float(pre_nbr.mean()):.3f}  post burned={float(post_nbr[:20].mean()):.3f}  unburned={float(post_nbr[20:].mean()):.3f}")
print(f"dNBR burned half={float(d[:20].mean()):.0f}  unburned half={float(d[20:].mean()):.0f}")
assert float(d[:20].mean()) > 500, "burned half should show high dNBR"
assert abs(float(d[20:].mean())) < 1, "unburned half should show ~zero dNBR"

# offset correction must remove a uniform phenological shift
d_shifted = d + 80
offset = fi.compute_dnbr_offset(d_shifted, grid(np.arange(n * n).reshape(n, n)) >= 20 * n)
corrected = d_shifted - offset
print(f"Phenological offset detected: {offset:.0f} (imposed 80)")
assert abs(offset - 80) < 5 and abs(float(corrected[20:].mean())) < 5

# RdNBR must reduce the dependence on pre-fire biomass.
# The realistic framing: the SAME fire effect produces a LARGER absolute NBR
# drop where pre-fire vegetation is denser. So impose a proportional effect
# (70% reduction in NBR) on dense and sparse ground and check that RdNBR brings
# the two closer together than dNBR does.
dense_pre = grid(np.full((n, n), 0.60))
sparse_pre = grid(np.full((n, n), 0.20))
FRACTION_REMAINING = 0.30
d_dense = fi.dnbr(dense_pre, dense_pre * FRACTION_REMAINING)
d_sparse = fi.dnbr(sparse_pre, sparse_pre * FRACTION_REMAINING)
r_dense = fi.rdnbr(dense_pre, dense_pre * FRACTION_REMAINING)
r_sparse = fi.rdnbr(sparse_pre, sparse_pre * FRACTION_REMAINING)

dnbr_ratio = float(d_dense.mean()) / float(d_sparse.mean())
rdnbr_ratio = float(r_dense.mean()) / float(r_sparse.mean())
print(f"\nSame proportional fire effect, dense vs sparse pre-fire vegetation:")
print(f"  dNBR:  {float(d_dense.mean()):.0f} vs {float(d_sparse.mean()):.0f}  (ratio {dnbr_ratio:.2f})")
print(f"  RdNBR: {float(r_dense.mean()):.0f} vs {float(r_sparse.mean()):.0f}  (ratio {rdnbr_ratio:.2f})")
assert rdnbr_ratio < dnbr_ratio, "RdNBR should be less sensitive to pre-fire biomass than dNBR"
# The sqrt normalizer is a compromise, not a complete fix, dividing by NBR
# itself would give ratio 1.0 exactly. Miller & Thode chose sqrt because full
# normalization overcorrects where pre-fire NBR is small. Some residual
# dependence is expected and correct.
assert rdnbr_ratio > 1.0, "sqrt normalization should not fully remove the dependence"

# severity classification
test_vals = grid(np.array([[50, 200, 400, 900]] * 4, dtype="float64"))
cls = fi.classify_severity(test_vals, metric="dnbr")
assert list(np.asarray(cls)[0]) == [0, 1, 2, 3], f"unexpected classes: {np.asarray(cls)[0]}"
summary = fi.severity_summary(cls)
print("\nSeverity summary:\n", summary[["label", "pixels", "pct"]])
assert abs(summary["pct"].sum() - 100) < 0.5

# fuel hazard
fuels = grid(np.array([[91, 101, 104, 108, 202]] * 4, dtype="float64"))
fh = fi.fuel_hazard(fuels)
assert list(np.asarray(fh)[0]) == [0.0, 0.25, 0.5, 0.75, 1.0]
print(f"\nFuel hazard mapping OK: {list(np.asarray(fh)[0])}")

# crown fire potential
cc = grid(np.array([[80.0, 80.0, 20.0, 80.0]] * 4))
cbh = grid(np.array([[2.0, 9.0, 2.0, 2.0]] * 4))
cbd = grid(np.array([[0.15, 0.15, 0.15, 0.01]] * 4))
cfp = np.asarray(fi.crown_fire_potential(cc, cbh, cbd))[0]
print(f"Crown fire potential [ideal, high CBH, sparse, low CBD]: {np.round(cfp, 3)}")
assert cfp[0] > 0.9, "ideal crown conditions should score high"
assert cfp[1] < 0.1 and cfp[3] < 0.5, "any limiting factor should collapse the score"

# recovery curve fitting 
years = np.arange(2021, 2031)
true_asym, true_rate, y0 = 0.72, 0.35, 0.18
clean = fi._recovery_model(years - 2021, true_asym, true_rate, y0) + rng.normal(0, 0.005, len(years))
fit = fi.fit_recovery_curve(years, clean)
print(f"\nRecovery fit: asymptote={fit['asymptote']:.3f} (true {true_asym}), "
      f"rate={fit['rate']:.3f} (true {true_rate}), half-life={fit['half_life_years']:.2f}y, R2={fit['r_squared']:.3f}")
assert fit["converged"] and abs(fit["asymptote"] - true_asym) < 0.05 and abs(fit["rate"] - true_rate) < 0.1
assert abs(fit["half_life_years"] - np.log(2) / true_rate) < 0.4

# a fast-to-low-ceiling trajectory (grass) must be distinguishable from slow-to-high (forest)
grass = fi.fit_recovery_curve(years, fi._recovery_model(years - 2021, 0.45, 1.2, 0.15))
forest = fi.fit_recovery_curve(years, fi._recovery_model(years - 2021, 0.85, 0.15, 0.15))
print(f"Grass-like:  asymptote={grass['asymptote']:.2f} half-life={grass['half_life_years']:.2f}y")
print(f"Forest-like: asymptote={forest['asymptote']:.2f} half-life={forest['half_life_years']:.2f}y")
assert grass["half_life_years"] < forest["half_life_years"] and grass["asymptote"] < forest["asymptote"]

# too few points must fail loudly rather than return a confident wrong answer
short = fi.fit_recovery_curve([2021, 2022], [0.2, 0.3])
assert not short["converged"] and np.isnan(short["asymptote"])

# severity vs predictor
pred = rng.uniform(0, 1, (200, 200))
sev = np.clip(np.round(pred * 3 + rng.normal(0, 0.4, pred.shape)), 0, 3)
tab = fi.severity_vs_predictor(grid(sev), grid(pred), n_bins=8)
print(f"\nSeverity-vs-predictor Spearman rho={tab.attrs['spearman_rho']}, p={tab.attrs['spearman_p']}")
assert tab.attrs["spearman_rho"] > 0.8, "should recover the imposed relationship"

# a null predictor must NOT produce a strong relationship
null_tab = fi.severity_vs_predictor(grid(sev), grid(rng.uniform(0, 1, pred.shape)), n_bins=8)
print(f"Null predictor Spearman rho={null_tab.attrs['spearman_rho']}, p={null_tab.attrs['spearman_p']}")
assert abs(null_tab.attrs["spearman_rho"]) < 0.7, "null predictor produced a spurious relationship"

# distance-weighted exposure
try:
    import geopandas as gpd
    from shapely.geometry import Point

    hazard = grid(np.zeros((60, 60)))
    hz = hazard.values.copy()
    hz[:30, :] = 1.0          # northern half is high hazard
    hazard = grid(hz)

    assets = gpd.GeoDataFrame(
        {"name": ["north", "south"]},
        geometry=[Point(-105.5, 40.85), Point(-105.5, 40.45)],
        crs="EPSG:4326",
    )
    exp = fi.distance_weighted_exposure(assets, hazard, decay_km=3.0)
    print("\nDistance-weighted exposure:\n", exp[["name", "weighted_hazard"]])
    north = exp.loc[exp["name"] == "north", "weighted_hazard"].squeeze()
    south = exp.loc[exp["name"] == "south", "weighted_hazard"].squeeze()
    assert north > south, "asset inside the hazard zone should score higher"
except ImportError:
    print("\n(geopandas unavailable; skipped exposure test)")

print("\nAll fire indicator checks passed.")
