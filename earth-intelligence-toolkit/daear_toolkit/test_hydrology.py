"""Sanity checks for daear_toolkit.hydrology on synthetic terrain."""
import numpy as np
import hydrology as h

rng = np.random.default_rng(0)

# synthetic V-shaped valley draining south
n = 60
rows, cols = np.mgrid[0:n, 0:n]
dem = 1000.0 - rows * 2.0 + np.abs(cols - n // 2) * 1.5

# Punch a pit that filling must remove.
dem[30, 30] -= 25.0

filled = h._fill_depressions_core(dem)
assert filled[30, 30] > dem[30, 30], "pit was not filled"
assert np.all(filled >= dem - 1e-9), "filling lowered cells"

direction = h._flow_direction_core(filled)
acc = h._flow_accumulation_core(filled, direction, None)

print(f"accumulation: min={acc.min():.0f} max={acc.max():.0f} total={acc[direction < 0].sum():.0f} cells={dem.size}")
# Mass balance: everything that leaves the grid at outlets must equal cell count.
assert np.isclose(acc[direction < 0].sum(), dem.size), "flow accumulation loses mass"
# The valley axis should collect far more than a random hillslope cell.
assert acc[n - 2, n // 2] > acc[n - 2, 5] * 5, "valley axis is not concentrating flow"

streams = acc >= 100
dist = h._downstream_traverse_core(filled, direction, np.full(dem.shape, 30.0), streams)
assert np.nanmax(dist) > 0 and np.all(dist[streams] == 0), "distance-to-stream is wrong at channels"
print(f"distance to stream: max={np.nanmax(dist):.0f} m, mean={np.nanmean(dist):.0f} m")

# connectivity: bare steep ground should out-score vegetated flat ground
slope = np.rad2deg(np.arctan(np.full(dem.shape, 0.2)))
impedance_bare = np.full(dem.shape, 0.9)
impedance_veg = np.full(dem.shape, 0.1)
ic_bare = h.sediment_connectivity_index(dem, slope, impedance_bare, stream_threshold=100)
ic_veg = h.sediment_connectivity_index(dem, slope, impedance_veg, stream_threshold=100)
print(f"IC bare mean={np.nanmean(ic_bare):.2f}  IC vegetated mean={np.nanmean(ic_veg):.2f}")
assert np.nanmean(ic_bare) > np.nanmean(ic_veg), "bare ground should be more connected than vegetated"

# NaN handling
dem_nan = dem.copy()
dem_nan[:5, :5] = np.nan
f2 = h._fill_depressions_core(dem_nan)
d2 = h._flow_direction_core(f2)
a2 = h._flow_accumulation_core(f2, d2, None)
assert np.isnan(a2[:5, :5]).all(), "NaN region leaked into accumulation"
assert np.isclose(np.nansum(a2[d2 < 0]), np.isfinite(dem_nan).sum()), "mass balance broken with NaNs"

# subwatersheds
labs = h.subwatershed_labels(dem, stream_threshold=100, min_cells=50)
uniq = np.unique(labs[np.isfinite(labs)])
print(f"subwatersheds delineated: {len(uniq)}")
assert len(uniq) >= 2, "expected multiple sub-basins"

print("\nAll hydrology checks passed.")
