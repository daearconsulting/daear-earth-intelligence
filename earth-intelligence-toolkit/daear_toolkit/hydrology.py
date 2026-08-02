from __future__ import annotations

"""
daear_toolkit.hydrology
Terrain-based flow routing for the soil-watershed-intelligence modules.

Why this module exists
`indicators.py` has slope from `np.gradient`, which is enough for erosion
*susceptibility* (a per-cell property) but not for *connectivity* (a
path-dependent property). Answering "where does this cell's runoff actually
go, and does it reach the mainstem?" requires routing water downhill across
the whole grid.

Everything here is pure NumPy and SciPy, deliberately: `pysheds` and `richdem`
are both excellent but pull in GDAL/Cython build chains that we do not want
as a hard dependency of the demo repos. The algorithms below are the standard
ones (priority-flood depression filling, D8 single-flow-direction routing,
topological-order accumulation), just written against arrays we already have.

Public API
    fill_depressions(dem)            -> DataArray
    flow_direction(dem)              -> DataArray (D8 pointer, 0-7, -1 = sink)
    flow_accumulation(dem)           -> DataArray (contributing cell count)
    extract_streams(acc, threshold)  -> DataArray (bool)
    distance_to_stream(dem, streams) -> DataArray (flow-path distance, m)
    sediment_connectivity_index(...)  -> DataArray (Borselli-style IC)

All functions accept and return xarray.DataArray objects so they compose with
`viz.plot_raster` and the rest of the toolkit. The `_core` functions underneath
take plain 2-D NumPy arrays and are what the unit tests exercise.
"""

import heapq

import numpy as np

try:  # xarray is a toolkit-level dependency, but keep the core importable without it
    import xarray as xr
except ImportError:  # pragma: no cover
    xr = None


# D8 neighbour geometry
# Neighbour order is the ESRI-style clockwise-from-east convention, stored as
# indices 0-7 rather than the 1/2/4/8/... bitmask (easier to index with).
#
#   3  2  1
#   4  .  0
#   5  6  7
#
_D8_OFFSETS = np.array(
    [(0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1)],
    dtype=np.int32,
)
# Diagonal neighbours are sqrt(2) cell-widths away; this matters for both
# slope-based direction choice and for flow-path distance.
_D8_DISTANCE = np.array([1.0, np.sqrt(2), 1.0, np.sqrt(2), 1.0, np.sqrt(2), 1.0, np.sqrt(2)])


# xarray <-> numpy helpers
def _as_array(obj) -> np.ndarray:
    """Return a float64 2-D NumPy view of a DataArray or array-like."""
    values = getattr(obj, "values", obj)
    arr = np.asarray(values, dtype="float64")
    if arr.ndim != 2:
        raise ValueError(f"expected a 2-D raster, got shape {arr.shape}")
    return arr


def _like(template, data: np.ndarray, name: str):
    """
    Wrap `data` in a DataArray carrying `template`'s coords/dims/attrs.

    Falls back to returning the bare array if the input was not a DataArray,
    so the core functions stay usable in tests and scripts.
    """
    if xr is None or not isinstance(template, xr.DataArray):
        return data
    out = xr.DataArray(data, coords=template.coords, dims=template.dims, attrs=dict(template.attrs))
    out.name = name
    return out


def _cell_size_m(dem, fallback: float = 30.0) -> float:
    """
    Best-effort horizontal cell size in metres.

    Order of preference: an explicit `cell_size_m` attr set by `data_access`,
    then the coordinate spacing (converted from degrees if the coords look
    geographic), then the fallback (30 m, matching 3DEP 1-arcsecond).
    """
    if hasattr(dem, "attrs") and "cell_size_m" in dem.attrs:
        return float(dem.attrs["cell_size_m"])

    if xr is not None and isinstance(dem, xr.DataArray) and len(dem.dims) == 2:
        ydim, xdim = dem.dims
        if xdim in dem.coords and dem.sizes[xdim] > 1:
            step = float(np.abs(np.diff(dem.coords[xdim].values)).mean())
            # Heuristic: sub-degree steps mean the raster is in lat/lon, so
            # convert using ~111.32 km per degree of longitude at the equator,
            # scaled by cos(latitude) at the raster's centre.
            if step < 0.01 and ydim in dem.coords:
                lat = float(np.mean(dem.coords[ydim].values))
                return step * 111_320.0 * float(np.cos(np.deg2rad(lat)))
            if step > 1.0:  # already projected metres
                return step
    return fallback


# Depression filling (priority-flood)
def _fill_depressions_core(dem: np.ndarray) -> np.ndarray:
    """
    Priority-flood depression filling (Barnes, Lehman & Mulla 2014).

    Real DEMs contain pits: single cells or basins with no downhill neighbour,
    caused by sensor noise, culverts, and genuine closed depressions. Flow
    routing on an unfilled DEM produces broken, stubby drainage networks
    because water gets trapped in every one of those pits.

    The algorithm: seed a min-heap with every edge cell, then repeatedly pop
    the lowest unprocessed cell and "flood" its neighbours up to at least its
    own elevation. Because we always expand from the lowest available cell,
    the elevation each interior cell gets raised to is exactly the lowest
    spill point on any path to the raster edge. O(n log n).
    """
    filled = dem.copy()
    nrows, ncols = filled.shape

    # NaN cells (outside the AOI, or DEM voids) are treated as off-map: they
    # neither route water nor get filled.
    nan_mask = np.isnan(filled)

    closed = np.zeros((nrows, ncols), dtype=bool)
    heap: list[tuple[float, int, int]] = []

    # Seed: every valid cell on the raster border, plus every valid cell that
    # touches a NaN (those are effectively border cells for routing purposes).
    border = np.zeros((nrows, ncols), dtype=bool)
    border[0, :] = border[-1, :] = True
    border[:, 0] = border[:, -1] = True
    if nan_mask.any():
        padded = np.pad(nan_mask, 1, constant_values=True)
        touches_nan = np.zeros_like(nan_mask)
        for dr, dc in _D8_OFFSETS:
            touches_nan |= padded[1 + dr : 1 + dr + nrows, 1 + dc : 1 + dc + ncols]
        border |= touches_nan

    for r, c in zip(*np.where(border & ~nan_mask)):
        heapq.heappush(heap, (float(filled[r, c]), int(r), int(c)))
        closed[r, c] = True
    closed |= nan_mask

    while heap:
        elev, r, c = heapq.heappop(heap)
        for dr, dc in _D8_OFFSETS:
            nr, nc = r + int(dr), c + int(dc)
            if not (0 <= nr < nrows and 0 <= nc < ncols) or closed[nr, nc]:
                continue
            # Raise the neighbour to the spill elevation if it sits below it.
            # This is what removes the pit.
            filled[nr, nc] = max(filled[nr, nc], elev)
            closed[nr, nc] = True
            heapq.heappush(heap, (float(filled[nr, nc]), nr, nc))

    return filled


def fill_depressions(dem):
    """Fill sinks in a DEM so that every valid cell drains to the raster edge."""
    filled = _fill_depressions_core(_as_array(dem))
    return _like(dem, filled, "filled_elevation")


# D8 flow direction
def _flow_direction_core(filled: np.ndarray) -> np.ndarray:
    """
    D8 single-flow-direction: each cell sends all of its water to whichever
    neighbour gives the steepest downhill drop per unit distance.

    Returns an int8 pointer array of neighbour indices 0-7, with -1 marking
    cells that have no downhill neighbour (raster outlets and NaN cells).

    D8 is the simplest defensible choice. It exaggerates flow convergence on
    planar hillslopes compared with D-infinity or MFD, which is a real
    limitation for hillslope-scale sediment work; it is fine for the
    "does this burn scar drain to the mainstem" question this module answers.
    """
    nrows, ncols = filled.shape
    padded = np.pad(filled, 1, constant_values=np.nan)
    centre = padded[1:-1, 1:-1]

    best_drop = np.zeros((nrows, ncols), dtype="float64")
    direction = np.full((nrows, ncols), -1, dtype="int8")

    for k, ((dr, dc), dist) in enumerate(zip(_D8_OFFSETS, _D8_DISTANCE)):
        neighbour = padded[1 + dr : 1 + dr + nrows, 1 + dc : 1 + dc + ncols]
        # Drop per unit distance, so diagonals are not unfairly favoured.
        drop = (centre - neighbour) / dist
        better = np.greater(drop, best_drop, where=~np.isnan(drop), out=np.zeros_like(drop, dtype=bool))
        best_drop = np.where(better, drop, best_drop)
        direction = np.where(better, np.int8(k), direction)

    direction[np.isnan(filled)] = -1
    return direction


def flow_direction(dem, prefilled: bool = False):
    """
    D8 flow direction pointer grid.

    Parameters
    dem : DataArray
        Elevation raster, ex. `terrain["elevation"]` from `data_access.get_terrain`.
    prefilled : bool
        Set True if you already ran `fill_depressions` and want to skip it.
    """
    arr = _as_array(dem)
    filled = arr if prefilled else _fill_depressions_core(arr)
    return _like(dem, _flow_direction_core(filled).astype("float64"), "flow_direction")


# Flow accumulation
def _flow_accumulation_core(filled: np.ndarray, direction: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """
    Accumulate weights downstream in topological order.

    The trick that makes this fast without a graph library: on a filled DEM,
    water only ever moves to a strictly lower cell, so processing cells in
    order of *descending elevation* guarantees every upstream contributor has
    already been handled by the time we reach a given cell. One pass, O(n log n)
    for the sort.

    `weights` defaults to 1.0 per cell, giving accumulation in units of
    contributing cells. Pass an array to accumulate something else effective
    precipitation, sediment yield, burn severity — which is how Module 2 gets a
    "burned area upstream of each cell" layer for free.
    """
    nrows, ncols = filled.shape
    acc = np.ones((nrows, ncols), dtype="float64") if weights is None else np.asarray(weights, dtype="float64").copy()
    acc[np.isnan(filled)] = np.nan

    valid = ~np.isnan(filled)
    flat_idx = np.flatnonzero(valid)
    # Descending elevation = upstream-to-downstream processing order.
    order = flat_idx[np.argsort(-filled.ravel()[flat_idx], kind="stable")]

    dir_flat = direction.ravel()
    acc_flat = acc.ravel()

    for idx in order:
        k = dir_flat[idx]
        if k < 0:
            continue  # outlet or sink: nothing downstream to hand off to
        r, c = divmod(idx, ncols)
        dr, dc = _D8_OFFSETS[k]
        nr, nc = r + int(dr), c + int(dc)
        if 0 <= nr < nrows and 0 <= nc < ncols:
            acc_flat[nr * ncols + nc] += acc_flat[idx]

    return acc_flat.reshape(nrows, ncols)


def flow_accumulation(dem, weights=None):
    """
    Upslope contributing area, in cells (or in `weights` units if supplied).

    This is the function the Module 2 scaffold called for. Typical use:

        acc = hydrology.flow_accumulation(terrain["elevation"])
        streams = hydrology.extract_streams(acc, threshold=500)

    Multiply by cell area to get m^2 if you need physical units.
    """
    arr = _as_array(dem)
    filled = _fill_depressions_core(arr)
    direction = _flow_direction_core(filled)
    w = None if weights is None else _as_array(weights)
    acc = _flow_accumulation_core(filled, direction, w)
    out = _like(dem, acc, "flow_accumulation")
    if hasattr(out, "attrs"):
        out.attrs["units"] = "cells" if weights is None else "weight-units"
    return out


def extract_streams(accumulation, threshold: float = 500.0):
    """
    Threshold an accumulation grid into a stream network.

    The threshold is the classic arbitrary-but-tunable parameter: it sets the
    minimum contributing area required to call a cell "channel". 500 cells at
    30 m is ~0.45 km^2, which gives a reasonably dense network in steep terrain
    like the upper Poudre. Tune it against NHD flowlines for the real thing 
    `data_access.get_watershed_boundaries` returns those for comparison.
    """
    acc = _as_array(accumulation)
    streams = (acc >= threshold) & ~np.isnan(acc)
    return _like(accumulation, streams.astype("float64"), "streams")


# Flow-path distance to the channel network
def _downstream_traverse_core(filled: np.ndarray, direction: np.ndarray, per_cell_cost: np.ndarray, stop_mask: np.ndarray) -> np.ndarray:
    """
    Accumulate a per-cell cost along each cell's downstream flow path until it
    hits `stop_mask` (normally the stream network) or leaves the grid.

    Same topological trick as accumulation, run the other way: process cells in
    *ascending* elevation so that every cell's downstream neighbour already has
    its total when we get to it. Then total(cell) = cost(cell) + total(downstream).
    """
    nrows, ncols = filled.shape
    total = np.full((nrows, ncols), np.nan, dtype="float64")

    valid = ~np.isnan(filled)
    total[valid] = 0.0

    flat_idx = np.flatnonzero(valid)
    order = flat_idx[np.argsort(filled.ravel()[flat_idx], kind="stable")]  # ascending

    dir_flat = direction.ravel()
    total_flat = total.ravel()
    cost_flat = per_cell_cost.ravel()
    stop_flat = stop_mask.ravel()

    for idx in order:
        if stop_flat[idx]:
            total_flat[idx] = 0.0  # already at a channel: zero remaining path
            continue
        k = dir_flat[idx]
        if k < 0:
            continue
        r, c = divmod(idx, ncols)
        dr, dc = _D8_OFFSETS[k]
        nr, nc = r + int(dr), c + int(dc)
        if not (0 <= nr < nrows and 0 <= nc < ncols):
            continue
        downstream = total_flat[nr * ncols + nc]
        total_flat[idx] = cost_flat[idx] * _D8_DISTANCE[k] + (0.0 if np.isnan(downstream) else downstream)

    return total_flat.reshape(nrows, ncols)


def distance_to_stream(dem, streams, cell_size_m: float | None = None):
    """
    Along-flow-path distance from each cell to the nearest downstream channel.

    Note this is *not* Euclidean distance to the nearest stream it follows
    the actual routing, which is what matters for sediment delivery. A cell
    50 m from a stream but on the far side of a divide is hundreds of metres
    away in flow-path terms.
    """
    arr = _as_array(dem)
    size = cell_size_m if cell_size_m is not None else _cell_size_m(dem)
    filled = _fill_depressions_core(arr)
    direction = _flow_direction_core(filled)
    cost = np.full(arr.shape, float(size))
    dist = _downstream_traverse_core(filled, direction, cost, _as_array(streams) > 0)
    out = _like(dem, dist, "distance_to_stream")
    if hasattr(out, "attrs"):
        out.attrs["units"] = "m"
    return out


# Sediment connectivity
def sediment_connectivity_index(
    dem,
    slope_deg,
    impedance,
    streams=None,
    stream_threshold: float = 500.0,
    cell_size_m: float | None = None,
    min_slope: float = 0.005,
):
    """
    Index of Connectivity (Borselli et al. 2008; Cavalli et al. 2013).

        IC = log10( D_up / D_dn )

        D_up = W_bar * S_bar * sqrt(A)          upslope component
        D_dn = sum_i ( d_i / (W_i * S_i) )      downslope component, along the
                                                flow path to the channel

    where W is a per-cell weighting factor representing resistance to sediment
    transfer (low W = vegetation, roughness, and litter intercepting runoff;
    high W = bare, hydrophobic, severely burned ground) and S is slope in m/m.

    High IC means a cell's runoff and sediment reach the channel network
    directly. That is the Module 2 question: which parts of the Cameron Peak
    burn scar are hydrologically wired straight into the Poudre mainstem, and
    which are buffered by intact vegetation or flatter ground downslope.

    Parameters
    impedance : DataArray
        The W factor, 0-1, where higher = *less* resistance (more connected).
        Build it from burn severity and NDVI: see `indicators.transfer_weight`.
    min_slope : float
        Floor applied to slope, since D_dn divides by S and flat cells would
        otherwise blow up to infinite impedance.

    Returns a DataArray of IC values. These are relative, not absolute: compare
    within a region, not across regions with different cell sizes.
    """
    arr = _as_array(dem)
    size = cell_size_m if cell_size_m is not None else _cell_size_m(dem)
    filled = _fill_depressions_core(arr)
    direction = _flow_direction_core(filled)

    # Slope as a tangent, floored so the division in D_dn stays finite.
    slope = np.clip(np.tan(np.deg2rad(_as_array(slope_deg))), min_slope, None)
    W = np.clip(_as_array(impedance), 0.01, 1.0)  # floored for the same reason

    # upslope component
    # A: contributing area in m^2.
    acc_cells = _flow_accumulation_core(filled, direction, None)
    area_m2 = acc_cells * size * size
    # W_bar, S_bar: the flow-weighted mean of W and S over the contributing
    # area. Accumulating W and S as weights and dividing by cell count is the
    # standard way to get this without a second traversal.
    W_bar = _flow_accumulation_core(filled, direction, W) / acc_cells
    S_bar = _flow_accumulation_core(filled, direction, slope) / acc_cells
    d_up = W_bar * S_bar * np.sqrt(area_m2)

    # downslope component
    if streams is None:
        stream_mask = acc_cells >= stream_threshold
    else:
        stream_mask = _as_array(streams) > 0
    per_cell_cost = size / (W * slope)
    d_dn = _downstream_traverse_core(filled, direction, per_cell_cost, stream_mask)
    # Cells that *are* channels have D_dn = 0; give them a half-cell floor so
    # the log stays finite and they come out maximally connected, as they are.
    d_dn = np.where(d_dn <= 0, size / 2.0, d_dn)

    with np.errstate(divide="ignore", invalid="ignore"):
        ic = np.log10(d_up / d_dn)
    ic[np.isnan(filled)] = np.nan

    out = _like(dem, ic, "sediment_connectivity_index")
    if hasattr(out, "attrs"):
        out.attrs["description"] = "Borselli-style index of connectivity; higher = more directly coupled to the channel network"
    return out


def subwatershed_labels(dem, streams=None, stream_threshold: float = 500.0, min_cells: int = 200):
    """
    Label sub-watersheds by tracing every cell down to its outlet.

    This is the fallback for when NHD/WBD polygons are not available or are too
    coarse for the AOI. It is genuinely derived from the terrain, unlike the
    illustrative grid-based split used in `wildfire-landscape-intelligence`
    Module 4 but prefer `data_access.get_watershed_boundaries` when you can,
    because those polygons are what agency partners already work in.
    """
    arr = _as_array(dem)
    filled = _fill_depressions_core(arr)
    direction = _flow_direction_core(filled)
    nrows, ncols = filled.shape

    labels = np.full(filled.size, -1, dtype="int64")
    dir_flat = direction.ravel()
    valid = ~np.isnan(filled).ravel()

    # Outlets (direction == -1) each seed a label; every other cell inherits
    # its downstream cell's label. Descending-elevation order would work, but
    # ascending is simpler here since outlets sit at the bottom.
    order = np.flatnonzero(valid)[np.argsort(filled.ravel()[valid], kind="stable")]
    next_label = 0
    for idx in order:
        k = dir_flat[idx]
        if k < 0:
            labels[idx] = next_label
            next_label += 1
            continue
        r, c = divmod(idx, ncols)
        dr, dc = _D8_OFFSETS[k]
        nr, nc = r + int(dr), c + int(dc)
        if 0 <= nr < nrows and 0 <= nc < ncols:
            labels[idx] = labels[nr * ncols + nc]

    # Drop slivers: single-cell edge outlets produce a lot of one-pixel basins.
    out = labels.reshape(nrows, ncols).astype("float64")
    for lab in np.unique(labels[labels >= 0]):
        if (labels == lab).sum() < min_cells:
            out[out == lab] = np.nan
    out[np.isnan(filled)] = np.nan
    return _like(dem, out, "subwatershed_id")
