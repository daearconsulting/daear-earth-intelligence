# earth-intelligence-toolkit

Shared library that makes `wildfire-landscape-intelligence`, `soil-watershed-intelligence`,
`tribal-wildfire-intelligence`, and `climate-resilience-indicators` behave like
one platform.

## What's in here

| Module | Purpose |
|---|---|
| `data_access.py` | Uniform interface to Sentinel-2, Landsat, MTBS, FIRMS, SSURGO/SoilGrids, DEM/terrain sources |
| `indicators.py` | Vegetation indices (NDVI, NBR/dNBR), terrain derivatives (slope, flow accumulation proxy), soil exposure indicators |
| `cube.py` | xarray/zarr-backed data cube builder |
| `governance.py` | OCAP®/CARE/FAIR-aligned tagging and access-control scaffolding for the public/Tribal two-track data pattern |
| `viz.py` | Shared plotting style |

## Data access: honest status

The synthetic generator (`data_access._smooth_random_field`
and friends) produces structurally realistic rasters with correct shape,
coordinate reference, plausible value ranges and spatial autocorrelation 
seeded so results are reproducible. Every downstream calculation (NDVI,
dNBR, erosion indicators, composite indices) runs on that synthetic data
using the *same* code path that would run on real imagery.

**`synthetic=False` is now a real implementation, not a stub**: see
`daear_toolkit/_live.py`. Real data:

| Source | Function | Confidence |
|---|---|---|
| Sentinel-2 (via Planetary Computer) | `_live.sentinel2_scene` | High stable public API, no credentials needed |
| Copernicus 30m DEM (via Planetary Computer) | `_live.copernicus_dem` | High |
| FIRMS active-fire detections | `_live.firms_fire_detections` | High needs a free `FIRMS_MAP_KEY` |
| Burn severity, no-MTBS route | `_live.burn_severity_from_sentinel2` | High real dNBR from two real Sentinel-2 scenes |
| MTBS burn severity, direct download | `_live.mtbs_burn_severity_from_direct_download` | Medium: supply the GeoTIFF URL; MTBS has no clean queryable API |
| SSURGO soil properties | `_live.ssurgo_soil_properties` | Medium: endpoint/query verified against SDA docs as of this build, but SDA's exact SQL function names have shifted before; returns a per-map-unit table, not a grid |

See `_live.py`'s module docstring for the full reasoning behind each
confidence rating before relying on this under deadline pressure.

## Demo region

All four downstream repos anchor their vertical-slice demo on the same
place: the **Cache la Poudre River watershed above Fort Collins, Larimer
County, Colorado**, centered on the area burned by the **2020 Cameron Peak
Fire** (the largest wildfire in Colorado history at the time, ~208,000
acres) and the debris-flow and water-quality impacts to the Poudre that
followed in 2021. This is a real, well-documented case that touches every
repo in the portfolio: pre-fire fuel conditions, a major fire event,
post-fire watershed degradation, and a municipal water supply (Fort
Collins/Greeley draw from the Poudre) with a direct resilience stake in the
outcome.

Bounding box used throughout: approximately `-105.85, 40.55, -105.45, 40.85`
(WGS84), covering the upper Poudre Canyon burn scar down to the Poudre's
exit onto the Front Range.

## Going live

1. `pip install 'daear-toolkit[live]'` (or from this repo: `pip install -e ".[live]"`)
2. Get a free FIRMS `MAP_KEY` (https://firms.modaps.eosdis.nasa.gov/api/map_key/) and set it: `export FIRMS_MAP_KEY=...`
   Planetary Computer needs no credentials for read access.
3. Call `data_access.get_optical_scene(..., synthetic=False)` or
   `data_access.get_terrain(..., synthetic=False)` directly; for burn
   severity and soil, call the `daear_toolkit._live` functions directly
   (see table above as their live paths don't map 1:1 onto the synthetic
   function signatures, for reasons explained inline).

## Install

```bash
pip install -e .
```

Then in any of the other four repos: `import daear_toolkit as dt`
