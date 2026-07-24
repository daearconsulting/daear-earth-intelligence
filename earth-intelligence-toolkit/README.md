# earth-intelligence-toolkit

Shared library underneath all Daear demo repos. Not a standalone demo — this is
what makes `wildfire-landscape-intelligence`, `soil-watershed-intelligence`,
`tribal-wildfire-intelligence`, and `climate-resilience-indicators` behave like
one platform instead of four disconnected notebooks.

## What's in here

| Module | Purpose |
|---|---|
| `data_access.py` | Uniform interface to Sentinel-2, Landsat, MTBS, FIRMS, SSURGO/SoilGrids, DEM/terrain sources |
| `indicators.py` | Vegetation indices (NDVI, NBR/dNBR), terrain derivatives (slope, flow accumulation proxy), soil exposure indicators |
| `cube.py` | xarray/zarr-backed data cube builder — same cube pattern used in the OLC Integrated Data Cube contract |
| `governance.py` | OCAP®/CARE/FAIR-aligned tagging and access-control scaffolding for the public/Tribal two-track data pattern |
| `viz.py` | Shared plotting style so every repo's figures look like one product line |

## Data access: honest status

This toolkit is built to call real Earth observation sources — `earthaccess`
(NASA), `pystac-client` (Microsoft Planetary Computer / Element84), and USDA
SSURGO web services — through `data_access.py`. **In this build**, those calls
are implemented but the demo notebooks run in `synthetic=True` mode, because
the sandbox this was built in does not have network access to Earthdata,
Planetary Computer, or Copernicus endpoints.

The synthetic generator (`data_access.synthetic_scene`) produces
structurally realistic rasters with correct shape, coordinate reference,
plausible value ranges and spatial autocorrelation seeded so results are
reproducible. Every downstream calculation (NDVI, dNBR, erosion indicators,
composite indices) runs on that synthetic data using the *same* code path
that would run on real imagery. Swapping `synthetic=False` and supplying
credentials is the only change needed to point this at live data — see
"Going live" below.

## Demo region

All four downstream repos anchor their vertical-slice demo on the same
place: the **Cache la Poudre River watershed above Fort Collins, Larimer
County, Colorado**, centered on the area burned by the **2020 Cameron Peak
Fire** (the largest wildfire in Colorado history at the time, ~208,000
acres) and the debris-flow and water-quality impacts to the Poudre that
followed in 2021. It's a real, well-documented case that touches every
repo in the portfolio: pre-fire fuel conditions, a major fire event,
post-fire watershed degradation, and a municipal water supply (Fort
Collins/Greeley draw from the Poudre) with a direct resilience stake in the
outcome — plus it sits inside the Colorado footprint that ARID/SHIELD care
about.

Bounding box used throughout: approximately `-105.85, 40.55, -105.45, 40.85`
(WGS84), covering the upper Poudre Canyon burn scar down to the Poudre's
exit onto the Front Range.

## Going live

To point this at real data:
1. `pip install earthaccess pystac-client rioxarray` (not installed in this build)
2. Set `EARTHDATA_USERNAME`/`EARTHDATA_TOKEN` (or Planetary Computer SAS token) as environment variables
3. Call any `data_access.get_*` function with `synthetic=False`

## Install

```bash
pip install -e .
```

Then in any of the other four repos: `import daear_toolkit as dt`
