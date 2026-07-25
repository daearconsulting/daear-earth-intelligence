# climate-resilience-indicators

The synthesis / atlas layer — the "Mountain Watershed Resilience Atlas" idea
from the SHIELD notes. Pulls indicators computed in the other three repos
and stitches them into one cross-cutting resilience picture, rather than
recomputing anything from scratch. This is the artifact that argues "Daear
is an Earth intelligence company," not "Daear has a wildfire tool and a
soil tool."

## Module plan

| Module | Status | Notebook |
|---|---|---|
| 1. Indicator catalog | Scaffolded | `01_indicator_catalog.ipynb` |
| 2. Regional coverage | Scaffolded | `02_regional_coverage.ipynb` |
| 3. Composite resilience index | **Built** (synthetic) + **live-ready** | `03_composite_resilience_index.ipynb` / `03_composite_resilience_index_LIVE.ipynb` |
| 4. Time-series view | Scaffolded | `04_time_series_view.ipynb` |

## Live-data notebook

`03_composite_resilience_index_LIVE.ipynb` calls real Sentinel-2, Copernicus
DEM, and SSURGO sources via `earth-intelligence-toolkit`'s `_live` module —
**not executed in this build** (the sandbox this was built in has no network
access to those providers), but syntax-validated and ready to run wherever
you have open internet access. See its first cell for setup, and
`earth-intelligence-toolkit/daear_toolkit/_live.py`'s module docstring for
a per-source confidence note (Sentinel-2/DEM/FIRMS are solid; MTBS/SSURGO
have caveats worth reading before a deadline-critical run).

Module 3 is built first because it's the single artifact that most directly
demonstrates the "one company, one continuous picture" story: it imports
the exact `erosion_susceptibility` and `watershed_resilience_index` outputs
already computed in `wildfire-landscape-intelligence` and
`soil-watershed-intelligence` for the Cache la Poudre / Cameron Peak region,
and combines them into one blended score — no recomputation, genuinely
reused.

## Regional scope (planned, beyond this build)

Per the SHIELD notes: Black Hills, Colorado Front Range, and Tribal
watersheds (the "Three Rivers" framing). This build demonstrates the
pattern on one region (Cache la Poudre) — extending to the other two is a
matter of pointing the same toolkit functions at new bounding boxes, not
new code.

## Run it

```bash
pip install -e ../earth-intelligence-toolkit
jupyter lab notebooks/
```
