# climate-resilience-indicators

The synthesis/atlas layer the "Mountain Watershed Resilience Atlas" idea. 
Pulls indicators computed in the other three repos
and stitches them into one cross-cutting resilience picture, rather than
recomputing anything from scratch."

## Live-data notebook

`03_composite_resilience_index_LIVE.ipynb` calls Sentinel-2, Copernicus
DEM, and SSURGO sources via `earth-intelligence-toolkit`. See its first cell for setup, and
`earth-intelligence-toolkit/daear_toolkit/_live.py`'s module docstring for
a per-source confidence note (Sentinel-2/DEM/FIRMS are solid; MTBS/SSURGO
have caveats worth reading before a deadline-critical run).

Module 3 imports the exact `erosion_susceptibility` 
and `watershed_resilience_index` outputs
already computed in `wildfire-landscape-intelligence` and
`soil-watershed-intelligence` for the Cache la Poudre/Cameron Peak region,
and combines them into one blended score with no recomputation.

## Regional scope (planned, beyond this demo)

This build demonstrates the pattern on one region (Cache la Poudre) 
extending to the other two is a matter of pointing the same toolkit 
functions at new bounding boxes.

## Run it

```bash
pip install -e ../earth-intelligence-toolkit
jupyter lab notebooks/
```
