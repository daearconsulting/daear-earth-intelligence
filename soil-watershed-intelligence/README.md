# soil-watershed-intelligence

Uses `earth-intelligence-toolkit` for all data access and indicator math 
same Cache la Poudre/Cameron Peak region as `wildfire-landscape-intelligence`, 
so this repo picks up exactly where that one's burn-severity and terrain layers leave off.

This repo demonstrates *quantifying soil–watershed relationships*, not direct soil
sensing. Every notebook here scales understanding *between* ground
measurements rather than claiming to replace them.

## Module plan

| Module | Status | Notebook |
|---|---|---|
| 1. Soil landscape indicators | Built | `01_soil_landscape_indicators.ipynb` |
| 2. Hydrologic connectivity | Built | `02_hydrologic_connectivity.ipynb` |
| 3. Fire Soil Water impacts | Built | `03_fire_soil_water_impacts.ipynb` |
| 4. Restoration monitoring | Built | `04_restoration_monitoring.ipynb` |

Module 3 directly consumes `wildfire-landscape-intelligence`'s burn-severity 
output and carries it through to a soil-vulnerability and watershed-risk result. 
The scaffolded notebooks lay out the intended data sources and outputs for each 
remaining module so they can be filled in without re-deriving the plan.

## Run it

```bash
pip install -e ../earth-intelligence-toolkit
jupyter lab notebooks/
```
