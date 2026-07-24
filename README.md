# Daear Consulting Demo Repo Portfolio

```
Daear-Consulting/
├── earth-intelligence-toolkit    # shared library for data access, indicators, cube, governance, viz
├── wildfire-landscape-intelligence   # fully built, 4/4 modules
├── soil-watershed-intelligence       # module 3 built, 1/2/4 scaffolded
├── tribal-wildfire-intelligence      # data sovereignty module 4 built, 1/2/3 scaffolded
└── climate-resilience-indicators     # synthesis/atlas layer module 3 built, 1/2/4 scaffolded
```

## What this is

A working demonstration that showcases how some of Daear's wildfire, soil-watershed, 
and data sovereignty work operate.
All four downstream repos share one library (`earth-intelligence-toolkit`)
and anchor on one location, the **Cache la Poudre watershed above Fort
Collins, Colorado**, burned by the **2020 Cameron Peak Fire**.

## What's actually built vs. scaffolded

Per the "thin vertical slice" approach: rather than fully building one repo
and leaving the other three empty, every repo has at least one fully
working, executed notebook using the shared toolkit against the shared
region and the remaining modules in each repo are scaffolded with a clear
plan (data sources, outputs, which toolkit functions they'll call) rather
than left blank.

**Fully built:** `wildfire-landscape-intelligence` (all 4 modules).

**One module built, rest scaffolded:** `soil-watershed-intelligence`
(fire→soil→water chain), `tribal-wildfire-intelligence` (governance
enforcement demo), `climate-resilience-indicators` (composite index,
reusing the other two repos' outputs directly).

Every notebook that's marked "Built" has been executed end-to-end in this
build and includes real output (plots, tables, printed results) — nothing
here is unrun.

## Data status: honest disclosure

**All data in this build is synthetic.** The sandbox this was built in
does not have network access to Earthdata, Planetary Computer, Copernicus,
MTBS, FIRMS, or SSURGO/SoilGrids endpoints. `earth-intelligence-toolkit`'s
`data_access.py` module has real function signatures matching those APIs,
with a `synthetic=True` demo path that generates structurally realistic
data (correct shapes, coordinate systems, plausible value ranges) so every
downstream calculation — NDVI, dNBR, erosion susceptibility, the composite
resilience index runs on the same code path that would run on live
imagery. Going live is a matter of installing three more packages
(`earthaccess`, `pystac-client`, `rioxarray`), setting credentials, and
flipping `synthetic=False` no changes to `indicators.py`, `cube.py`,
`governance.py`, or any notebook logic.

**If sharing this externally (e.g. as an accelerator demo link):** say so
plainly rather than letting the polish imply live data. Something like
*"a working analytical pipeline demonstrated on synthetic data structured
to match real Sentinel-2/MTBS/SSURGO sources, ready to connect to live
feeds"* is accurate; presenting the outputs as if they're from real 2020
Cameron Peak imagery would not be.

## Quickstart

```bash
cd earth-intelligence-toolkit && pip install -e .
cd ../wildfire-landscape-intelligence && jupyter lab notebooks/
```

Every notebook runs top-to-bottom with no credentials required.

## Next steps toward a real demo

1. Get `earthaccess`/Planetary Computer credentials and flip `synthetic=False`
   for at least one region/date to replace synthetic data with a real
   Cameron Peak scene
2. Fill in the scaffolded modules in `soil-watershed-intelligence`,
   `tribal-wildfire-intelligence`, and `climate-resilience-indicators` —
   each README lists exactly what's planned
3. Replace the illustrative grid-based "sub-watershed" split in
   `wildfire-landscape-intelligence` Module 4 with real NHD/WBD HUC-12
   polygons
4. Extend `climate-resilience-indicators` to the Black Hills and a Tribal
   watershed region (new `Region` entries in
   `earth_intelligence_toolkit.regions` no new code needed)
