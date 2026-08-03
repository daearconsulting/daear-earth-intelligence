# Daear Consulting Earth Intelligence Platform

```
Daear-Consulting/
├── earth-intelligence-toolkit        # shared library for data access, indicators, cube, governance, viz
├── wildfire-landscape-intelligence   # fully built, 4/4 modules
├── soil-watershed-intelligence       # module 3 built, 1/2/4 scaffolded
├── tribal-wildfire-intelligence      # data sovereignty module 4 built, 1/2/3 scaffolded
└── climate-resilience-indicators     # synthesis/atlas layer: module 3 built, 1/2/4 scaffolded
```

## What this is

A working demonstration of Daear's wildfire, soil-watershed, and data
sovereignty work are facets of one platform. All four downstream repos 
share one library (`earth-intelligence-toolkit`) and anchor on one place: 
the **Cache la Poudre watershed above Fort Collins, Colorado**, burned by 
the **2020 Cameron Peak Fire**.

## What's actually built vs. scaffolded

Per the "thin vertical slice" approach: every repo has at least one fully
working, executed notebook using the shared toolkit against the shared
region and the remaining modules in each repo are scaffolded with a clear
plan (data sources, outputs, which toolkit functions they'll call).

**Fully built:** `wildfire-landscape-intelligence` (all 4 modules).

**One module built, rest scaffolded:** `soil-watershed-intelligence`
(fire/soil/water chain), `tribal-wildfire-intelligence` (governance
enforcement demo), `climate-resilience-indicators` (composite index,
reusing the other two repos' outputs directly).

Every notebook that's marked "Built" has been executed end-to-end in this
build and includes real output (plots, tables, printed results).

## Quickstart

```bash
cd earth-intelligence-toolkit && pip install -e .
cd ../wildfire-landscape-intelligence && jupyter lab notebooks/
```

Every notebook runs top-to-bottom with no credentials required.

## Next steps

1. Fill in the scaffolded modules in `soil-watershed-intelligence`,
   `tribal-wildfire-intelligence`, and `climate-resilience-indicators` 
   each README lists exactly what's planned
2. Replace the illustrative grid-based "sub-watershed" split in
   `wildfire-landscape-intelligence` Module 4 with real NHD/WBD HUC-12
   polygons
3. Extend `climate-resilience-indicators` to the Black Hills and a Tribal
   watershed region (new `Region` entries in
   `earth_intelligence_toolkit.regions` no new code needed)
