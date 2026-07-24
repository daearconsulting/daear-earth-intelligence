# wildfire-landscape-intelligence

Uses `earth-intelligence-toolkit` for all
data access and indicator math.

**Demo region:** Cache la Poudre watershed / 2020 Cameron Peak Fire burn
scar, Larimer County, Colorado (see toolkit README for why).

## Module plan

| Module | Status | Notebook |
|---|---|---|
| 1. Pre-fire landscape conditions | **Built** | `01_pre_fire_conditions.ipynb` |
| 2. Fire event analysis (dNBR, burn severity) | **Built** | `02_fire_event_analysis.ipynb` |
| 3. Post-fire recovery | **Built** | `03_post_fire_recovery.ipynb` |
| 4. Community risk dashboard | **Built** | `04_community_risk_dashboard.ipynb` |

This is the one repo in the portfolio built out fully end-to-end as the
"thin vertical slice" — the other three repos share its data/indicator
backbone but are scaffolded at varying depth (see each repo's own README).

## Run it

```bash
pip install -e ../earth-intelligence-toolkit
jupyter lab notebooks/
```

All notebooks run against synthetic demo data by default (see toolkit
README) no credentials required to execute top to bottom.

## Output

`04_community_risk_dashboard.ipynb` produces the headline artifact:
a map + summary table identifying which sub-areas of the watershed show
the highest combined burn severity + erosion susceptibility the
"three watersheds requiring post-fire monitoring" framing from the pitch.
