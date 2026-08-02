# tribal-wildfire-intelligence

The data-sovereignty differentiator repo. Where the other repos *describe*
Daear's governance commitments, this one demonstrates them running against
sample data and access is actually gated by governance metadata, not just
documented in a policy paragraph.

## Module plan

Module 4 was built first because a working governance mechanism is more
convincing to reviewers than a description of one. It uses
`daear_toolkit.governance` (`GovernanceRecord`, `AccessTier`,
`gated_release`, `aggregate_to_public`) against a synthetic sample dataset
standing in for something like site-level fire history near
culturally-sensitive locations so the notebook shows a public-tier request
being correctly denied, a community-tier request being served, and a
restricted layer being safely released only after aggregation.

## Data layout

```
data/
  tribal_restricted/   # would hold anything requiring COMMUNITY or
                        # RESTRICTED-tier access in a real deployment.
                        # Empty here — this repo demonstrates the *access
                        # control mechanism*, not real restricted data.
```

## Run it

```bash
pip install -e ../earth-intelligence-toolkit
jupyter lab notebooks/
```
