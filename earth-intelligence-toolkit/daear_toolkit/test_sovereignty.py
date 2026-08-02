"""Sanity checks for the sovereignty and tribal indicator modules."""
import sys, types, datetime as dt
import numpy as np, pandas as pd
from pathlib import Path

# Folder containing sovereignty.py and tribal_indicators.py
TOOLKIT_PATH = Path(r"C:\Users\gekek\Documents\Daear-Consulting demos\daear-earth-intelligence\earth-intelligence-toolkit\daear_toolkit")

pkg = types.ModuleType("daear_toolkit")
pkg.__path__ = [str(TOOLKIT_PATH)]
sys.modules["daear_toolkit"] = pkg
sys.path.insert(0, str(TOOLKIT_PATH))

# Imports must come after sys.path is updated
import sovereignty as sv
import tribal_indicators as ti

# governance 
ctx = sv.GovernanceContext(nation="Oglala Sioux Tribe")
ctx.check(sv.Tier.PUBLIC, dataset="mtbs")           # allowed without agreement
print("PUBLIC tier without agreement: permitted")

for tier in (sv.Tier.PARTNER, sv.Tier.COMMUNITY):
    try:
        ctx.check(tier, dataset="mtbs"); raise AssertionError(f"{tier.name} should have been denied")
    except PermissionError as e:
        print(f"{tier.name} tier without agreement: denied ({str(e)[:48]}...)")

# RESTRICTED is denied even WITH a maximal agreement
future = (dt.date.today() + dt.timedelta(days=365)).isoformat()
dua = sv.DataUseAgreement(
    nation="Oglala Sioux Tribe", counterparty="Daear Consulting LLC",
    purpose="post-fire watershed assessment", executed="2026-01-15", expires=future,
    release_authority="OST Natural Resources Regulatory Agency",
    approved_tiers=(sv.Tier.PUBLIC, sv.Tier.PARTNER, sv.Tier.COMMUNITY, sv.Tier.RESTRICTED),
    data_scope=("mtbs", "fod", "housing"))
ctx2 = sv.GovernanceContext(nation="Oglala Sioux Tribe", agreement=dua)
ctx2.check(sv.Tier.COMMUNITY, dataset="housing")
print("COMMUNITY tier with agreement: permitted")
try:
    ctx2.check(sv.Tier.RESTRICTED, dataset="housing"); raise AssertionError("RESTRICTED must always deny")
except PermissionError:
    print("RESTRICTED tier even with a maximal agreement: denied (correct)")

# scope is enumerated, not open-ended
try:
    ctx2.check(sv.Tier.PARTNER, dataset="health_records"); raise AssertionError("out-of-scope must deny")
except PermissionError:
    print("Out-of-scope dataset: denied (correct)")

# expiry
expired = sv.DataUseAgreement(nation="N", counterparty="C", purpose="p", executed="2020-01-01",
                              expires="2021-01-01", release_authority="A",
                              approved_tiers=(sv.Tier.PARTNER,), data_scope=("mtbs",))
assert not expired.is_current and not expired.permits(sv.Tier.PARTNER, "mtbs")
print(f"Expired agreement: denied ({expired.days_until_expiry:+d} days)")

# an agreement approving nothing is rejected at construction
try:
    sv.DataUseAgreement(nation="N", counterparty="C", purpose="p", executed="2026-01-01",
                        expires=future, release_authority="A", approved_tiers=(), data_scope=())
    raise AssertionError("empty tiers must be rejected")
except ValueError:
    print("Agreement approving no tiers: rejected at construction")

# publication gate fails closed
try:
    ctx.check_publication("fire history maps"); raise AssertionError("must fail closed")
except PermissionError:
    print("\nPublication gate with no approval: BLOCKED (fails closed)")
ctx.public_release_approved = True
try:
    ctx.check_publication(); raise AssertionError("approval without a record must fail")
except PermissionError:
    print("Publication approved but unattributed: BLOCKED (correct)")
ctx.approval_record = "OST NRRA, reviewed 2026-07-20, tribal council resolution 26-114"
ctx.check_publication()
print("Publication with recorded approval: permitted")

log = ctx.audit_log()
print(f"Audit log: {len(log)} gated operations, {int((log['result']=='DENIED').sum())} denied")
assert len(log) >= 3

# cultural resource guard
try:
    import geopandas as gpd
    from shapely.geometry import Point
    safe = gpd.GeoDataFrame({"name": ["fire station"], "kind": ["facility"]},
                            geometry=[Point(-102.5, 43.2)], crs="EPSG:4326")
    assert sv.CulturalResourceGuard.scan(safe) == []
    print("\nBenign dataset: passes the guard")

    for bad in [
        gpd.GeoDataFrame({"burial_site_id": [1]}, geometry=[Point(-102.5, 43.2)], crs="EPSG:4326"),
        gpd.GeoDataFrame({"site_type": ["ceremonial"]}, geometry=[Point(-102.5, 43.2)], crs="EPSG:4326"),
    ]:
        try:
            sv.CulturalResourceGuard.scan(bad); raise AssertionError("should have been flagged")
        except PermissionError:
            pass
    print("Sensitive column name AND sensitive column VALUE: both flagged")
except ImportError:
    print("(geopandas unavailable; guard test skipped)")

# CARE assessment
rep = sv.care_assessment({q: True for q in sv.CARE_QUESTIONS["Ethics"]}, verbose=False)
assert rep["Ethics"]["met"] == 4 and rep["Collective Benefit"]["met"] == 0
assert len(rep["Collective Benefit"]["unaddressed"]) == 4
print(f"\nCARE: unanswered questions count as unmet, not neutral "
      f"(overall {rep['_overall']['fraction']:.0%} with one principle fully answered)")

# provenance
prov = sv.ProvenanceRecord(
    product="fire history summary", created="2026-07-26", creator="Daear Consulting LLC",
    nation="Oglala Sioux Tribe", data_sources=("MTBS 1984-2024", "Short FOD 1992-2020"),
    methods=("fire rotation", "ignition cause profile"), tier=sv.Tier.PUBLIC,
    indigenous_knowledge_contributions=("Fire history context provided by OST BIA Fire staff",))
md = prov.to_markdown()
assert "Indigenous knowledge contributions" in md and len(prov.fingerprint()) == 16
print(f"Provenance record renders; fingerprint {prov.fingerprint()}")

# fire rotation math
# 30-year period, exactly half the area burned -> rotation must be 60 years
r = ti.fire_rotation(total_area_ha=100_000, burned_area_by_year={y: 50_000/30 for y in range(1994, 2024)})
print(f"\nFire rotation (50% of area over 30 yrs): {r['rotation_years']} years, "
      f"{r['pct_area_burned']}% burned")
assert abs(r["rotation_years"] - 60.0) < 0.5
# whole area burned once over the period -> rotation == period
r2 = ti.fire_rotation(100_000, {y: 100_000/30 for y in range(1994, 2024)})
assert abs(r2["rotation_years"] - 30.0) < 0.5
# no fire -> infinite rotation, not a crash or a zero
assert ti.fire_rotation(100_000, {y: 0 for y in range(1994, 2024)})["rotation_years"] == np.inf
print("Edge cases: full-area burn -> rotation == period; zero burn -> infinite")

comp = ti.suppression_era_comparison(observed_rotation_years=300, reference_fri_years=25)
print(f"Suppression comparison: ratio {comp['ratio']} -> '{comp['interpretation']}'")
assert comp["ratio"] == 12.0 and "deficit" in comp["interpretation"]

# ignition profile
rng = np.random.default_rng(3)
n = 900
ign = pd.DataFrame({
    "fire_year": rng.integers(1992, 2021, n),
    "nwcg_general_cause": rng.choice(
        ["Natural", "Equipment and vehicle use", "Debris and open burning",
         "Missing data/not specified/undetermined"], n, p=[0.3, 0.2, 0.2, 0.3]),
    "fire_size": rng.lognormal(1.0, 2.0, n),
})
prof = ti.ignition_profile(ign)
print(f"\nIgnition profile: {len(prof)} causes; undetermined = {prof.attrs['undetermined_pct']}%")
assert 25 < prof.attrs["undetermined_pct"] < 36, "undetermined fraction must be surfaced"

# adaptive capacity refuses invented data
try:
    ti.adaptive_capacity({"tribal_fire_program_capacity": 0.8})
    raise AssertionError("must require a source")
except ValueError:
    print("\nAdaptive capacity without a source: rejected (these are community assessments)")

comps = ti.adaptive_capacity({
    "tribal_fire_program_capacity": (0.75, "OST BIA Fire, structural example"),
    "mutual_aid_agreements": (0.60, "structural example"),
    "land_based_knowledge": (0.90, "structural example"),
    "kinship_response_networks": (0.85, "structural example"),
})
assert len(comps) == 4

try:
    ti.community_weighted_index(comps)
    raise AssertionError("must refuse analyst-chosen weights")
except ValueError as e:
    assert "belongs to the community" in str(e)
    print("Composite index without community weights: refused (correct)")

idx = ti.community_weighted_index(
    comps, weights={"tribal_fire_program_capacity": 3, "mutual_aid_agreements": 2,
                    "land_based_knowledge": 3, "kinship_response_networks": 2},
    weights_source="structural example -- NOT a real community determination")
print(f"With sourced weights: index = {idx['index']}")
assert 0 <= idx["index"] <= 1
assert abs(sum(c["weight"] for c in idx["components"]) - 1.0) < 1e-9

# unknown component names are rejected rather than silently ignored
try:
    ti.adaptive_capacity({"median_income": (0.4, "census")}, require_source=False)
    raise AssertionError("unknown component must be rejected")
except ValueError:
    print("Deficit-framed variable smuggled in as a component: rejected")

print("\nAll sovereignty and tribal indicator checks passed.")
