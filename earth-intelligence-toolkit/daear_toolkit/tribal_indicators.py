from __future__ import annotations

"""
daear_toolkit.tribal_indicators

Indicators for `tribal-wildfire-intelligence`.

    Module 1  fire_rotation, fire_return_interval, suppression_era_comparison,
              ignition_profile
    Module 2  jurisdictional_complexity, exposure_by_land_status
    Module 3  adaptive_capacity, community_weighted_index

Standard social vulnerability indices (CDC SVI, SoVI and their descendants)
perform badly on Tribal communities, and not by accident. They are built from
census variables that encode a deficit frame: poverty rate, unemployment, no
vehicle, limited English, single-parent households, mobile homes. Applied to a
reservation, they produce a high score reliably and explain nothing, because:

- The variables measure distance from a white suburban norm, not capacity to
  withstand and recover from fire.
- They systematically miss the actual sources of resilience: kinship networks
  that mobilize faster than any agency, Tribal government emergency authority,
  land relationship and land-based knowledge, cultural continuity, and in many
  Nations a fire program with more local knowledge than the agencies around it.
- Census undercount on reservations is severe and well documented, so the input
  data is weakest exactly where the index is applied most confidently.
- A high vulnerability score is used to justify external intervention, which is
  how a measurement instrument becomes a political one.

The alternative implemented here is asset-based: it measures capacity that
exists, and it **refuses to run with analyst-chosen weights**. Which components
matter, and how much, is a community determination. That refusal is enforced in
code: see `community_weighted_index`.
"""

import numpy as np
import pandas as pd
from scipy import stats


# Module 1 fire history
def fire_rotation(total_area_ha: float, burned_area_by_year: dict, period_years: int | None = None) -> dict:
    """
    Fire rotation: the time required to burn an area equal to the study area.

        rotation = period_length/(cumulative burned area/study area)

    Rotation, not "mean fire return interval", is the right statistic from
    perimeter data. Fire return interval is a point property i.e. how often *this
    spot* burns and estimating it needs tree-ring or charcoal evidence.
    Rotation is a landscape property computable from mapped area, and conflating
    the two is a common and consequential error: they differ by a large factor
    in landscapes with skewed fire-size distributions, which is all of them.

    Note that reburns are counted. If 30% of the area burns twice within the
    period, that area contributes twice to the cumulative total, which is
    correct for rotation and wrong if you wanted "area burned at least once".
    """
    years = sorted(burned_area_by_year)
    period = period_years or (max(years) - min(years) + 1)
    cumulative = float(sum(burned_area_by_year.values()))

    if cumulative <= 0:
        return {"rotation_years": np.inf, "pct_area_burned": 0.0, "period_years": period,
                "cumulative_burned_ha": 0.0, "annual_pct": 0.0}

    fraction = cumulative / total_area_ha
    return {
        "rotation_years": round(period / fraction, 1),
        "pct_area_burned": round(fraction * 100, 2),
        "annual_pct": round(fraction / period * 100, 4),
        "cumulative_burned_ha": round(cumulative, 1),
        "period_years": period,
        "n_fire_years": int(sum(1 for v in burned_area_by_year.values() if v > 0)),
    }


def suppression_era_comparison(observed_rotation_years: float, reference_fri_years: float) -> dict:
    """
    Compare an observed satellite-era fire rotation against a reference
    pre-suppression fire return interval.

    `reference_fri_years` comes from LANDFIRE Biophysical Settings/Mean Fire
    Return Interval, or better, from local fire-scar chronologies where they
    exist.

    **What the ratio means, and what it does not.** A ratio far above 1 means
    substantially less fire than the reference regime a fire deficit. It does
    *not* by itself mean the landscape is unhealthy, and it does not license the
    conclusion that more fire of any kind would restore it. Contemporary fires
    under contemporary fuels and climate are not the fires the reference regime
    describes.

    What the deficit does establish is that the 1984-present satellite record is
    not a natural baseline. It is a record of a suppression regime, layered on
    top of the deliberate interruption of Indigenous burning. Treating it as
    "the historical range of variability" builds the policy into the baseline
    and then measures departure from it.
    """
    ratio = observed_rotation_years/reference_fri_years if reference_fri_years > 0 else np.nan
    if ratio > 3:
        interp = "severe fire deficit relative to the reference regime"
    elif ratio > 1.5:
        interp = "moderate fire deficit"
    elif ratio > 0.67:
        interp = "observed rotation broadly comparable to the reference"
    else:
        interp = "burning more frequently than the reference regime"

    return {
        "observed_rotation_years": round(observed_rotation_years, 1),
        "reference_fri_years": reference_fri_years,
        "ratio": round(ratio, 2),
        "interpretation": interp,
        "caution": ("The reference regime included Indigenous cultural burning. Comparing a "
                    "suppression-era record against it measures the effect of policy, not of "
                    "climate or of the landscape."),
    }


def ignition_profile(ignitions: pd.DataFrame, cause_col: str = "nwcg_general_cause", year_col: str = "fire_year") -> pd.DataFrame:
    """
    Ignition counts and size distribution by cause.

    Report the undetermined fraction prominently. It is usually large, it varies
    by reporting jurisdiction, and analyses that quietly drop it convert a data
    limitation into a confident cause attribution.
    """
    if ignitions.empty or cause_col not in ignitions.columns:
        return pd.DataFrame()

    df = ignitions.copy()
    size_col = "fire_size" if "fire_size" in df.columns else None

    grouped = df.groupby(cause_col)
    out = pd.DataFrame({"n_ignitions": grouped.size()})
    if size_col:
        out["median_size_acres"] = grouped[size_col].median().round(2)
        out["mean_size_acres"] = grouped[size_col].mean().round(1)
        out["max_size_acres"] = grouped[size_col].max().round(0)
        out["n_over_100ac"] = grouped[size_col].apply(lambda s: int((s > 100).sum()))
    out["pct_of_ignitions"] = (out["n_ignitions"] / len(df) * 100).round(1)

    out = out.sort_values("n_ignitions", ascending=False)
    undetermined = out.index.astype(str).str.contains("Missing|undeterm|not specified", case=False, regex=True)
    out.attrs["undetermined_pct"] = round(float(out.loc[undetermined, "pct_of_ignitions"].sum()), 1)
    return out


# Module 2 jurisdiction and exposure
def jurisdictional_complexity(land_status_gdf, status_col: str | None = None, grid_km: float = 5.0) -> pd.DataFrame:
    """
    Measure how fragmented land status is, on a grid.

    The output is the count of distinct ownership/status categories intersecting
    each cell. It is a proxy for how many authorities have to coordinate for a
    fire in that cell.

    This exists because checkerboarding is an operational fire problem with an
    identifiable spatial signature. Allotment and subsequent fee patenting
    produced interleaved trust, allotted-trust, and fee parcels inside
    reservation boundaries. Response authority, funding eligibility, and
    treatment permitting can differ parcel to parcel so a fire crossing three
    ownership types in one drainage may need three authorizations, and that
    delay is measured in hours during initial attack.

    High-complexity cells are where pre-negotiated mutual aid agreements pay for
    themselves. That is an actionable finding an agency partner can use.
    """
    if land_status_gdf.empty:
        return pd.DataFrame()

    if status_col is None:
        candidates = [c for c in land_status_gdf.columns
                      if any(k in c.lower() for k in ("status", "owner", "type", "class", "categ"))]
        status_col = candidates[0] if candidates else land_status_gdf.columns[0]

    projected = land_status_gdf.to_crs(epsg=5070)     # CONUS Albers, metres
    minx, miny, maxx, maxy = projected.total_bounds
    step = grid_km * 1000

    from shapely.geometry import box
    import geopandas as gpd

    cells = [box(x, y, x + step, y + step)
             for x in np.arange(minx, maxx, step)
             for y in np.arange(miny, maxy, step)]
    grid = gpd.GeoDataFrame(geometry=cells, crs=projected.crs)

    joined = gpd.sjoin(grid.reset_index(names="cell_id"), projected[[status_col, "geometry"]],
                       how="inner", predicate="intersects")
    agg = joined.groupby("cell_id")[status_col].agg([("n_statuses", "nunique"),
                                                     ("statuses", lambda s: "|".join(sorted(set(map(str, s)))))])
    out = grid.reset_index(names="cell_id").merge(agg, on="cell_id", how="left").fillna({"n_statuses": 0})
    return out.to_crs("EPSG:4326")


def exposure_by_land_status(assets_gdf, land_status_gdf, status_col: str | None = None) -> pd.DataFrame:
    """
    Count assets by land status category.

    The reason to break exposure out this way rather than reporting one
    reservation-wide number: funding eligibility and response authority follow
    land status, so "200 structures exposed" is not an actionable figure until
    it is split into which program can act on which of them.
    """
    import geopandas as gpd

    if assets_gdf.empty or land_status_gdf.empty:
        return pd.DataFrame()
    if status_col is None:
        candidates = [c for c in land_status_gdf.columns
                      if any(k in c.lower() for k in ("status", "owner", "type", "class"))]
        status_col = candidates[0] if candidates else land_status_gdf.columns[0]

    joined = gpd.sjoin(assets_gdf, land_status_gdf[[status_col, "geometry"]], how="left", predicate="within")
    out = joined.groupby(status_col, dropna=False).size().reset_index(name="n_assets")
    out["pct"] = (out["n_assets"] / out["n_assets"].sum() * 100).round(1)
    return out.sort_values("n_assets", ascending=False)


# Asset-based components/Module 3. Each is framed as capacity that exists, not as a
# deficit relative to an external norm. The contrast with a standard SVI
# variable list is the point and should survive into any presentation of this.
ADAPTIVE_CAPACITY_COMPONENTS = {
    "tribal_fire_program_capacity": "Staffing, equipment, and qualifications held by the Nation's own fire program",
    "mutual_aid_agreements": "Pre-negotiated agreements with adjacent jurisdictions, reducing authorization delay",
    "land_based_knowledge": "Active knowledge of fire behaviour, fuels, and terrain held in the community",
    "cultural_burning_practice": "Existing or revivable cultural burning practice and the authority to exercise it",
    "kinship_response_networks": "Informal networks that mobilize evacuation and sheltering faster than agencies",
    "governance_continuity": "Stable Tribal emergency management authority and continuity of program staff",
    "communications_reach": "Ability to reach households quickly, including where cell coverage is absent",
    "youth_workforce_pipeline": "Local pathways into fire and natural-resource careers",
}

# For contrast in the notebook, not for use. These are the variables a standard
# SVI would apply, listed here so the critique can be made concrete rather than
# asserted.
STANDARD_SVI_VARIABLES = (
    "below poverty line", "unemployed", "no high school diploma", "aged 65+",
    "aged 17 or younger", "with a disability", "single-parent household",
    "speaks English less than well", "mobile homes", "crowded housing",
    "no vehicle available", "group quarters",
)


def adaptive_capacity(component_scores: dict, require_source: bool = True) -> pd.DataFrame:
    """
    Assemble asset-based adaptive capacity components into a table.

    `component_scores` maps component name -> (score 0-1, source). The source is
    mandatory by default and there is a reason: every one of these components is
    something only the community can assess. A satellite cannot see mutual-aid
    agreements or kinship networks, and an analyst assigning them a number from
    a desk has invented data and given it the same visual weight as a measured
    layer.
    """
    rows = []
    for name, value in component_scores.items():
        if name not in ADAPTIVE_CAPACITY_COMPONENTS:
            raise ValueError(f"Unknown component {name!r}. Known: {sorted(ADAPTIVE_CAPACITY_COMPONENTS)}")
        if isinstance(value, (tuple, list)):
            score, source = value[0], value[1]
        else:
            score, source = value, None
        if require_source and not source:
            raise ValueError(
                f"Component {name!r} has no source. These components are community "
                f"assessments, not remotely sensed quantities record who provided the "
                f"score and when. Pass require_source=False only for a structural example "
                f"with clearly fabricated values."
            )
        rows.append({"component": name, "score": float(score), "source": source,
                     "description": ADAPTIVE_CAPACITY_COMPONENTS[name]})
    return pd.DataFrame(rows)


def community_weighted_index(components: pd.DataFrame, weights: dict | None = None,
                             weights_source: str | None = None) -> dict:
    """
    Combine adaptive-capacity components using **community-determined** weights.

    Raises if weights are not supplied with a source. This is the executable
    version of the argument in this module's docstring: which components matter,
    and how much, is a sovereignty question. An analyst who picks the weights has
    decided what resilience means for a community that was not asked.

    Equal weighting is not a neutral escape from this. It is the claim that every
    component matters identically, which is itself a substantive position and one
    no community has ever actually held.

    The right sequence is: build the component list *with* the community, have
    them weight it, then compute. If you are computing before that conversation,
    compute the components and show them unaggregated which is what this
    function tells you to do when it refuses.
    """
    if weights is None or weights_source is None:
        raise ValueError(
            "community_weighted_index requires both `weights` and `weights_source`.\n"
            "\n"
            "Weights determine what 'resilience' means in the output. That determination "
            "belongs to the community, not the analyst, and equal weighting is a position "
            "rather than a neutral default.\n"
            "\n"
            "Until weights have been set with the community, present the components "
            "separately, an unaggregated component table is still useful. "
            "Once they have been set, pass them with a source:\n"
            "    community_weighted_index(components, weights={...},\n"
            "                             weights_source='Tribal fire program workshop, <date>')"
        )

    missing = set(components["component"]) - set(weights)
    if missing:
        raise ValueError(f"No weight supplied for: {sorted(missing)}")

    w = np.array([weights[c] for c in components["component"]], dtype="float64")
    if w.sum() <= 0:
        raise ValueError("Weights must sum to a positive value")
    w = w / w.sum()
    scores = components["score"].values

    return {
        "index": round(float((scores * w).sum()), 4),
        "weights_source": weights_source,
        "components": components.assign(weight=w.round(3),
                                        contribution=(scores * w).round(4)).to_dict("records"),
        "n_components": len(components),
    }


def compare_framings(svi_score: float, capacity_index: float) -> str:
    """
    Narrative contrast between a deficit-framed vulnerability score and an
    asset-framed capacity index, for use in a notebook or a briefing.

    Both numbers can be high at once, and that combination is the common case.
    It is not a contradiction it means a community faces real material
    constraints *and* holds real capacity. A deficit index alone reports the
    first half and, presented on its own, implies the second half is absent.
    """
    return (
        f"Standard vulnerability framing: {svi_score:.2f} (higher = more 'vulnerable')\n"
        f"Asset-based capacity framing:   {capacity_index:.2f} (higher = more capacity)\n"
        f"\n"
        f"These are not opposites and both can be high. Material constraints and community "
        f"capacity coexist; a reservation with high poverty rates may also have a fire program "
        f"with deeper local knowledge than any surrounding agency and kinship networks that "
        f"evacuate faster than a formal system.\n"
        f"\n"
        f"Reporting only the first number is not a neutral simplification. It produces a "
        f"description in which the community appears as a problem to be managed rather than "
        f"a partner with capabilities, and that description shapes who gets funded to do what."
    )
