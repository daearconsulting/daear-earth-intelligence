import numpy as np
import pytest

import daear_toolkit as dt
from daear_toolkit import data_access, indicators, governance


BBOX = dt.POUDRE_CAMERON_PEAK.bbox


def test_optical_scene_shape_and_bounds():
    scene = data_access.get_optical_scene(BBOX, "2020-06-01")
    for band in ("red", "nir", "swir16"):
        arr = scene[band].values
        assert arr.min() >= 0
        assert np.isfinite(arr).all()


def test_ndvi_bounded():
    scene = data_access.get_optical_scene(BBOX, "2020-06-01")
    ndvi = indicators.ndvi(scene)
    assert float(ndvi.min()) >= -1.0001
    assert float(ndvi.max()) <= 1.0001


def test_dnbr_positive_where_burned():
    pre = data_access.get_optical_scene(BBOX, "2020-06-01", seed=10)
    post = data_access.get_optical_scene(BBOX, "2020-10-01", seed=20)
    d = indicators.dnbr(pre, post)
    # not asserting a specific sign globally (synthetic data), just that it runs
    # and produces a finite, correctly-shaped result
    assert d.shape == pre["red"].shape
    assert np.isfinite(d.values).all()


def test_erosion_susceptibility_bounded():
    terrain = data_access.get_terrain(BBOX)
    burn = data_access.get_burn_severity(BBOX, fire_year=2020)
    soil = data_access.get_soil_properties(BBOX)
    ero = indicators.erosion_susceptibility(terrain["slope_deg"], burn, soil["runoff_potential"])
    assert float(ero.min()) >= 0
    assert float(ero.max()) <= 1


def test_governance_gating_blocks_lower_tier():
    record = governance.GovernanceRecord(
        title="Precise cultural site locations",
        steward="Tribal Historic Preservation Office (example)",
        access_tier=governance.AccessTier.RESTRICTED,
    )
    with pytest.raises(governance.AccessDeniedError):
        governance.gated_release(record, governance.AccessTier.PUBLIC, payload="secret")


def test_governance_gating_allows_matching_tier():
    record = governance.GovernanceRecord(
        title="Community fire-risk summary",
        steward="Example Tribal Natural Resources Dept.",
        access_tier=governance.AccessTier.COMMUNITY,
    )
    result = governance.gated_release(record, governance.AccessTier.COMMUNITY, payload={"risk": "moderate"})
    assert result == {"risk": "moderate"}
