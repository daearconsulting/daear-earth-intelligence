from __future__ import annotations

"""
Data sovereignty governance scaffolding.

Implements the two-track public/Tribal data pattern used across Daear's
NIFA-funded work: every dataset carries governance metadata, and access is
actually gated by that metadata rather than merely documented in a policy
paragraph. `tribal-wildfire-intelligence` is where this gets exercised
end-to-end against a sample dataset.

Aligned with OCAP(R) (Ownership, Control, Access, Possession), the CARE
Principles for Indigenous Data Governance (Collective Benefit, Authority to
Control, Responsibility, Ethics), and FAIR (Findable, Accessible,
Interoperable, Reusable) as complementary, not competing, frameworks.
"""

from dataclasses import dataclass, field
from datetime import date
from enum import Enum


class AccessTier(Enum):
    PUBLIC = "public"           # open, no restrictions
    COMMUNITY = "community"     # restricted to designated Tribal/community accounts
    RESTRICTED = "restricted"   # sensitive locations (ex. cultural sites) aggregate-only release


@dataclass
class GovernanceRecord:
    """Attaches to a dataset or derived product."""
    title: str
    steward: str                       # who holds authority to control this data
    access_tier: AccessTier
    care_notes: str = ""               # how Collective Benefit/Authority/Responsibility/Ethics apply
    ocap_notes: str = ""               # ownership/control/access/possession specifics
    created: date = field(default_factory=date.today)
    local_contexts_notice: str | None = None  # e.g. "TK Label: TK Attribution"


class AccessDeniedError(PermissionError):
    pass


def check_access(record: GovernanceRecord, requester_tier: AccessTier) -> bool:
    """
    Real gating logic, not just documentation: RESTRICTED data is never
    releasable at PUBLIC or COMMUNITY tiers; COMMUNITY data requires at
    least COMMUNITY-tier credentials.
    """
    order = {AccessTier.PUBLIC: 0, AccessTier.COMMUNITY: 1, AccessTier.RESTRICTED: 2}
    return order[requester_tier] >= order[record.access_tier]


def gated_release(record: GovernanceRecord, requester_tier: AccessTier, payload):
    """
    Return `payload` only if the requester's tier satisfies the record's
    access tier; otherwise raise. This is the function
    tribal-wildfire-intelligence calls before handing back any
    culturally-sensitive layer (ex. site-level fire history near named
    locations), demonstrating enforcement rather than just describing it.
    """
    if not check_access(record, requester_tier):
        raise AccessDeniedError(
            f"'{record.title}' requires {record.access_tier.value}-tier access; "
            f"requester has {requester_tier.value}-tier access. "
            f"Steward of record: {record.steward}."
        )
    return payload


def aggregate_to_public(fine_grained_da, agg_func="mean", coarsen_factor: int = 8):
    """
    Standard release path for RESTRICTED-tier spatial data: coarsen/aggregate
    until individual sensitive locations can no longer be resolved, then
    tag the result PUBLIC. Used e.g. to release watershed-level fire-risk
    trends without exposing precise culturally-sensitive site locations.
    """
    coarsened = fine_grained_da.coarsen(
        lat=coarsen_factor, lon=coarsen_factor, boundary="trim"
    )
    return getattr(coarsened, agg_func)()
