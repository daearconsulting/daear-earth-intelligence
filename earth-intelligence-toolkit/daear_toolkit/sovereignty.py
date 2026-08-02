from __future__ import annotations

"""
daear_toolkit.sovereignty
Indigenous data governance as executable code.

Why this is a module and not a README section
Governance commitments that live in prose get skipped under deadline. Every
Tribal data agreement I have seen described as "we were careful" was careful
until the week before a deliverable was due. So the rules here are functions
that raise, checks that fail closed, and defaults that withhold rather than
publish.

That is also the honest answer to the obvious objection: nothing in this file
enforces anything against a determined analyst who deletes the check. It is not
security. It is a design that makes the governance decision *visible at the
point of use* the user has to write the line that says "yes, this is approved,
here is the agreement".

Frameworks this implements
    CARE           Collective benefit, Authority to control, Responsibility,
                   Ethics (Carroll et al. 2020, Global Indigenous Data Alliance)
    OCAP(R)        Ownership, Control, Access, Possession (First Nations
                   Information Governance Centre; a registered trademark of
                   FNIGC, cited here with attribution)
    FAIR           Findable, Accessible, Interoperable, Reusable included
                   because FAIR alone is insufficient and the gap is the point
    IEEE 2890-2025 Recommended practice for provenance of Indigenous data

CARE and FAIR are not alternatives; CARE is what FAIR omits. FAIR asks whether
data can be found and reused. It says nothing about whether it *should* be, or
by whom, or who benefits. An open dataset about a community that the community
did not consent to and cannot control is maximally FAIR and a CARE failure.

A note on scope
This module encodes process, not authority. It cannot tell you what a Nation
would consent to, and no code can. Every function here assumes a relationship
exists and gives it structure; none of them substitute for one.
"""

import datetime as dt
import hashlib
import json
import warnings
from dataclasses import dataclass, field, asdict
from enum import IntEnum


# Publication tiers
class Tier(IntEnum):
    """
    Publication tiers, ordered from most public to most restricted.

    Note the ordering: **higher tier = finer resolution AND tighter control.**
    Communities receive more detail about their own lands, not less.

    An architecture where the public tier is the detailed one and the community
    receives a summary has the sovereignty relationship exactly backwards, and
    it is a surprisingly common accident, it happens whenever the public
    product is built first and the community version is derived from it by
    subtraction.
    """

    PUBLIC = 0      # regional aggregate; no sub-jurisdictional structure
    PARTNER = 1     # sub-regional, under an executed agreement
    COMMUNITY = 2   # native resolution, held by the Nation
    RESTRICTED = 3  # cultural resources; not held in analytical systems at all


TIER_RULES = {
    Tier.PUBLIC: {
        "spatial": "jurisdiction-level aggregate (whole reservation or larger)",
        "temporal": "decadal or multi-year",
        "audience": "open repository, publication, accelerator demo",
        "requires": "public federal data sources only; Nation notified before release",
    },
    Tier.PARTNER: {
        "spatial": "sub-jurisdictional (district, watershed, community area)",
        "temporal": "annual",
        "audience": "named agency or academic partners",
        "requires": "executed data-use agreement naming the partner and the purpose",
    },
    Tier.COMMUNITY: {
        "spatial": "native raster resolution (10-30 m)",
        "temporal": "daily to seasonal",
        "audience": "the Nation whose lands are described",
        "requires": "Nation holds the data and the release authority",
    },
    Tier.RESTRICTED: {
        "spatial": "not applicable -- not stored",
        "temporal": "not applicable",
        "audience": "knowledge holders as determined by the Nation",
        "requires": "these data do not enter analytical systems; see CulturalResourceGuard",
    },
}


# Data use agreements
@dataclass
class DataUseAgreement:
    """
    A record of an executed agreement, carried through the analysis.

    Every field is required except `notes` and `restrictions`, and that is
    deliberate. An agreement that cannot name its parties, its purpose, its
    expiry, and who holds release authority is not an agreement it is an
    understanding, and understandings are what people remember differently two
    years later when someone wants to publish.

    `expires` is required for the same reason. Open-ended consent is not
    consent; it is a permission someone gave once, to a project that has since
    changed, that nobody can revisit because there is no renewal moment.
    """

    nation: str
    counterparty: str
    purpose: str
    executed: str                    # ISO date
    expires: str                     # ISO date
    release_authority: str           # who decides what gets published
    approved_tiers: tuple[Tier, ...]
    data_scope: tuple[str, ...]      # which datasets, explicitly enumerated
    restrictions: tuple[str, ...] = ()
    notes: str = ""

    def __post_init__(self):
        for f in ("nation", "counterparty", "purpose", "executed", "expires", "release_authority"):
            if not getattr(self, f):
                raise ValueError(f"DataUseAgreement requires a non-empty '{f}'")
        if not self.approved_tiers:
            raise ValueError("An agreement approving no tiers approves nothing; state the tiers explicitly.")

    @property
    def is_current(self) -> bool:
        return dt.date.fromisoformat(self.expires) >= dt.date.today()

    @property
    def days_until_expiry(self) -> int:
        return (dt.date.fromisoformat(self.expires) - dt.date.today()).days

    def permits(self, tier: Tier, dataset: str | None = None) -> bool:
        if not self.is_current:
            return False
        if tier not in self.approved_tiers:
            return False
        # Scope is enumerated, not open-ended: an agreement covering fire history
        # does not silently cover housing data because both are "fire related".
        if dataset is not None and dataset not in self.data_scope:
            return False
        return True

    def summary(self) -> str:
        status = "CURRENT" if self.is_current else "EXPIRED"
        return (f"{self.nation} <-> {self.counterparty} | {status} "
                f"(expires {self.expires}, {self.days_until_expiry:+d} days) | "
                f"tiers: {[t.name for t in self.approved_tiers]}")


@dataclass
class GovernanceContext:
    """
    The governance state an analysis runs under. Pass this to anything that
    reads, derives from, or publishes Indigenous data.

    `agreement=None` is a valid and common state it means public federal data
    only, PUBLIC tier only. That is the default this repo ships in, and it is
    why the demo notebooks run at all without an agreement in hand.
    """

    nation: str
    agreement: DataUseAgreement | None = None
    public_release_approved: bool = False
    approval_record: str = ""
    _log: list = field(default_factory=list, repr=False)

    def max_tier(self) -> Tier:
        return max(self.agreement.approved_tiers) if self.agreement and self.agreement.is_current else Tier.PUBLIC

    def check(self, tier: Tier, dataset: str | None = None, action: str = "access") -> None:
        """
        Gate an operation. Raises PermissionError if not permitted.

        Raising rather than returning False is the whole design: a boolean gets
        ignored, an exception stops the notebook. The failure should be loud and
        it should happen at the point of use.
        """
        entry = {"when": dt.datetime.now().isoformat(timespec="seconds"),
                 "tier": tier.name, "dataset": dataset, "action": action}

        if tier == Tier.RESTRICTED:
            entry["result"] = "DENIED"
            self._log.append(entry)
            raise PermissionError(
                "RESTRICTED-tier data (cultural resources, sacred sites, burial grounds, "
                "ceremonial locations) does not enter analytical systems. This is not a "
                "permissions question that an agreement can resolve; the protection is "
                "in not holding the data. See CulturalResourceGuard."
            )

        if tier > Tier.PUBLIC:
            if self.agreement is None:
                entry["result"] = "DENIED"
                self._log.append(entry)
                raise PermissionError(
                    f"{tier.name}-tier access requires an executed data-use agreement with "
                    f"{self.nation}. None is recorded in this context. "
                    f"Public federal data at PUBLIC tier is available without one."
                )
            if not self.agreement.permits(tier, dataset):
                entry["result"] = "DENIED"
                self._log.append(entry)
                reason = ("agreement has expired" if not self.agreement.is_current
                          else f"dataset '{dataset}' is outside the agreed scope" if dataset and dataset not in self.agreement.data_scope
                          else f"{tier.name} tier not among approved tiers")
                raise PermissionError(f"Not permitted: {reason}. {self.agreement.summary()}")

        entry["result"] = "PERMITTED"
        self._log.append(entry)

    def check_publication(self, what: str = "analysis outputs") -> None:
        """
        Gate public release. **Fails closed** approval must be recorded
        explicitly, with a reference to how it was obtained.

        The distinction this enforces: using public federal data does not
        require anyone's permission, but *publishing an analysis about a Nation*
        is a governance decision regardless of where the inputs came from. The
        aggregate product is a new thing. Source licensing does not transfer
        release authority over it.
        """
        if not self.public_release_approved:
            raise PermissionError(
                f"Public release of {what} concerning {self.nation} is not approved in this context.\n"
                f"\n"
                f"Federal open data as an input does not make the OUTPUT public by default. "
                f"An analysis about a Nation's lands is a new artifact and its release is a "
                f"governance decision.\n"
                f"\n"
                f"To proceed, obtain sign-off from the Nation's designated authority and record it:\n"
                f"    ctx.public_release_approved = True\n"
                f"    ctx.approval_record = '<who approved, when, in what forum>'\n"
                f"\n"
                f"If you are about to set these without having obtained sign-off, that is the "
                f"moment this check exists for."
            )
        if not self.approval_record:
            raise PermissionError(
                "public_release_approved is set but approval_record is empty. "
                "Record who approved release, when, and in what forum an unattributed "
                "approval cannot be verified later and will not survive a partner asking."
            )

    def audit_log(self):
        """Every gated operation attempted in this context, permitted or denied."""
        import pandas as pd
        return pd.DataFrame(self._log)


# Cultural resources
# Column-name fragments that suggest a dataset contains culturally sensitive
# location information. Deliberately over-inclusive: a false positive costs one
# explicit override, a false negative costs something that cannot be undone.
_SENSITIVE_HINTS = (
    "sacred", "ceremon", "burial", "cemeter", "grave", "ancestral", "tcp",
    "traditional_cultural", "petroglyph", "pictograph", "rock_art", "archaeolog",
    "artifact", "nagpra", "shrine", "medicine_wheel", "sundance", "sun_dance",
    "vision_quest", "gathering_site", "plant_gather", "sensitive_site",
)


class CulturalResourceGuard:
    """
    Refuses to process datasets that appear to contain cultural site locations.

    The premise: **the protection is in not holding the data.** Access controls,
    encryption, and masking all assume the data is in the system and something
    stands between it and disclosure. Every one of those layers has failed
    somewhere. Not collecting the coordinates has not.

    This matters because the archaeological and cultural-resource record has a
    specific history: site location databases assembled for protective purposes
    have repeatedly become the targeting information for looting. That is not a
    hypothetical risk profile.
    """

    @staticmethod
    def scan(gdf, raise_on_match: bool = True) -> list[str]:
        """
        Scan a GeoDataFrame's columns and values for cultural-site indicators.

        Returns the list of matches. Raises by default, because a warning in a
        long notebook is a warning nobody reads.
        """
        matches = []
        for col in gdf.columns:
            low = str(col).lower()
            for hint in _SENSITIVE_HINTS:
                if hint in low:
                    matches.append(f"column '{col}' (matched '{hint}')")

        # Also scan the values of small string columns; a generic 'site_type'
        # column with 'burial' among its values is the same problem as a column
        # named 'burial_sites', and is more common in practice.
        #
        # Do NOT gate this on `dtype == object`. Recent pandas represents string
        # columns as a dedicated `str` dtype, so an object-dtype check silently
        # skips them -- and this guard failing open is the one failure mode it
        # cannot have. Attempt the scan on every non-geometry column and let the
        # exception handler discard the ones that are not text.
        import pandas as _pd

        geom_col = getattr(getattr(gdf, "geometry", None), "name", None)
        for col in gdf.columns:
            if col == geom_col:
                continue
            try:
                series = gdf[col]
                if _pd.api.types.is_numeric_dtype(series) or _pd.api.types.is_datetime64_any_dtype(series):
                    continue
                uniques = series.dropna().unique()
                if len(uniques) > 200:
                    continue
                vals = " ".join(str(v).lower() for v in uniques)
                for hint in _SENSITIVE_HINTS:
                    if hint in vals:
                        matches.append(f"values in column '{col}' (matched '{hint}')")
                        break
            except (TypeError, AttributeError, ValueError):
                continue

        if matches and raise_on_match:
            raise PermissionError(
                "This dataset appears to contain cultural resource location information:\n  "
                + "\n  ".join(sorted(set(matches)))
                + "\n\nThis data does not belong in an analytical pipeline or a public repository. "
                  "If it was provided to you, return it to the Nation and confirm deletion. "
                  "If protection of these locations is the goal, the correct approach is a "
                  "Nation-held avoidance buffer applied on their side they mask, then share "
                  "the mask. You never hold the points.\n\n"
                  "If this is a genuine false positive, pass raise_on_match=False and document "
                  "the review in your data management plan."
            )
        return sorted(set(matches))

    @staticmethod
    def masking_is_insufficient_because() -> str:
        """
        Why geographic masking is not an answer for cultural sites.

        This is a method, not a comment, so it can be cited in a data management
        plan and so nobody has to reconstruct the argument under time pressure.
        """
        return (
            "Geographic masking (jittering, aggregation, random displacement) is standard "
            "for health and household privacy and is not sufficient for cultural sites, for "
            "four reasons:\n"
            " 1. Re-identification. Sites correlate strongly with terrain i.e. ridgelines, "
            "    springs, confluences, specific landforms. A jittered point plus a DEM "
            "    narrows the search area dramatically; the terrain does the deanonymizing.\n"
            " 2. Aggregation across releases. Multiple masked versions of the same points, "
            "    or a masked layer alongside an unmasked buffer, can be intersected to "
            "    recover the original locations.\n"
            " 3. Presence is itself sensitive. Even a coarse polygon saying 'there are sites "
            "    in this drainage' directs attention to that drainage.\n"
            " 4. It is not the Nation's decision to delegate. Choosing a masking radius is "
            "    an authority-to-control question, and an analyst picking 1 km because it "
            "    seemed adequate has made a sovereignty decision without the sovereign.\n"
            "\n"
            "The workable pattern: the Nation applies avoidance buffers on their side and "
            "shares only the resulting mask a 'do not treat here' polygon with no "
            "indication of why. The analysis consumes the mask. The points never leave."
        )


# CARE assessment
CARE_QUESTIONS = {
    "Collective Benefit": [
        "Does the Nation define what benefit means here, or does the analyst?",
        "Does a usable product reach the community, or only a publication and a repo?",
        "Is capacity transferred does someone in the Nation end up able to run and modify this?",
        "Who holds the outputs when the funding ends?",
    ],
    "Authority to Control": [
        "Has the Nation's designated authority reviewed the questions being asked, not just the methods?",
        "Can the Nation halt, amend, or withdraw from the work after it has started?",
        "Who decides what is published, at what resolution, and in what venue?",
        "Are derived products and models covered by the agreement, or only the raw data?",
    ],
    "Responsibility": [
        "Are limitations stated in terms a non-specialist reader can act on?",
        "Is the analysis reviewed by people with ground knowledge before release?",
        "Have foreseeable harms from misinterpretation been named, including by third parties?",
        "Is there a correction and retraction path if the analysis turns out to be wrong?",
    ],
    "Ethics": [
        "Does the framing avoid deficit narratives is the community described as capable?",
        "Are Indigenous knowledge contributions attributed as knowledge, or extracted as 'input data'?",
        "Would the community recognize itself in how it is described?",
        "Is the relationship ongoing, or does it end at the deliverable?",
    ],
}


def care_assessment(responses: dict | None = None, verbose: bool = True) -> dict:
    """
    Work through the CARE principles as a structured checklist.

    `responses` maps question text to True/False/None. Unanswered questions
    count as unmet, not as neutral an unexamined governance question is an
    unmet one, and scoring it as neutral produces a comfortable middling number
    that means nothing.

    This is a structured prompt for a conversation with the Nation, not a
    compliance score to put in a proposal. A high number here means the
    questions were asked; it does not mean the answers were good, and it
    certainly does not certify anything.
    """
    responses = responses or {}
    report = {}
    for principle, questions in CARE_QUESTIONS.items():
        answered = [responses.get(q) for q in questions]
        met = sum(1 for a in answered if a is True)
        unaddressed = [q for q, a in zip(questions, answered) if a is None]
        failed = [q for q, a in zip(questions, answered) if a is False]
        report[principle] = {
            "met": met, "total": len(questions),
            "unaddressed": unaddressed, "not_met": failed,
        }

    total_met = sum(r["met"] for r in report.values())
    total_q = sum(r["total"] for r in report.values())
    report["_overall"] = {"met": total_met, "total": total_q, "fraction": round(total_met / total_q, 2)}

    if verbose:
        print("CARE PRINCIPLES ASSESSMENT")
        for principle, r in report.items():
            if principle.startswith("_"):
                continue
            print(f"\n{principle}: {r['met']}/{r['total']}")
            for q in r["not_met"]:
                print(f"   NOT MET      {q}")
            for q in r["unaddressed"]:
                print(f"   UNADDRESSED  {q}")
        print("\n" )
        print(f"Overall: {total_met}/{total_q} ({report['_overall']['fraction']:.0%})")
        print("\nThis is a prompt for a conversation, not a certification. A high score means")
        print("the questions were asked. It says nothing about whether the answers were good.")

    return report


# Provenance and attribution
@dataclass
class ProvenanceRecord:
    """
    Provenance for a derived product, in the spirit of IEEE 2890-2025.

    The field that distinguishes this from ordinary lineage metadata is
    `indigenous_knowledge_contributions`. Standard provenance tracks datasets
    and code. When Indigenous knowledge shapes an analysis which watersheds
    matter, what counts as a healthy landscape, where fire historically moved 
    that contribution is usually absorbed into the method and disappears from
    the record. Naming it is both an attribution obligation and a factual
    correction to the lineage.
    """

    product: str
    created: str
    creator: str
    nation: str
    data_sources: tuple[str, ...]
    methods: tuple[str, ...]
    tier: Tier
    agreement_ref: str | None = None
    indigenous_knowledge_contributions: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()

    def fingerprint(self) -> str:
        payload = json.dumps({k: str(v) for k, v in asdict(self).items()}, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def to_markdown(self) -> str:
        lines = [
            f"# Provenance: {self.product}",
            "",
            f"- **Created:** {self.created} by {self.creator}",
            f"- **Concerning:** {self.nation}",
            f"- **Publication tier:** {self.tier.name}",
            f"- **Agreement:** {self.agreement_ref or 'none (public federal data, PUBLIC tier)'}",
            f"- **Fingerprint:** `{self.fingerprint()}`",
            "",
            "## Data sources",
            *[f"- {s}" for s in self.data_sources],
            "",
            "## Methods",
            *[f"- {m}" for m in self.methods],
        ]
        if self.indigenous_knowledge_contributions:
            lines += ["", "## Indigenous knowledge contributions",
                      "*Attributed as knowledge contributions, not as input data.*", ""]
            lines += [f"- {c}" for c in self.indigenous_knowledge_contributions]
        if self.limitations:
            lines += ["", "## Limitations", *[f"- {l}" for l in self.limitations]]
        return "\n".join(lines)


def data_acknowledgment(nation: str, sources, agreement: DataUseAgreement | None = None) -> str:
    """
    Acknowledgment text for any product derived from work on Tribal lands.

    Names the Nation first, then the federal sources. The ordering is not
    decoration: it states whose lands the analysis concerns before it states
    which agency collected the pixels.
    """
    parts = [
        f"This analysis concerns lands of the {nation}.",
    ]
    if agreement:
        parts.append(
            f"It was conducted under a data-use agreement executed {agreement.executed}, "
            f"with release authority held by {agreement.release_authority}."
        )
    else:
        parts.append(
            "It uses publicly available federal datasets only. Public availability of the "
            "inputs does not constitute the Nation's endorsement of this analysis or its "
            "conclusions."
        )
    parts.append("Federal data sources: " + "; ".join(sources) + ".")
    parts.append(
        "Interpretations are those of the analyst and should not be attributed to the Nation "
        "or to any Tribal program."
    )
    return " ".join(parts)


def ocap_note() -> str:
    """OCAP(R) attribution: it is a registered trademark and citing it correctly matters."""
    return (
        "OCAP(R): Ownership, Control, Access, and Possession, is a registered trademark of "
        "the First Nations Information Governance Centre (FNIGC). It was developed in and for "
        "a First Nations context in Canada. It is referenced here because its principles are "
        "widely applied, and it should not be assumed to be adopted by, or appropriate for, "
        "any given Tribal Nation in the United States without that Nation saying so. Several "
        "Nations have their own research codes and IRBs that take precedence."
    )
