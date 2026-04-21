"""Post-hoc aggregation of per-facade outcomes into an enriched audit log.

Lives as its own stage rather than inside ``pipeline.selection`` because it
consumes ``facade_works`` (from vision) *and* raw_photo_metadata +
canal_facades (from ingestion/spatial) to join winner-perpendicularity and
occluder-ratio onto each record. Running this downstream of vision means
adding new audit fields does NOT invalidate ``chosen_work_for_facade``'s SAM
cache.
"""

from hamilton.function_modifiers import tag

from facade_grammar.pipeline.selection import (
    batch_facade_geoms,
    perpendicularity_deg,
    reproject_photos,
)
from facade_grammar.schemas.buildings import AuditRecord, Facade, FacadeWork
from facade_grammar.schemas.photos import PhotoMetadata


@tag(stage="audit")
def audit_records(
    facade_works: dict[str, FacadeWork],
    raw_photo_metadata: list[PhotoMetadata],
    canal_facades: list[Facade],
) -> list[AuditRecord]:
    """Enrich each worker's AuditRecord with the winner's perpendicularity_deg
    and occluder_ratio."""
    photos_rd = reproject_photos(raw_photo_metadata)
    photo_rd_by_id = {p.photo_id: rd for p, rd in zip(raw_photo_metadata, photos_rd, strict=True)}
    facades_by_bid = {
        f.building_id: (f.normal_deg, midpoint)
        for f, (_, midpoint) in zip(canal_facades, batch_facade_geoms(canal_facades), strict=True)
    }
    out: list[AuditRecord] = []
    for bid, work in facade_works.items():
        perp = _winner_perpendicularity(work.audit, bid, facades_by_bid, photo_rd_by_id)
        occluder_ratio = work.mask.occluder_ratio if work.mask is not None else None
        out.append(
            work.audit.model_copy(
                update={
                    "perpendicularity_deg": perp,
                    "occluder_ratio": occluder_ratio,
                }
            )
        )
    return sorted(out, key=lambda r: r.building_id)


def _winner_perpendicularity(
    audit: AuditRecord,
    building_id: str,
    facades_by_bid: dict[str, tuple[float, tuple[float, float]]],
    photo_rd_by_id: dict[str, tuple[float, float]],
) -> float | None:
    if not audit.photo_id or building_id not in facades_by_bid:
        return None
    rd = photo_rd_by_id.get(audit.photo_id)
    if rd is None:
        return None
    normal_deg, midpoint = facades_by_bid[building_id]
    return perpendicularity_deg(normal_deg, midpoint, rd)
