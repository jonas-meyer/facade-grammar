"""Shared base class for frozen Pydantic domain records.

Hamilton fingerprints inputs by identity for cache keys; mutating a model
after validation would silently corrupt downstream cache entries. Every
domain record in ``schemas.buildings`` / ``schemas.grammar`` / ``schemas.photos``
inherits from ``FrozenModel`` so the invariant is enforced once instead
of repeated on every class.
"""

from pydantic import BaseModel, ConfigDict


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True)
