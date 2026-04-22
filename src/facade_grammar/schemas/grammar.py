"""Pydantic types for the induced split grammar.

Produced by ``pipeline.induction.induced_grammar`` and serialised via
``pydantic.TypeAdapter`` at the output boundary. All records are frozen —
Hamilton fingerprints inputs by identity, so mutating an ``InducedGrammar``
after induction would silently break downstream cache entries.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from facade_grammar.schemas.base import FrozenModel
from facade_grammar.schemas.buildings import FeatureClass


class SplitChild(FrozenModel):
    """One child slot inside a split rule, sized as a fraction of the parent dim."""

    rule_name: str
    size: float = Field(gt=0.0, le=1.0)


class TerminalRule(FrozenModel):
    """Uniform region of one class."""

    kind: Literal["terminal"] = "terminal"
    name: str
    cls: FeatureClass | Literal["wall"]


class SplitRule(FrozenModel):
    """Axis-aligned split of a region into ``children``. Binary in Phase 1
    (exactly two children); Phase 2 would collapse chains into n-ary."""

    kind: Literal["split"] = "split"
    name: str
    axis: Literal["horizontal", "vertical"]
    children: list[SplitChild]


Rule = Annotated[TerminalRule | SplitRule, Field(discriminator="kind")]


class ParseNode(FrozenModel):
    """One node of a per-facade parse tree, referencing a rule by name.

    Terminal parses have no children. Split-rule parses carry one ``ParseNode``
    per ``SplitChild`` in the rule, in the same order. ``children`` is a tuple
    so the node itself is hashable (useful for downstream deduplication).
    """

    rule_name: str
    children: tuple[ParseNode, ...] = ()


class InducedGrammar(FrozenModel):
    """Shared split grammar discovered across the corpus.

    Variable facade shapes mean there is no single global root rule — each
    building's root sits in ``facade_roots``. Rule reuse happens at the
    sub-region level via the shared memo table inside ``_solve``.
    """

    rules: dict[str, Rule]
    facade_roots: dict[str, str]
    mdl_bits: float


class NonTerminalFamily(FrozenModel):
    """A group of concrete rules that share the same structural signature.

    Families abstract over ``SplitChild.size``: two rules with the same axis
    and structurally-equivalent children (to the configured depth) belong to
    the same family regardless of concrete size ratios. Each family acts as
    a non-terminal with multiple productions weighted by usage.
    """

    name: str
    key: tuple[str, ...]
    productions: tuple[str, ...]
    weights: tuple[int, ...]


class GrammarFamilies(FrozenModel):
    """Non-terminal families mined from ``InducedGrammar`` + parse trees."""

    families: dict[str, NonTerminalFamily]
    rule_to_family: dict[str, str]
    root_family_weights: dict[str, int]


class SynthesizedFacade(FrozenModel):
    """One facade sampled from the mined families.

    ``parse_tree`` references concrete rules from the source ``InducedGrammar``
    so rendering reuses the same size-resolution path as ``facade_parses``.
    ``family_trace`` records the families visited during sampling, root-first,
    for debugging lineage.
    """

    seed: int
    family_trace: tuple[str, ...] = ()
    parse_tree: ParseNode


ParseNode.model_rebuild()
