"""Sample novel parse trees from the mined non-terminal families.

Walks each family as a choice point: pick a production weighted by its
recorded usage from the training parses + a Dirichlet prior
(``SynthesisConfig.dirichlet_concentration``), then recurse into each
child through its own family. Terminates at a terminal rule or at a hard
depth cap (``max_depth * 2``) which forces a fallback terminal.

Position semantics are NOT modelled at the grammar level — this matches
Martinović CVPR 2013 §4 / Wu & Wonka TOG 2014 / Kozinski 2016. The
rendering layer (``viz/synthesis.py``) applies a canonical-order
cosmetic pass so gables float up and doors sink to the bottom at draw
time, same as those papers do in their figure-generation pipelines.

The output ``parse_tree`` references concrete rules from the source
``InducedGrammar`` — same renderer as ``facade_parses``.
"""

import numpy as np
from hamilton.function_modifiers import tag

from facade_grammar.config import SynthesisConfig
from facade_grammar.schemas.grammar import (
    GrammarFamilies,
    InducedGrammar,
    NonTerminalFamily,
    ParseNode,
    Rule,
    SplitRule,
    SynthesizedFacade,
    TerminalRule,
)

# Hard recursion cap — a family with zero terminal-routed productions could
# otherwise loop forever. At 2x max_depth we force a leaf using any available
# TerminalRule, guaranteeing termination.
_HARD_DEPTH_MULTIPLIER = 2


@tag(stage="synthesis")
def synthesized_facades(
    grammar_families: GrammarFamilies,
    induced_grammar: InducedGrammar,
    synthesis: SynthesisConfig,
) -> list[SynthesizedFacade]:
    """Sample ``n_samples`` novel parse trees from the mined families."""
    if not induced_grammar.rules or not grammar_families.root_family_weights:
        return []

    fallback_terminal = _first_terminal_name(induced_grammar.rules)
    seed_seq = np.random.SeedSequence(synthesis.rng_seed)
    child_seeds = seed_seq.spawn(synthesis.n_samples)
    samples: list[SynthesizedFacade] = []
    for idx, child_seq in enumerate(child_seeds):
        rng = np.random.default_rng(child_seq)
        trace: list[str] = []
        # Root family selection stays Dirichlet-free: root families have few
        # candidates and smoothing distorts the narrow choice.
        root_items = list(grammar_families.root_family_weights.items())
        root_weights = np.asarray([w for _, w in root_items], dtype=np.float64)
        root_probs = _smoothed_probs(root_weights, alpha=0.0, denom_n=len(root_items))
        root_family = _choice_from_probs([name for name, _ in root_items], root_probs, rng)
        trace.append(root_family)
        tree = _sample_tree(
            root_family,
            grammar_families,
            induced_grammar.rules,
            rng,
            depth=0,
            max_depth=synthesis.max_depth,
            trace=trace,
            min_weight=synthesis.min_production_weight,
            fallback_terminal=fallback_terminal,
            alpha=synthesis.dirichlet_concentration,
        )
        # ``seed`` is a lineage tag derived from (rng_seed, sample_index);
        # not sufficient to reproduce the sample in isolation — that requires
        # the same ``SynthesisConfig`` + sample index, because the sampler is
        # driven by the Generator seeded from ``child_seq``.
        samples.append(
            SynthesizedFacade(
                seed=synthesis.rng_seed * 10_000 + idx,
                family_trace=tuple(trace),
                parse_tree=tree,
            )
        )
    return samples


_ARCHETYPE_TOP_K = 6
_ARCHETYPE_SAMPLES_PER_FAMILY = 2


@tag(stage="synthesis")
def photogenic_facades(
    grammar_families: GrammarFamilies,
    induced_grammar: InducedGrammar,
    synthesis: SynthesisConfig,
) -> list[SynthesizedFacade]:
    """``synthesis.n_samples`` facades restricted to the single top root family.

    Used for a blog header image — every tile is a variation of the canonical
    canal-house archetype rather than a broad tour of the corpus.
    """
    if not grammar_families.root_family_weights:
        return []
    top_name, top_weight = max(grammar_families.root_family_weights.items(), key=lambda kv: kv[1])
    narrowed = grammar_families.model_copy(update={"root_family_weights": {top_name: top_weight}})
    return synthesized_facades(narrowed, induced_grammar, synthesis)


@tag(stage="synthesis")
def archetype_facades(
    grammar_families: GrammarFamilies,
    induced_grammar: InducedGrammar,
    synthesis: SynthesisConfig,
) -> list[SynthesizedFacade]:
    """Two samples from each of the top-6 root families — a structural tour.

    Shows that canal-house facades cluster into a handful of archetypes.
    Each family gets its own RNG seed so samples differ across archetypes
    even if the inner sampler would otherwise produce identical draws.
    """
    if not grammar_families.root_family_weights:
        return []
    ranked = sorted(
        grammar_families.root_family_weights.items(), key=lambda kv: kv[1], reverse=True
    )
    samples: list[SynthesizedFacade] = []
    for idx, (fam_name, fam_weight) in enumerate(ranked[:_ARCHETYPE_TOP_K]):
        narrowed = grammar_families.model_copy(
            update={"root_family_weights": {fam_name: fam_weight}}
        )
        # Prime offset decorrelates per-family sample seeds.
        per_fam_cfg = synthesis.model_copy(
            update={
                "n_samples": _ARCHETYPE_SAMPLES_PER_FAMILY,
                "rng_seed": synthesis.rng_seed + idx * 1009,
            }
        )
        samples.extend(synthesized_facades(narrowed, induced_grammar, per_fam_cfg))
    return samples


def _sample_tree(
    family_name: str,
    families: GrammarFamilies,
    rules: dict[str, Rule],
    rng: np.random.Generator,
    *,
    depth: int,
    max_depth: int,
    trace: list[str],
    min_weight: int = 1,
    fallback_terminal: str | None = None,
    alpha: float = 0.0,
) -> ParseNode:
    # Hard cap: a family with no terminal-routed production can otherwise loop.
    if fallback_terminal is not None and depth >= max_depth * _HARD_DEPTH_MULTIPLIER:
        return ParseNode(rule_name=fallback_terminal, children=())
    family = families.families[family_name]
    production = _pick_production(
        family, families, rules, rng, depth, max_depth, min_weight=min_weight, alpha=alpha
    )
    rule = rules[production]
    if isinstance(rule, TerminalRule):
        return ParseNode(rule_name=production, children=())
    assert isinstance(rule, SplitRule)
    child_nodes: list[ParseNode] = []
    for child in rule.children:
        child_family = families.rule_to_family[child.rule_name]
        trace.append(child_family)
        child_nodes.append(
            _sample_tree(
                child_family,
                families,
                rules,
                rng,
                depth=depth + 1,
                max_depth=max_depth,
                trace=trace,
                min_weight=min_weight,
                fallback_terminal=fallback_terminal,
                alpha=alpha,
            )
        )
    return ParseNode(rule_name=production, children=tuple(child_nodes))


def _first_terminal_name(rules: dict[str, Rule]) -> str | None:
    for name, rule in rules.items():
        if isinstance(rule, TerminalRule):
            return name
    return None


def _pick_production(
    family: NonTerminalFamily,
    families: GrammarFamilies,
    rules: dict,
    rng: np.random.Generator,
    depth: int,
    max_depth: int,
    min_weight: int = 1,
    alpha: float = 0.0,
) -> str:
    """Pick a production for a family via Dirichlet-smoothed sampling.

    Filter chain, in order:
    1. Drop productions with total weight < ``min_weight`` (fallback to full
       set if that would empty the family).
    2. At ``depth >= max_depth`` prefer productions whose children all route
       to terminals (fallback to previous set if none qualify).
    3. Apply Dirichlet smoothing with concentration ``alpha`` and denominator
       N = full family production count. ``alpha=0`` reproduces MLE exactly.
    """
    full_n = len(family.productions)
    candidates = list(zip(family.productions, family.weights, strict=True))
    pruned = [pair for pair in candidates if pair[1] >= min_weight]
    if pruned:
        candidates = pruned
    if depth >= max_depth:
        terminal_routed = [pair for pair in candidates if _routes_to_terminals(pair[0], rules)]
        if terminal_routed:
            candidates = terminal_routed
    weights = np.asarray([w for _, w in candidates], dtype=np.float64)
    probs = _smoothed_probs(weights, alpha=alpha, denom_n=full_n)
    names = [name for name, _ in candidates]
    return _choice_from_probs(names, probs, rng)


def _routes_to_terminals(rule_name: str, rules: dict) -> bool:
    """A rule whose immediate children are all terminals (one hop)."""
    rule = rules[rule_name]
    if isinstance(rule, TerminalRule):
        return True
    assert isinstance(rule, SplitRule)
    return all(isinstance(rules[c.rule_name], TerminalRule) for c in rule.children)


def _smoothed_probs(weights: np.ndarray, *, alpha: float, denom_n: int) -> np.ndarray:
    """Dirichlet-posterior-mean: ``(w_i + alpha) / (sum_w + N * alpha)``.

    ``denom_n`` is the full family production count (not the filtered length)
    so smoothing magnitude stays stable across different filter passes.
    Falls back to uniform when ``alpha == 0`` and ``sum_w == 0``.
    """
    total = float(weights.sum())
    denom = total + denom_n * alpha
    if denom <= 0.0:
        return np.ones_like(weights) / len(weights)
    return (weights + alpha) / denom


def _choice_from_probs(names: list[str], probs: np.ndarray, rng: np.random.Generator) -> str:
    return str(rng.choice(names, p=probs))
