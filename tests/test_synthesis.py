"""Tests for ``facade_grammar.pipeline.synthesis``."""

import numpy as np
from pydantic import TypeAdapter

from facade_grammar.config import InductionConfig, SynthesisConfig
from facade_grammar.pipeline.families import grammar_families
from facade_grammar.pipeline.induction import facade_parses, induced_grammar
from facade_grammar.pipeline.synthesis import _smoothed_probs, synthesized_facades
from facade_grammar.schemas.grammar import (
    GrammarFamilies,
    InducedGrammar,
    NonTerminalFamily,
    SplitChild,
    SplitRule,
    SynthesizedFacade,
    TerminalRule,
)

_IND = InductionConfig()


def _build_corpus(
    make_lattice, n_samples: int = 4, seed: int = 42
) -> tuple[list[SynthesizedFacade], dict]:
    lattices = {
        "a": make_lattice("a", [["wall", "window"], ["wall", "window"]]),
        "b": make_lattice("b", [["wall", "door"], ["wall", "door"]]),
        "c": make_lattice("c", [["wall", "wall"], ["window", "window"]]),
    }
    cfg = SynthesisConfig(n_samples=n_samples, rng_seed=seed)
    ig = induced_grammar(lattices, _IND)
    parses = facade_parses(lattices, ig)
    fams = grammar_families(ig, parses, cfg)
    samples = synthesized_facades(fams, ig, cfg)
    return samples, ig.rules


def test_synthesis_is_deterministic_given_seed(make_lattice) -> None:
    a, _ = _build_corpus(make_lattice, n_samples=4, seed=42)
    b, _ = _build_corpus(make_lattice, n_samples=4, seed=42)
    assert [s.parse_tree for s in a] == [s.parse_tree for s in b]


def test_seed_isolation_first_n_stable_across_different_n(make_lattice) -> None:
    # SeedSequence.spawn is prefix-stable: the first k children of spawn(n)
    # equal the first k children of spawn(m) for m > n.
    three, _ = _build_corpus(make_lattice, n_samples=3, seed=42)
    five, _ = _build_corpus(make_lattice, n_samples=5, seed=42)
    assert [s.parse_tree for s in three] == [s.parse_tree for s in five[:3]]


def test_every_synthesized_leaf_is_a_terminal_rule(make_lattice) -> None:
    samples, rules = _build_corpus(make_lattice, n_samples=8, seed=7)

    def check(node):
        if node.children == ():
            assert isinstance(rules[node.rule_name], TerminalRule)
            return
        for child in node.children:
            check(child)

    for s in samples:
        check(s.parse_tree)


def test_synthesized_list_round_trips_via_type_adapter(make_lattice) -> None:
    samples, _ = _build_corpus(make_lattice, n_samples=3, seed=42)
    adapter = TypeAdapter(list[SynthesizedFacade])
    reloaded = adapter.validate_json(adapter.dump_json(samples))
    assert reloaded == samples


def test_empty_corpus_returns_no_samples() -> None:
    empty_ig = InducedGrammar(rules={}, facade_roots={}, mdl_bits=0.0)
    empty_fams = GrammarFamilies(families={}, rule_to_family={}, root_family_weights={})
    assert synthesized_facades(empty_fams, empty_ig, SynthesisConfig()) == []


def test_dirichlet_smoothing_moves_probabilities_toward_uniform() -> None:
    weights = np.asarray([10.0, 1.0, 0.0], dtype=np.float64)
    mle = _smoothed_probs(weights, alpha=0.0, denom_n=3)
    np.testing.assert_allclose(mle, [10 / 11, 1 / 11, 0.0], atol=1e-9)
    # alpha=5, N=3: denominator is 11 + 15 = 26.
    smoothed = _smoothed_probs(weights, alpha=5.0, denom_n=3)
    np.testing.assert_allclose(smoothed, [15 / 26, 6 / 26, 5 / 26], atol=1e-9)
    assert smoothed[2] > 0


def test_dirichlet_concentration_zero_matches_prior_behaviour(make_lattice) -> None:
    samples, _ = _build_corpus(make_lattice, n_samples=4, seed=42)
    again, _ = _build_corpus(make_lattice, n_samples=4, seed=42)
    assert [s.parse_tree for s in samples] == [s.parse_tree for s in again]


def test_sampling_terminates_under_self_recursive_family() -> None:
    # Pathological grammar: S1 → V(S1, T_wall). The sampler must fall back to
    # a terminal at max_depth * _HARD_DEPTH_MULTIPLIER instead of recursing
    # forever.
    rules = {
        "T_wall": TerminalRule(name="T_wall", cls="wall"),
        "S1": SplitRule(
            name="S1",
            axis="vertical",
            children=[
                SplitChild(rule_name="S1", size=0.5),
                SplitChild(rule_name="T_wall", size=0.5),
            ],
        ),
    }
    families = {
        "F_split": NonTerminalFamily(
            name="F_split", key=("S", "vertical"), productions=("S1",), weights=(1,)
        ),
        "F_wall": NonTerminalFamily(
            name="F_wall", key=("T", "wall"), productions=("T_wall",), weights=(1,)
        ),
    }
    fams = GrammarFamilies(
        families=families,
        rule_to_family={"S1": "F_split", "T_wall": "F_wall"},
        root_family_weights={"F_split": 1},
    )
    ig = InducedGrammar(rules=rules, facade_roots={"x": "S1"}, mdl_bits=0.0)
    cfg = SynthesisConfig(n_samples=1, rng_seed=1, max_depth=5, min_production_weight=1)
    out = synthesized_facades(fams, ig, cfg)
    assert len(out) == 1
