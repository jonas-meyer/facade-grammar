"""Tests for ``facade_grammar.pipeline.families``."""

from facade_grammar.config import InductionConfig, SynthesisConfig
from facade_grammar.pipeline.families import grammar_families
from facade_grammar.pipeline.induction import facade_parses, induced_grammar
from facade_grammar.schemas.grammar import InducedGrammar

_IND = InductionConfig()
_SYN = SynthesisConfig()


def test_rule_to_family_is_total(make_lattice) -> None:
    lattices = {
        "a": make_lattice("a", [["wall", "window"], ["wall", "window"]]),
        "b": make_lattice("b", [["wall", "door"], ["wall", "door"]]),
    }
    ig = induced_grammar(lattices, _IND)
    parses = facade_parses(lattices, ig)
    fams = grammar_families(ig, parses, _SYN)
    assert set(fams.rule_to_family) == set(ig.rules)


def test_two_splits_with_same_structure_but_different_sizes_share_family(make_lattice) -> None:
    lattices = {
        "square": make_lattice("square", [["wall", "window"], ["wall", "window"]]),
        "tall": make_lattice(
            "tall",
            [["wall", "window"], ["wall", "window"], ["wall", "window"]],
        ),
    }
    ig = induced_grammar(lattices, _IND)
    parses = facade_parses(lattices, ig)
    fams = grammar_families(ig, parses, _SYN)
    root_a = ig.facade_roots["square"]
    root_b = ig.facade_roots["tall"]
    assert fams.rule_to_family[root_a] == fams.rule_to_family[root_b]


def test_family_weights_sum_matches_total_parse_usage(make_lattice) -> None:
    lattices = {
        "a": make_lattice("a", [["wall", "window"], ["wall", "window"]]),
        "b": make_lattice("b", [["wall", "wall"], ["window", "window"]]),
    }
    ig = induced_grammar(lattices, _IND)
    parses = facade_parses(lattices, ig)
    fams = grammar_families(ig, parses, _SYN)
    total_weight = sum(sum(fam.weights) for fam in fams.families.values())
    total_nodes = sum(_count_nodes(t) for t in parses.values())
    assert total_weight == total_nodes


def test_root_family_weights_sum_to_facade_count(make_lattice) -> None:
    lattices = {
        "a": make_lattice("a", [["wall", "window"], ["wall", "window"]]),
        "b": make_lattice("b", [["wall", "window"], ["wall", "window"]]),
        "c": make_lattice("c", [["wall", "door"], ["wall", "door"]]),
    }
    ig = induced_grammar(lattices, _IND)
    parses = facade_parses(lattices, ig)
    fams = grammar_families(ig, parses, _SYN)
    assert sum(fams.root_family_weights.values()) == len(lattices)


def test_empty_induced_grammar_returns_empty_families() -> None:
    empty = InducedGrammar(rules={}, facade_roots={}, mdl_bits=0.0)
    fams = grammar_families(empty, {}, _SYN)
    assert fams.families == {}
    assert fams.rule_to_family == {}
    assert fams.root_family_weights == {}


def _count_nodes(node) -> int:
    return 1 + sum(_count_nodes(c) for c in node.children)
