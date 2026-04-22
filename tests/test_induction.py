"""Tests for ``facade_grammar.pipeline.induction``."""

import numpy as np
from pydantic import TypeAdapter

from facade_grammar.config import InductionConfig
from facade_grammar.pipeline.induction import (
    _sig,
    facade_parses,
    induced_grammar,
)
from facade_grammar.schemas.grammar import InducedGrammar, SplitRule, TerminalRule

_CFG = InductionConfig()


def test_uniform_grid_yields_single_terminal(make_lattice) -> None:
    lat = make_lattice("b", [["wall", "wall"], ["wall", "wall"]])
    g = induced_grammar({"b": lat}, _CFG)
    assert len(g.rules) == 1
    rule = next(iter(g.rules.values()))
    assert isinstance(rule, TerminalRule)
    assert rule.cls == "wall"
    assert g.facade_roots == {"b": rule.name}
    assert g.mdl_bits >= 0


def test_two_uniform_halves_split_horizontally(make_lattice) -> None:
    lat = make_lattice("b", [["wall", "wall"], ["window", "window"]])
    g = induced_grammar({"b": lat}, _CFG)
    assert len(g.rules) == 3
    root = g.rules[g.facade_roots["b"]]
    assert isinstance(root, SplitRule)
    assert root.axis == "horizontal"
    child_classes = set()
    for child in root.children:
        rule = g.rules[child.rule_name]
        assert isinstance(rule, TerminalRule)
        child_classes.add(rule.cls)
    assert child_classes == {"wall", "window"}


def test_identical_subregions_share_rule_via_memoisation(make_lattice) -> None:
    lat1 = make_lattice("b1", [["wall", "wall"], ["wall", "wall"]])
    lat2 = make_lattice("b2", [["wall", "wall"], ["wall", "wall"]])
    g = induced_grammar({"b1": lat1, "b2": lat2}, _CFG)
    assert len(g.rules) == 1
    assert g.facade_roots["b1"] == g.facade_roots["b2"]


def test_repeated_pattern_lowers_mdl_bits_than_distinct(make_lattice) -> None:
    a = make_lattice("a", [["wall", "window"], ["wall", "window"]])
    a2 = make_lattice("a2", [["wall", "window"], ["wall", "window"]])
    b = make_lattice("b", [["wall", "door"], ["wall", "door"]])
    repeated = induced_grammar({"a": a, "a2": a2}, _CFG)
    distinct = induced_grammar({"a": a, "b": b}, _CFG)
    assert len(repeated.rules) < len(distinct.rules)


def test_non_contiguous_slice_signature_regression() -> None:
    grid = np.arange(16, dtype=np.uint8).reshape(4, 4)
    slice_view = grid[::2, ::2]
    contig = np.ascontiguousarray(slice_view)
    assert _sig(slice_view) == _sig(contig)


def test_empty_corpus_returns_empty_grammar() -> None:
    g = induced_grammar({}, _CFG)
    assert g.rules == {}
    assert g.facade_roots == {}
    assert g.mdl_bits == 0.0


def test_induced_grammar_roundtrips_through_type_adapter(make_lattice) -> None:
    lat = make_lattice("b", [["wall", "wall"], ["window", "window"]])
    g = induced_grammar({"b": lat}, _CFG)
    adapter = TypeAdapter(InducedGrammar)
    reloaded = adapter.validate_json(adapter.dump_json(g))
    assert reloaded.rules == g.rules
    assert reloaded.facade_roots == g.facade_roots
    assert reloaded.mdl_bits == g.mdl_bits


def test_facade_parses_builds_tree_matching_root_rule(make_lattice) -> None:
    lat = make_lattice("b", [["wall", "wall"], ["window", "window"]])
    g = induced_grammar({"b": lat}, _CFG)
    parses = facade_parses({"b": lat}, g)
    tree = parses["b"]
    assert tree.rule_name == g.facade_roots["b"]
    assert len(tree.children) == 2
    assert all(len(c.children) == 0 for c in tree.children)
