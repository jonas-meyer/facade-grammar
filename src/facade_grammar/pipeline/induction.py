"""MDL-guided split-grammar induction over the facade-lattice corpus.

Top-down recursive DP on labelled row x col lattices, memoised by canonical
region signature. Identical sub-regions across all facades collapse into one
non-terminal. Scoring is a two-part MDL objective (rule dictionary bits +
per-facade root bits) reported post-hoc; the DP uses a greedy "prefer cuts
that yield uniform children" heuristic rather than globally optimising MDL.

Phase 1 scope: binary splits, corpus-wide rule reuse. Binary
over-penalises natural n-ary patterns (a 3-way split costs
``1 + 3 * log2|N|`` as a chained binary vs ``1 + 2 * log2|N|`` as
n-ary). An n-ary collapse post-pass is deferred to Phase 2.
"""

import math
from dataclasses import dataclass, field
from typing import Literal, get_args

import numpy as np
from hamilton.function_modifiers import tag
from numpy.typing import NDArray

from facade_grammar.config import InductionConfig
from facade_grammar.schemas.buildings import FacadeLattice, FeatureClass
from facade_grammar.schemas.grammar import (
    InducedGrammar,
    ParseNode,
    Rule,
    SplitChild,
    SplitRule,
    TerminalRule,
)

_ClsLabel = FeatureClass | Literal["wall"]

# Slot 0 is "wall" (the wall-fill label used by regularization); the rest are
# FeatureClass values in declared order. Keeps memoisation signatures compact.
_CLASSES: tuple[_ClsLabel, ...] = ("wall", *get_args(FeatureClass))
_CLASS_TO_U8: dict[_ClsLabel, int] = {c: i for i, c in enumerate(_CLASSES)}

_U8Grid = NDArray[np.uint8]


@dataclass
class _InductionState:
    """Mutable state threaded through the DP."""

    rules: dict[str, Rule] = field(default_factory=dict)
    memo: dict[tuple[tuple[int, int], bytes], str] = field(default_factory=dict)
    terminal_counter: int = 0
    split_counter: int = 0


@tag(stage="induction")
def induced_grammar(
    facade_lattices: dict[str, FacadeLattice],
    induction: InductionConfig,
) -> InducedGrammar:
    """Run DP induction across the corpus; return rules + per-facade roots + mdl_bits.

    A post-hoc rule-unification pass collapses terminals by class and splits
    by ``(axis, child-references, sizes)`` until fixed point — the simplified
    Bayesian-merging layer from Martinović 2013. Drops redundant size-variant
    terminals (54 → ~8 on the canal-house corpus) and cascades into splits
    whose children are now identical.
    """
    if not facade_lattices:
        return InducedGrammar(rules={}, facade_roots={}, mdl_bits=0.0)

    state = _InductionState()
    facade_roots: dict[str, str] = {}
    for bid, lattice in facade_lattices.items():
        grid = _to_grid(lattice)
        facade_roots[bid] = _solve(grid, state)

    rules, facade_roots = _collapse_rules(state.rules, facade_roots)
    mdl_bits = _mdl_bits(rules, facade_roots, induction)
    return InducedGrammar(rules=rules, facade_roots=facade_roots, mdl_bits=mdl_bits)


@tag(stage="induction")
def facade_parses(
    facade_lattices: dict[str, FacadeLattice],
    induced_grammar: InducedGrammar,
) -> dict[str, ParseNode]:
    """Walk each facade's root rule into a parse tree (deterministic given rules)."""
    return {
        bid: _build_parse_tree(induced_grammar.facade_roots[bid], induced_grammar.rules)
        for bid in facade_lattices
        if bid in induced_grammar.facade_roots
    }


def _to_grid(lattice: FacadeLattice) -> _U8Grid:
    """Pack a lattice's cells into a contiguous (n_rows, n_cols) uint8 array."""
    grid = np.zeros((lattice.n_rows, lattice.n_cols), dtype=np.uint8)
    for cell in lattice.cells:
        grid[cell.row, cell.col] = _CLASS_TO_U8[cell.cls]
    return grid


def _sig(region: _U8Grid) -> tuple[tuple[int, int], bytes]:
    """Canonical memoisation key; forces contiguity so slices don't alias."""
    arr = np.ascontiguousarray(region)
    return arr.shape, arr.tobytes()


def _is_uniform(region: _U8Grid) -> bool:
    return bool(region.size > 0 and np.all(region == region.flat[0]))


def _solve(region: _U8Grid, state: _InductionState) -> str:
    """DP: return the rule name for ``region``, creating or reusing via memo.

    Tie-break across candidate splits is deterministic: (axis="horizontal"
    < "vertical", cut_index ascending) by iteration order plus strict `>`
    comparison on score. Required so Hamilton's cache fingerprint is
    stable across runs.
    """
    sig = _sig(region)
    cached = state.memo.get(sig)
    if cached is not None:
        return cached

    if _is_uniform(region):
        state.terminal_counter += 1
        name = f"T{state.terminal_counter}"
        state.rules[name] = TerminalRule(name=name, cls=_CLASSES[int(region.flat[0])])
        state.memo[sig] = name
        return name

    n_rows, n_cols = region.shape
    # Score each candidate cut by how many halves are immediately uniform (0,1,2).
    # Picks first-found at the best score — lexicographic (axis, cut).
    best_score = -1
    best: tuple[str, int] | None = None
    for axis, n in (("horizontal", n_rows), ("vertical", n_cols)):
        for cut in range(1, n):
            left, right = _split(region, axis, cut)
            score = int(_is_uniform(left)) + int(_is_uniform(right))
            if score > best_score:
                best_score, best = score, (axis, cut)

    assert best is not None  # non-uniform region always has at least one candidate cut
    axis, cut = best
    left, right = _split(region, axis, cut)
    left_name = _solve(left, state)
    right_name = _solve(right, state)
    n = n_rows if axis == "horizontal" else n_cols
    children = [
        SplitChild(rule_name=left_name, size=cut / n),
        SplitChild(rule_name=right_name, size=(n - cut) / n),
    ]
    state.split_counter += 1
    name = f"S{state.split_counter}"
    state.rules[name] = SplitRule(name=name, axis=axis, children=children)
    state.memo[sig] = name
    return name


def _split(region: _U8Grid, axis: str, cut: int) -> tuple[_U8Grid, _U8Grid]:
    if axis == "horizontal":
        return region[:cut, :], region[cut:, :]
    return region[:, :cut], region[:, cut:]


def _collapse_rules(
    rules: dict[str, Rule], facade_roots: dict[str, str]
) -> tuple[dict[str, Rule], dict[str, str]]:
    """Unify rules that are semantically identical.

    Two passes: (1) map every terminal rule of a given class to one canonical
    terminal, (2) iteratively fold split rules with identical body
    ``(axis, (child_name, size)...)`` signatures into a single canonical rule.
    Pass (1) exposes more identical splits for pass (2); pass (2) cascades
    until a fixed point. Rule names are reused (no renumbering).
    """
    rename: dict[str, str] = {name: name for name in rules}

    def resolve(name: str) -> str:
        while rename[name] != name:
            name = rename[name]
        return name

    term_canonical: dict[str, str] = {}
    for name, rule in rules.items():
        if isinstance(rule, TerminalRule):
            canonical = term_canonical.setdefault(rule.cls, name)
            rename[name] = canonical

    changed = True
    while changed:
        changed = False
        split_canonical: dict[tuple[str, tuple[tuple[str, float], ...]], str] = {}
        for name, rule in rules.items():
            if resolve(name) != name or not isinstance(rule, SplitRule):
                continue
            sig = (
                rule.axis,
                tuple((resolve(c.rule_name), round(c.size, 6)) for c in rule.children),
            )
            canonical = split_canonical.setdefault(sig, name)
            if canonical != name:
                rename[name] = canonical
                changed = True

    final: dict[str, Rule] = {}
    for name, rule in rules.items():
        if resolve(name) != name:
            continue
        if isinstance(rule, TerminalRule):
            final[name] = rule
        else:
            final[name] = SplitRule(
                name=name,
                axis=rule.axis,
                children=[
                    SplitChild(rule_name=resolve(c.rule_name), size=c.size) for c in rule.children
                ],
            )
    new_roots = {bid: resolve(r) for bid, r in facade_roots.items()}
    return final, new_roots


def _mdl_bits(rules: dict[str, Rule], facade_roots: dict[str, str], cfg: InductionConfig) -> float:
    """Post-hoc two-part description length of the induced grammar.

    ``DL(G)`` is the rule-dictionary size; ``DL(D|G)`` is the bits needed
    to name each facade's root rule given the dictionary. ``grammar_prior_weight``
    tilts the trade-off between grammar size and per-facade encoding cost.
    """
    if not rules:
        return 0.0
    n_rules = len(rules)
    n_classes = len(_CLASSES)
    log2_rules = math.log2(n_rules) if n_rules > 1 else 0.0
    log2_classes = math.log2(n_classes)
    # Coarse upper bound for the cut-position quantisation; the stored SplitChild
    # size is a fraction, but the underlying cut is an integer in [1, dim-1] with
    # dim <= 255 for any plausible facade lattice.
    log2_max_cut = 8.0
    dl_g = 0.0
    for rule in rules.values():
        if isinstance(rule, TerminalRule):
            dl_g += log2_classes
        else:
            dl_g += 1.0 + log2_max_cut + 2.0 * log2_rules
    dl_d_given_g = len(facade_roots) * log2_rules
    return cfg.grammar_prior_weight * dl_g + dl_d_given_g


def _build_parse_tree(rule_name: str, rules: dict[str, Rule]) -> ParseNode:
    rule = rules[rule_name]
    if isinstance(rule, TerminalRule):
        return ParseNode(rule_name=rule_name, children=())
    children = tuple(_build_parse_tree(c.rule_name, rules) for c in rule.children)
    return ParseNode(rule_name=rule_name, children=children)
