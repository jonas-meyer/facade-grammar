"""Kozinski-style non-terminal family mining.

Groups concrete rules that share the same structural signature to a bounded
depth, ignoring concrete ``SplitChild.size`` ratios. Each family then acts
as a non-terminal with multiple productions, weighted by usage across the
parse trees from ``facade_parses``. Downstream ``pipeline.synthesis``
samples novel parse trees by walking these families.

Family-key recurrence (budget-style, see Kozinski et al. 2016):

- ``depth == 0``: ``("*",)`` — truncation marker for every rule.
- Terminal at ``depth >= 1``: ``("T", cls)`` — class is the whole signature.
- Split at ``depth >= 1``: ``("S", axis, *(child_key_at_depth-1))`` — ordered
  child keys, concrete sizes dropped.
"""

from collections import Counter

from hamilton.function_modifiers import tag

from facade_grammar.config import SynthesisConfig
from facade_grammar.schemas.grammar import (
    GrammarFamilies,
    InducedGrammar,
    NonTerminalFamily,
    ParseNode,
    Rule,
    TerminalRule,
)


@tag(stage="families")
def grammar_families(
    induced_grammar: InducedGrammar,
    facade_parses: dict[str, ParseNode],
    synthesis: SynthesisConfig,
) -> GrammarFamilies:
    """Mine non-terminal families from the induced grammar + parse-tree usage."""
    if not induced_grammar.rules:
        return GrammarFamilies(families={}, rule_to_family={}, root_family_weights={})

    keys = _family_keys(induced_grammar.rules, synthesis.family_depth)
    usage = _count_usage(facade_parses)

    # Group rule names by key, sort families deterministically for stable
    # naming across runs (Hamilton's downstream cache fingerprint depends on it).
    by_key: dict[tuple[str, ...], list[str]] = {}
    for rule_name, key in keys.items():
        by_key.setdefault(key, []).append(rule_name)
    ordered_keys = sorted(by_key.keys(), key=lambda k: (k, min(by_key[k])))

    families: dict[str, NonTerminalFamily] = {}
    rule_to_family: dict[str, str] = {}
    for idx, key in enumerate(ordered_keys, start=1):
        name = f"F{idx}"
        members = sorted(by_key[key])
        weights = tuple(usage.get(m, 0) for m in members)
        families[name] = NonTerminalFamily(
            name=name,
            key=key,
            productions=tuple(members),
            weights=weights,
        )
        for m in members:
            rule_to_family[m] = name

    assert set(rule_to_family) == set(induced_grammar.rules), (
        "rule_to_family must be total over induced_grammar.rules"
    )

    root_family_weights: dict[str, int] = {}
    for root_rule in induced_grammar.facade_roots.values():
        fam = rule_to_family[root_rule]
        root_family_weights[fam] = root_family_weights.get(fam, 0) + 1

    return GrammarFamilies(
        families=families,
        rule_to_family=rule_to_family,
        root_family_weights=root_family_weights,
    )


def _family_keys(rules: dict[str, Rule], depth: int) -> dict[str, tuple[str, ...]]:
    """Compute a structural key per rule, memoised on ``(rule_name, depth)``."""
    memo: dict[tuple[str, int], tuple[str, ...]] = {}

    def key_of(rule_name: str, budget: int) -> tuple[str, ...]:
        cached = memo.get((rule_name, budget))
        if cached is not None:
            return cached
        if budget <= 0:
            result: tuple[str, ...] = ("*",)
        else:
            rule = rules[rule_name]
            if isinstance(rule, TerminalRule):
                result = ("T", str(rule.cls))
            else:
                child_keys = tuple(key_of(child.rule_name, budget - 1) for child in rule.children)
                flat: list[str] = ["S", rule.axis]
                for ck in child_keys:
                    flat.append("(")
                    flat.extend(ck)
                    flat.append(")")
                result = tuple(flat)
        memo[(rule_name, budget)] = result
        return result

    return {name: key_of(name, depth) for name in rules}


def _count_usage(parses: dict[str, ParseNode]) -> Counter[str]:
    """Count every ParseNode occurrence, root + descendants, across all facades."""
    counter: Counter[str] = Counter()
    for tree in parses.values():
        _walk_parse(tree, counter)
    return counter


def _walk_parse(node: ParseNode, counter: Counter[str]) -> None:
    counter[node.rule_name] += 1
    for child in node.children:
        _walk_parse(child, counter)
