"""Matplotlib renderer for ``SynthesizedFacade`` parse trees.

Draws each synthesised facade as a unit-square decomposed top-down by
``SplitRule`` axis + ``SplitChild.size`` into terminal cells. Each
terminal class renders as a small feature shape over a wall background
(window panes with mullions, doors with knobs, chimneys with caps, etc.)
— solid blocks make the grammar look like Mondrian rather than a facade.
"""

from pathlib import Path

from matplotlib.patches import Circle, Rectangle

from facade_grammar.schemas.grammar import (
    InducedGrammar,
    ParseNode,
    Rule,
    SplitRule,
    SynthesizedFacade,
    TerminalRule,
)
from facade_grammar.viz.plots import _init_contact_sheet, _save_contact_sheet

_WALL_COLOR = "#d0d0d0"
_FRAME_COLOR = "#404040"
_GLASS_COLOR = "#a9c8e6"
_DOOR_COLOR = "#6b4a2b"
_DOOR_TRIM_COLOR = "#3a2915"
_KNOB_COLOR = "#c8a951"
_CHIMNEY_BRICK = "#8c564b"
_CHIMNEY_CAP = "#3a2915"
_RAIL_COLOR = "#606060"
_HOIST_COLOR = "#17becf"

# Canonical y-position per terminal class (0=top, 1=bottom). Used as a
# render-time cosmetic reorder — the grammar itself is position-naive
# (matches Martinović 2013 / Wu-Wonka 2014), so at draw time we sort
# horizontal-split children by their subtree's mean canonical position
# to float gables/chimneys up and sink doors to the bottom. This is
# purely presentational; the stored parse tree is untouched.
_CANONICAL_POSITION: dict[str, float] = {
    "gable": 0.0,
    "chimney": 0.05,
    "hoist_beam": 0.08,
    "balcony": 0.3,
    "floor": 0.5,
    "window": 0.5,
    "wall": 0.5,
    "door": 1.0,
}


def plot_synthesized_facades_contact_sheet(
    *,
    synthesized_facades: list[SynthesizedFacade],
    induced_grammar: InducedGrammar,
    out_path: Path,
) -> Path:
    """3x4 grid of synthesised facades rendered as coloured rectangles."""
    fig, cells, sample = _init_contact_sheet(synthesized_facades, out_path)

    for ax, facade in zip(cells, sample, strict=False):
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.invert_yaxis()
        ax.set_aspect("equal")
        _draw_node(
            ax,
            facade.parse_tree,
            induced_grammar.rules,
            x=0.0,
            y=0.0,
            w=1.0,
            h=1.0,
        )
        trace_label = " → ".join(facade.family_trace[:3])
        ax.set_title(f"seed={facade.seed}  {trace_label}", fontsize=7)

    fig.suptitle(
        f"{len(synthesized_facades)} synthesised facades "
        f"(from {len(induced_grammar.rules)} induced rules)",
        fontsize=11,
    )
    return _save_contact_sheet(fig, out_path)


def _draw_node(
    ax,
    node: ParseNode,
    rules: dict[str, Rule],
    *,
    x: float,
    y: float,
    w: float,
    h: float,
) -> None:
    rule = rules[node.rule_name]
    if isinstance(rule, TerminalRule):
        _draw_terminal(ax, rule.cls, x=x, y=y, w=w, h=h)
        return
    assert isinstance(rule, SplitRule)
    pairs = list(zip(rule.children, node.children, strict=True))
    if rule.axis == "horizontal":
        # Cosmetic canonical-order pass: float gables/chimneys up, sink doors
        # down. Presentational only — stored parse tree is untouched.
        pairs.sort(key=lambda p: _subtree_canonical_position(p[1], rules))
        cursor_y = y
        for spec, child_node in pairs:
            child_h = h * spec.size
            _draw_node(ax, child_node, rules, x=x, y=cursor_y, w=w, h=child_h)
            cursor_y += child_h
    else:
        cursor_x = x
        for spec, child_node in pairs:
            child_w = w * spec.size
            _draw_node(ax, child_node, rules, x=cursor_x, y=y, w=child_w, h=h)
            cursor_x += child_w


def _subtree_canonical_position(node: ParseNode, rules: dict[str, Rule]) -> float:
    """Average canonical y-position of terminals in this subtree."""
    rule = rules[node.rule_name]
    if isinstance(rule, TerminalRule):
        return _CANONICAL_POSITION.get(rule.cls, 0.5)
    positions = [_subtree_canonical_position(c, rules) for c in node.children]
    return sum(positions) / len(positions) if positions else 0.5


def _draw_terminal(ax, cls: str, *, x: float, y: float, w: float, h: float) -> None:
    """Wall background + feature shape overlay. ``cls=="wall"`` is just stucco."""
    ax.add_patch(Rectangle((x, y), w, h, facecolor=_WALL_COLOR, edgecolor="none", linewidth=0))
    if cls == "window":
        _draw_window(ax, x, y, w, h)
    elif cls == "door":
        _draw_door(ax, x, y, w, h)
    elif cls == "chimney":
        _draw_chimney(ax, x, y, w, h)
    elif cls == "balcony":
        _draw_balcony(ax, x, y, w, h)
    elif cls == "hoist_beam":
        _draw_hoist_beam(ax, x, y, w, h)
    # "wall", "gable", "floor" fall through as plain wall stucco.


def _draw_window(ax, x: float, y: float, w: float, h: float) -> None:
    mw, mh = w * 0.18, h * 0.15
    ax.add_patch(
        Rectangle(
            (x + mw, y + mh),
            w - 2 * mw,
            h - 2 * mh,
            facecolor=_GLASS_COLOR,
            edgecolor=_FRAME_COLOR,
            linewidth=0.8,
        )
    )
    mx = x + w / 2
    ax.plot(
        [mx, mx],
        [y + mh, y + h - mh],
        color=_FRAME_COLOR,
        linewidth=0.6,
    )
    my = y + mh + (h - 2 * mh) * 0.5
    ax.plot(
        [x + mw, x + w - mw],
        [my, my],
        color=_FRAME_COLOR,
        linewidth=0.4,
    )


def _draw_door(ax, x: float, y: float, w: float, h: float) -> None:
    mw = w * 0.2
    ax.add_patch(
        Rectangle(
            (x + mw, y),
            w - 2 * mw,
            h,
            facecolor=_DOOR_COLOR,
            edgecolor=_DOOR_TRIM_COLOR,
            linewidth=0.8,
        )
    )
    knob_x = x + w - mw - (w - 2 * mw) * 0.12
    knob_y = y + h * 0.5
    ax.add_patch(Circle((knob_x, knob_y), radius=min(w, h) * 0.03, facecolor=_KNOB_COLOR))


def _draw_chimney(ax, x: float, y: float, w: float, h: float) -> None:
    stack_w = w * 0.35
    stack_h = h * 0.55
    sx = x + (w - stack_w) / 2
    ax.add_patch(
        Rectangle(
            (sx, y),
            stack_w,
            stack_h,
            facecolor=_CHIMNEY_BRICK,
            edgecolor=_DOOR_TRIM_COLOR,
            linewidth=0.5,
        )
    )
    cap_w = stack_w * 1.2
    cap_h = stack_h * 0.12
    ax.add_patch(
        Rectangle(
            (sx - (cap_w - stack_w) / 2, y),
            cap_w,
            cap_h,
            facecolor=_CHIMNEY_CAP,
            edgecolor="none",
        )
    )


def _draw_balcony(ax, x: float, y: float, w: float, h: float) -> None:
    rail_h = h * 0.12
    ax.add_patch(
        Rectangle(
            (x + w * 0.08, y + h - rail_h - h * 0.1),
            w * 0.84,
            rail_h,
            facecolor=_RAIL_COLOR,
            edgecolor="none",
        )
    )


def _draw_hoist_beam(ax, x: float, y: float, w: float, h: float) -> None:
    beam_h = h * 0.2
    ax.add_patch(
        Rectangle(
            (x, y),
            w,
            beam_h,
            facecolor=_HOIST_COLOR,
            edgecolor=_FRAME_COLOR,
            linewidth=0.5,
        )
    )
