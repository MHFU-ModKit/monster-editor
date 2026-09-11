"""The behaviour pairs as a graph: nodes are `(main, sub)`, arrows are the hand-offs
a handler makes when its action ends (`PairIntel.next`, from `tools/em_chain.py`).

Why a drawing and not another column: the engine does not run one pair, it walks a
SEQUENCE — the Tigrex charge is `(1,4)` until its run budget is spent, then `(0,3)`
(the skid), then `(0,1)`/`(0,2)` where the brain picks again — and a script that
writes one pair and holds it is fighting that walk, which is how a forced charge
parks with its hitbox spent (`monster-ai`). Seeing the arrows is what makes "which
pair do I hand to next" a lookup instead of a guess.

The view is a tab beside the Viewport (same dock space), so the model and the graph
swap with one click and neither has to be small. The layout is a layered DAG from a
set of ROOTS — the manifest's moves, or the selected pair — with the hub pairs
(`SpeciesIntel.hubs`: where most hand-offs land) drawn once, as terminals, or every
chain would be three arrows into the same three boxes.

Pure imgui draw-list work, no GL. Runs under the panel test's `imgui` stand-in.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

Pair = Tuple[int, int]

#: layered-layout geometry, in unzoomed points
NODE_W, NODE_H = 148.0, 58.0
GAP_X, GAP_Y = 96.0, 22.0
PAD = 18.0

SCOPES = ("moves", "selected", "attacks")


@dataclass
class Node:
    pair: Pair
    lines: List[str]
    layer: int = 0
    row: int = 0
    x: float = 0.0
    y: float = 0.0
    hub: bool = False
    move: Optional[str] = None
    entry: bool = False
    attacks: bool = False
    #: other pairs this node stands for: same handler, same hand-offs, same attack
    #: ids (em75's main-3 bank is 42 such siblings) — drawn once, listed on hover
    siblings: Tuple[Pair, ...] = ()


@dataclass
class Arrow:
    src: Pair
    dst: Pair
    label: str
    guards: Tuple[str, ...] = ()
    mode: Optional[int] = None


@dataclass
class Layout:
    nodes: Dict[Pair, Node] = field(default_factory=dict)
    arrows: List[Arrow] = field(default_factory=list)
    width: float = 0.0
    height: float = 0.0
    note: str = ""

    @property
    def empty(self) -> bool:
        return not self.nodes


# --------------------------------------------------------------------------- #
# building
# --------------------------------------------------------------------------- #
def _label(p, move: Optional[str]) -> List[str]:
    lines = ["(%d,%d)%s" % (p.main, p.sub, "  " + move if move else "")]
    if p.a1:
        lines.append("a1 %s%s" % (",".join(str(a) for a in p.a1[:4]),
                                  "+" if p.a1_computed else ""))
    elif p.a1_computed:
        lines.append("a1 computed")
    bits = []
    if p.attack_ids:
        bits.append("atk %s" % ",".join(str(a) for a in p.attack_ids[:3]))
    if p.ends_on and p.ends_on != "unknown":
        bits.append(p.ends_on)
    if bits:
        lines.append("  ".join(bits))
    return lines


def build(intel, moves: dict, selected: Optional[Pair], scope: str = "moves",
          depth: int = 6) -> Layout:
    """`moves` is `{name: Move}` from the manifest (its `.main`/`.sub` are read);
    `scope` picks the roots. Hubs are never expanded, and always shown once.

    In the `moves` scope the Action tab's selected pair joins the roots only when
    it is not in the picture already: selecting a hub that is on screen as a
    terminal must not turn it into a root and unfold its own hand-offs over the
    chain you were reading (a selected `(0,1)` grew a self-loop and six arrows)."""
    lay = _build(intel, moves, selected, scope, depth, extra_root=False)
    if (scope == "moves" and selected is not None and not lay.empty
            and selected not in lay.nodes and intel.pair(*selected) is not None):
        lay = _build(intel, moves, selected, scope, depth, extra_root=True)
    return lay


def _build(intel, moves: dict, selected: Optional[Pair], scope: str, depth: int,
           extra_root: bool) -> Layout:
    lay = Layout()
    if intel is None or not getattr(intel, "has_chain", False):
        lay.note = ("no hand-off intel for this overlay — rebuild species/*.json with "
                    "`python tools/em_intel.py --all`")
        return lay
    bound: Dict[Pair, str] = {}
    for name in sorted(moves):
        mv = moves[name]
        bound.setdefault((int(mv.main), int(mv.sub)), name)
    hubs = set(intel.hubs)

    siblings: Dict[Pair, Tuple[Pair, ...]] = {}
    if scope == "attacks":
        # every pair whose handler spawns a hitbox: the moves a port could ride.
        # Pairs that share a handler, its hand-offs and its attack ids are ONE node:
        # em75's main-3 bank is 42 siblings, and 42 boxes with the same 12 arrows
        # each is a wall, not a picture.
        groups: Dict[tuple, List[Pair]] = {}
        for p in intel:
            if p.attack_ids and p.next is not None:
                sig = (p.handler, tuple(p.successors), tuple(p.attack_ids))
                groups.setdefault(sig, []).append((p.main, p.sub))
        roots = []
        for members in groups.values():
            members.sort()
            head = (selected if selected in members
                    else next((m for m in members if m in bound), members[0]))
            roots.append(head)
            if len(members) > 1:
                siblings[head] = tuple(m for m in members if m != head)
    elif scope == "selected":
        roots = [selected] if selected else []
    else:
        roots = sorted(bound)
        if extra_root and selected and selected not in roots:
            roots.append(selected)
    roots = [r for r in roots if intel.pair(*r) is not None]
    if not roots:
        lay.note = {"moves": "[moves] is empty and nothing is selected — pick a pair "
                             "in the Action tab, or bind one, to see its chain",
                    "selected": "select a pair in the Action tab",
                    "attacks": "no pair in this overlay names an attack id"}[scope]
        return lay

    # breadth-first from the roots; a hub is a terminal
    layer_of: Dict[Pair, int] = {}
    order: List[Pair] = []
    frontier = list(roots)
    for r in roots:
        layer_of[r] = 0
        order.append(r)
    for d in range(1, depth + 1):
        nxt: List[Pair] = []
        for key in frontier:
            if key in hubs and key not in roots:
                continue
            p = intel.pair(*key)
            for t in p.successors:
                if intel.pair(*t) is None:
                    continue
                if t not in layer_of:
                    layer_of[t] = d
                    order.append(t)
                    nxt.append(t)
                elif t not in roots and layer_of[t] < d and t not in hubs:
                    layer_of[t] = d              # longest path keeps arrows forward
        frontier = nxt
        if not frontier:
            break
    # hubs sit in one layer past the deepest ordinary node, once each
    inner = [layer_of[k] for k in order if not (k in hubs and k not in roots)]
    last = (max(inner) + 1) if inner else 0
    for key in order:
        if key in hubs and key not in roots:
            layer_of[key] = last

    # predecessors of the selected pair, one column to the left of everything
    preds: List[Pair] = []
    if selected and scope == "selected":
        preds = [(q.main, q.sub) for q in intel.predecessors(*selected)
                 if (q.main, q.sub) not in layer_of][:14]
        if preds:
            for key in layer_of:
                layer_of[key] += 1
            for key in preds:
                layer_of[key] = 0
                order.insert(0, key)

    for key in order:
        p = intel.pair(*key)
        lines = _label(p, bound.get(key))
        if key in hubs and key not in roots:
            lines = lines[:2]                  # the footer "brain picks next" is line 3
        sib = siblings.get(key, ())
        if sib:
            lines[0] += "  +%d alike" % len(sib)
        n = Node(pair=key, lines=lines, layer=layer_of[key],
                 hub=key in hubs, move=bound.get(key), entry=key in roots,
                 attacks=bool(p.attack_ids), siblings=sib)
        lay.nodes[key] = n
    # one arrow per (from, to); the alternative reasons join on the label
    seen_arrow: Dict[Tuple[Pair, Pair], Arrow] = {}
    for key in order:
        p = intel.pair(*key)
        if key in hubs and key not in roots:
            continue
        for e in p.next or []:
            for t in e.to:
                if t not in lay.nodes:
                    continue
                a = seen_arrow.get((key, t))
                if a is None:
                    a = Arrow(key, t, e.reason, e.guards, e.mode)
                    seen_arrow[(key, t)] = a
                    lay.arrows.append(a)
                elif e.reason and e.reason not in a.label:
                    a.label = (a.label + " / " + e.reason) if a.label else e.reason

    # rows: by barycenter of predecessors, one pass, then stable by pair
    cols: Dict[int, List[Node]] = {}
    for n in lay.nodes.values():
        cols.setdefault(n.layer, []).append(n)
    pos_of: Dict[Pair, float] = {}
    for layer in sorted(cols):
        col = cols[layer]
        if layer == 0:
            col.sort(key=lambda n: n.pair)
        else:
            def bary(n):
                ys = [pos_of[a.src] for a in lay.arrows if a.dst == n.pair
                      and a.src in pos_of]
                return (sum(ys) / len(ys)) if ys else 1e9, n.pair
            col.sort(key=bary)
        for i, n in enumerate(col):
            n.row = i
            pos_of[n.pair] = float(i)
    tallest = max(len(c) for c in cols.values())
    for layer, col in cols.items():
        top = (tallest - len(col)) * (NODE_H + GAP_Y) / 2.0
        for n in col:
            n.x = PAD + layer * (NODE_W + GAP_X)
            n.y = PAD + top + n.row * (NODE_H + GAP_Y)
    lay.width = PAD * 2 + (max(cols) + 1) * (NODE_W + GAP_X) - GAP_X
    lay.height = PAD * 2 + tallest * (NODE_H + GAP_Y) - GAP_Y
    return lay


# --------------------------------------------------------------------------- #
# drawing
# --------------------------------------------------------------------------- #
class MoveGraph:
    """The panel's state: scope, camera, the cached layout, what is hovered, what
    is PICKED (a single click — highlight and the info box) and, separately, what
    the Action tab has selected (a double click sends a node there).

    Left-drag on a node moves it (the position lives in the cached layout until
    the next re-layout); left-drag on empty canvas, or any right/middle drag, pans.
    Edge labels are drawn only for the picked or hovered node's edges — with every
    label on, a five-node chain was unreadable — and the info box in the corner
    lists the same hand-offs as text, one per line, which is where a condition is
    actually read."""

    def __init__(self) -> None:
        self.scope = "moves"
        self.pan = [0.0, 0.0]
        self.zoom = 1.0
        self.hover: Optional[Pair] = None
        self.picked: Optional[Pair] = None
        self._layout: Optional[Layout] = None
        self._key = None
        self._fit_pending = True
        self._press: Optional[Tuple[Optional[Pair], float, float]] = None
        self._moved = False

    def layout(self, intel, moves: dict, selected: Optional[Pair]) -> Layout:
        key = (id(intel), self.scope, selected,
               tuple(sorted((n, m.main, m.sub) for n, m in moves.items())))
        if self._layout is None or key != self._key:
            self._layout = build(intel, moves, selected, self.scope)
            self._key = key
            self._fit_pending = True
            if self.picked is not None and self.picked not in self._layout.nodes:
                self.picked = None
        return self._layout

    def invalidate(self) -> None:
        self._layout, self._key = None, None

    def fit(self, avail_w: float, avail_h: float) -> None:
        lay = self._layout
        if lay is None or lay.empty:
            return
        z = min(avail_w / max(lay.width, 1.0), avail_h / max(lay.height, 1.0), 1.25)
        self.zoom = max(0.35, z)
        self.pan = [(avail_w - lay.width * self.zoom) / 2.0,
                    (avail_h - lay.height * self.zoom) / 2.0]
        self._fit_pending = False

    # -- the panel body ------------------------------------------------------
    def draw(self, imgui, app) -> None:
        intel = app.intel
        m = app.scene.manifest
        moves = dict(m.moves) if m is not None else {}
        selected = app._pair
        self._toolbar(imgui, app, intel, selected)
        lay = self.layout(intel, moves, selected)
        avail = imgui.get_content_region_avail()
        w, h = max(float(avail.x), 1.0), max(float(avail.y), 1.0)
        if lay.empty:
            imgui.text_wrapped(lay.note)
            return
        if self._fit_pending:
            self.fit(w, h)
        origin = imgui.get_cursor_screen_pos()
        size = imgui.ImVec2(w, h)
        imgui.invisible_button("##movegraph", size, _button_flags(imgui))
        hovered = imgui.is_item_hovered()
        active = imgui.is_item_active()
        mouse = imgui.get_mouse_pos()
        self.hover = None
        if hovered:
            for n in lay.nodes.values():
                if _inside(self._rect(origin, n), mouse):
                    self.hover = n.pair
        self._input(imgui, app, lay, origin, hovered, active, mouse)
        draw = imgui.get_window_draw_list()
        p0 = imgui.ImVec2(float(origin.x), float(origin.y))
        p1 = imgui.ImVec2(float(origin.x) + w, float(origin.y) + h)
        draw.add_rect_filled(p0, p1, _col(imgui, 0.09, 0.10, 0.12, 1.0))
        imgui.push_clip_rect(p0, p1, True)
        focus = self.picked if self.picked is not None else self.hover
        # arrows that touch the focus node draw last, so their labels sit on top
        plain = [a for a in lay.arrows if focus not in (a.src, a.dst)]
        hot = [a for a in lay.arrows if focus in (a.src, a.dst)]
        for a in plain:
            self._arrow(imgui, draw, origin, lay, a, selected, focus, labelled=False)
        for n in lay.nodes.values():
            self._node(imgui, draw, origin, n, selected)
        for i, a in enumerate(hot):
            self._arrow(imgui, draw, origin, lay, a, selected, focus, labelled=True,
                        slot=i)
        _info_box(imgui, draw, origin, w, intel, lay, focus, moves)
        imgui.pop_clip_rect()
        _legend(imgui, draw, origin, w, h)

    def _toolbar(self, imgui, app, intel, selected) -> None:
        imgui.set_next_item_width(110.0)
        idx = SCOPES.index(self.scope) if self.scope in SCOPES else 0
        changed, pick = imgui.combo("##scope", idx, ["from [moves]", "from selected",
                                                       "every attack"])
        if changed:
            self.scope = SCOPES[pick]
            self.invalidate()
        imgui.same_line()
        if imgui.small_button("fit"):
            self._fit_pending = True
        imgui.same_line()
        if imgui.small_button("re-layout"):
            self.invalidate()
        imgui.same_line()
        if intel is None or not getattr(intel, "has_chain", False):
            imgui.text_disabled("em%02d: no hand-off intel" % app.browsing_species)
            return
        hubs = " ".join("(%d,%d)" % h for h in intel.hubs)
        if selected is not None and intel.pair(*selected) is not None:
            imgui.text_disabled(_walk_line(intel, selected))
        else:
            imgui.text_disabled("em%02d — arrows are what the HANDLER does when its "
                                "action ends; hubs %s are where the brain picks again"
                                % (app.browsing_species, hubs))
        if imgui.is_item_hovered():
            imgui.set_tooltip("Static: read from the overlay (tools/em_chain.py). A pair "
                              "with no arrow out never ends by itself — the engine only "
                              "enters it through the translator, which provisions it.\n"
                              "Click a node to read its hand-offs; DOUBLE-click to select "
                              "it in the Action tab. Drag a node to move it, drag the "
                              "canvas to pan, wheel to zoom.")

    def _input(self, imgui, app, lay: Layout, origin, hovered: bool, active: bool,
               mouse) -> None:
        # a press remembers what was under it: a node = a drag moves that node, a
        # release without movement picks it; empty canvas = a drag pans
        if hovered and imgui.is_mouse_clicked(0):
            self._press = (self.hover, float(mouse.x), float(mouse.y))
            self._moved = False
        if active and self._press is not None and imgui.is_mouse_dragging(0):
            d = imgui.get_mouse_drag_delta(0)
            node = lay.nodes.get(self._press[0]) if self._press[0] is not None else None
            if node is not None:
                node.x += float(d.x) / self.zoom
                node.y += float(d.y) / self.zoom
            else:
                self.pan[0] += float(d.x)
                self.pan[1] += float(d.y)
            imgui.reset_mouse_drag_delta(0)
            self._moved = True
        if self._press is not None and imgui.is_mouse_released(0):
            if not self._moved:
                self.picked = self._press[0]          # None clears the pick
            self._press = None
            self._moved = False
        if hovered and self.hover is not None and imgui.is_mouse_double_clicked(0):
            self.picked = self.hover
            app.select_pair(*self.hover)
        for btn in (1, 2):
            if active and imgui.is_mouse_dragging(btn):
                d = imgui.get_mouse_drag_delta(btn)
                self.pan[0] += float(d.x)
                self.pan[1] += float(d.y)
                imgui.reset_mouse_drag_delta(btn)
        wheel = float(imgui.get_io().mouse_wheel)
        if hovered and wheel:
            mx, my = float(mouse.x) - float(origin.x), float(mouse.y) - float(origin.y)
            old = self.zoom
            self.zoom = max(0.25, min(2.5, self.zoom * (1.12 if wheel > 0 else 1 / 1.12)))
            k = self.zoom / old
            self.pan[0] = mx - (mx - self.pan[0]) * k
            self.pan[1] = my - (my - self.pan[1]) * k

    # -- geometry --------------------------------------------------------------
    def _pt(self, origin, x: float, y: float):
        return (float(origin.x) + self.pan[0] + x * self.zoom,
                float(origin.y) + self.pan[1] + y * self.zoom)

    def _rect(self, origin, n: Node):
        x0, y0 = self._pt(origin, n.x, n.y)
        return (x0, y0, x0 + NODE_W * self.zoom, y0 + NODE_H * self.zoom)

    def _node(self, imgui, draw, origin, n: Node, selected) -> None:
        x0, y0, x1, y1 = self._rect(origin, n)
        sel = n.pair == selected
        pick = n.pair == self.picked
        hov = n.pair == self.hover
        if n.hub:
            fill = _col(imgui, 0.18, 0.19, 0.21, 1.0)
            edge = _col(imgui, 0.45, 0.47, 0.50, 1.0)
        elif n.move:
            fill = _col(imgui, 0.16, 0.27, 0.36, 1.0)
            edge = _col(imgui, 0.35, 0.70, 0.95, 1.0)
        elif n.attacks:
            fill = _col(imgui, 0.32, 0.20, 0.14, 1.0)
            edge = _col(imgui, 0.95, 0.60, 0.30, 1.0)
        else:
            fill = _col(imgui, 0.16, 0.17, 0.20, 1.0)
            edge = _col(imgui, 0.40, 0.42, 0.46, 1.0)
        if sel:
            edge = _col(imgui, 1.0, 0.90, 0.35, 1.0)
        if pick:
            edge = _col(imgui, 0.95, 0.95, 0.95, 1.0)
        elif hov and not sel:
            edge = _col(imgui, 0.75, 0.76, 0.78, 1.0)
        r = 6.0 * self.zoom
        draw.add_rect_filled(imgui.ImVec2(x0, y0), imgui.ImVec2(x1, y1), fill, r)
        # this binding's order is (rounding, thickness, flags) — not ImGui's C++ one
        draw.add_rect(imgui.ImVec2(x0, y0), imgui.ImVec2(x1, y1), edge, r,
                      2.5 if pick else 2.0 if (sel or hov) else 1.0)
        if self.zoom < 0.5:
            draw.add_text(imgui.ImVec2(x0 + 4, y0 + 3),
                          _col(imgui, 0.9, 0.9, 0.9, 1.0), n.lines[0])
            return
        text = _col(imgui, 0.92, 0.93, 0.95, 1.0)
        dim = _col(imgui, 0.62, 0.65, 0.70, 1.0)
        for i, line in enumerate(n.lines[:3]):
            draw.add_text(imgui.ImVec2(x0 + 7, y0 + 5 + i * 16 * min(self.zoom, 1.0)),
                          text if i == 0 else dim, line)
        if n.hub:
            draw.add_text(imgui.ImVec2(x0 + 7, y1 - 15 * min(self.zoom, 1.0)),
                          dim, "brain picks next")

    def _arrow(self, imgui, draw, origin, lay: Layout, a: Arrow, selected, focus,
               labelled: bool, slot: int = 0) -> None:
        s, d = lay.nodes[a.src], lay.nodes[a.dst]
        sx0, sy0, sx1, sy1 = self._rect(origin, s)
        dx0, dy0, dx1, dy1 = self._rect(origin, d)
        hot = focus in (a.src, a.dst)
        sel = selected in (a.src, a.dst)
        if hot:
            col = _col(imgui, 0.95, 0.88, 0.45, 1.0)
        elif sel:
            col = _col(imgui, 0.75, 0.68, 0.35, 0.9)
        else:
            col = _col(imgui, 0.42, 0.44, 0.48, 0.6)
        th = 2.2 if hot else 1.6 if sel else 1.0
        if a.src == a.dst:
            p = imgui.ImVec2(sx1 - 10, sy0)
            draw.add_bezier_cubic(p, imgui.ImVec2(sx1 + 24, sy0 - 26),
                                  imgui.ImVec2(sx1 - 44, sy0 - 26),
                                  imgui.ImVec2(sx1 - 30, sy0), col, th)
            return
        if d.x > s.x + NODE_W * 0.5:
            p1 = (sx1, (sy0 + sy1) / 2)
            p4 = (dx0, (dy0 + dy1) / 2)
            bend = max(30.0, (dx0 - sx1) * 0.5)
            p2 = (sx1 + bend, p1[1])
            p3 = (dx0 - bend, p4[1])
        elif d.x < s.x - NODE_W * 0.5:
            p1 = (sx0, (sy0 + sy1) / 2)
            p4 = (dx1, (dy0 + dy1) / 2)
            bend = max(30.0, (sx0 - dx1) * 0.5)
            p2 = (sx0 - bend, p1[1])
            p3 = (dx1 + bend, p4[1])
        else:
            # stacked: leave from the bottom (or top), arrive at the top (or bottom)
            down = d.y > s.y
            p1 = ((sx0 + sx1) / 2, sy1 if down else sy0)
            p4 = ((dx0 + dx1) / 2, dy0 if down else dy1)
            p2 = (p1[0], p1[1] + (40 if down else -40) * self.zoom)
            p3 = (p4[0], p4[1] - (40 if down else -40) * self.zoom)
        v = [imgui.ImVec2(*q) for q in (p1, p2, p3, p4)]
        draw.add_bezier_cubic(v[0], v[1], v[2], v[3], col, th)
        _head(imgui, draw, v[2], v[3], col, 7.0 * min(self.zoom, 1.2))
        if labelled and a.label:
            # on the curve itself, staggered per edge so a fan of labels off one
            # node does not pile up at the same point
            t = (0.30, 0.50, 0.70, 0.40, 0.60)[slot % 5]
            x, y = _bezier(p1, p2, p3, p4, t)
            lab = a.label if len(a.label) < 44 else a.label[:41] + "..."
            _pill(imgui, draw, x, y - 9, lab, _col(imgui, 0.98, 0.94, 0.70, 1.0))


# --------------------------------------------------------------------------- #
def _walk_line(intel, pair: Pair) -> str:
    p = intel.pair(*pair)
    if p.next is None:
        return "(%d,%d): no hand-off intel" % pair
    if not p.next:
        return ("(%d,%d) never ends itself — it stays until the brain or a flinch "
                "moves it" % pair)
    bits = []
    for e in p.next:
        tgt = "/".join("(%d,%d)" % t for t in e.to)
        bits.append("%s%s" % (tgt, " when " + e.reason if e.reason else ""))
    return "(%d,%d) ends -> " % pair + "  |  ".join(bits)


def _bezier(p1, p2, p3, p4, t: float):
    u = 1.0 - t
    x = u**3 * p1[0] + 3 * u * u * t * p2[0] + 3 * u * t * t * p3[0] + t**3 * p4[0]
    y = u**3 * p1[1] + 3 * u * u * t * p2[1] + 3 * u * t * t * p3[1] + t**3 * p4[1]
    return x, y


def _pill(imgui, draw, x: float, y: float, text: str, col: int) -> None:
    """Text on a dark rounded backing, so it reads over arrows and boxes."""
    ts = imgui.calc_text_size(text)
    pad = 4.0
    draw.add_rect_filled(imgui.ImVec2(x - pad, y - 2),
                         imgui.ImVec2(x + float(ts.x) + pad, y + float(ts.y) + 2),
                         _col(imgui, 0.05, 0.06, 0.07, 0.92), 4.0)
    draw.add_text(imgui.ImVec2(x, y), col, text)


def _info_lines(intel, pair: Pair, lay: Layout, moves: dict) -> List[str]:
    """The picked/hovered node as text — the readable form of its arrows."""
    p = intel.pair(*pair)
    n = lay.nodes.get(pair)
    head = "(%d,%d)" % pair
    for name, mv in sorted(moves.items()):
        if (mv.main, mv.sub) == pair:
            head += "  %s -> clip %s" % (name, mv.clip or "?")
            if getattr(mv, "after", None):
                head += "  (after = %s)" % mv.after
    lines = [head]
    if n is not None and n.siblings:
        lines.append("  + %d alike: %s%s" % (len(n.siblings),
                                             " ".join("(%d,%d)" % q for q in n.siblings[:8]),
                                             " ..." if len(n.siblings) > 8 else ""))
    bits = []
    if p.handler:
        bits.append("handler 0x%08X" % p.handler)
    if p.a1:
        bits.append("a1 " + ",".join(str(a) for a in p.a1[:5]) + ("+" if p.a1_computed else ""))
    if p.attack_ids:
        bits.append("attack " + ",".join(str(a) for a in p.attack_ids))
    if p.ends_on and p.ends_on != "unknown":
        bits.append("ends on " + p.ends_on)
    if bits:
        lines.append("  " + "   ".join(bits))
    if p.next:
        lines.append("hands to:")
        for e in p.next[:10]:
            tgt = "/".join("(%d,%d)" % t for t in e.to) or "(computed)"
            lines.append("  -> %-12s %s" % (tgt, " & ".join(e.guards) or "always"))
        if len(p.next) > 10:
            lines.append("  ... %d more" % (len(p.next) - 10))
    elif p.next is not None:
        lines.append("hands to: nothing — never ends by itself")
    if p.prev:
        lines.append("entered from: " + " ".join("(%d,%d)" % q for q in p.prev[:10])
                     + (" ..." if len(p.prev) > 10 else ""))
    else:
        lines.append("entered from: the brain (no handler hands here)")
    if p.measured:
        lines.append("census: entered %s, dwell %.1f ticks" % (p.entered, p.dwell_ticks))
    lines.append("double-click: select in the Action tab")
    return lines


def _info_box(imgui, draw, origin, w: float, intel, lay: Layout, pair: Optional[Pair],
              moves: dict) -> None:
    """Top-right corner of the canvas: the focus node's hand-offs as text."""
    if pair is None or intel is None or intel.pair(*pair) is None:
        lines = ["click a node to read its hand-offs, double-click to select it"]
        dim = True
    else:
        lines = _info_lines(intel, pair, lay, moves)
        dim = False
    lh = float(imgui.get_text_line_height()) + 3.0
    tw = max(float(imgui.calc_text_size(t).x) for t in lines)
    bw, bh = tw + 20.0, lh * len(lines) + 12.0
    x0 = float(origin.x) + w - bw - 8.0
    y0 = float(origin.y) + 8.0
    draw.add_rect_filled(imgui.ImVec2(x0, y0), imgui.ImVec2(x0 + bw, y0 + bh),
                         _col(imgui, 0.05, 0.06, 0.07, 0.90), 6.0)
    draw.add_rect(imgui.ImVec2(x0, y0), imgui.ImVec2(x0 + bw, y0 + bh),
                  _col(imgui, 0.35, 0.37, 0.40, 1.0), 6.0, 1.0)
    for i, t in enumerate(lines):
        if dim:
            col = _col(imgui, 0.55, 0.58, 0.62, 1.0)
        elif i == 0:
            col = _col(imgui, 0.95, 0.95, 0.97, 1.0)
        elif t.startswith("  ->"):
            col = _col(imgui, 0.98, 0.94, 0.70, 1.0)
        elif t.startswith("double-click"):
            col = _col(imgui, 0.50, 0.52, 0.56, 1.0)
        else:
            col = _col(imgui, 0.72, 0.75, 0.80, 1.0)
        draw.add_text(imgui.ImVec2(x0 + 10.0, y0 + 6.0 + i * lh), col, t)


def _legend(imgui, draw, origin, w: float, h: float) -> None:
    x = float(origin.x) + 8
    y = float(origin.y) + h - 18
    for text, rgb in (("move", (0.35, 0.70, 0.95)), ("attacks", (0.95, 0.60, 0.30)),
                      ("hub", (0.45, 0.47, 0.50)), ("picked", (0.95, 0.95, 0.95)),
                      ("Action tab", (1.0, 0.90, 0.35))):
        draw.add_rect_filled(imgui.ImVec2(x, y + 3), imgui.ImVec2(x + 10, y + 13),
                             _col(imgui, *rgb, 1.0), 2.0)
        draw.add_text(imgui.ImVec2(x + 14, y), _col(imgui, 0.7, 0.72, 0.76, 1.0), text)
        x += 14 + 8 * (len(text) + 2)


def _head(imgui, draw, frm, to, col, size: float) -> None:
    dx, dy = float(to.x) - float(frm.x), float(to.y) - float(frm.y)
    n = (dx * dx + dy * dy) ** 0.5 or 1.0
    ux, uy = dx / n, dy / n
    bx, by = float(to.x) - ux * size, float(to.y) - uy * size
    draw.add_triangle_filled(to, imgui.ImVec2(bx - uy * size * 0.5, by + ux * size * 0.5),
                             imgui.ImVec2(bx + uy * size * 0.5, by - ux * size * 0.5), col)


def _inside(rect, pos) -> bool:
    x0, y0, x1, y1 = rect
    return x0 <= float(pos.x) <= x1 and y0 <= float(pos.y) <= y1


def _col(imgui, r: float, g: float, b: float, a: float) -> int:
    return imgui.get_color_u32(imgui.ImVec4(r, g, b, a))


def _button_flags(imgui) -> int:
    return (imgui.ButtonFlags_.mouse_button_left.value
            | imgui.ButtonFlags_.mouse_button_right.value
            | imgui.ButtonFlags_.mouse_button_middle.value)
