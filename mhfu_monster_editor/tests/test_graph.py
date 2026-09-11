"""The Moves tab's LAYOUT, without imgui: `ui.graph.build` over a small chain doc
and over the real em75 file when it is built. The drawing is the panel test's job
(`test_ui_panels.py` runs `_moves_panel` under the imgui stand-in)."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

from mhfu_monster_editor import intel as I                          # noqa: E402
from mhfu_monster_editor.ui import graph as G                       # noqa: E402

DOC = {
    "host_species": 75, "main_states": [],
    "chain": {"hubs": [[0, 1], [0, 2], [2, 2]]},
    "pairs": [
        {"main": 1, "sub": 4, "handler": "0x1", "a1": [17], "attack_ids": [6], "measured": None,
         "next": [{"to": [[0, 3]], "guards": ["phase==3", "!collided", "budget spent"]},
                  {"to": [[0, 6]], "guards": ["phase==3", "collided"], "mode": 1},
                  {"to": [[2, 2]], "guards": ["phase==3", "+0x280!=0"]}], "prev": []},
        {"main": 1, "sub": 3, "handler": "0x2", "a1": [2], "measured": None,
         "next": [{"to": [[0, 1], [0, 2]], "guards": ["phase==1"]}], "prev": []},
        {"main": 0, "sub": 3, "handler": "0x3", "measured": None,
         "next": [{"to": [[0, 1], [0, 2]], "guards": ["phase==1"]},
                  {"to": [[2, 2]], "guards": ["phase==1", "+0x280!=0"]}], "prev": [[1, 4]]},
        {"main": 0, "sub": 6, "handler": "0x4", "measured": None,
         "next": [{"to": [[0, 1], [0, 2]], "guards": []}], "prev": [[1, 4]]},
        {"main": 0, "sub": 1, "handler": "0x5", "measured": None, "next": [], "prev": [[0, 3]]},
        {"main": 0, "sub": 2, "handler": "0x6", "measured": None, "next": [], "prev": [[0, 3]]},
        {"main": 2, "sub": 2, "handler": "0x7", "measured": None, "next": [], "prev": [[1, 4]]},
        {"main": 3, "sub": 9, "handler": "0x8", "attack_ids": [1], "measured": None,
         "next": [{"to": [[0, 1], [0, 2]], "guards": ["phase==2"]}], "prev": []},
        {"main": 3, "sub": 10, "handler": "0x8", "attack_ids": [1], "measured": None,
         "next": [{"to": [[0, 1], [0, 2]], "guards": ["phase==2"]}], "prev": []},
    ]}
MOVES = {"lunge": SimpleNamespace(main=1, sub=4, clip="c", after="lunge_stop"),
         "lunge_stop": SimpleNamespace(main=1, sub=3, clip="s", after=None)}


def _si():
    return I.SpeciesIntel.from_dict(DOC)


def test_moves_scope_roots_the_bound_pairs_and_puts_the_hubs_last():
    lay = G.build(_si(), MOVES, None, "moves")
    assert not lay.empty and lay.note == ""
    layers = {k: n.layer for k, n in lay.nodes.items()}
    assert layers[(1, 4)] == 0 and layers[(1, 3)] == 0
    assert layers[(0, 3)] == 1 and layers[(0, 6)] == 1
    assert layers[(0, 1)] == layers[(0, 2)] == layers[(2, 2)] == 2, layers
    assert lay.nodes[(1, 4)].move == "lunge" and lay.nodes[(1, 4)].attacks
    assert lay.nodes[(0, 1)].hub and not lay.nodes[(0, 3)].hub
    # one arrow per (from, to); the charge's three exits all present, with reasons
    pairs = {(a.src, a.dst): a for a in lay.arrows}
    assert pairs[((1, 4), (0, 3))].label == "!collided & run budget spent"
    assert pairs[((1, 4), (0, 6))].label == "collided"
    assert ((0, 3), (0, 1)) in pairs and ((0, 3), (0, 2)) in pairs
    # a hub is a terminal: nothing leaves it
    assert not any(a.src in ((0, 1), (0, 2), (2, 2)) for a in lay.arrows)
    assert lay.width > 0 and lay.height > 0
    # nodes in one layer never overlap
    ys = sorted(n.y for n in lay.nodes.values() if n.layer == 2)
    assert all(b - a >= G.NODE_H for a, b in zip(ys, ys[1:]))


def test_selected_scope_puts_the_predecessors_in_a_column_to_the_left():
    lay = G.build(_si(), MOVES, (0, 3), "selected")
    layers = {k: n.layer for k, n in lay.nodes.items()}
    assert layers[(1, 4)] == 0 and layers[(0, 3)] == 1 and layers[(0, 1)] == 2, layers
    assert (0, 6) not in lay.nodes            # a sibling, not on the selected path
    assert lay.nodes[(0, 3)].entry


def test_moves_scope_never_reroots_a_pair_that_is_already_in_the_picture():
    """Selecting a hub that is on screen as a terminal must not unfold it — that is
    where the self-loop and the six extra arrows came from. A selected pair that is
    NOT in the picture joins the roots, so the Action tab's pick is always visible."""
    lay = G.build(_si(), MOVES, (0, 1), "moves")
    assert lay.nodes[(0, 1)].hub and not lay.nodes[(0, 1)].entry
    assert not any(a.src == (0, 1) for a in lay.arrows)
    assert set(lay.nodes) == set(G.build(_si(), MOVES, None, "moves").nodes)
    lay2 = G.build(_si(), MOVES, (3, 9), "moves")
    assert (3, 9) in lay2.nodes and lay2.nodes[(3, 9)].entry and lay2.nodes[(3, 9)].layer == 0


def test_the_info_lines_read_the_hand_offs_as_text():
    lay = G.build(_si(), MOVES, None, "moves")
    lines = G._info_lines(_si(), (1, 4), lay, MOVES)
    assert lines[0].startswith("(1,4)  lunge -> clip c  (after = lunge_stop)")
    assert any("-> (0,3)" in t and "run budget spent" in t for t in lines), lines
    assert any("-> (0,6)" in t and "collided" in t for t in lines)
    assert any(t.startswith("entered from: the brain") for t in lines)
    hub = G._info_lines(_si(), (0, 1), lay, MOVES)
    assert any("never ends by itself" in t for t in hub)


def test_attacks_scope_groups_pairs_with_the_same_handler_exits_and_ids():
    lay = G.build(_si(), MOVES, None, "attacks")
    assert (3, 9) in lay.nodes and (3, 10) not in lay.nodes
    n = lay.nodes[(3, 9)]
    assert n.siblings == ((3, 10),) and "+1 alike" in n.lines[0]
    assert (1, 4) in lay.nodes and lay.nodes[(1, 4)].move == "lunge"


def test_no_intel_and_no_roots_are_stated_not_drawn():
    assert G.build(None, MOVES, None).note.startswith("no hand-off intel")
    plain = I.SpeciesIntel.from_dict({"host_species": 75, "main_states": [],
                                      "pairs": [{"main": 1, "sub": 4, "measured": None}]})
    assert G.build(plain, MOVES, None).empty
    lay = G.build(_si(), {}, None, "moves")
    assert lay.empty and "[moves] is empty" in lay.note
    assert "select a pair" in G.build(_si(), {}, None, "selected").note


def test_em75_the_zinogres_chain_lays_out_as_charge_skid_think():
    si = I.find_intel(75)
    if si is None or not si.has_chain:
        print("SKIP: species/em75.json without the chain")
        return
    from mhfu_monster_editor.manifest import load
    m = load(_ROOT / "ports" / "zinogre.toml")
    lay = G.build(si, m.moves, (1, 4), "moves")
    layers = {k: n.layer for k, n in lay.nodes.items()}
    assert layers[(1, 4)] == 0 and layers[(0, 3)] == 1 and layers[(0, 1)] == 2, layers
    assert lay.nodes[(1, 4)].move == "lunge"
    big = G.build(si, m.moves, None, "attacks")
    assert 20 <= len(big.nodes) <= 60, len(big.nodes)
    assert len(big.arrows) < 400, len(big.arrows)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    bad = 0
    for f in fns:
        try:
            f()
            print("  PASS  %s" % f.__name__)
        except Exception as e:
            bad += 1
            import traceback
            traceback.print_exc()
            print("  FAIL  %s: %s" % (f.__name__, e))
    print("\n%d tests, %d failed" % (len(fns), bad))
    sys.exit(1 if bad else 0)
