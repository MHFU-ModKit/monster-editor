"""The collision half of an edit list: does the broadphase still find everything?

`stage_collide.plan` is the piece that makes a map edit honest — a moved object that
leaves its collision behind is a wall you cannot see and a chest you walk through. The
planner is pure (no emulator), so everything it does can be checked here: that a moved
triangle is unlinked from the cells it left and linked into the ones it entered, that a
synthesised solid box is closed and outward-wound, and that the relinked cells are a
SUPERSET of the exact overlap set — the one property the shipped data guarantees and a
rebuild must not lose (docs/STAGE_MAP_FORMAT.md §4).
"""
from __future__ import annotations

import os
import sys

TOOLS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)

from mhfu_model import stage as ST                                   # noqa: E402
import stage_collide as SC                                           # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                    "workspace", "extracted", "data_files")
STAGE = 98


def _st():
    return ST.load(DATA, STAGE)


def test_box_is_closed_and_wound_outward():
    tris = SC.box_tris((0, 0, 0), (100, 200, 300))
    assert len(tris) == 12
    cen = (50.0, 100.0, 150.0)
    for t in tris:
        mid = [sum(v[k] for v in (t.v0, t.v1, t.v2)) / 3.0 for k in range(3)]
        out = [mid[k] - cen[k] for k in range(3)]
        assert sum(out[k] * t.normal[k] for k in range(3)) > 0, "inward-facing face"
        assert abs(sum(t.normal[k] * t.v0[k] for k in range(3)) + t.plane_d) < 1e-3
    print("  [collide] 12 faces, all outward, all planes exact")


def test_a_move_relinks_the_cells_it_left_and_entered():
    st = _st()
    ops = [{"op": "move", "group": 11, "sphere": [14450, 1100, 16250, 400],
            "by": [0, 0, -3000], "collision": {"chunks": [1]}}]
    p = SC.plan(st, ops, verbose=False)
    assert p["moved"], "nothing selected — the fixture stopped matching the stage"
    ch = p["chunks"][1]
    before = st.cells(1)
    for ci, ti, t in p["moved"]:
        want = set(ST.cells_for_tri(t, ch.grid, ch.cell, ch.origin))
        for cidx in want:
            assert ("t", 1, ti) in p["cells"][1][cidx], \
                "triangle %d is not listed in cell %d it now overlaps" % (ti, cidx)
        for cidx, members in enumerate(before):
            if ti in members and cidx not in want:
                assert ("t", 1, ti) not in p["cells"][1].get(cidx, []), \
                    "triangle %d still listed in cell %d it left" % (ti, cidx)
    print("  [collide] %d moved triangles relinked across %d cells"
          % (len(p["moved"]), len(p["cells"][1])))


def test_added_triangles_are_reachable_from_every_cell_they_overlap():
    st = _st()
    obj = os.path.join(os.path.dirname(__file__), "_slab.obj")
    with open(obj, "w") as fh:
        fh.write("v 13000 1400 15400\nv 13600 1400 15400\n"
                 "v 13600 1400 16000\nv 13000 1400 16000\nf 1 2 3\nf 1 3 4\n")
    try:
        ops = [{"op": "pack", "group": 11, "obj": obj, "solid": True, "chunk": 1}]
        p = SC.plan(st, ops, verbose=False)
        assert len(p["added"]) == 2, p["added"]
        ch = p["chunks"][1]
        for k, (ci, t) in enumerate(p["added"]):
            for cidx in ST.cells_for_tri(t, ch.grid, ch.cell, ch.origin):
                assert ("a", k) in p["cells"][ci][cidx], \
                    "added triangle %d missing from cell %d" % (k, cidx)
        print("  [collide] %d added triangles listed in %d cells"
              % (len(p["added"]), len(p["cells"][1])))
    finally:
        os.remove(obj)


def test_a_relinked_cell_never_loses_a_triangle_it_used_to_have():
    """Conservative is safe, missing is not — a rebuild may list MORE, never less."""
    st = _st()
    ops = [{"op": "move", "group": 11, "sphere": [14450, 1100, 16250, 400],
            "by": [40, 0, 40], "collision": {"chunks": [0, 1]}}]
    p = SC.plan(st, ops, verbose=False)
    for ci, cells in p["cells"].items():
        was = st.cells(ci)
        movedset = {ti for c, ti, _t in p["moved"] if c == ci}
        for cidx, members in cells.items():
            have = {m[2] for m in members if m[0] == "t"}
            lost = set(was[cidx]) - have - movedset
            assert not lost, "cell %d of chunk %d lost %s" % (cidx, ci, sorted(lost)[:4])
    print("  [collide] no untouched triangle fell out of a relinked cell")


if __name__ == "__main__":
    test_box_is_closed_and_wound_outward()
    test_a_move_relinks_the_cells_it_left_and_entered()
    test_added_triangles_are_reachable_from_every_cell_they_overlap()
    test_a_relinked_cell_never_loses_a_triangle_it_used_to_have()
    print("OK stage_collision_plan")
