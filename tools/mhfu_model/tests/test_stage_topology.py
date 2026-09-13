"""Guard: ADDING visible geometry to a stage (pmo_topology on stage PMOs).

  * parse -> serialize with no edits is BYTE-IDENTICAL on every shipped stage
    terrain/props PMO. This is the oracle the whole add path rests on: it means the
    re-layout reproduces retail's exactly, so any diff after an edit is the edit.
  * a triangle LIST does not alternate winding (pmo.run_ge). Retail never shows this
    -- every shipped tri-list PRIM is a single triangle -- but the grow path emits
    multi-triangle lists, so it has to be pinned here.
  * growing a group lands the new vertices where they were asked for (within the s16
    step), the INDEPENDENT decoder sees them, and nothing else in the PAC moves.

Run: PYTHONPATH=tools python tools/mhfu_model/tests/test_stage_topology.py
"""
from __future__ import annotations

import os

from mhfu_model import pmo, stage as ST
from mhfu_model import pmo_topology as topo

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                    "workspace", "extracted", "data_files")


def _stage_pmos():
    for n in sorted({s for _r, _f, ss in ST.map_table(DATA) for s in ss if s}):
        s = ST.load(DATA, n)
        for k in (0, 2):
            blob = s.sub(k)
            if blob and blob[:4] == b"pmo\x00":
                yield n, k, s, blob


def test_roundtrip_is_byte_identical():
    seen = bad = 0
    for n, k, _s, blob in _stage_pmos():
        seen += 1
        header, groups = topo.parse(blob)
        if topo.serialize(blob, header, groups) != blob:
            bad += 1
            print("   st%03d sub[%d] DIFFERS" % (n, k))
    assert seen > 400, seen
    assert bad == 0, "%d/%d stage PMOs did not round-trip byte-identically" % (bad, seen)
    print("  [roundtrip] %d stage PMOs re-encode byte for byte" % seen)


def test_triangle_list_winding_does_not_alternate():
    """A quad added as one tri-list PRIM must come back (0,1,2),(0,2,3) -- not with the
    second triangle mirrored, which is what the strip rule does to a list."""
    s = ST.load(DATA, 98)
    blob = s.sub(0)
    header, groups = topo.parse(blob)
    g = groups[6]
    base = g.vcount
    X, Y, Z, R = 15000.0, 1500.0, 15000.0, 300.0
    verts = [dict(x=X - R, y=Y, z=Z - R, u=0.0, v=0.0),
             dict(x=X + R, y=Y, z=Z - R, u=1.0, v=0.0),
             dict(x=X + R, y=Y, z=Z + R, u=1.0, v=1.0),
             dict(x=X - R, y=Y, z=Z + R, u=0.0, v=1.0)]
    tris = [(base, base + 1, base + 2), (base, base + 2, base + 3)]
    topo.grow_group_explicit(g, verts, tris, scale=header[2:5])
    out = topo.serialize(blob, header, groups)
    d = [x for x in pmo.parse(out).mesh_groups if x.vg_rec == 6][0]
    got = [(f["v1"], f["v2"], f["v3"]) for f in d.faces if max(f.values()) >= base]
    assert got == tris, got
    print("  [winding] tri-list keeps its winding: %s" % (got,))


def test_grow_lands_where_asked_and_moves_nothing_else():
    s = ST.load(DATA, 98)
    blob = s.sub(0)
    header, groups = topo.parse(blob)
    g = groups[11]
    base = g.vcount
    want = [(14600.0, 1500.0, 14600.0), (15400.0, 1500.0, 14600.0),
            (15400.0, 1500.0, 15400.0), (14600.0, 1500.0, 15400.0)]
    verts = [dict(x=x, y=y, z=z, u=0.0, v=0.0) for x, y, z in want]
    topo.grow_group_explicit(
        g, verts, [(base, base + 1, base + 2), (base, base + 2, base + 3)],
        scale=header[2:5], src_indices=[0, 1, 2, 3])
    out = topo.serialize(blob, header, groups)

    step = max(header[2:5]) / 32767.0
    d = [x for x in pmo.parse(out).mesh_groups if x.vg_rec == 11][0]
    assert len(d.vertices) == base + 4, len(d.vertices)
    for got, (x, y, z) in zip(d.vertices[base:], want):
        off = max(abs(got["x"] - x), abs(got["y"] - y), abs(got["z"] - z))
        assert off <= step, (got, (x, y, z), off, step)

    # every OTHER group is untouched, and so is every other sub-resource
    before = {x.vg_rec: len(x.vertices) for x in pmo.parse(blob).mesh_groups}
    after = {x.vg_rec: len(x.vertices) for x in pmo.parse(out).mesh_groups}
    assert {k: v for k, v in after.items() if k != 11} == \
           {k: v for k, v in before.items() if k != 11}
    pac = s.with_sub(0, out)
    chk = ST.Stage(98, pac, ST.subresources(pac))
    for k in (1, 2, 3, 4, 5):
        assert chk.sub(k) == s.sub(k), "sub[%d] changed" % k
    print("  [grow] +4 verts within %.1f units, 12 other groups and subs 1-5 untouched"
          % step)


def test_solid_keeps_the_broadphase_complete():
    """The --solid half: the same triangles in the collision mesh must be reachable
    through the shipped lattice, or the hunter falls through what he can see."""
    s = ST.load(DATA, 98)
    quad = [(14600.0, 1500.0, 14600.0), (15400.0, 1500.0, 14600.0),
            (15400.0, 1500.0, 15400.0), (14600.0, 1500.0, 15400.0)]
    extra = [ST.Tri.from_verts(quad[0], quad[1], quad[2]),
             ST.Tri.from_verts(quad[0], quad[2], quad[3])]
    chunks = []
    for ch in s.hits_chunks():
        lst = s.tri_list(ch.index) + (extra if ch.index == 1 else [])
        chunks.append(ST.build_hits(lst, ch.grid, ch.cell, ch.origin))
    pac = s.with_collision(chunks)
    chk = ST.Stage(98, pac, ST.subresources(pac))
    assert chk.verify()["bad"] == 0, chk.verify()
    for ch in chk.hits_chunks():
        assert chk.grid_check(ch.index)["missing"] == 0, (ch.index,
                                                          chk.grid_check(ch.index))
    print("  [solid] +2 collision triangles, planes clean, broadphase complete")


if __name__ == "__main__":
    test_roundtrip_is_byte_identical()
    test_triangle_list_winding_does_not_alternate()
    test_grow_lands_where_asked_and_moves_nothing_else()
    test_solid_keeps_the_broadphase_complete()
    print("OK stage_topology")
