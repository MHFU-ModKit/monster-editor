"""Guard: ADDING visible geometry to a stage (pmo_topology on stage PMOs).

  * parse -> serialize with no edits is BYTE-IDENTICAL on every shipped stage
    terrain/props PMO. This is the oracle the whole add path rests on: it means the
    re-layout reproduces retail's exactly, so any diff after an edit is the edit.
  * a triangle LIST does not alternate winding (pmo.run_ge). Retail never shows this
    -- every shipped tri-list PRIM is a single triangle -- but the grow path emits
    multi-triangle lists, so it has to be pinned here.
  * growing a group lands the new vertices where they were asked for (within the s16
    step), the INDEPENDENT decoder sees them, and nothing else in the PAC moves.
  * sculpt_group -- the edit that actually SHOWS UP in a running game -- changes not a
    single GE word, and puts the triangles it was given where they were asked for.
  * a PRIM added to a GE list lands before the list's closing OFFSETADDR, not after it:
    on the far side of that reset the draw resolves its vertex pointer to nothing and
    is silently invisible.
  * overwrite_group -- the RESIDENT path -- leaves the vgroup TABLE byte-identical and
    the PMO the same length. That is the property that decides whether an injection
    into a running game shows up cleanly or corrupts the frame on the spot, so it is
    checked rather than argued.

Run: PYTHONPATH=tools python tools/mhfu_model/tests/test_stage_topology.py
"""
from __future__ import annotations

import os

from mhfu_model import pmo, stage as ST
from mhfu_model import pmo_topology as topo

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                    "workspace", "extracted", "data_files")


def _stage_pmos():
    for n in sorted({s for _r, _n, ss in ST.map_table(DATA) for s in ss}):
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


def test_overwrite_leaves_the_vgroup_table_byte_identical():
    """The resident-injection contract: same length, and not one byte of the vgroup
    table moves. A changed I3/I4/I5 is read by the engine mid-frame and corrupts it."""
    s = ST.load(DATA, 98)
    blob = s.sub(0)
    header, groups = topo.parse(blob)
    verts = [{"x": 14300.0 + 100 * (i % 3), "y": 2100.0 + 100 * (i // 3),
              "z": 15600.0 + 60 * i, "u": 0.0, "v": 0.0} for i in range(6)]
    tris = [(0, 1, 2), (0, 2, 3), (3, 4, 5)]
    topo.overwrite_group(groups[9], verts, tris, scale=header[2:5])
    topo.clear_group(groups[12], drop_vertices=False, keep_layout=True)
    out, moved = topo.serialize_inplace(blob, header, groups)
    t0, t1 = header[8], header[9]
    assert len(out) == len(blob), (len(out), len(blob))
    assert moved == [], moved
    assert out[t0:t1] == blob[t0:t1], "vgroup table moved"

    m = pmo.parse(out)
    by = {d.vg_rec: d for d in m.mesh_groups}
    assert len(by[9].faces) == 3, len(by[9].faces)
    assert len(by[12].faces) == 0, len(by[12].faces)
    step = max(header[2:5]) / 32767.0
    got = by[9].vertices
    for want, have in zip(verts, got):
        assert abs(want["x"] - have["x"]) <= step, (want, have)
        assert abs(want["y"] - have["y"]) <= step, (want, have)
        assert abs(want["z"] - have["z"]) <= step, (want, have)
    nb = sum(1 for i in range(len(blob)) if out[i] != blob[i])
    assert nb < 4000, "%d bytes differ -- a resident edit has to be small" % nb
    print("  [overwrite] st098 g9 replaced, g12 cleared: table identical, %d bytes diff"
          % nb)


def test_overwrite_refuses_what_will_not_fit():
    """The two capacity limits are errors, not silent corruption."""
    s = ST.load(DATA, 98)
    blob = s.sub(0)
    header, groups = topo.parse(blob)
    g = groups[12]
    big = [{"x": 0.0, "y": 0.0, "z": 0.0} for _ in range(g.vcount + 1)]
    for fn, args in ((topo.overwrite_group, (g, big, [(0, 1, 2)])),
                     (topo.overwrite_group,
                      (g, [{"x": 0.0, "y": 0.0, "z": 0.0} for _ in range(3)],
                       [(0, 1, 2)] * 5000))):
        try:
            fn(*args, scale=header[2:5])
        except ValueError:
            pass
        else:
            raise AssertionError("%s accepted an edit that does not fit" % fn.__name__)
    print("  [overwrite] too many vertices / too many indices both refused")


def test_sculpt_changes_no_ge_word_and_lands_where_asked():
    s = ST.load(DATA, 98)
    blob = s.sub(0)
    header, groups = topo.parse(blob)
    before = list(groups[9].words)
    verts = [{"x": 14300.0, "y": 1500.0, "z": 16200.0},
             {"x": 14700.0, "y": 1500.0, "z": 16200.0},
             {"x": 14700.0, "y": 1900.0, "z": 16200.0},
             {"x": 14300.0, "y": 1900.0, "z": 16200.0}]
    tris = [(0, 1, 2), (0, 2, 3)]
    r = topo.sculpt_group(groups[9], verts, tris, scale=header[2:5])
    assert r["used"] == 2, r
    assert groups[9].words == before, "sculpt rewrote a GE command"
    out, moved = topo.serialize_inplace(blob, header, groups)
    t0, t1 = header[8], header[9]
    assert moved == [] and len(out) == len(blob) and out[t0:t1] == blob[t0:t1]

    m = pmo.parse(out)
    d = {x.vg_rec: x for x in m.mesh_groups}[9]
    step = max(header[2:5]) / 32767.0
    placed = {(round(v["x"]), round(v["y"]), round(v["z"])) for v in d.vertices}
    for want in verts:
        assert any(abs(want["x"] - x) <= step and abs(want["y"] - y) <= step
                   and abs(want["z"] - z) <= step for x, y, z in placed), want
    print("  [sculpt] st098 g9: 2 triangles placed, GE list untouched")


def test_sculpt_can_leave_the_rest_of_the_group_alone():
    """`collapse=False` is what lets a small object be carved out of a big group
    without flattening the scenery that supplies its texture."""
    s = ST.load(DATA, 98)
    blob = s.sub(0)
    header, groups = topo.parse(blob)
    kept = bytes(groups[11].vbuf)
    verts = [{"x": 14000.0, "y": 1500.0, "z": 15000.0},
             {"x": 14200.0, "y": 1500.0, "z": 15000.0},
             {"x": 14200.0, "y": 1700.0, "z": 15000.0}]
    r = topo.sculpt_group(groups[11], verts, [(0, 1, 2)], scale=header[2:5],
                          first=32, collapse=False)
    assert r["used"] == 1, r
    changed = sum(1 for a, b in zip(kept, groups[11].vbuf) if a != b)
    assert 0 < changed < 200, "%d vertex bytes changed for one triangle" % changed
    print("  [sculpt] collapse=False touched %d bytes of g11 for one triangle" % changed)


def test_a_new_prim_lands_before_the_lists_offset_reset():
    """🔴 GE lists here open with ORIGIN_ADDR and close with OFFSETADDR 0. A PRIM
    appended after that reset draws nothing at all -- the failure that looks exactly
    like the edit never happening."""
    s = ST.load(DATA, 98)
    blob = s.sub(0)
    header, groups = topo.parse(blob)
    g = groups[9]
    resets = [i for i, w in enumerate(g.words) if (w >> 24) == 0x13]
    assert resets, "st098 g9 was chosen because it HAS a closing OFFSETADDR"
    verts = [{"x": 14400.0 + 100 * i, "y": 1500.0, "z": 16200.0} for i in range(3)]
    topo.overwrite_group(g, verts, [(0, 1, 2)], scale=header[2:5])
    prims = [i for i, w in enumerate(g.words) if (w >> 24) == 0x04]
    resets = [i for i, w in enumerate(g.words) if (w >> 24) == 0x13]
    assert prims and max(prims) < min(resets), (prims, resets)
    print("  [order] PRIM at %d, OFFSETADDR reset at %d" % (prims[0], resets[0]))


def _tetra(cx, cy, cz, r=200.0):
    """A closed 4-face shape — connected, so the stripifier has real work to do."""
    v = [{"x": cx, "y": cy + r, "z": cz},
         {"x": cx - r, "y": cy - r, "z": cz - r},
         {"x": cx + r, "y": cy - r, "z": cz - r},
         {"x": cx, "y": cy - r, "z": cz + r}]
    return v, [(0, 1, 2), (0, 2, 3), (0, 3, 1), (1, 3, 2)]


def _grid_mesh(cx, cy, cz, n=6, step=80.0):
    """An n x n quad grid — the connected case a strip packer should pack densely."""
    v = [{"x": cx + i * step, "y": cy, "z": cz + j * step}
         for j in range(n + 1) for i in range(n + 1)]
    t = []
    for j in range(n):
        for i in range(n):
            a = j * (n + 1) + i
            t += [(a, a + 1, a + n + 1), (a + 1, a + n + 2, a + n + 1)]
    return v, t


def test_pack_fills_a_strip_instead_of_wasting_it():
    """The whole point: a 4-vertex strip is a QUAD, not one triangle."""
    blob = ST.load(DATA, 98).sub(0)
    header, groups = topo.parse(blob)
    g = groups[9]                                   # 158 strips of exactly 4 vertices
    b = topo.group_budget(g)
    assert b["tri"] == 158 and b["strip"] == 316, b
    verts, tris = _grid_mesh(14000.0, 1500.0, 16000.0, n=6)      # 72 triangles
    before = list(g.words)
    r = topo.pack_group(g, verts, tris, scale=header[2:5], collapse=False)
    assert g.words == before, "pack_group must not touch a GE word"
    assert r["triangles"] == len(tris), r
    assert r["used_prims"] <= len(tris) // 2, r      # every 4-slot strip drew a quad
    print("  [pack] %d triangles into %d of %d prims, budget %d (one-per-prim %d)"
          % (r["triangles"], r["used_prims"], r["prims"], b["strip"], b["tri"]))


def test_pack_lands_where_asked_and_keeps_the_table():
    blob = ST.load(DATA, 98).sub(0)
    verts, tris = _tetra(14500.0, 1500.0, 16000.0)
    for reindex in (True, False):
        header, groups = topo.parse(blob)
        g = groups[11]
        ibytes = list(g.indices)
        # collapse=True over an explicit RANGE is the reindex-friendly shape: every
        # primitive in [first, first+count) is ours, so every vertex in it is ours too.
        r = topo.pack_group(g, verts, tris, scale=header[2:5], first=0, count=64,
                            collapse=True, reindex=reindex)
        assert r["triangles"] == len(tris), (reindex, r)
        if not reindex:
            assert g.indices == ibytes, "reindex=False must not write an index"
        else:
            assert g.indices != ibytes, "reindex=True should have rewritten indices"
        out, moved = topo.serialize_inplace(blob, header, groups)
        assert not moved and len(out) == len(blob)
        t0, t1 = header[8], header[9]
        assert out[t0:t1] == blob[t0:t1], "vgroup table moved"
        m = pmo.parse(out)
        d = [x for x in m.mesh_groups if x.vg_rec == 11][0]
        src = [(v["x"], v["y"], v["z"]) for v in verts]
        on = 0
        for f in d.faces:
            vv = [d.vertices[f[k]] for k in ("v1", "v2", "v3")]
            if any(v is None for v in vv):
                continue
            if all(min(abs(v["x"] - s[0]) + abs(v["y"] - s[1]) + abs(v["z"] - s[2])
                       for s in src) <= 4.0 for v in vv):
                on += 1
        assert on >= len(tris), (reindex, on, len(tris))
        print("  [pack] reindex=%-5s %d faces land on the shape (source %d)"
              % (reindex, on, len(tris)))


def test_pack_never_writes_a_vertex_another_primitive_owns():
    """reindex=False has to skip a primitive that shares a vertex — check it does."""
    blob = ST.load(DATA, 98).sub(0)
    header, groups = topo.parse(blob)
    g = groups[11]
    keep_from = 300
    before = bytes(g.vbuf)
    owned = set(g.indices[sum(c for _w, _t, c in g.prims[:keep_from]):])
    verts, tris = _grid_mesh(14000.0, 1500.0, 16000.0, n=8)
    topo.pack_group(g, verts, tris, scale=header[2:5], first=0, count=keep_from,
                    collapse=False, reindex=False)
    vs = g.vtype.vsize
    hurt = [i for i in sorted(owned)
            if i < len(before) // vs and g.vbuf[i * vs:(i + 1) * vs] != before[i * vs:(i + 1) * vs]]
    assert not hurt, "wrote %d vertices that primitives 300+ still draw" % len(hurt)
    print("  [pack] %d vertices owned by the untouched primitives are intact" % len(owned))


if __name__ == "__main__":
    test_roundtrip_is_byte_identical()
    test_triangle_list_winding_does_not_alternate()
    test_grow_lands_where_asked_and_moves_nothing_else()
    test_solid_keeps_the_broadphase_complete()
    test_overwrite_leaves_the_vgroup_table_byte_identical()
    test_overwrite_refuses_what_will_not_fit()
    test_sculpt_changes_no_ge_word_and_lands_where_asked()
    test_sculpt_can_leave_the_rest_of_the_group_alone()
    test_a_new_prim_lands_before_the_lists_offset_reset()
    test_pack_fills_a_strip_instead_of_wasting_it()
    test_pack_lands_where_asked_and_keeps_the_table()
    test_pack_never_writes_a_vertex_another_primitive_owns()
    print("OK stage_topology")
