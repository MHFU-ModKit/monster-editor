"""Phase 5 guard: the PMO topology-GROW encoder (pmo_topology.py).

  * a no-edit region rebuild re-parses (via the proven pmo.py) to the SAME vertex
    and face counts on every big-monster PMO (the rebuild is layout-different but
    geometry-faithful);
  * growing a vertex group adds exactly +N verts / +(N-2) faces, survives a PAC
    repack, and re-decodes correctly;
  * the 8-bit index 256-vert cap and shared-block / non-8-bit cases raise clearly.

Run: PYTHONPATH=tools python tools/mhfu_model/tests/test_pmo_topology.py
"""
from __future__ import annotations

import glob
import os

from mhfu_model import pac, pmo
from mhfu_model import pmo_topology as topo

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                    "workspace", "extracted", "data_files")
LO, HI = 6111, 6159
TIGREX_INGAME = os.path.join(DATA, "file_06185.bin")   # the real native-quest model


def _pacs():
    out = []
    for f in sorted(glob.glob(os.path.join(DATA, "file_0[0-9][0-9][0-9][0-9].bin"))):
        n = int(os.path.basename(f)[5:10])
        if LO <= n <= HI:
            out.append(f)
    if os.path.exists(TIGREX_INGAME):
        out.append(TIGREX_INGAME)
    return out


def _pmo_of(path):
    P = pac.MonsterPac.from_bytes(open(path, "rb").read())
    sub = next((s for s in P.subs if s.magic == b"pmo\x00"), None)
    return (P, sub) if sub else (None, None)


def _counts(blob):
    m = pmo.parse(blob)
    return (sum(g.vertex_count for g in m.mesh_groups),
            sum(g.face_count for g in m.mesh_groups))


def test_roundtrip_preserves_geometry():
    """No-edit parse+serialize -> same vert/face totals (via pmo.py) on every PMO."""
    bad = []
    n = 0
    for f in _pacs():
        _, sub = _pmo_of(f)
        if not sub:
            continue
        try:
            v0, fc0 = _counts(sub.data)
            rebuilt = topo.roundtrip_region(sub.data)
            v1, fc1 = _counts(rebuilt)
        except Exception as e:                       # noqa
            bad.append((os.path.basename(f), "exc:%s" % e))
            continue
        n += 1
        if (v0, fc0) != (v1, fc1):
            bad.append((os.path.basename(f), (v0, fc0), (v1, fc1)))
    assert n > 0, "no PMOs found under %s" % DATA
    assert not bad, "topology round-trip changed geometry: %s" % bad[:8]
    print("  [roundtrip] %d PMOs preserved geometry" % n)


def test_grow_adds_geometry_and_repacks():
    path = TIGREX_INGAME if os.path.exists(TIGREX_INGAME) else _pacs()[0]
    P, sub = _pmo_of(path)
    blob = sub.data
    v0, fc0 = _counts(blob)

    header, groups = topo.parse(blob)
    g = next(x for x in groups if x.shared_with is None and x.vtype.index_char == "B")
    topo.grow_group(g, 6, shift=(0, 200, 0), scale=header[2:5])
    grown = topo.serialize(blob, header, groups)
    v1, fc1 = _counts(grown)
    assert (v1, fc1) == (v0 + 6, fc0 + 4), ((v0, fc0), (v1, fc1))
    assert len(grown) > len(blob), "grown PMO should be larger"

    # PAC repack (offsets re-flow) -> re-decode still shows the growth
    idx = sub.index
    P.subs[idx] = pac.SubResource(idx, grown)
    repacked = P.to_bytes()
    P2 = pac.MonsterPac.from_bytes(repacked)
    blob2 = next(s for s in P2.subs if s.magic == b"pmo\x00").data
    v2, _ = _counts(blob2)
    assert v2 == v0 + 6, (v2, v0)
    print("  [grow] %s %d->%d verts, %d->%d faces, PAC %d->%d B"
          % (os.path.basename(path), v0, v1, fc0, fc1, len(blob), len(grown)))


def test_grow_past_256_auto_promotes_to_16bit():
    """Growing an 8-bit group past 256 verts auto-promotes it to 16-bit indices and
    succeeds; the engine-format re-decode (pmo.py) confirms +N verts."""
    import struct
    path = TIGREX_INGAME if os.path.exists(TIGREX_INGAME) else _pacs()[0]
    P, sub = _pmo_of(path)
    blob = sub.data
    v0, fc0 = _counts(blob)
    header, groups = topo.parse(blob)
    g = next(x for x in groups if x.shared_with is None and x.vtype.index_char == "B")
    base = g.vcount
    n = 256 - base + 12                       # push comfortably past 256
    assert n >= 3
    topo.grow_group(g, n, shift=(0, 200, 0), scale=header[2:5], spread=80.0)
    assert g.vtype.index_char == "H", "should have promoted to 16-bit"
    # the VTYPE word's index field must now read 2 (16-bit)
    vw = g.words[g.vtype_widx]
    assert (vw >> 11) & 3 == 2, "VTYPE index field not promoted"
    grown = topo.serialize(blob, header, groups)
    v1, fc1 = _counts(grown)
    assert (v1, fc1) == (v0 + n, fc0 + (n - 2)), ((v0, fc0), (v1, fc1), n)
    print("  [16bit] promoted group, %d->%d verts past the 256 cap" % (v0, v1))


def test_grow_respects_hard_cap():
    """Even 16-bit can't exceed 65536 verts in a group."""
    path = TIGREX_INGAME if os.path.exists(TIGREX_INGAME) else _pacs()[0]
    _, sub = _pmo_of(path)
    header, groups = topo.parse(sub.data)
    g = next(x for x in groups if x.shared_with is None and x.vtype.index_char == "B")
    try:
        topo.grow_group(g, 70000, scale=header[2:5], spread=80.0)
        raise AssertionError("expected a cap error")
    except ValueError as e:
        assert "65536" in str(e) or "cap" in str(e), e
    print("  [cap] 65536-vert hard cap enforced")


def test_new_verts_have_real_attributes():
    """New verts get varied UVs, ~unit normals, and a single-bone clean weight (not a
    copy of vertex0)."""
    import struct
    path = TIGREX_INGAME if os.path.exists(TIGREX_INGAME) else _pacs()[0]
    _, sub = _pmo_of(path)
    header, groups = topo.parse(sub.data)
    g = next(x for x in groups if x.shared_with is None and x.vtype.index_char == "B"
             and x.vtype.uv_off is not None and x.vtype.wt_count)
    vt = g.vtype
    base = g.vcount
    n = 12
    topo.grow_group(g, n, shift=(0, 200, 0), scale=header[2:5], spread=80.0,
                    weight_slot=0)
    uvs, wsums, wnz = set(), [], []
    for k in range(base, base + n):
        v = g.vbuf[k * vt.vsize:(k + 1) * vt.vsize]
        u, w = struct.unpack_from("<2%s" % vt.uv_char, v, vt.uv_off)
        uvs.add((u, w))
        ws = struct.unpack_from("<%d%s" % (vt.wt_count, vt.wt_char), v, vt.wt_off)
        nz = [i for i, x in enumerate(ws) if x]
        wnz.append(nz)
        wsums.append(sum(ws))
    assert len(uvs) > 1, "UVs should vary across the new verts, got %d" % len(uvs)
    assert all(nz == [0] for nz in wnz), "weights should be single-bone on slot 0: %s" % wnz
    # weight slot 0 should be the format's full-weight quantum
    full = {"B": 0x80, "H": 0x8000, "f": 1.0}[vt.wt_char]
    assert all(abs(s - full) < 1e-3 or s == full for s in wsums), wsums
    print("  [attrs] %d new verts: %d distinct UVs, single-bone weights" % (n, len(uvs)))


def test_grow_explicit_blender_path():
    """grow_group_explicit (the Blender add path): author-supplied verts + tris with
    real positions/UV/normals re-decode correctly. Positions must stay within the
    group's per-axis scale bounds (existing verts are stored as fractions of `scale`,
    so coords past it saturate)."""
    path = TIGREX_INGAME if os.path.exists(TIGREX_INGAME) else _pacs()[0]
    _, sub = _pmo_of(path)
    blob = sub.data
    v0, fc0 = _counts(blob)
    header, groups = topo.parse(blob)
    g = next(x for x in groups if x.shared_with is None
             and x.vtype.uv_off is not None and x.vtype.wt_count)
    sx, sy, sz = header[2:5]
    base = g.vcount
    # in-bounds quad (|coord| < per-axis scale) so positions round-trip cleanly
    qx, qy, qz = sx * 0.1, sy * 0.5, sz * 0.1
    nv = [{"x": 0, "y": qy, "z": 0, "i": 0, "j": 1, "k": 0, "u": 0.1, "v": 0.1},
          {"x": qx, "y": qy, "z": 0, "i": 0, "j": 1, "k": 0, "u": 0.9, "v": 0.1},
          {"x": qx, "y": qy, "z": qz, "i": 0, "j": 1, "k": 0, "u": 0.9, "v": 0.9},
          {"x": 0, "y": qy, "z": qz, "i": 0, "j": 1, "k": 0, "u": 0.1, "v": 0.9}]
    tris = [(base, base + 1, base + 2), (base, base + 2, base + 3)]
    topo.grow_group_explicit(g, nv, tris, scale=header[2:5])
    grown = topo.serialize(blob, header, groups)
    v1, fc1 = _counts(grown)
    assert (v1, fc1) == (v0 + 4, fc0 + 2), ((v0, fc0), (v1, fc1))
    tg = [gg for gg in pmo.parse(grown).mesh_groups if gg.vg_rec == g.rec_index][0]
    last = tg.vertices[-4:]
    assert abs(last[1]["x"] - qx) < 1.0, last[1]
    assert all(v["weights"][0] == 1.0 and not any(v["weights"][1:]) for v in last), last
    assert len({(round(v["u"], 3), round(v["v"], 3)) for v in last}) > 1
    print("  [explicit] Blender add path: +4 verts, real UV/normal/clean weights")


if __name__ == "__main__":
    test_roundtrip_preserves_geometry()
    test_grow_adds_geometry_and_repacks()
    test_grow_past_256_auto_promotes_to_16bit()
    test_grow_respects_hard_cap()
    test_new_verts_have_real_attributes()
    test_grow_explicit_blender_path()
    print("OK pmo_topology")
