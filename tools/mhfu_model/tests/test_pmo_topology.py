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


def test_grow_respects_256_cap():
    path = TIGREX_INGAME if os.path.exists(TIGREX_INGAME) else _pacs()[0]
    _, sub = _pmo_of(path)
    header, groups = topo.parse(sub.data)
    g = next(x for x in groups if x.shared_with is None and x.vtype.index_char == "B")
    try:
        topo.grow_group(g, 260, scale=header[2:5])   # >256 verts -> 8-bit index overflow
        raise AssertionError("expected a cap error")
    except ValueError as e:
        assert "256" in str(e), e
    print("  [cap] 256-vert 8-bit-index cap enforced")


if __name__ == "__main__":
    test_roundtrip_preserves_geometry()
    test_grow_adds_geometry_and_repacks()
    test_grow_respects_256_cap()
    print("OK pmo_topology")
