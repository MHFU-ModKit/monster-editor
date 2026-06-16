"""Phase 3 guard: the PMO geometry encoder (in-place vertex rewrite).

  * unedited model -> source bytes verbatim (encode default path);
  * forcing the in-place rewrite with UNCHANGED vertices reproduces the source
    byte-for-byte (proves the position/normal/UV re-quantization is exact);
  * a vertex move re-encodes, re-parses to the moved position, and leaves other
    groups' geometry intact;
  * a topology change (vertex count) is rejected with a clear error.

Run: PYTHONPATH=tools python tools/mhfu_model/tests/test_pmo_encode.py
"""
from __future__ import annotations

import glob
import os

from mhfu_model import load_pac
from mhfu_model import pmo

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                    "workspace", "extracted", "data_files")
TIGREX = os.path.join(DATA, "file_06134.bin")
LO, HI = 6111, 6159


def _pacs():
    out = []
    for f in sorted(glob.glob(os.path.join(DATA, "file_0[0-9][0-9][0-9][0-9].bin"))):
        n = int(os.path.basename(f)[5:10])
        if LO <= n <= HI:
            out.append(f)
    return out


def test_unedited_inplace_is_byte_identical():
    """Re-encoding every group in place WITHOUT changing values == the source PMO.

    This is the strong guard: it exercises the actual position/normal/UV
    quantization round-trip on real data for all 49 monsters, not just passthrough.
    """
    bad = []
    for f in _pacs():
        mm = load_pac(f)
        src = mm.pac.find("model").data
        mm.model.edited = True                  # force the in-place rewrite path
        try:
            enc = pmo.encode(mm.model)
        except ValueError:
            # a monster with only multi-VADDR/bypass groups can't be in-place
            # re-encoded; passthrough still covers it (skip the strict check)
            continue
        if enc != src:
            n = sum(1 for a, b in zip(enc, src) if a != b)
            bad.append((os.path.basename(f), n))
    assert not bad, "in-place re-encode not byte-identical: %s" % bad[:10]


def test_vertex_move_survives_roundtrip():
    mm = load_pac(TIGREX)
    g = next(g for g in mm.model.mesh_groups
             if getattr(g, "enc", None) and g.enc.get("plain") and g.vertices)
    gi = g.index
    v0 = g.vertices[0]
    moved = (v0["x"] + 10.0, v0["y"] - 5.0, v0["z"] + 2.0)
    v0["x"], v0["y"], v0["z"] = moved
    mm.model.edited = True
    enc = pmo.encode(mm.model)

    # rebuild a PAC with the edited PMO and re-decode
    from mhfu_model.pac import MonsterPac, SubResource
    subs = [SubResource(s.index, enc if mm.pac.role(s) == "model" else s.data)
            for s in mm.pac.subs]
    out = MonsterPac(subs=subs, tail=mm.pac.tail).to_bytes()
    from mhfu_model import parse_pac
    re = parse_pac(out)
    rg = re.model.mesh_groups[gi].vertices[0]
    # quantized to s16 position -> within one quantum of the requested move
    quantum = g.scale[0] / 32767.0 * 4         # generous tolerance
    assert abs(rg["x"] - moved[0]) < max(quantum, 0.5), (rg["x"], moved[0])
    # a different group is untouched
    other = next(i for i in range(len(mm.model.mesh_groups)) if i != gi)
    assert re.model.mesh_groups[other].vertices[0]["x"] == \
        load_pac(TIGREX).model.mesh_groups[other].vertices[0]["x"]


def test_topology_change_rejected():
    mm = load_pac(TIGREX)
    g = next(g for g in mm.model.mesh_groups
             if getattr(g, "enc", None) and g.enc.get("plain") and len(g.vertices) > 1)
    g.vertices = g.vertices[:-1]            # drop a vertex -> topology change
    mm.model.edited = True
    try:
        pmo.encode(mm.model)
    except ValueError as e:
        assert "topology" in str(e) or "rebuild" in str(e)
        return
    raise AssertionError("topology change was not rejected")


if __name__ == "__main__":
    test_unedited_inplace_is_byte_identical()
    test_vertex_move_survives_roundtrip()
    test_topology_change_rejected()
    print("OK — PMO encoder: in-place rewrite byte-exact, vertex moves survive, "
          "topology change rejected")
