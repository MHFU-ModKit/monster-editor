"""Phase 0 guard: parse -> re-serialize every big-monster PAC byte-identical.

This is the keystone test for every future encoder. Today it exercises the
container writer + the lossless sub-resource passthroughs; when Phase 3 adds real
encoders, the same assertions guard them.

Run:
    PYTHONPATH=tools python -m pytest tools/mhfu_model/tests -q
    # or, without pytest:
    PYTHONPATH=tools python tools/mhfu_model/tests/test_roundtrip.py
"""
from __future__ import annotations

import glob
import os

from mhfu_model import load_pac, parse_pac
from mhfu_model.pac import MonsterPac

# Big-monster model PACs = file_06111 .. file_06159 (em_id + 0x17AB, the species
# that have an em*.ovl AI overlay and thus a skeleton sub-resource).
DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "..",
    "workspace", "extracted", "data_files")
LO, HI = 6111, 6159


def _pacs():
    out = []
    for f in sorted(glob.glob(os.path.join(DATA_DIR, "file_0[0-9][0-9][0-9][0-9].bin"))):
        n = int(os.path.basename(f)[5:10])
        if LO <= n <= HI:
            out.append(f)
    return out


def test_pac_container_roundtrip():
    """Container: from_bytes -> to_bytes is byte-identical."""
    files = _pacs()
    assert files, "no big-monster PACs found under %s" % DATA_DIR
    for f in files:
        data = open(f, "rb").read()
        assert MonsterPac.from_bytes(data).to_bytes() == data, \
            "container mismatch: %s" % os.path.basename(f)


def test_subresource_passthrough_roundtrip():
    """Decode EACH sub-resource into the data model and re-encode it; assert
    byte-identical, then repack the whole PAC and assert byte-identical.

    Each sub is parsed/encoded independently (a dual-set PAC has two skeletons /
    models — set 1 in slots 0-3, set 2 in slots 4-6 — so the parsed object must
    match the sub it came from). This guards skeleton/anim/pmo encoders as they
    replace the passthroughs in Phase 3."""
    from mhfu_model import anim, pmo, skeleton
    for f in _pacs():
        data = open(f, "rb").read()
        pac = MonsterPac.from_bytes(data)
        for s in pac.subs:
            role = pac.role(s)
            if role == "skeleton":
                enc = skeleton.encode(skeleton.parse(s.data))
            elif role == "anim":
                enc = anim.encode(anim.parse(s.data))
            elif role == "model":
                enc = pmo.encode(pmo.parse(s.data))
            else:
                continue
            assert enc == s.data, "sub %d (%s) encode mismatch in %s" % (
                s.index, role, os.path.basename(f))
        assert pac.to_bytes() == data, "repack mismatch: %s" % os.path.basename(f)


def test_decode_smoke():
    """Every PAC decodes without raising and exposes the four sub-resources.

    Detailed geometry/anim decode is best-effort: single-set monsters (anim
    header 0x18) fully decode; the dual-model-set variant (0x38) decodes skeleton
    + geometry but its anim layout is a Phase-1 follow-up (bytes still preserved).
    """
    for f in _pacs():
        mm = load_pac(f)
        assert mm.skeleton and mm.skeleton.bone_count > 0
        assert mm.model and mm.model.mesh_groups, "geometry: %s" % f
        assert mm.anim and mm.anim.animations, "anim: %s" % f
    # Tigrex (single-set) is the Phase 1 reference target — must fully decode.
    tig = load_pac(os.path.join(DATA_DIR, "file_06134.bin"))
    assert tig.skeleton.bone_count == 25
    # 21 clips in slots 1..22 (slot 4 empty). It read 22 until 2026-08-24: the
    # slot table was taken from hsize-4, which is the data_start word, so every
    # monster gained a phantom slot 0 aliasing its first real block. Slot 0 is
    # empty in every anim pack checked, in both games.
    assert len(tig.anim.animations) == 21
    assert 0 not in {a.slot for a in tig.anim.animations}
    assert tig.model.mesh_groups and tig.model.mesh_groups[0].vertex_count > 0


if __name__ == "__main__":
    test_pac_container_roundtrip()
    test_subresource_passthrough_roundtrip()
    test_decode_smoke()
    n = len(_pacs())
    print("OK — round-trip byte-identical + decode smoke on %d big-monster PACs" % n)
