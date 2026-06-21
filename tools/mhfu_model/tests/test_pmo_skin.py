"""Blend-skinning reader/encoder round-trip on the native Tigrex (file_06185).

The native big-monster PMO uses real multi-bone blend skinning (a running bone
palette + per-vertex weights). These tests prove `pmo_skin.read` resolves it and
`pmo_skin.encode` reproduces it exactly (geometry + per-vertex influences), so the
Brute port can emit faithful blend skinning instead of lossy rigid binding.
"""
import os
import struct
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from mhfu_model import pmo_skin as PS

_DATA = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                     "workspace", "extracted", "data_files", "file_06185.bin")


def _pmo_sub(path):
    with open(path, "rb") as f:
        blob = f.read()
    if blob[:4] == b"pmo\x00":
        return blob
    n = struct.unpack_from("<I", blob, 0)[0]
    off = 4
    for _ in range(n):
        o, s = struct.unpack_from("<II", blob, off)
        off += 8
        if blob[o:o + 4] == b"pmo\x00":
            return blob[o:o + s]
    raise AssertionError("no PMO sub")


_HAVE = os.path.exists(_DATA)
pytestmark = pytest.mark.skipif(not _HAVE, reason="native file_06185 not extracted")


def _infl_dict(infl):
    d = {}
    for b, w in infl:
        if w != 0:
            d[b] = d.get(b, 0.0) + w
    return d


def test_native_is_blend_skinned():
    sm = PS.read(_pmo_sub(_DATA))
    # native Tigrex really uses multi-bone blend (not rigid one-bone)
    multi = sum(1 for g in sm.vgroups for inf in g.influences
                if len(_infl_dict(inf)) > 1)
    assert multi > 100                       # plenty of genuinely-blended verts
    # every weight resolves to a real bone, sums to ~1
    for g in sm.vgroups:
        for inf in g.influences:
            d = _infl_dict(inf)
            assert all(b >= 0 for b in d)
            assert abs(sum(d.values()) - 1.0) < 0.05


def test_roundtrip_exact_geometry_and_skinning():
    orig = _pmo_sub(_DATA)
    sm1 = PS.read(orig)
    enc = PS.encode(sm1)
    sm2 = PS.read(enc)
    assert len(sm1.vgroups) == len(sm2.vgroups)
    for g1, g2 in zip(sm1.vgroups, sm2.vgroups):
        assert len(g1.vertices) == len(g2.vertices)
        assert len(g1.faces) == len(g2.faces)
        for v1, v2 in zip(g1.vertices, g2.vertices):
            assert abs(v1["x"] - v2["x"]) < 1e-3
            assert abs(v1["y"] - v2["y"]) < 1e-3
            assert abs(v1["z"] - v2["z"]) < 1e-3
        for i1, i2 in zip(g1.influences, g2.influences):
            d1, d2 = _infl_dict(i1), _infl_dict(i2)
            assert set(d1) == set(d2)
            for b in d1:
                assert abs(d1[b] - d2[b]) < 1e-3
        # winding preserved
        for f1, f2 in zip(g1.faces, g2.faces):
            assert (f1["v1"], f1["v2"], f1["v3"]) == (f2["v1"], f2["v2"], f2["v3"])


def test_encoded_is_valid_pmo():
    sm = PS.read(_pmo_sub(_DATA))
    enc = PS.encode(sm)
    assert enc[:4] == b"pmo\x00"
    assert enc[4:8] == b"1.0\x00"
    # header filesize == actual length
    fsz = struct.unpack_from("<I", enc, 8)[0]
    assert fsz == len(enc)
