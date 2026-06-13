"""Phase 1 headless guard: the Blender-independent conversion math.

The Blender addon (blender_mhfu/) can only be verified visually in Blender, but
all its math lives in mhfu_model.convert and is checked here against the Tigrex
reference PAC.

Run: PYTHONPATH=tools python tools/mhfu_model/tests/test_convert.py
"""
from __future__ import annotations

import math
import os

from mhfu_model import load_pac
from mhfu_model import convert as C

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                    "workspace", "extracted", "data_files")
TIGREX = os.path.join(DATA, "file_06134.bin")


def test_quantization_units():
    assert abs(math.degrees(C.dequantize("rot", 4096)) - 90.0) < 1e-6
    assert C.dequantize("loc", 16) == 1.0
    assert C.dequantize("scl", 256) == 1.0


def test_quantize_inverse():
    for kind, unit in (("rot", None), ("loc", 16), ("scl", 256)):
        for raw in (-30000, -1234, 0, 1, 4096, 30000):
            assert C.quantize(kind, C.dequantize(kind, raw)) == raw, (kind, raw)


def test_bone_tree_well_formed():
    sk = load_pac(TIGREX).skeleton
    by = {b.index: b for b in sk.bones}
    # exactly one root, every non-root parent exists, no cycles
    roots = sk.roots()
    assert roots == [0], roots
    seen = set()
    for b in sk.bones:
        depth = 0
        i = b.index
        while by[i].parent != -1:
            i = by[i].parent
            assert i in by, "dangling parent on bone %d" % b.index
            depth += 1
            assert depth < len(sk.bones), "cycle through bone %d" % b.index
        seen.add(b.index)
    # children map is consistent with parents
    kids = C.children_of(sk)
    for b in sk.bones:
        for c in kids[b.index]:
            assert by[c].parent == b.index
    # world positions resolve for every bone
    wp = C.bone_world_positions(sk)
    assert len(wp) == len(sk.bones)


def test_animation_fcurves():
    mm = load_pac(TIGREX)
    a = mm.anim.animations[0]
    n_bones = len(mm.skeleton.bones)
    seen_kinds = set()
    n = 0
    for bone_index, kind, axis, pts in C.animation_fcurves(a):
        assert 0 <= bone_index < n_bones + 8        # track index ~ bone index
        assert kind in ("rot", "loc", "scl")
        assert 0 <= axis <= 2
        assert pts and all(isinstance(f, int) for f, _ in pts)
        # rotations in a sane radian range
        if kind == "rot":
            assert all(abs(v) < 2 * math.pi for _, v in pts)
        seen_kinds.add(kind)
        n += 1
    assert "rot" in seen_kinds and n > 0


if __name__ == "__main__":
    test_quantization_units()
    test_quantize_inverse()
    test_bone_tree_well_formed()
    test_animation_fcurves()
    print("OK — convert math verified against Tigrex (bone tree, dequant, fcurves)")
