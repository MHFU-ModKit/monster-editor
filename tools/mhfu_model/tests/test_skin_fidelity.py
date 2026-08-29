"""A ported monster ships its AUTHENTIC per-vertex skin, bit for bit.

MHP3rd big monsters are blend-skinned (the Brute: 1492 of 2689 vertices carry
fractional weights) and so are MHFU's — both bound by the PSP GE's 8-matrix limit,
which is why the port is lossless rather than approximate. These tests run the
porter's skinning stage, encode a native MHFU PMO, decode it with the independent
reader and compare every vertex against the source.
"""
import os
import sys

_TOOLS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _TOOLS)

import skin_fidelity as SF

_P3 = os.path.join(_TOOLS, "..", "workspace", "extracted_mhp3", "data_files")
_MHFU = os.path.join(_TOOLS, "..", "workspace", "extracted", "data_files")
FRAME = os.path.join(_MHFU, "file_06185.bin")


def _run(model, geo, source_skeleton=True):
    return SF.analyse(open(os.path.join(_P3, model), "rb").read(),
                      open(os.path.join(_P3, geo), "rb").read(),
                      open(FRAME, "rb").read(),
                      source_skeleton=source_skeleton)


def test_brute_skin_round_trips_exactly():
    r = _run("file_05248.bin", "file_05249.bin")
    assert r["verts"] == 2689
    assert r["verts_with_weights"] == r["verts"]
    assert r["frac_verts"] == 1492, "the source really is blend-skinned"
    assert r["vgroups_wrong_length"] == 0
    assert r["verts_wrong_bone_set"] == 0
    assert r["max_weight_error"] == 0.0


def test_zinogre_skin_round_trips_exactly():
    """A monster with no similar MHFU native — the case the pipeline exists for."""
    r = _run("file_05339.bin", "file_05340.bin")
    assert r["verts"] == 4180
    assert r["verts_wrong_bone_set"] == 0
    assert r["max_weight_error"] == 0.0


def test_the_psp_palette_limit_can_never_bite():
    """Both engines are capped at 8 bone matrices per draw, so the encoder's
    `max_pal=8` never has to drop an influence. Measured across every skinned
    MHP3rd model: the worst vgroup uses exactly 8."""
    for model, geo in (("file_05248.bin", "file_05249.bin"),
                       ("file_05339.bin", "file_05340.bin")):
        r = _run(model, geo)
        assert r["vgroups_over_palette"] == 0
        assert r["max_bones_per_vgroup"] <= 8


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    bad = 0
    for f in fns:
        try:
            f(); print("  PASS  %s" % f.__name__)
        except Exception as e:
            bad += 1; print("  FAIL  %s: %s" % (f.__name__, e))
    print("\n%d tests, %d failed" % (len(fns), bad))
    sys.exit(1 if bad else 0)
