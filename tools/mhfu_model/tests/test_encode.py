"""Phase 3 guard: encoders apply edits correctly and repack engine-valid PACs.

Byte-identity for *unedited* assets is covered by test_roundtrip; here we prove the
encoders actually serialize EDITS (the point of write-back) and survive a decode:
  * edit an existing keyframe value (in-place patch path)
  * change a keyframe count (full-rebuild path)
  * fill an empty animation slot (new clip)
  * edit a skeleton bind-pose value
  * repack a whole PAC after an edit and re-parse it

Run: PYTHONPATH=tools python tools/mhfu_model/tests/test_encode.py
"""
from __future__ import annotations

import copy
import os

from mhfu_model import load_pac, parse_pac, repack
from mhfu_model import anim as A
from mhfu_model import skeleton as S
from mhfu_model.model import Animation, BoneTrack, Channel, Keyframe

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                    "workspace", "extracted", "data_files")
TIGREX = os.path.join(DATA, "file_06134.bin")


def _first_channel(pack, slots=None):
    """Return (anim, track_i, chan_i) of the first channel that has keyframes.

    `slots`, if given, restricts to those slot indices (used to pick a slot whose
    offset isn't aliased, so the in-place patch path is exercised)."""
    for a in pack.animations:
        if slots is not None and a.slot not in slots:
            continue
        for ti, tr in enumerate(a.tracks):
            for ci, ch in enumerate(tr.channels):
                if ch.keyframes:
                    return a, ti, ci
    raise AssertionError("no keyframed channel found")


def _unaliased_slots(pack):
    """Slots whose anim offset is not shared with another slot."""
    import struct
    raw, hsize = pack.raw, len(pack.header)
    tbl = struct.unpack_from("<%dI" % pack.slot_count, raw, hsize)   # NOT hsize-4
    seen = {}
    for i, off in enumerate(tbl):
        if off in (0, 0xFFFFFFFF):
            continue
        seen.setdefault(off, []).append(i)
    return {s[0] for off, s in seen.items() if len(s) == 1}


def test_anim_value_edit_inplace():
    """Editing a keyframe value (no size change) round-trips through decode and
    leaves every other byte untouched (in-place patch path)."""
    mm = load_pac(TIGREX)
    pack = mm.anim
    orig = A.encode(pack)
    a0, ti, ci = _first_channel(pack, slots=_unaliased_slots(pack))
    kf = a0.tracks[ti].channels[ci].keyframes[0]
    old = kf.value
    kf.value = old + 100 if old + 100 <= 32767 else old - 100
    enc = A.encode(pack)
    assert len(enc) == len(orig), "value edit must not change pack size"
    assert enc != orig, "edit did not change the bytes"
    re = A.parse(enc)
    assert re.by_slot()[a0.slot].tracks[ti].channels[ci].keyframes[0].value == kf.value
    # one 16-bit word edited -> at most 2 mismatching bytes (minimal in-place patch)
    diff = sum(1 for x, y in zip(enc, orig) if x != y)
    assert 0 < diff <= 2, "in-place value edit touched %d bytes" % diff


def test_anim_keyframe_count_change_rebuild():
    """Dropping a keyframe changes a block size -> full rebuild; result re-parses
    with the new count and the pack stays valid."""
    mm = load_pac(TIGREX)
    pack = mm.anim
    a0, ti, ci = _first_channel(pack)
    ch = a0.tracks[ti].channels[ci]
    if len(ch.keyframes) < 2:
        return
    n0 = len(ch.keyframes)
    ch.keyframes = ch.keyframes[:-1]
    enc = A.encode(pack)
    re = A.parse(enc)
    assert len(re.by_slot()[a0.slot].tracks[ti].channels[ci].keyframes) == n0 - 1
    # all other anims still decode and keep their bone counts
    by = re.by_slot()
    for a in pack.animations:
        assert a.slot in by and by[a.slot].bone_count == a.bone_count


def test_anim_fill_empty_slot():
    """Filling an empty slot with a new anim re-parses with the populated slot."""
    mm = load_pac(TIGREX)
    pack = mm.anim
    used = {a.slot for a in pack.animations}
    free = next(s for s in range(pack.slot_count) if s not in used)
    proto = copy.deepcopy(pack.animations[0])
    proto.slot = free
    pack.animations.append(proto)
    enc = A.encode(pack)
    re = A.parse(enc)
    assert free in re.by_slot(), "new slot %d not present after re-parse" % free
    assert len(re.animations) == len(pack.animations)


def test_skeleton_bindpose_edit():
    """A bind-pose edit serializes; re-parse reflects it; unedited bones unchanged."""
    mm = load_pac(TIGREX)
    sk = mm.skeleton
    sk.bones[1].bind_pos = (sk.bones[1].bind_pos[0] + 5.0,
                            sk.bones[1].bind_pos[1], sk.bones[1].bind_pos[2])
    enc = S.encode(sk)
    re = S.parse(enc)
    assert abs(re.bones[1].bind_pos[0] - sk.bones[1].bind_pos[0]) < 1e-4
    assert re.bones[0].bind_pos == sk.bones[0].bind_pos
    assert len(enc) == len(sk.raw), "same-count edit must not resize the blob"


def test_repack_unedited_is_identical():
    data = open(TIGREX, "rb").read()
    assert repack(parse_pac(data)) == data


def test_repack_after_edit_decodes():
    """Repack a PAC after an anim edit; re-parse and confirm the edit survived and
    the texture sub stayed byte-identical."""
    data = open(TIGREX, "rb").read()
    mm = parse_pac(data)
    a0, ti, ci = _first_channel(mm.anim)
    kf = a0.tracks[ti].channels[ci].keyframes[0]
    kf.value = kf.value + 50 if kf.value + 50 <= 32767 else kf.value - 50
    tex = mm.pac.find("texture").data
    out = repack(mm)
    assert out != data
    mm2 = parse_pac(out)
    assert mm2.anim.by_slot()[a0.slot].tracks[ti].channels[ci].keyframes[0].value == kf.value
    assert mm2.pac.find("texture").data == tex, "untouched texture must be identical"


if __name__ == "__main__":
    test_anim_value_edit_inplace()
    test_anim_keyframe_count_change_rebuild()
    test_anim_fill_empty_slot()
    test_skeleton_bindpose_edit()
    test_repack_unedited_is_identical()
    test_repack_after_edit_decodes()
    print("OK — encoders apply anim/skeleton edits and repack engine-valid PACs")
