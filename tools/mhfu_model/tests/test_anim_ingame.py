"""Tests for the MHFU in-game (recursive 3-stream) animation codec."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from mhfu_model.pac import MonsterPac
from mhfu_model import anim_ingame as ai

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                    "workspace", "extracted", "data_files", "file_06185.bin")


def _native_anim():
    return MonsterPac.from_bytes(open(DATA, "rb").read()).subs[3].data


@pytest.mark.skipif(not os.path.exists(DATA), reason="needs extracted game data")
def test_roundtrip_byte_exact():
    blob = _native_anim()
    m = ai.parse_ingame(blob)
    assert ai.encode_ingame(m) == blob


@pytest.mark.skipif(not os.path.exists(DATA), reason="needs extracted game data")
def test_parse_structure():
    m = ai.parse_ingame(_native_anim())
    assert m.magic == 0x64 and m.hsize == 0x38 and m.num_slots == 100
    pop = [len(s.clips) for s in m.streams]
    assert pop == [63, 0, 64, 0, 62, 0]            # main + sub1 + sub3
    # stream bone counts 31/9/5
    bc = lambda st: len(next(iter(st.clips.values())).bones)
    assert bc(m.streams[0]) == 31
    assert bc(m.streams[2]) == 9
    assert bc(m.streams[4]) == 5


def test_static_pose_empty_single_stream():
    # all 42 animated bones in main, EMPTY sections (round-trip fidelity form)
    pose = ai.make_static_pose([(42, list(range(100)))], bone_factory=ai.empty_bone)
    blob = ai.encode_ingame(pose)
    rt = ai.parse_ingame(blob)
    assert len(rt.streams[0].clips) == 100
    blk = next(iter(rt.streams[0].clips.values()))
    assert len(blk.bones) == 42
    assert all(b.mask == ai.FLAG and not b.channels for b in blk.bones)
    assert ai.encode_ingame(rt) == blob       # generator output round-trips


def test_rest_pose_default_has_identity_keyframes():
    # default bone_factory = rest_bone: every bone gets 3 rot channels, identity kfs.
    # static_pose_for_bonecount places the split on native's streams (main/sub1/sub3).
    pose = ai.static_pose_for_bonecount(42, split=[28, 9, 5])
    blob = ai.encode_ingame(pose)
    rt = ai.parse_ingame(blob)
    assert [len(s.clips) for s in rt.streams] == [100, 0, 100, 0, 100, 0]
    blk = next(iter(rt.streams[0].clips.values()))
    for b in blk.bones:
        assert b.mask == (ai.FLAG | 0x38)
        assert len(b.channels) == 3
        for c in b.channels:
            assert all(kf.value == 0 for kf in c.keyframes)
    assert ai.encode_ingame(rt) == blob


def test_swap_to_restpose_keeps_size():
    # the addon path: swap a big-mon PAC's anim for a rest pose, same total size
    import os
    p = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                     "tmp", "brute_tigrex_v12.bin")
    if not os.path.exists(p):
        return
    src = open(p, "rb").read()
    out, info = ai.swap_anim_to_bindpose(src)
    assert len(out) == len(src)
    assert info["animated"] == sum(info["split"])
