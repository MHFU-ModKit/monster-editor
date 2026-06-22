"""Tests for the skeleton-to-skeleton bone matcher + the from_flat_anim bone_map path."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from mhfu_model import bone_match as BM


def test_identity_match():
    # identical skeletons -> identity mapping
    parents = [-1, 0, 1, 2, 2]
    local = [(0, 0, 0), (0, 0, 10), (0, 0, 10), (5, 0, 0), (-5, 0, 0)]
    m = BM.match_skeletons(parents, local, parents, local)
    assert m == {i: i for i in range(5)}


def test_leading_root_offset():
    # target has an EXTRA leading root bone -> source bones shift by one; the
    # matcher must align by position, leaving the unmatched extra root None.
    src_p = [-1, 0, 1, 1]
    src_l = [(0, 0, 0), (0, 0, 10), (5, 0, 5), (-5, 0, 5)]
    # target = same creature but with one more root joint at the very front
    dst_p = [-1, 0, 1, 2, 2]
    dst_l = [(0, 0, 0), (0, 0, 0), (0, 0, 10), (5, 0, 5), (-5, 0, 5)]
    m = BM.match_skeletons(src_p, src_l, dst_p, dst_l)
    # dst joints 2,3,4 should map to src 1,2,3 (the body); a leading dst root None
    assert m[2] == 1
    assert m[3] == 2
    assert m[4] == 3
    assert None in m.values()  # the extra leading root is unmatched/static


def test_remap_tracks():
    tracks = ["t0", "t1", "t2"]
    bm = {0: None, 1: 0, 2: 1, 3: 2, 4: None}
    out = BM.remap_tracks(tracks, bm, 5, empty_factory=lambda: "EMPTY")
    assert out == ["EMPTY", "t0", "t1", "t2", "EMPTY"]


def test_walk_order_storage():
    assert BM.walk_order([-1, 0, 1, 1]) == [0, 1, 2, 3]


def test_from_flat_anim_bone_map_places_tracks():
    # a 5-bone flat clip -> a host of 7 joints where joints 0,1 are static (None)
    # and the 5 source tracks land on joints 2..6.
    from mhfu_model.anim_ingame import from_flat_anim, summary

    class _KF:
        def __init__(s, v): s.value, s.frame, s.ease_in, s.ease_out = v, 0, 0, 0

    class _Ch:
        def __init__(s, t, v): s.type, s.keyframes = t, [_KF(v)]

    class _Trk:
        def __init__(s, tag, v): s.tag, s.channels = tag, [_Ch(0x08, v)]

    class _An:
        slot = 0; loop = 0; loop_start = 0.0
        def __init__(s, trks): s.tracks = trks

    class _Pack:
        def __init__(s, an): s.animations = [an]

    trks = [_Trk(0x08, 100 + i) for i in range(5)]
    pack = _Pack(_An(trks))
    bm = {0: None, 1: None, 2: 0, 3: 1, 4: 2, 5: 3, 6: 4}
    ig = from_flat_anim(pack, split=[7], num_slots=4, bone_map=bm)
    blk = ig.streams[0].clips[0]
    assert len(blk.bones) == 7
    # joints 0,1 unmatched -> REST pose (identity rotation channels, all value 0), NOT
    # an empty section (empty zeroes the matrix -> collapse). 2..6 carry source tracks.
    for j in (0, 1):
        assert blk.bones[j].channels, "unmatched joint must get rest channels, not empty"
        assert all(k.value == 0 for c in blk.bones[j].channels for k in c.keyframes)
    assert (blk.bones[2].mask & 0xFFFF) != 0
    assert blk.bones[2].channels[0].keyframes[0].value == 100
    assert blk.bones[6].channels[0].keyframes[0].value == 104
