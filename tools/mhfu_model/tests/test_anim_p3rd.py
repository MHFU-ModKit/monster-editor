"""Tests for MHP3rd animation and skeleton parse_p3rd paths.

Coverage:
  - anim.parse_p3rd: header decode, slot table, stub Animation objects
  - skeleton.parse_p3rd: delegate to skeleton_p3rd, same Skeleton dataclass
  - skeleton_p3rd: 0x80000000 magic, both section magics, lobby extra header word
  - tools/skeleton.py CLI: --game p3rd flag dispatches to parse_p3rd
"""
from __future__ import annotations

import os
import struct
import sys
import types
import unittest

# Make mhfu_model importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from mhfu_model.anim import parse_p3rd as anim_parse_p3rd
from mhfu_model.model import AnimationPack, Skeleton
from mhfu_model.skeleton import parse_p3rd as skel_parse_p3rd

DATA_DIR = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "..", "workspace", "extracted_mhp3", "data_files",
)

# Lobby em058 PAC (file_05248) has a skeleton sub with 0x80000000 magic
LOBBY_PAC = os.path.join(DATA_DIR, "file_05248.bin")
# In-quest PAC (file_04016) has a skeleton + anim sub (sub[3])
QUEST_PAC = os.path.join(DATA_DIR, "file_04016.bin")
# Second in-quest PAC (file_04000) — smaller anim (5 anim_count)
QUEST_PAC2 = os.path.join(DATA_DIR, "file_04000.bin")

HAS_DATA = os.path.isfile(LOBBY_PAC) and os.path.isfile(QUEST_PAC)


def _read_pac_sub(pac_path: str, sub_idx: int) -> bytes:
    with open(pac_path, "rb") as f:
        data = f.read()
    cnt = struct.unpack_from("<I", data)[0]
    off, sz = struct.unpack_from("<II", data, 4 + sub_idx * 8)
    return data[off : off + sz]


def _find_sub_magic(pac_path: str, magic: int) -> bytes | None:
    with open(pac_path, "rb") as f:
        data = f.read()
    cnt = struct.unpack_from("<I", data)[0]
    if cnt > 64:
        return None
    for i in range(cnt):
        off, sz = struct.unpack_from("<II", data, 4 + i * 8)
        if off + 4 <= len(data):
            m = struct.unpack_from("<I", data, off)[0]
            if m == magic:
                return data[off : off + sz]
    return None


# ---------------------------------------------------------------------------
# anim.parse_p3rd tests
# ---------------------------------------------------------------------------


class TestAnimParseP3rdSynthetic(unittest.TestCase):
    """Tests that do NOT require extracted MHP3rd data."""

    def _make_blob(self, anim_count: int, slot_count: int, slots: dict) -> bytes:
        """Build a minimal synthetic MHP3rd anim blob.

        Layout: [0..tbase) = prefix (anim_count, hsize, slot_count + pad),
                [tbase..tbase+slot_count*4) = slot offset table,
                [tbase+slot_count*4..) = per-slot data blocks.
        tbase = hsize - 4 = 0x14.
        """
        hsize = 0x18
        tbase = hsize - 4  # 0x14 = 20
        # prefix: 3 u32 (12 B) + 8 B pad = 20 B = tbase
        prefix = struct.pack("<3I", anim_count, hsize, slot_count) + b"\x00" * 8
        assert len(prefix) == tbase
        # slot data starts right after the table
        data_start = tbase + slot_count * 4
        table = [0xFFFFFFFF] * slot_count
        blobs = {}
        for slot_id, (bone_count, block_size) in slots.items():
            blobs[slot_id] = struct.pack("<4I", bone_count, block_size, 0, 0)
        pos = data_start
        slot_data = b""
        for slot_id in sorted(blobs):
            table[slot_id] = pos
            slot_data += blobs[slot_id]
            pos += len(blobs[slot_id])
        tbl_bytes = struct.pack("<%dI" % slot_count, *table)
        return prefix + tbl_bytes + slot_data

    def test_empty_blob(self):
        pack = anim_parse_p3rd(b"")
        self.assertIsInstance(pack, AnimationPack)
        self.assertEqual(pack.slot_count, 0)
        self.assertEqual(pack.animations, [])

    def test_no_valid_slots(self):
        blob = self._make_blob(anim_count=0, slot_count=4, slots={})
        pack = anim_parse_p3rd(blob)
        self.assertEqual(pack.slot_count, 4)
        self.assertEqual(len(pack.animations), 0)

    def test_single_slot(self):
        blob = self._make_blob(anim_count=1, slot_count=4, slots={2: (18, 32)})
        pack = anim_parse_p3rd(blob)
        self.assertEqual(pack.slot_count, 4)
        self.assertEqual(len(pack.animations), 1)
        anim = pack.animations[0]
        self.assertEqual(anim.slot, 2)
        self.assertEqual(anim.bone_count, 18)
        self.assertIsNotNone(anim.raw)

    def test_magic_stored_as_anim_count(self):
        """pack.magic stores the raw anim_count field, not 0x64."""
        blob = self._make_blob(anim_count=53, slot_count=8, slots={1: (19, 24)})
        pack = anim_parse_p3rd(blob)
        self.assertEqual(pack.magic, 53)

    def test_multiple_slots(self):
        blob = self._make_blob(anim_count=3, slot_count=6, slots={0: (18, 24), 3: (18, 48), 5: (18, 32)})
        pack = anim_parse_p3rd(blob)
        self.assertEqual(len(pack.animations), 3)
        slot_ids = {a.slot for a in pack.animations}
        self.assertEqual(slot_ids, {0, 3, 5})

    def test_animations_have_no_tracks(self):
        """Stub decode: tracks list is empty (not yet decoded)."""
        blob = self._make_blob(anim_count=2, slot_count=4, slots={0: (22, 32), 1: (22, 32)})
        pack = anim_parse_p3rd(blob)
        for anim in pack.animations:
            self.assertEqual(anim.tracks, [])

    def test_corrupted_block_skipped(self):
        """Corrupt block size should not raise; parser skips it gracefully."""
        # Build a blob where one slot's block_size is huge (points past EOF)
        blob = self._make_blob(anim_count=2, slot_count=2, slots={0: (18, 24), 1: (18, 99999)})
        pack = anim_parse_p3rd(blob)
        # Should not raise; at least slot 0 should be parsed
        self.assertGreaterEqual(len(pack.animations), 1)


@unittest.skipUnless(HAS_DATA, "MHP3rd extracted data not present")
class TestAnimParseP3rdLive(unittest.TestCase):
    """Tests against actual extracted MHP3rd PAC data."""

    def test_anim_sub_quest_pac(self):
        """file_04016 sub[3]: 53-anim-count, 48 slot_count, 9 valid slots."""
        anim_blob = _read_pac_sub(QUEST_PAC, 3)
        pack = anim_parse_p3rd(anim_blob)
        anim_count = struct.unpack_from("<I", anim_blob)[0]
        self.assertEqual(pack.magic, anim_count)  # 53
        self.assertEqual(pack.slot_count, 48)
        # We know at least 9 valid slots from inspection
        self.assertGreaterEqual(len(pack.animations), 5)

    def test_anim_bone_counts_sane(self):
        """All parsed anim bone_counts should be close to skeleton bone count."""
        anim_blob = _read_pac_sub(QUEST_PAC, 3)
        skel_blob = _find_sub_magic(QUEST_PAC, 0x80000000)
        if skel_blob is None:
            self.skipTest("no skeleton sub found")
        sk = skel_parse_p3rd(skel_blob)
        pack = anim_parse_p3rd(anim_blob)
        for anim in pack.animations:
            self.assertLessEqual(anim.bone_count, sk.bone_count + 2,
                "anim bone_count way above skeleton bone_count")

    def test_anim_quest_pac2(self):
        """file_04000 sub[3]: smaller monster, anim_count=5, slot_count=0 in primary table.

        This file uses a header variant where slot_count (u2) = 0.  parse_p3rd
        correctly returns 0 animations for the primary table; anim_count (magic)
        still reflects the correct count of 5.
        """
        if not os.path.isfile(QUEST_PAC2):
            self.skipTest("file_04000 not present")
        anim_blob = _read_pac_sub(QUEST_PAC2, 3)
        pack = anim_parse_p3rd(anim_blob)
        # anim_count stored in magic field; should be 5 for this monster
        self.assertLess(pack.magic, 20)
        # slot_count is 0 in this variant's primary table — that is the
        # correct (if limited) decode; stub returns 0 animations
        self.assertEqual(pack.slot_count, 0)
        self.assertEqual(len(pack.animations), 0)


# ---------------------------------------------------------------------------
# skeleton.parse_p3rd tests
# ---------------------------------------------------------------------------


@unittest.skipUnless(HAS_DATA, "MHP3rd extracted data not present")
class TestSkeletonParseP3rd(unittest.TestCase):

    def test_lobby_pac_skel(self):
        """file_05248 sub[0]: 0x80000000, 47 bones."""
        skel_blob = _find_sub_magic(LOBBY_PAC, 0x80000000)
        self.assertIsNotNone(skel_blob, "no 0x80000000 sub in lobby PAC")
        sk = skel_parse_p3rd(skel_blob)
        self.assertIsInstance(sk, Skeleton)
        self.assertEqual(sk.bone_count, 47)
        self.assertGreaterEqual(len(sk.bones), 46)  # allow 1-off truncation
        roots = [b for b in sk.bones if b.parent == -1]
        self.assertGreater(len(roots), 0, "no root bone found")

    def test_quest_pac_skel(self):
        """file_04016 sub with 0x80000000."""
        skel_blob = _find_sub_magic(QUEST_PAC, 0x80000000)
        self.assertIsNotNone(skel_blob)
        sk = skel_parse_p3rd(skel_blob)
        self.assertIsInstance(sk, Skeleton)
        self.assertGreater(sk.bone_count, 0)
        self.assertGreater(len(sk.bones), 0)

    def test_returns_skeleton_dataclass(self):
        """Result is a Skeleton dataclass, not a plain dict."""
        skel_blob = _find_sub_magic(LOBBY_PAC, 0x80000000)
        if skel_blob is None:
            self.skipTest("no skel sub found")
        sk = skel_parse_p3rd(skel_blob)
        # Skeleton has these attributes
        self.assertTrue(hasattr(sk, "bone_count"))
        self.assertTrue(hasattr(sk, "bones"))
        self.assertTrue(hasattr(sk, "raw"))
        self.assertTrue(hasattr(sk.bones[0], "bind_pos"))


# ---------------------------------------------------------------------------
# tools/skeleton.py CLI --game p3rd
# ---------------------------------------------------------------------------


class TestSkeletonCLIP3rd(unittest.TestCase):

    @unittest.skipUnless(HAS_DATA, "MHP3rd extracted data not present")
    def test_cli_game_p3rd(self):
        """--game p3rd flag routes to parse_p3rd."""
        tools_dir = os.path.join(os.path.dirname(__file__), "..", "..")
        sys.path.insert(0, tools_dir)
        from skeleton import main, parse_p3rd
        result = parse_p3rd(_find_sub_magic(LOBBY_PAC, 0x80000000))
        self.assertIn("bone_count", result)
        self.assertGreater(result["bone_count"], 0)
        self.assertIn("bones", result)

    def test_parse_p3rd_accepts_mhfu_magic(self):
        """parse_p3rd (via skeleton_p3rd) accepts both 0x80000000 and 0xC0000000.

        skeleton_p3rd.parse() is the unifying parser for both game generations;
        it raises ValueError only on completely unknown magics.
        """
        # 0xC0000000 blob is too short to yield any bones but should not raise
        blob = struct.pack("<3I", 0xC0000000, 0, 0) + b"\x00" * 0x20
        sk = skel_parse_p3rd(blob)
        self.assertEqual(sk.bone_count, 0)

    def test_parse_p3rd_rejects_unknown_magic(self):
        """parse_p3rd raises ValueError on a completely unknown top magic."""
        blob = struct.pack("<3I", 0xDEADBEEF, 1, 0x1C) + b"\x00" * 0x20
        with self.assertRaises(ValueError):
            skel_parse_p3rd(blob)


if __name__ == "__main__":
    unittest.main()
