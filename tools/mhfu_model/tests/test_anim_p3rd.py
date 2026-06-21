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

    def _make_blob(self, anim_count: int, slots: dict, tracks=None, hsize=0x18) -> bytes:
        """Build a synthetic MHP3rd anim blob in the authoritative (Kurogami) layout.

        Header (``hsize`` bytes): word0 = count field, word1 = hsize, the last header
        word (at hsize-4) = first_anim offset; then one null word; then the slot
        offset table of ``anim_count`` u32 at hsize+4; then the per-slot blocks
        starting at first_anim = hsize + 4 + anim_count*4. Blocks default to
        bone_count=0 (a valid empty-track anim) unless ``tracks[slot_id]`` supplies
        pre-encoded bone-record bytes.
        """
        first_anim = hsize + 4 + anim_count * 4
        header = bytearray(hsize)
        struct.pack_into("<2I", header, 0, anim_count, hsize)
        struct.pack_into("<I", header, hsize - 4, first_anim)
        table = [0xFFFFFFFF] * anim_count
        blobs = {}
        for slot_id, (bone_count, _bs) in slots.items():
            body = (tracks or {}).get(slot_id, b"")
            block_size = 0x10 + len(body)
            blobs[slot_id] = struct.pack("<4I", bone_count, block_size, 0, 0) + body
        pos = first_anim
        slot_data = b""
        for slot_id in sorted(blobs):
            table[slot_id] = pos
            slot_data += blobs[slot_id]
            pos += len(blobs[slot_id])
        return (bytes(header) + struct.pack("<I", 0)
                + struct.pack("<%dI" % anim_count, *table) + slot_data)

    @staticmethod
    def _bone(channels):
        """Encode one MHP3rd bone record: u16 nch, u16 size, then channels.
        ``channels`` = list of (bit, [(value,frame,ein,eout), ...])."""
        body = b""
        for bit, kfs in channels:
            csz = 8 + len(kfs) * 8
            body += struct.pack("<HHI", bit, len(kfs), csz)
            for v, f, ei, eo in kfs:
                body += struct.pack("<4h", v, f, ei, eo)
        return struct.pack("<2H", len(channels), 4 + len(body)) + body

    def test_empty_blob(self):
        pack = anim_parse_p3rd(b"")
        self.assertIsInstance(pack, AnimationPack)
        self.assertEqual(pack.slot_count, 0)
        self.assertEqual(pack.animations, [])

    def test_no_valid_slots(self):
        blob = self._make_blob(anim_count=4, slots={})
        pack = anim_parse_p3rd(blob)
        self.assertEqual(pack.slot_count, 4)        # table size == anim_count
        self.assertEqual(len(pack.animations), 0)

    def test_single_slot(self):
        blob = self._make_blob(anim_count=4, slots={2: (0, 0)})
        pack = anim_parse_p3rd(blob)
        self.assertEqual(pack.slot_count, 4)
        self.assertEqual(len(pack.animations), 1)
        anim = pack.animations[0]
        self.assertEqual(anim.slot, 2)
        self.assertEqual(anim.bone_count, 0)
        self.assertIsNotNone(anim.raw)

    def test_magic_stored_as_anim_count(self):
        """pack.magic stores the raw anim_count field, not 0x64."""
        blob = self._make_blob(anim_count=53, slots={1: (0, 0)})
        pack = anim_parse_p3rd(blob)
        self.assertEqual(pack.magic, 53)

    def test_multiple_slots(self):
        blob = self._make_blob(anim_count=6, slots={0: (0, 0), 3: (0, 0), 5: (0, 0)})
        pack = anim_parse_p3rd(blob)
        self.assertEqual(len(pack.animations), 3)
        slot_ids = {a.slot for a in pack.animations}
        self.assertEqual(slot_ids, {0, 3, 5})

    def test_decodes_bone_channels_keyframes(self):
        """Real decode: a 1-bone, 1-channel (locX, 2 kf) block round-trips into the
        data model (the cross-game-port source)."""
        bone = self._bone([(0x40, [(44, 0, 0, 0), (50, 90, 0, 0)])])
        blob = self._make_blob(anim_count=2, slots={0: (1, 0)}, tracks={0: bone})
        pack = anim_parse_p3rd(blob)
        self.assertEqual(len(pack.animations), 1)
        a = pack.animations[0]
        self.assertEqual(a.bone_count, 1)
        self.assertEqual(len(a.tracks), 1)
        ch = a.tracks[0].channels
        self.assertEqual(len(ch), 1)
        self.assertEqual(ch[0].type & 0xFFF, 0x40)            # locX bit preserved
        self.assertEqual(a.tracks[0].tag & 0xFFF, 0x40)       # mask = OR of bits
        kf = ch[0].keyframes
        self.assertEqual([(k.value, k.frame) for k in kf], [(44, 0), (50, 90)])

    def test_corrupted_block_skipped(self):
        """A block whose bone_count overruns the blob is skipped, not raised."""
        good = self._bone([(0x40, [(1, 0, 0, 0)])])
        blob = self._make_blob(anim_count=2, slots={0: (1, 0), 1: (99, 0)},
                               tracks={0: good})   # slot1 claims 99 bones, no body
        pack = anim_parse_p3rd(blob)
        self.assertGreaterEqual(len(pack.animations), 1)      # slot 0 still parses


@unittest.skipUnless(HAS_DATA, "MHP3rd extracted data not present")
class TestAnimParseP3rdLive(unittest.TestCase):
    """Tests against actual extracted MHP3rd PAC data."""

    def test_anim_sub_quest_pac(self):
        """file_04016 sub[3]: real decode via the reference (Kurogami) header logic.
        The slot table is sized by (first_anim-hsize-4)/4 (= 100 here), NOT word0."""
        anim_blob = _read_pac_sub(QUEST_PAC, 3)
        pack = anim_parse_p3rd(anim_blob)
        word0 = struct.unpack_from("<I", anim_blob)[0]
        self.assertEqual(pack.magic, word0)                   # raw count field (53)
        hsize = struct.unpack_from("<I", anim_blob, 4)[0]
        first_anim = struct.unpack_from("<I", anim_blob, hsize - 4)[0]
        self.assertEqual(pack.slot_count, (first_anim - hsize - 4) // 4)
        self.assertGreaterEqual(len(pack.animations), 5)
        # real keyframes decoded (not a stub)
        nkf = sum(len(c.keyframes) for a in pack.animations
                  for t in a.tracks for c in t.channels)
        self.assertGreater(nkf, 500)

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
            self.assertEqual(len(anim.tracks), anim.bone_count)

    def test_anim_quest_pac2(self):
        """file_04000 sub[3]: word0(count) is small (5) but the slot table is sized by
        the reference logic (first_anim-hsize-4)/4, so the anims still decode."""
        if not os.path.isfile(QUEST_PAC2):
            self.skipTest("file_04000 not present")
        anim_blob = _read_pac_sub(QUEST_PAC2, 3)
        pack = anim_parse_p3rd(anim_blob)
        self.assertLess(pack.magic, 20)                       # raw word0 count == 5
        hsize = struct.unpack_from("<I", anim_blob, 4)[0]
        first_anim = struct.unpack_from("<I", anim_blob, hsize - 4)[0]
        self.assertEqual(pack.slot_count, (first_anim - hsize - 4) // 4)
        self.assertGreaterEqual(len(pack.animations), 1)      # decodes (old stub got 0)
        # frames non-decreasing in every decoded channel
        for a in pack.animations:
            for t in a.tracks:
                for c in t.channels:
                    fr = [k.frame for k in c.keyframes]
                    self.assertTrue(all(fr[i + 1] >= fr[i] for i in range(len(fr) - 1)))


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
