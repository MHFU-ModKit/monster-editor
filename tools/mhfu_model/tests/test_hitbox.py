"""The attack side: the record shape, the join, and the two claims that were measured.

Written against what the engine's own code does and what the live edits proved, not
against a table that looks plausible — the same discipline `test_hitzone.py` adopted
after the "these four spheres are the hurtbox" mistake.

The two anchors every test here leans on, both from 2026-09-11:

  * `0x09B674D0` copies `attack_table[id]`, **0x18 bytes**, into `node+0x1C..+0x33`.
    So `rec+0x02 -> node+0x1E` (the raw damage the damage path reads) and
    `rec+0x0A -> node+0x26` (the volume-set index).
  * editing em75's record 6 `+0x02` from 64 to 10 took the Tigrex charge from -72 HP
    to -11, and replacing volume set 2 with one bone-1 sphere moved the hit from
    645 units to 152 (r=150) and 1381 (r=1500).

Structure tests need the extracts and return early without them — no game data lives
in this repo (`docs/ASSETS.md`).
"""
import os
import struct
import sys

_TOOLS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _TOOLS)

from mhfu_model import hitbox as HB
from mhfu_model import hitzone as HZ

_DATA = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                     "workspace", "extracted", "data_files")
EM75 = os.path.join(_DATA, "file_06108.bin")
EM_FILES = [os.path.join(_DATA, "file_%05d.bin" % (6094 + i)) for i in range(17)]

#: the Tigrex charge, measured live: attack id 6, power 64, volume set 2.
CHARGE_ID, CHARGE_POWER, CHARGE_VOLUME = 6, 64, 2


def _have():
    return os.path.exists(EM75)


def _img(path):
    with open(path, "rb") as fh:
        return HZ.Image.parse(fh.read())


# ---- shape, no game data needed -------------------------------------------

def test_stride_matches_the_copy():
    """0x09B674D0's `sll 1 / addu / sll 3` is *24*, and it copies 0x18 bytes."""
    assert HB.ATTACK_STRIDE == 0x18
    assert HB.NODE_RECORD_OFF + HB.ATTACK_STRIDE - 1 == 0x33


def test_record_fields_land_where_the_node_reads_them():
    raw = bytearray(HB.ATTACK_STRIDE)
    raw[0x02], raw[0x09], raw[0x0A] = 99, 0x21, 7
    a = HB.Attack(index=3, va=0x09D60848, raw=bytes(raw))
    assert a.power == 99                       # -> node+0x1E, the raw damage
    assert a.element == 0x21                   # -> node+0x25, the resistance gate
    assert a.volume == 7                       # -> node+0x26, the volume index
    assert a.pack() == bytes(raw)


def test_edits_are_byte_surgical():
    raw = bytes(range(HB.ATTACK_STRIDE))
    a = HB.Attack(0, 0, raw)
    assert a.with_power(10).raw[0x02] == 10
    assert a.with_volume(4).raw[0x0A] == 4
    # nothing else moves
    for edited in (a.with_power(10), a.with_volume(4)):
        diff = [i for i in range(HB.ATTACK_STRIDE) if edited.raw[i] != raw[i]]
        assert len(diff) == 1


def test_marker_records_are_accepted():
    """125/126/127 carry shape=1, row=1 — rejecting them truncates most sets."""
    rec = bytearray(HB.SPHERE_STRIDE)
    struct.pack_into("<4H", rec, 0, 125, 1, 1, 0)
    assert HB._volume_record(bytes(rec), 0)
    assert not HB._volume_record(bytes(HB.SPHERE_STRIDE), 0)   # all-zero is not one


# ---- against the game's own bytes ------------------------------------------

def test_em75_charge_joins_to_the_set_that_was_measured():
    if not _have():
        return
    t = HB.tables(_img(EM75))[0]
    a = t.attacks[CHARGE_ID]
    assert a.power == CHARGE_POWER, (a.power, CHARGE_POWER)
    assert a.volume == CHARGE_VOLUME, (a.volume, CHARGE_VOLUME)
    v = t.volume_for(CHARGE_ID)
    assert v is not None and v.va == 0x09D5E8A0
    # the ten spheres the live dump walked, head to tail
    assert [s.bone for s in v.spheres] == [10, 18, 34, 4, 2, 41, 125, 42, 125, 43]


def test_em75_roar_is_a_huge_sphere_that_does_nothing():
    """Attack 8 is power 0 on a 1200-unit sphere — the signature of a roar, and a
    check that `power` is not just "the first non-zero byte"."""
    if not _have():
        return
    t = HB.tables(_img(EM75))[0]
    assert t.attacks[8].power == 0
    v = t.volume_for(8)
    assert v is not None and len(v.spheres) == 1
    assert v.spheres[0].radius == 1200.0


def test_the_tables_round_trip():
    if not _have():
        return
    for path in EM_FILES:
        if not os.path.exists(path):
            continue
        img = _img(path)
        for t in HB.tables(img):
            for v in t.volumes:
                assert v.pack(sentinel=False) == b"".join(s.raw for s in v.spheres)
            for a in t.attacks:
                assert a.pack() == img.data[img.off(a.va):img.off(a.va) + 0x18]


def test_most_overlays_are_covered():
    """15 of 17. em1 and em33 never call the setter; that is a stated gap, not a
    silent one, and this test is what would catch it getting worse."""
    if not _have():
        return
    found = sum(1 for p in EM_FILES if os.path.exists(p) and HB.tables(_img(p)))
    present = sum(1 for p in EM_FILES if os.path.exists(p))
    assert found >= present - 2, (found, present)
