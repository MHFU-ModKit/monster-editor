"""The part system: the record shape, the grid shape, and the claims about both.

The old module asserted that four bone+radius spheres in the host overlay were the
monster's hurtbox. A cold-boot test disproved it in 2026-06-28 and the module kept
saying it, which is why these tests are written against the DISASSEMBLY's arithmetic
rather than against a plausible-looking table:

    lh   v1, 4(s2)   /  sll+addu+sll  /  addu s0, a0, v0     -> block + row*10

Everything below either falls out of that or falls out of the game's own bytes.
Structure tests need the extracts and return early without them, like
`test_stream_partition.py` — no game data lives in this repo (`docs/ASSETS.md`).
"""
import os
import struct
import sys

_TOOLS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _TOOLS)

from mhfu_model import hitzone as HZ

_DATA = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                     "workspace", "extracted", "data_files")
EM75 = os.path.join(_DATA, "file_06108.bin")
GAME_TASK = os.path.join(_DATA, "file_00070.bin")
#: the 17 big-monster overlays, file_06094 onward
EM_FILES = [os.path.join(_DATA, "file_%05d.bin" % (6094 + i)) for i in range(17)]

TIGREX = 75


def _have():
    return os.path.exists(EM75) and os.path.exists(GAME_TASK)


def _img(path):
    with open(path, "rb") as fh:
        return HZ.Image.parse(fh.read())


# --------------------------------------------------------------------------- #
# the record, with no game data at all
# --------------------------------------------------------------------------- #
def test_a_sphere_round_trips_through_pack():
    s = HZ.Sphere(bone=31, shape=1, hitzone_row=2, part=1, flags=0x101,
                  radius=110.0, a=(0.0, -40.0, -75.0), b=(0.0, 0.0, 330.0))
    back = HZ.parse_sphere(s.pack(), 0)
    for f in ("bone", "shape", "hitzone_row", "part", "flags", "radius", "a", "b"):
        assert getattr(back, f) == getattr(s, f), f


def test_pack_preserves_bytes_it_does_not_understand():
    """`raw` carries the whole 0x28 record forward, so editing a radius cannot
    silently drop a field nobody has decoded yet."""
    raw = bytes(range(0x28))
    s = HZ.parse_sphere(raw, 0)
    s.radius = 123.0
    out = s.pack()
    assert len(out) == 0x28
    assert struct.unpack_from("<f", out, 0x0C)[0] == 123.0
    assert out[0x08:0x0C] == raw[0x08:0x0C]      # the flags word, untouched


def test_part_is_masked_the_way_the_engine_masks_it():
    """The deposit does `s2[+0x6] & 7`, so a part of 9 lands in slot 1 — the field
    is a u16 and only three bits of it reach `entity+0x3B8`."""
    assert HZ.Sphere(bone=0, shape=0, hitzone_row=0, part=9, flags=0,
                     radius=1.0).part_index == 1


def test_part_and_row_are_not_the_same_field():
    """The mistake this module exists to prevent. They sit two bytes apart and a
    real record uses different values for them."""
    s = HZ.Sphere(bone=12, shape=0, hitzone_row=5, part=6, flags=0, radius=90.0)
    assert s.hitzone_row != s.part
    back = HZ.parse_sphere(s.pack(), 0)
    assert (back.hitzone_row, back.part) == (5, 6)


def test_a_row_is_addressable_by_column_name_and_refuses_a_bad_percentage():
    r = HZ.HitzoneRow([100, 75, 65, 40, 0, 15, 5, 30, 20, 110])
    assert r["cut"] == 75 and r["thunder"] == 30 and r[0] == 100
    r["fire"] = 45
    assert r.values[4] == 45
    try:
        r["fire"] = 300
    except ValueError:
        pass
    else:
        raise AssertionError("a hitzone of 300 was accepted")


def test_every_column_says_what_its_name_rests_on():
    """Six of the ten names are inferred. A UI that renders an inference like a
    disassembled fact is the failure mode; it can only avoid it if the data says
    which is which."""
    assert set(HZ.COLUMN_PROVENANCE) == set(HZ.COLUMNS)
    kinds = {c: HZ.COLUMN_PROVENANCE[c].split()[0] for c in HZ.COLUMNS}
    assert kinds["cut"] == "static" and kinds["impact"] == "static"
    assert kinds["shot"] == "static"
    assert kinds["fire"] == "inferred" and kinds["thunder"] == "inferred"
    assert kinds["ko"] == "inferred"
    assert kinds["raw"] == "unnamed"          # nothing in the damage path reads it
    assert set(kinds.values()) <= {"static", "inferred", "unnamed"}, kinds


def test_the_element_bits_are_not_in_column_order():
    """Columns 6 and 7 are gated by 0x80 and 0x40 — the pair a reader assuming bit
    order would swap. Pinned so a refactor cannot quietly re-sort them."""
    order = [HZ.ELEMENT_BITS[c] for c in HZ.COLUMNS[4:9]]
    assert order == [0x10, 0x20, 0x80, 0x40, 0x100]
    assert order != sorted(order)


def test_a_block_packs_back_to_exactly_0x48():
    blk = HZ.HitzoneBlock(va=0, rows=[HZ.HitzoneRow([i] * 10) for i in range(7)])
    assert len(blk.pack()) == HZ.GRID_BLOCK
    assert HZ.GRID_ROWS * HZ.GRID_COLS + 2 == HZ.GRID_BLOCK


def test_a_short_block_is_refused_rather_than_padded():
    blk = HZ.HitzoneBlock(va=0, rows=[HZ.HitzoneRow() for _ in range(6)])
    try:
        blk.pack()
    except ValueError:
        return
    raise AssertionError("a six-row block packed without complaint")


# --------------------------------------------------------------------------- #
# the game's own bytes
# --------------------------------------------------------------------------- #
def test_the_grid_block_is_seven_rows_of_ten():
    """The load-bearing claim, and the one with an independent check: if the row
    stride were 8 or 9 the two spare bytes would not be where they are. All 114
    blocks in the game end in two zero bytes."""
    if not _have():
        return
    grid = _img(GAME_TASK)
    all_hz = HZ.all_species_hitzones(grid)
    assert len(all_hz) >= 80, len(all_hz)
    pads = {b.pad for h in all_hz.values() for b in h.states}
    assert pads == {b"\0\0"}, pads


def test_every_grid_block_round_trips_byte_for_byte():
    if not _have():
        return
    grid = _img(GAME_TASK)
    for hz in HZ.all_species_hitzones(grid).values():
        for b in hz.states:
            off = grid.off(b.va)
            assert b.pack() == grid.data[off:off + HZ.GRID_BLOCK], "0x%08X" % b.va


def test_the_state_table_sits_right_after_the_blocks_it_points_at():
    """That adjacency is what supplies the state COUNT, which is stored nowhere.
    If it ever failed, `species_hitzones` would have to guess instead."""
    if not _have():
        return
    grid = _img(GAME_TASK)
    for hz in HZ.all_species_hitzones(grid).values():
        last = hz.states[-1].va + HZ.GRID_BLOCK
        assert last == hz.state_table_va, "%d: 0x%08X vs 0x%08X" % (
            hz.species, last, hz.state_table_va)


ELEMENTS = ("fire", "water", "dragon", "thunder", "ice")


def test_the_tigrex_grid_reads_as_the_tigrex_we_know():
    """The cross-check behind the element names, stated as what is actually there.

    Column 7 is the largest element in **all seven rows of both states**, and
    column 4 totals 15 across a whole state against thunder's 145 — the native
    Tigrex is thunder-weak and fire barely scratches him. (Fire is 0 in six rows,
    not seven: row 2 has 15. An earlier draft of this test claimed seven and this
    is what caught it.) If a future edit re-orders the columns, this notices."""
    if not _have():
        return
    hz = HZ.species_hitzones(_img(GAME_TASK), TIGREX)
    assert hz is not None and hz.n_states == 2, hz
    for state in hz.states:
        for row in state.rows:
            assert row["thunder"] == max(row[c] for c in ELEMENTS), row.values
        total = {c: sum(r[c] for r in state.rows) for c in ELEMENTS}
        assert total["thunder"] == max(total.values()), total
        assert total["fire"] == min(total.values()), total
        assert total["thunder"] > 8 * total["fire"], total


def test_ko_lands_on_exactly_one_row_per_state():
    """Column 9's name rests on this: in MH a monster is knocked out by hits to the
    head and nowhere else. Across the game it is nonzero on about one row in seven,
    and on the Tigrex that row is the one with the highest values."""
    if not _have():
        return
    grid = _img(GAME_TASK)
    counts = []
    for hz in HZ.all_species_hitzones(grid).values():
        for b in hz.states:
            counts.append(sum(1 for r in b.rows if r["ko"]))
    assert counts, "no blocks"
    avg = sum(counts) / len(counts)
    assert 0.5 <= avg <= 2.0, avg


def test_the_hurtbox_sets_use_seven_rows_and_eight_parts_and_no_more():
    """The other side of the 7x10 claim, from a different file. `hitzone_row` is
    what indexes the grid, so a row of 7 in any overlay would break the block; the
    engine masks `part` with 7, so an 8 would be silently folded."""
    if not all(os.path.exists(p) for p in EM_FILES):
        return
    for path in EM_FILES:
        img = _img(path)
        sets = HZ.hurtbox_sets(img)
        for st in sets:
            for s in st.spheres:
                assert s.hitzone_row < HZ.GRID_ROWS, (img.name, s)
                assert s.part < 8, (img.name, s)


def test_every_overlay_has_a_hurtbox_table():
    """A monster with no hurtbox cannot be hit, so an empty result is a bug in the
    finder, not a fact about the game. em58 is the one that catches it: it has a
    single sentinel in its whole data section, so a sentinel-anchored search finds
    nothing there and reports 483 valid records as no table at all."""
    if not all(os.path.exists(p) for p in EM_FILES):
        return
    thin = []
    for path in EM_FILES:
        img = _img(path)
        n = sum(len(s.spheres) for s in HZ.hurtbox_sets(img))
        if n < 3:
            thin.append((img.name, n))
    assert not thin, thin


def test_a_hurtbox_set_round_trips_byte_for_byte():
    if not os.path.exists(EM75):
        return
    img = _img(EM75)
    for st in HZ.find_sets(img):
        assert st.pack(sentinel=False) == b"".join(s.raw for s in st.spheres)


def test_the_four_record_table_is_reported_as_not_a_hurtbox():
    """🔴 The correction this module carries. `VOL_VA` is bone+radius spheres with
    no part, no row and no offset — zeroing all four radii did not stop damage. It
    must not come back classified as something a hit resolves against."""
    if not os.path.exists(EM75):
        return
    img = _img(EM75)
    vols = HZ.parse_volumes(img.data, HZ.VOL_FILE_OFF)
    assert [v.bone for v in vols] == [10, 18, 41, 42]
    assert [v.radius for v in vols] == [150.0, 150.0, 170.0, 170.0]
    for st in HZ.hurtbox_sets(img):
        assert st.va != HZ.VOL_VA, "the attack-volume table came back as a hurtbox"


def test_the_old_tables_still_round_trip():
    """The one thing the earlier pass got right, kept: the overwrite mechanism is
    real even though the label was wrong."""
    if not os.path.exists(EM75):
        return
    blob = open(EM75, "rb").read()
    vols = HZ.parse_volumes(blob)
    wks = HZ.parse_weakness(blob)
    assert HZ.overwrite_in_place(blob, vols, wks) == blob


def test_the_tigrex_sphere_named_in_the_memory_map_decodes_as_documented():
    """`0x09D591D0` is the record `docs/agent_memory_map.md` calls "the overlay
    collision-node descriptor". It is one row of a table, and it carries the part
    index the deposit reads."""
    if not os.path.exists(EM75):
        return
    img = _img(EM75)
    s = HZ.parse_sphere(img.data, img.off(0x09D591D0))
    assert (s.bone, s.part_index, s.hitzone_row) == (2, 1, 2), s
    assert s.radius == 97.0 and s.a == (0.0, -30.0, 30.0), s


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    if not _have():
        print("  workspace/ extracts absent — data tests skipped (docs/ASSETS.md)")
    bad = 0
    for f in fns:
        try:
            f()
            print("  PASS  %s" % f.__name__)
        except Exception as e:
            bad += 1
            print("  FAIL  %s: %s" % (f.__name__, e))
    print("\n%d tests, %d failed" % (len(fns), bad))
    sys.exit(1 if bad else 0)
