"""Monster hit data — the part system, decoded: where he can be hit, and for how much.

Two tables, in two different files, joined by one number.

  **1. The collision spheres** — `0x28` records in the host AI overlay (`emNN.ovl`),
  bone-attached, each tagged with a `part` and a `hitzone_row`. This is the geometry
  a weapon actually touches. `docs/agent_memory_map.md` calls one of these records
  "the overlay collision-node descriptor" (`0x09D591D0` in em75); it is a whole table.

  **2. The damage grid** — `0x48` blocks in `game_task.ovl` (`file_00070`), reached
  from the species row. Seven rows of ten damage-type percentages. Species have one
  to three of them, and the engine picks by `entity.s8[+0x481]`, which is what makes
  "break the part first and *then* it gets weak" expressible at all.

A hit resolves as: sphere -> `hitzone_row` -> grid row -> the column for the damage
type -> a percentage; and separately sphere -> `part` -> `entity+0x3B8[part]`, the
per-part damage accumulator the break/sever logic totals.

    dmg = raw(node+0x1E) * row[col] / 100     (`0x09C43458` in `game_sub.ovl`)

## How the shape was established (2026-08-31, offline)

`game_sub.ovl` = `file_00075` @ `0x09C19000`, function `0x09C43458`:

    lbu   a1, 0x1E8(s3)      ; species
    sll   a0, a1, 3 / subu / sll 2 / addu / sll 4      ; a1 * 0x1D0   <- the row stride
    addiu v1, v1, -30020     ; 0x09BB8ABC = species_table + 0x2FC
    lb    v0, 0x481(s3)      ; the STATE index
    lw    v0, 0(v1)          ; -> the state pointer table
    lh    v1, 4(s2)          ; the sphere's `hitzone_row`
    lw    a0, 0(v0)          ; -> the 0x48 block for this state
    sll/addu/sll             ; row * 10
    addu  s0, a0, v0         ; s0 = block + row*10        <- ROW STRIDE IS TEN

so a block is **7 rows x 10 bytes + 2 zero pad** = `0x48`. Confirmed three ways:
bytes 70..71 are zero in all 106 blocks in the game; `hitzone_row` never exceeds 6 in
any of the 17 overlays; and `s0 = 0x09BC67AC` — the pointer caught live in the
2026-06-28 session — is exactly `0x09BC6798 + 2*10`, row 2.

⚠️ **`part` and `hitzone_row` are DIFFERENT numbers on the same record.** `part`
(0..7) is the damage accumulator and what breaks; `hitzone_row` (0..6) is which
percentage row applies. A Tigrex wing is part 6 but row 5. Conflating them is the
mistake this module exists to prevent — the old sketch had only one field.

## The columns

| col | | how we know |
|---|---|---|
| 0 | unread by the damage path; 100/90/80/70/60/50/40 on most species | shape only |
| 1 | **cut** | `0x09C43B04`, scaled onto the weapon's cut share (`player+0x10F0`) |
| 2 | **impact** | `0x09C43B40`, the blunt share (`player+0x10F1`); the engine takes whichever product is LARGER |
| 3 | **shot** | `0x09C441B8` |
| 4..8 | five element channels | each gated by a bit of the node's element mask `node+0x28`, magnitude `node+0x2A` |
| 9 | scaled against `node+0x30` | `0x09C44828` / `0x09C44C78` |

The five element columns are gated, in column order, by bits
`0x10, 0x20, 0x80, 0x40, 0x100` — note that 6 and 7 are **not** in bit order, which
is exactly the kind of thing a guess gets wrong.

🔴 **The element NAMES are inferred, not read.** Under the usual `1 << (id+3)` for
MHFU's element ids (fire 1, water 2, thunder 3, dragon 4, ice 5) the columns come out
fire / water / dragon / thunder / ice, and the native Tigrex agrees on both anchors:
column 7 is the largest element in **all seven rows of both states** and totals 145
against column 4's **15** — he is thunder-weak and all but fire-proof, which is what
MHFU's Tigrex is. The middle three rest on the shift rule alone. `COLUMN_PROVENANCE`
says so per column, and nothing here presents an inference as a measurement.

## The other two tables, and what they are NOT

`VOL_FILE_OFF` / `WK_FILE_OFF` below are the tables an earlier pass called "hurtbox
volumes" and "weakness". 🔴 **The volume table is not the player's hurtbox** — a
cold-boot test (2026-06-28) zeroed all four radii and the monster kept taking damage.

✅ **Both are now decoded, and they are the two halves of the ATTACK side** (2026-09-11,
issue #33). The `KIND_VOLUME` sets are the monster's **attack volumes** — proven by
replacing em75's set 2 in RAM with one bone-1 sphere and watching the Tigrex charge's
reach follow the radius (645 -> 152 at r=150, -> 1381 at r=1500). The `0x18` records
are the **attack table**: `0x09B674D0` copies one, by attack id, straight into
`node+0x1C..+0x33`, and its `+0x02` is the damage (64 -> 10 took the charge from -72 HP
to -11) while its `+0x0A` picks the volume set. The `0x2127`/`0x2128` that looked like a
`part_id` is two separate bytes, `+0x08` and `+0x09`, and `+0x09` is the element gate.

`hitbox.py` owns that side: it finds both tables in any overlay from the species
overlay's own call to `0x09B674C0`, and joins attack id -> record -> volume set.
Nothing here changes; `find_sets()` still finds and classifies both kinds.

## Which set is HIS (2026-09-11, offline) — the species row points at it

em75 holds four hurtbox sets and nothing in the overlay references any of them. The
pointer is in the OTHER file: `game_task.ovl`'s species row, `0x09BB8A00 + species*0x1D0`
(field `+0x240` of the `0x09BB87C0` table), holds the VA of the set that species uses.
Tigrex 75 -> `0x09BC11F0` -> `0x09D58CD0` (42 records); species 76, 81 and 88 own the
other three. So the four sets are per SPECIES ID, not per state or per action — the
record caught live in 2026-06-28 (`0x09D591D0`) is set 0's head sphere, as it should be.

Both readers walk from that pointer to the `bone == -1` sentinel — `0x09C37F30` in
`game_sub.ovl` (`lh a2,0(a1); beq a2,-1`) and `0x09A84AFC` in `game_task.ovl` — and
they gate each record on its flag bytes (`+0x08 & 0x04`, `+0x09 & 0x0A`/`0x0E`,
`+0x0A & 0x0A`/`0x0F`, `+0x0B & 0x05`/`0x01` skip). Bones `0x7D`/`0x7E`/`0x7F` are
markers, not joints: `0x7D` hands the cursor to `0x09C38658` (the tail-sever skip —
set 0 has one before each tail sphere). Which means the runtime seam is simply:
overwrite the records IN PLACE through the pointer, fewer than the original and
sentinel-terminated, or repoint the u32 at a table of your own.

→ `docs/agent_memory_map.md` "Hitzone / collision data", issues #10 / #11 / #19
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

Vec3 = Tuple[float, float, float]

# --------------------------------------------------------------------------- #
# the overlay image (MWo3), inlined so this module stays stdlib-only
# --------------------------------------------------------------------------- #
_HDR = struct.Struct("<4sIIIIIII")
TEXT_OFF = 0x80                      # 64-byte header + 64-byte pad, per ovl_explore


@dataclass
class Image:
    """Just enough of an MWo3 overlay to turn a VA into a file offset."""
    data: bytes
    load: int
    text_size: int
    data_size: int
    name: str = ""

    @classmethod
    def parse(cls, blob: bytes) -> "Image":
        magic, _oid, load, text, data, _bss, _cs, _ce = _HDR.unpack_from(blob, 0)
        if magic != b"MWo3":
            raise ValueError("not an MWo3 overlay: %r" % (magic,))
        return cls(data=blob, load=load, text_size=text, data_size=data,
                   name=blob[32:64].split(b"\0")[0].decode(errors="replace"))

    # file offset == VA - load: text starts at file 0x80 and at `load + 0x80`.
    def off(self, va: int) -> int:
        return va - self.load

    def va(self, off: int) -> int:
        return self.load + off

    @property
    def data_range(self) -> Tuple[int, int]:
        lo = TEXT_OFF + self.text_size
        return lo, lo + self.data_size

    def u32(self, va: int) -> int:
        return struct.unpack_from("<I", self.data, self.off(va))[0]

    def holds(self, va: int) -> bool:
        return 0 <= self.off(va) < len(self.data)


# --------------------------------------------------------------------------- #
# 1. the collision spheres  (host AI overlay, emNN.ovl)
# --------------------------------------------------------------------------- #
SPHERE_STRIDE = 0x28
#: a record of eight 0xFF bytes then zeros — the end of one set.
SENTINEL = (0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF)

SPHERE = "sphere"
CAPSULE = "capsule"

#: a set that names parts and hitzone rows — the geometry a hit resolves against.
KIND_HURTBOX = "hurtbox"
#: bone-attached spheres with NO part and NO row. NOT a hurtbox — see the module
#: docstring's cold-boot note. These are the ATTACK volumes (confirmed 2026-09-11);
#: `hitbox.py` reads them through the engine's own pointer table instead of
#: structurally, which is the accurate route.
KIND_VOLUME = "volume"
#: a run of bytes that parses as records and says nothing: one bone, part 0, row 0.
#: The data section is full of float arrays that validate by accident, and calling
#: one of them a hurtbox would be the same class of mistake this module corrects.
KIND_UNKNOWN = "unknown"


@dataclass
class Sphere:
    """One collision volume: a sphere on a bone, or a capsule between two points.

    `a` and `b` are BONE-RELATIVE, in engine units. A sphere uses `a` only.
    """
    bone: int
    shape: int                 # 0 = sphere, 1 = capsule (b is the far end)
    hitzone_row: int           # -> the grid row (0..6)
    part: int                  # -> entity+0x3B8[part]; the engine masks it & 7
    flags: int                 # +0x08; 0x101 / 0x300 / 0x05 / 0x02 / 0x08000000 seen
    radius: float
    a: Vec3 = (0.0, 0.0, 0.0)
    b: Vec3 = (0.0, 0.0, 0.0)
    raw: bytes = b""

    @property
    def is_capsule(self) -> bool:
        return self.shape == 1

    @property
    def part_index(self) -> int:
        """What the damage deposit actually uses: `node+0x6 & 7`."""
        return self.part & 7

    def pack(self) -> bytes:
        b = bytearray(self.raw if len(self.raw) == SPHERE_STRIDE
                      else bytes(SPHERE_STRIDE))
        struct.pack_into("<4H", b, 0, self.bone & 0xFFFF, self.shape & 0xFFFF,
                         self.hitzone_row & 0xFFFF, self.part & 0xFFFF)
        struct.pack_into("<I", b, 0x08, self.flags & 0xFFFFFFFF)
        struct.pack_into("<f", b, 0x0C, float(self.radius))
        struct.pack_into("<3f", b, 0x10, *[float(v) for v in self.a])
        struct.pack_into("<3f", b, 0x1C, *[float(v) for v in self.b])
        return bytes(b)


@dataclass
class SphereSet:
    """One sentinel-delimited run of records."""
    va: int
    spheres: List[Sphere]
    kind: str = KIND_HURTBOX

    @property
    def parts(self) -> List[int]:
        return sorted({s.part_index for s in self.spheres})

    @property
    def rows(self) -> List[int]:
        return sorted({s.hitzone_row for s in self.spheres})

    @property
    def bones(self) -> List[int]:
        return sorted({s.bone for s in self.spheres})

    def by_part(self) -> Dict[int, List[Sphere]]:
        out: Dict[int, List[Sphere]] = {}
        for s in self.spheres:
            out.setdefault(s.part_index, []).append(s)
        return out

    def pack(self, sentinel: bool = True) -> bytes:
        body = b"".join(s.pack() for s in self.spheres)
        return body + (build_sentinel() if sentinel else b"")


def build_sentinel() -> bytes:
    b = bytearray(SPHERE_STRIDE)
    struct.pack_into("<4H", b, 0, *SENTINEL)
    return bytes(b)


def parse_sphere(buf: bytes, off: int) -> Sphere:
    rec = buf[off:off + SPHERE_STRIDE]
    bone, shape, row, part = struct.unpack_from("<4H", rec, 0)
    flags = struct.unpack_from("<I", rec, 0x08)[0]
    radius = struct.unpack_from("<f", rec, 0x0C)[0]
    a = struct.unpack_from("<3f", rec, 0x10)
    b = struct.unpack_from("<3f", rec, 0x1C)
    return Sphere(bone=bone, shape=shape, hitzone_row=row, part=part, flags=flags,
                  radius=radius, a=tuple(a), b=tuple(b), raw=rec)


#: bones above this are not rig indices. 125 shows up as a tail marker on some sets
#: and is kept so a round trip stays byte-identical.
MAX_BONE = 127
MAX_RADIUS = 1000.0


def _plausible(buf: bytes, off: int) -> bool:
    if off < 0 or off + SPHERE_STRIDE > len(buf):
        return False
    bone, shape, row, part = struct.unpack_from("<4H", buf, off)
    if (bone, shape, row, part) == SENTINEL:
        return True
    if bone > MAX_BONE or shape > 1 or row > 6 or part > 7:
        return False
    radius = struct.unpack_from("<f", buf, off + 0x0C)[0]
    if not (0.0 <= radius < MAX_RADIUS):
        return False
    # offsets are metres-ish in engine units; a garbage float fails this fast
    for v in struct.unpack_from("<6f", buf, off + 0x10):
        if not (-4000.0 <= v <= 4000.0):
            return False
    return True


def _is_sentinel(buf: bytes, off: int) -> bool:
    return (off + SPHERE_STRIDE <= len(buf)
            and struct.unpack_from("<4H", buf, off) == SENTINEL)


def _degenerate(buf: bytes, off: int) -> bool:
    """An all-zero record. It parses as a legal sphere of radius 0 on bone 0, so it
    has to be excluded explicitly or a run walks straight into the next zero-filled
    region of the data section and swallows it."""
    return buf[off:off + SPHERE_STRIDE] == bytes(SPHERE_STRIDE)


#: below this a "run" is more likely to be a float array that happens to validate.
MIN_SET_RECORDS = 3


def find_sets(img: Image) -> List[SphereSet]:
    """Every sphere set in an overlay's DATA section, sentinel-delimited.

    There is no header pointer to find them by — nothing in `file_06108` points at
    the table's first record — so they are recovered structurally: find each
    MAXIMAL run of records at `0x28` stride that all validate, then cut it at the
    sentinels inside it.

    ⚠️ It cannot anchor on the sentinels alone. em58 has exactly ONE sentinel in
    its whole data section and 483 valid records, so a sentinel-anchored walk finds
    nothing there and reports a monster with no hurtbox — which is why `--verify`
    prints the record count per overlay rather than just "parsed ok".
    """
    buf = img.data
    lo, hi = img.data_range

    def member(off: int) -> bool:
        return _plausible(buf, off) and not _degenerate(buf, off)

    # 1. every maximal run, at every 4-byte phase. A run is a *candidate* only:
    #    a misaligned phase can validate over the same bytes as the real table.
    runs: List[Tuple[int, int]] = []                   # (start, record count)
    for off in range(lo, hi - SPHERE_STRIDE, 4):
        if not member(off) or member(off - SPHERE_STRIDE):
            continue
        n = 0
        while member(off + n * SPHERE_STRIDE):
            n += 1
        if n >= MIN_SET_RECORDS:
            runs.append((off, n))

    # 2. longest wins. Two phases cannot both be right about the same bytes, and
    #    the real table is the one that explains more of them.
    taken: List[Tuple[int, int]] = []
    for start, n in sorted(runs, key=lambda r: (-r[1], r[0])):
        end = start + n * SPHERE_STRIDE
        if any(start < e and s < end for s, e in taken):
            continue
        taken.append((start, end))

    # 3. cut each accepted run at its sentinels
    sets: List[SphereSet] = []
    for start, end in sorted(taken):
        base, cur = start, []
        for at in range(start, end, SPHERE_STRIDE):
            if _is_sentinel(buf, at):
                _emit(sets, img, base, cur)
                base, cur = at + SPHERE_STRIDE, []
            else:
                cur.append(parse_sphere(buf, at))
        _emit(sets, img, base, cur)
    return sets


def _classify(recs: List[Sphere]) -> str:
    """What a run of records is, judged only on what is in it.

    A real hurtbox set spreads over the rig and names several parts or several
    hitzone rows — em75's four sets each touch 26-30 bones and use all 8 parts and
    all 7 rows. A run on ONE bone with part 0 and row 0 throughout carries no
    information a hit could use, and the overlay data sections are full of float
    arrays that parse as exactly that.
    """
    bones = {s.bone for s in recs}
    if len(bones) < 2:
        return KIND_UNKNOWN
    if len({s.part_index for s in recs}) > 1 or len({s.hitzone_row for s in recs}) > 1:
        return KIND_HURTBOX
    if all(s.part == 0 and s.hitzone_row == 0 and s.a == (0.0,) * 3 for s in recs):
        return KIND_VOLUME
    return KIND_UNKNOWN


def _emit(sets: List[SphereSet], img: Image, va_off: int,
          recs: List[Sphere]) -> None:
    if len(recs) < MIN_SET_RECORDS or not any(s.radius > 0.0 for s in recs):
        return
    sets.append(SphereSet(va=img.va(va_off), spheres=recs, kind=_classify(recs)))


def hurtbox_sets(img: Image) -> List[SphereSet]:
    """The sets that carry parts — the ones a hit can land on."""
    return [s for s in find_sets(img) if s.kind == KIND_HURTBOX]


# --------------------------------------------------------------------------- #
# 2. the damage grid  (game_task.ovl, file_00070)
# --------------------------------------------------------------------------- #
GAME_TASK_LOAD = 0x09A5F200
SPECIES_TABLE = 0x09BB87C0
SPECIES_STRIDE = 0x1D0
#: the species row field holding the pointer to this species' state table.
STATE_TABLE_FIELD = 0x2FC
#: the species row field the break/sever logic reads (byte0 = part, byte1 = mask).
#: Located, not decoded — `0x09BB8AFA + species*0x1D0`.
BREAK_FIELD = 0x33A

#: the species row field holding the VA of THIS species' hurtbox set (`0x09BB8A00`
#: for species 0). The overlay never references its own sets; this does.
SPHERE_TABLE_FIELD = 0x240
#: a record's flag bytes that make BOTH walkers skip it (`+0x08..+0x0B`, little
#: endian). A record with `flags & WALK_SKIP_MASK == 0` is seen by the weapon-hit
#: walker `0x09C37F30` and by `0x09A84AFC` alike; set 0's hurtboxes are all 0/0x101.
WALK_SKIP_MASK = 0x050A0A04
#: bones that are markers, not joints. 0x7D calls `0x09C38658` with the cursor.
MARKER_BONES = (0x7D, 0x7E, 0x7F)

GRID_ROWS = 7
GRID_COLS = 10
GRID_BLOCK = 0x48                    # 7 * 10, then two zero bytes
MAX_STATES = 8                       # a walk bound; the real max in game is 3

COLUMNS = ("raw", "cut", "impact", "shot",
           "fire", "water", "dragon", "thunder", "ice", "ko")

#: what each column name rests on. The UI must not render an inference the same way
#: it renders a disassembled fact.
COLUMN_PROVENANCE = {
    "raw": "unnamed — no site in the damage path reads column 0; the name is a "
           "placeholder for a value that is 100/90/80/70/60/50/40 on most species",
    "cut": "static — 0x09C43B04, scaled onto the weapon's cut share",
    "impact": "static — 0x09C43B40, the blunt share; the larger product wins",
    "shot": "static — 0x09C441B8",
    "fire": "inferred — column 4, element-mask bit 0x10. Cross-check: the lowest "
            "element on the native Tigrex by a factor of ten (15 against thunder's "
            "145 over a state), and fire is famously useless on him",
    "water": "inferred — column 5, bit 0x20, from `1 << (id+3)` alone",
    "dragon": "inferred — column 6, bit 0x80, from `1 << (id+3)` alone",
    "thunder": "inferred — column 7, bit 0x40. Cross-check: the largest element in "
               "all seven Tigrex rows of both states, and thunder is his known "
               "weakness",
    "ice": "inferred — column 8, bit 0x100, from `1 << (id+3)` alone",
    "ko": "inferred — column 9, scaled against node+0x30 at 0x09C44C78. "
          "Cross-check: nonzero on about ONE row per block across the whole game, "
          "and on the Tigrex that row is the one with the highest values — which "
          "is what a knock-out channel looks like. The name was not read",
}

#: element column -> the bit of `node+0x28` that gates it. NOT in bit order.
ELEMENT_BITS = {"fire": 0x10, "water": 0x20, "dragon": 0x80,
                "thunder": 0x40, "ice": 0x100}


@dataclass
class HitzoneRow:
    """Ten damage-type percentages for one hitzone row."""
    values: List[int] = field(default_factory=lambda: [0] * GRID_COLS)

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.values[COLUMNS.index(key)]
        return self.values[key]

    def __setitem__(self, key, value):
        i = COLUMNS.index(key) if isinstance(key, str) else key
        if not 0 <= int(value) <= 255:
            raise ValueError("hitzone %r out of range 0..255" % (value,))
        self.values[i] = int(value)

    def as_dict(self) -> Dict[str, int]:
        return dict(zip(COLUMNS, self.values))


@dataclass
class HitzoneBlock:
    """One state's grid: seven rows, plus the two trailing bytes, preserved."""
    va: int
    rows: List[HitzoneRow]
    pad: bytes = b"\0\0"

    def pack(self) -> bytes:
        out = bytearray()
        for r in self.rows:
            out += bytes(r.values)
        out += self.pad
        if len(out) != GRID_BLOCK:
            raise ValueError("block is %d bytes, not 0x48" % len(out))
        return bytes(out)


@dataclass
class SpeciesHitzones:
    species: int
    row_va: int
    state_table_va: int
    states: List[HitzoneBlock]

    @property
    def n_states(self) -> int:
        return len(self.states)


def parse_block(buf: bytes, off: int, va: int = 0) -> HitzoneBlock:
    rows = [HitzoneRow(list(buf[off + r * GRID_COLS: off + (r + 1) * GRID_COLS]))
            for r in range(GRID_ROWS)]
    return HitzoneBlock(va=va, rows=rows,
                        pad=bytes(buf[off + GRID_ROWS * GRID_COLS: off + GRID_BLOCK]))


def species_hitzones(img: Image, species: int) -> Optional[SpeciesHitzones]:
    """The grid for one species, or None if its row has no state table.

    The state COUNT is not stored anywhere. It does not have to be: the pointer
    table sits immediately after the last block it points at, so the count falls
    out of `(table - first_block) / 0x48` — and every entry is checked to be a
    pointer into the file before it is believed.
    """
    row_va = SPECIES_TABLE + species * SPECIES_STRIDE
    if not img.holds(row_va + SPECIES_STRIDE):
        return None
    tbl = img.u32(row_va + STATE_TABLE_FIELD)
    if not img.holds(tbl + 4):
        return None
    blocks: List[int] = []
    for i in range(MAX_STATES):
        va = img.u32(tbl + 4 * i)
        if not (img.holds(va + GRID_BLOCK) and va % 4 == 0 and va < tbl):
            break
        blocks.append(va)
    if not blocks:
        return None
    # the self-check: contiguous 0x48 blocks running up to the table itself
    if blocks != [blocks[0] + i * GRID_BLOCK for i in range(len(blocks))]:
        return None
    return SpeciesHitzones(
        species=species, row_va=row_va, state_table_va=tbl,
        states=[parse_block(img.data, img.off(va), va) for va in blocks])


def species_sphere_table(img: Image, species: int) -> Optional[int]:
    """The VA of the hurtbox set THIS species walks, from its `game_task.ovl` row.

    Returns None when the row is outside the file. The value is only meaningful
    together with the overlay it points into — `species_sets` does that join.
    """
    at = SPECIES_TABLE + SPHERE_TABLE_FIELD + species * SPECIES_STRIDE
    if not img.holds(at + 4):
        return None
    return img.u32(at)


#: a walk bound for `walk_set` — the longest set in the game is em59's 49.
MAX_SET_RECORDS = 512


def walk_set(img: Image, va: int, limit: int = MAX_SET_RECORDS) -> Optional[SphereSet]:
    """The set at a KNOWN VA, read the way the engine reads it: record after
    record until the sentinel. No plausibility gate — this is the authoritative
    reader once the species row has said where the table is, and the gate is what
    makes `find_sets` miss em58 (a 1100-unit sphere) and em55 (a marker with
    shape 8). Returns None if the VA is outside the image or no sentinel turns up
    within ``limit`` records.
    """
    if not img.holds(va + SPHERE_STRIDE):
        return None
    recs: List[Sphere] = []
    off = img.off(va)
    for _ in range(limit):
        if off + SPHERE_STRIDE > len(img.data):
            return None
        if _is_sentinel(img.data, off):
            return SphereSet(va=va, spheres=recs, kind=KIND_HURTBOX)
        recs.append(parse_sphere(img.data, off))
        off += SPHERE_STRIDE
    return None


def own_set(overlay: Image, game_task: Image, species: int) -> Optional[SphereSet]:
    """The hurtbox set ``species`` actually walks: its row's pointer, followed.

    This is the one a weapon resolves against and the one a runtime table replaces
    IN PLACE — its record count is the capacity. All 17 big-monster species resolve
    (2026-09-11); a None here means the pointer left the overlay.
    """
    va = species_sphere_table(game_task, species)
    if va is None:
        return None
    lo, hi = overlay.data_range
    if not lo <= overlay.off(va) < hi:
        return None
    return walk_set(overlay, va)


def species_sets(overlay: Image, game_task: Image,
                 limit: int = 0x100) -> Dict[int, int]:
    """`{set_va: species}` for every species whose row points INTO this overlay.

    em75 answers `{0x09D58CD0: 75, 0x09D599A0: 76, 0x09D59388: 81, 0x09D59FB8: 88}`:
    one overlay, four species ids, one set each. A set no row points at is dead
    data as far as a weapon is concerned.
    """
    lo, hi = overlay.data_range
    out: Dict[int, int] = {}
    for sp in range(limit):
        va = species_sphere_table(game_task, sp)
        if va is not None and lo <= overlay.off(va) < hi and va not in out:
            out[va] = sp
    return out


def all_species_hitzones(img: Image, limit: int = 0x100) -> Dict[int, SpeciesHitzones]:
    out = {}
    for s in range(limit):
        hz = species_hitzones(img, s)
        if hz is not None:
            out[s] = hz
    return out


def overwrite_block(blob: bytes, block: HitzoneBlock, load: int = GAME_TASK_LOAD) -> bytes:
    """Write one state's grid back at its own VA. Same size, always."""
    out = bytearray(blob)
    off = block.va - load
    out[off:off + GRID_BLOCK] = block.pack()
    return bytes(out)


# --------------------------------------------------------------------------- #
# 3. the two older tables — kept, and correctly labelled
# --------------------------------------------------------------------------- #
# ⚠️ VOL_* names one of the SIX part-less sphere tables in em75. Zeroing all four
# radii did not stop player damage (cold boot, 2026-06-28), so these are attack or
# broadphase volumes. `find_sets()` reports every one of them, classified.
VOL_FILE_OFF = 0x449C8
VOL_VA = 0x09D5EB48
VOL_COUNT = 4
VOL_STRIDE = 0x28

WK_FILE_OFF = 0x466E0          # first record (0x18 lead-in after the table base VA)
WK_VA = 0x09D60848 + 0x18
WK_COUNT = 6
WK_STRIDE = 0x18

HDR_PTR_VOL = 0x46600          # file offset where the volumes-table VA is stored
HDR_PTR_WK = 0x470D0           # file offset where the weakness-table VA is stored


@dataclass
class Volume:
    bone: int
    radius: float
    raw: bytes = b""           # full 0x28 record (reserved bytes preserved on edit)

    def pack(self) -> bytes:
        b = bytearray(self.raw if len(self.raw) == VOL_STRIDE else bytes(VOL_STRIDE))
        struct.pack_into("<I", b, 0x00, self.bone & 0xFFFFFFFF)
        struct.pack_into("<f", b, 0x0C, float(self.radius))
        return bytes(b)


@dataclass
class Weakness:
    """UNDECODED. Parsed and round-tripped; `part` holds 0x2127/0x2128 plus a
    variant index and nothing here knows what those mean."""
    part: int
    type: int
    value: int
    w0: int = 0
    w1: int = 0
    flag: int = 1
    raw: bytes = b""

    def pack(self) -> bytes:
        b = bytearray(self.raw if len(self.raw) == WK_STRIDE else bytes(WK_STRIDE))
        struct.pack_into("<6I", b, 0,
                         self.w0, self.w1, self.part & 0xFFFFFFFF,
                         self.flag, self.type, self.value & 0xFFFFFFFF)
        return bytes(b)


def parse_volumes(buf: bytes, off: int = VOL_FILE_OFF, n: int = VOL_COUNT) -> List[Volume]:
    out = []
    for i in range(n):
        o = off + i * VOL_STRIDE
        rec = buf[o:o + VOL_STRIDE]
        bone = struct.unpack_from("<I", rec, 0)[0]
        radius = struct.unpack_from("<f", rec, 0x0C)[0]
        out.append(Volume(bone=bone, radius=radius, raw=rec))
    return out


def parse_weakness(buf: bytes, off: int = WK_FILE_OFF, n: int = WK_COUNT) -> List[Weakness]:
    out = []
    for i in range(n):
        o = off + i * WK_STRIDE
        rec = buf[o:o + WK_STRIDE]
        w0, w1, part, flag, typ, val = struct.unpack_from("<6I", rec, 0)
        out.append(Weakness(part=part, type=typ, value=val, w0=w0, w1=w1,
                            flag=flag, raw=rec))
    return out


def build_volume_table(vols: Sequence[Volume]) -> bytes:
    return b"".join(v.pack() for v in vols)


def build_weakness_table(wks: Sequence[Weakness]) -> bytes:
    return b"".join(w.pack() for w in wks)


def overwrite_in_place(overlay: bytes, vols: Sequence[Volume] = None,
                       wks: Sequence[Weakness] = None) -> bytes:
    """Return a copy of the overlay with the two older tables overwritten IN PLACE
    (same record count — the safe case). Raises if a table would change size."""
    out = bytearray(overlay)
    if vols is not None:
        if len(vols) != VOL_COUNT:
            raise ValueError(f"volume count {len(vols)} != {VOL_COUNT} (use relocate)")
        out[VOL_FILE_OFF:VOL_FILE_OFF + VOL_COUNT * VOL_STRIDE] = build_volume_table(vols)
    if wks is not None:
        if len(wks) != WK_COUNT:
            raise ValueError(f"weakness count {len(wks)} != {WK_COUNT} (use relocate)")
        out[WK_FILE_OFF:WK_FILE_OFF + WK_COUNT * WK_STRIDE] = build_weakness_table(wks)
    return bytes(out)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
_DATA = "workspace/extracted/data_files"
#: the 17 big-monster overlays, in file order — same list as tools/em_abi.py
EM_SPECIES = [1, 2, 7, 14, 15, 17, 20, 21, 33, 40, 54, 55, 58, 59, 75, 82, 83]


def _load(path) -> Image:
    import os
    p = str(path)
    if not os.path.exists(p):
        cand = os.path.join(_DATA, os.path.basename(p))
        if os.path.exists(cand):
            p = cand
    with open(p, "rb") as fh:
        return Image.parse(fh.read())


def report(img: Image, grid: Optional[Image] = None,
           species: Optional[int] = None) -> str:
    out = ["%s  load 0x%08X" % (img.name or "?", img.load), ""]
    sets = find_sets(img)
    hb = [s for s in sets if s.kind == KIND_HURTBOX]
    vol = [s for s in sets if s.kind == KIND_VOLUME]
    unk = [s for s in sets if s.kind == KIND_UNKNOWN]
    out.append("collision spheres — %d hurtbox set(s), %d part-less volume set(s), "
               "%d run(s) that say nothing" % (len(hb), len(vol), len(unk)))
    owners = species_sets(img, grid) if grid is not None else {}
    for s in hb + vol:
        caps = sum(1 for x in s.spheres if x.is_capsule)
        who = owners.get(s.va)
        out.append("  0x%08X  %-8s %3d recs (%d capsule)  bones %2d  parts %s  rows %s%s"
                   % (s.va, s.kind, len(s.spheres), caps, len(s.bones),
                      s.parts, s.rows,
                      "" if who is None else "  <- species %d" % who))
    if grid is not None and species is not None:
        mine = species_sphere_table(grid, species)
        if mine is not None and mine in {s.va for s in hb}:
            out.append("  species %d walks 0x%08X (row field +0x%X); the others "
                       "belong to other species ids" % (species, mine, SPHERE_TABLE_FIELD))
    if vol:
        out += ["", "  ⚠️ a part-less set is NOT the player's hurtbox: zeroing the "
                "four radii", "     at 0x09D5EB48 did not stop damage (cold boot, "
                "2026-06-28)."]
    if grid is not None and species is not None:
        hz = species_hitzones(grid, species)
        out += ["", "damage grid — species %d, row 0x%08X" % (species, hz.row_va)
                if hz else "", ]
        if hz:
            out.append("  %d state(s) at 0x%08X" % (hz.n_states, hz.state_table_va))
            head = "  row  " + " ".join("%7s" % c for c in COLUMNS)
            out.append(head)
            for si, blk in enumerate(hz.states):
                out.append("  -- state %d  (0x%08X)" % (si, blk.va))
                for ri, r in enumerate(blk.rows):
                    out.append("  %3d  " % ri + " ".join("%7d" % v for v in r.values))
        else:
            out.append("damage grid — species %d has no state table" % species)
    return "\n".join(x for x in out if x is not None)


def verify() -> int:
    """Parse every overlay and every species grid, and assert the shape holds."""
    bad = 0
    grid = _load("file_00070.bin")
    print("== collision spheres, all 17 overlays")
    for i, sp in enumerate(EM_SPECIES):
        img = _load("file_%05d.bin" % (6094 + i))
        sets = hurtbox_sets(img)
        rows = sorted({s.hitzone_row for st in sets for s in st.spheres})
        parts = sorted({s.part_index for st in sets for s in st.spheres})
        n = sum(len(st.spheres) for st in sets)
        ok = (not rows or rows[-1] <= 6) and (not parts or parts[-1] <= 7)
        bad += not ok
        print("  em%-3d %-12s %d set(s) %3d recs  rows<=%s parts<=%s  %s"
              % (sp, img.name, len(sets), n, rows[-1] if rows else "-",
                 parts[-1] if parts else "-", "ok" if ok else "SHAPE VIOLATION"))
        for st in find_sets(img):
            if st.pack(sentinel=False) != b"".join(s.raw for s in st.spheres):
                print("    ROUND TRIP FAILED at 0x%08X" % st.va)
                bad += 1
    print("\n== the set each species WALKS (row +0x%X), read to the sentinel"
          % SPHERE_TABLE_FIELD)
    for i, sp in enumerate(EM_SPECIES):
        img = _load("file_%05d.bin" % (6094 + i))
        own = own_set(img, grid, sp)
        if own is None:
            print("  em%-3d NO OWN SET — the row pointer left the overlay" % sp)
            bad += 1
            continue
        found = own.va in {st.va for st in find_sets(img)}
        print("  em%-3d 0x%08X %3d recs  %s" % (sp, own.va, len(own.spheres),
              "also found structurally" if found else "MISSED by find_sets"))
    print("\n== damage grid, every species with a state table")
    hz = all_species_hitzones(grid)
    pad = {b.pad for h in hz.values() for b in h.states}
    blocks = sum(h.n_states for h in hz.values())
    print("  %d species, %d blocks, trailing pad values %s"
          % (len(hz), blocks, sorted(pad)))
    if pad != {b"\0\0"}:
        print("  PAD IS NOT ALWAYS ZERO — the 7x10 reading is wrong")
        bad += 1
    for h in hz.values():
        for b in h.states:
            if b.pack() != grid.data[grid.off(b.va):grid.off(b.va) + GRID_BLOCK]:
                print("  ROUND TRIP FAILED at 0x%08X" % b.va)
                bad += 1
    print("\n%s" % ("all checks passed" if not bad else "%d PROBLEM(S)" % bad))
    return 1 if bad else 0


def main(argv=None) -> int:                                    # pragma: no cover
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("overlay", nargs="?", default="file_06108.bin",
                    help="a big-monster AI overlay (emNN.ovl / file_06094+)")
    ap.add_argument("--species", type=int, default=75,
                    help="species id for the damage grid (Tigrex = 75)")
    ap.add_argument("--no-grid", action="store_true",
                    help="skip the grid (do not read file_00070.bin)")
    ap.add_argument("--verify", action="store_true",
                    help="parse all 17 overlays + every grid and check the shape")
    a = ap.parse_args(argv)
    if a.verify:
        return verify()
    grid = None if a.no_grid else _load("file_00070.bin")
    print(report(_load(a.overlay), grid, a.species))
    return 0


if __name__ == "__main__":                                     # pragma: no cover
    raise SystemExit(main())
