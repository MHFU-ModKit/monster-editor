"""Stage (map) files — the geometry, the collision and the table that picks them.

A playable area in MHFU is a **stage**, numbered 0..266, and a stage number is the
same number the runtime calls `area_index`. Each one is two files:

    st<NNN>.pac     geometry + textures + environment + COLLISION   file id NNN + 5808
    stage<NNN>.ovl  the per-stage MWo3 code overlay                 file id NNN + 5542

(Those are the ENGINE's file ids. `extract_iso.py` writes the TOC one lower, so on
disk they are `file_0{NNN+5807}.bin` and `file_0{NNN+5541}.bin` — same off-by-one as
the monster PACs.)

`st<NNN>.pac` is a plain package with a fixed six-entry shape:

    [0] PMO 1.0   terrain / main mesh
    [1] TMH       textures
    [2] PMO 1.0   props, water, skybox (empty in 5 stages)
    [3] 170 B     environment: `u16 2`, RGBA fog colour, clip/fog floats
    [4] ~2-5 KB   `0x02 0x01` parameter block (pointer-fixed at load)
    [5] HITS      the collision, as a nested package of 2 `HITS` chunks
    [6..7]        two more `0x02 0x00` blocks in 8 of the 282 files

A `HITS` chunk is a 0x28-byte header, an `nx * nz` spatial grid, the per-cell
triangle lists, then a flat array of **56-byte triangles**:

    u32 flags;  float v0[3], v1[3], v2[3];  float normal[3];  float plane_d

with `normal == normalize(cross(v1 - v0, v2 - v0))` and `dot(normal, v) + plane_d == 0`
— the self-check `verify()` runs, and what proves the record layout (245 of the 282
files pass at >99%; the other 20 are 2 KB placeholder stages with no sub-resource at
all, and st046 + its 16 variants have one chunk at 94%).

The **broadphase grid** is a uniform xz lattice over the world, `cell` units square
(500 or 501, never anything else) with its origin at (0, 0) in every one of the 492
chunks that ship. Cell `(ix, iz)` is entry **`ix * nz + iz`**, and its u32 is the
offset of a list of triangles overlapping that cell: `tri_index * 56` each, strictly
ascending, `0xFFFFFFFF`-terminated, at most 124 long. On disk the lists are laid out
back to back in cell order so `grid[i]` runs to `grid[i + 1]`, but nothing at runtime
relies on that — the terminator ends the list.

Membership is **conservative**: every cell a triangle actually overlaps is listed
(exact triangle-vs-rect, verified against all 246 stages), plus a few percent of
extras. The only systematic omissions are perfectly vertical triangles, whose xz
projection is a line. So a rebuilt grid may safely list more than the original; it
must never list less.

Chunk **1** is the walkable floor. Chunk 0 is the taller set (walls / ceilings /
camera). Vertices are WORLD space, the same frame the player position is read in,
so a stage can be identified from a recorded coordinate — with the caveat that the
player world position sits **~230-250 units above** the floor it stands on.

All offsets inside a HITS chunk are relative to `chunk_base + 8` — except the cell-list
entries, which are relative to the **triangle array**. The loader rewrites every one of
them to an absolute VA in place, so at runtime the structure is pure pointers: header
`+0x20`/`+0x24` hold the grid and triangle VAs, each grid word holds its cell list's VA,
and each list word holds a triangle's VA. Terminators and float vertices are untouched.

🔴 That is what makes a map editable **memory-only, including adding geometry**: because
nothing at runtime is offset-relative, a new grid, a new cell list or a new triangle can
be written ANYWHERE in free RAM and linked in by overwriting one pointer. You are not
confined to the resident chunk's original size.

🔴 **That is true of the COLLISION only.** The visible mesh (`sub[0]`/`sub[2]`, PMO) is
consumed ONCE at load: zeroing the entire terrain PMO in the resident PAC changes nothing
on screen, and a savestate round trip rules out an emulator cache. So a geometry edit
needs a reload — the inject path (`mhfu_model.inject`, "geometry edits apply on the next
engine rebuild") or a repacked PAC — while a collision edit is live. Editing a map means
two different write paths for the two halves of it.
"""

import os
import struct
from dataclasses import dataclass

PAC_BASE = 5808      # engine file id of st001.pac ... st266.pac is 5808 + 266
OVL_BASE = 5542      # engine file id of stage000.ovl
EXTRACT_SKEW = -1    # extracted file_NNNNN index = engine file id - 1

STAGE_MIN, STAGE_MAX = 0, 266
TRI_STRIDE = 56
GRID_OFF = 0x28          # the grid array always starts here (and header +0x20 says so)
TERM = 0xFFFFFFFF        # every cell list ends with this
SUB_ALIGN = 16           # a PAC's sub-resources are 16-byte aligned

#: resource slot each stage file is requested into (2nd arg of 0x088DDBD8)
SLOT_PAC, SLOT_OVL = 21, 41
SLOT_SND = (38, 39, 40)


def pac_file(stage: int, extracted: bool = True) -> int:
    return stage + PAC_BASE + (EXTRACT_SKEW if extracted else 0)


def ovl_file(stage: int, extracted: bool = True) -> int:
    return stage + OVL_BASE + (EXTRACT_SKEW if extracted else 0)


def read_extracted(data_dir, file_index) -> bytes:
    with open(os.path.join(data_dir, "file_%05d.bin" % file_index), "rb") as fh:
        return fh.read()


# --- the stage OVERLAY: surface-property table ---------------------------
#
# `stage<NNN>.ovl` is an MWo3 overlay, fully linked at its load address, whose
# parameter object is returned by vtable slot 10 of the object the map manager
# holds at `*(0x09A4F060) + 0`. That slot is always a constant getter
#
#     lui v0, HI ; jr ra ; addiu v0, v0, LO      ->  P
#
# and `P + 0x0C` is the SURFACE-PROPERTY TABLE: one u32 bitmask per surface id,
# indexed by the low byte of a HITS triangle's `flags`. See `TriFlags`.

OVL_MAGIC = b"MWo3"
OVL_HDR = struct.Struct("<4sIIIIIII")
PARAM_TBL_OFF = 0x0C     # the parameter object field holding the table
VT_GETTER_SLOT = 10      # vtable slot that returns the parameter object


class Overlay:
    """Just enough of an MWo3 overlay to find the stage parameter object."""

    def __init__(self, data: bytes):
        magic, self.oid, self.load, self.text_size, self.data_size, \
            self.bss_size, self.ctor_s, self.ctor_e = OVL_HDR.unpack_from(data, 0)
        if magic != OVL_MAGIC:
            raise ValueError("not an MWo3 overlay: %r" % magic)
        self.data = data
        self.text_off = 0x80                       # 64-byte header + 64-byte pad
        self.image_end = self.text_off + self.text_size + self.data_size
        self.mem_end = self.image_end + self.bss_size

    def word(self, va):
        """u32 at a VA, or None outside the file image (bss reads as 0)."""
        off = va - self.load
        if off < 0 or off >= self.mem_end:
            return None
        if off + 4 > self.image_end:
            return 0                               # bss: zero-initialised
        return struct.unpack_from("<I", self.data, off)[0]

    def param_object(self):
        """VA of the stage parameter object, or None.

        Every `lui v0 / jr ra / addiu v0` in text is a candidate; the parameter
        object is the one with the run of 0xFFFFFFFF at +0x14 and a table
        pointer at +0x0C.
        """
        best = None
        for off in range(self.text_off, self.text_off + self.text_size - 8, 4):
            w0, w1, w2 = struct.unpack_from("<3I", self.data, off)
            if (w0 >> 26) != 0x0F or ((w0 >> 16) & 31) != 2:        # lui v0
                continue
            if w1 != 0x03E00008:                                     # jr ra
                continue
            if (w2 >> 26) != 0x09 or ((w2 >> 21) & 31) != 2 or ((w2 >> 16) & 31) != 2:
                continue                                             # addiu v0, v0
            P = ((w0 & 0xFFFF) << 16) + (w2 & 0xFFFF)
            t = self.word(P + PARAM_TBL_OFF)
            if t is None or not (self.load <= t < self.load + self.mem_end):
                continue
            # +0x14 is a u16[10] of small slot ids padded with 0xFFFF, and no other
            # constant this idiom returns looks remotely like it — that is the whole
            # filter, and it picks the right object in 267 of the 267 overlays.
            ws = [self.word(P + 0x14 + 4 * k) for k in range(5)]
            if any(w is None for w in ws):
                continue
            slots = [h for w in ws for h in (w & 0xFFFF, w >> 16)]
            if any(h != 0xFFFF and h >= 0x40 for h in slots):
                continue
            run = slots.count(0xFFFF)
            if run < 2:
                continue
            if best is None or run > best[0]:
                best = (run, P)
        return best[1] if best else None

    def surface_table(self, count=8):
        """[u32] surface-property bitmasks, or None if there is no param object.

        `count` entries are read; the table's real length is not stored anywhere,
        so bound it by the largest surface id the stage's triangles actually use
        (`Stage.surface_ids()`). A table in bss reads as all zeroes, which is
        what a stage with no special surfaces ships.
        """
        P = self.param_object()
        if P is None:
            return None
        t = self.word(P + PARAM_TBL_OFF)
        return [self.word(t + 4 * k) for k in range(count)]


def read_overlay(data_dir, stage: int) -> Overlay:
    return Overlay(read_extracted(data_dir, ovl_file(stage)))


def subresources(blob: bytes):
    """[(offset, size)] of a package.

    ⚠️ `pac_file(0)` is NOT a stage PAC — engine file 5808 is `stage266.ovl`, and
    the PAC range starts at st001. A bad count here means "not a package", so say
    so rather than over-reading a 2 KB overlay into a 6 GB slice list.
    """
    if len(blob) < 12:
        raise ValueError("too short to be a package (%d bytes)" % len(blob))
    n = struct.unpack_from("<I", blob)[0]
    if not 0 < n < 64 or 4 + 8 * n > len(blob):
        raise ValueError("not a package: sub-resource count %d in a %d-byte blob"
                         % (n, len(blob)))
    return [struct.unpack_from("<II", blob, 4 + 8 * k) for k in range(n)]


@dataclass
class TriFlags:
    """The u32 at a HITS triangle's offset 0 — three fields, not one number.

    Every engine site reads it as `lbu +0 / lbu +1 / lhu +2`, and the packer at
    `0x09A70CBC` masks the first two to 3 and 4 bits:

        u8  surface_id;   // 0..7  -> index into the STAGE OVERLAY's property table
        u8  material;     // 0..15 -> passed to the effect spawner (footstep class)
        u16 exclude;      // 16-bit query mask: a query that shares a bit SKIPS
                          // this triangle (`0x09C3E404`, `0x09C40EA4`)

    The whole word is copied onto the actor at `+0x290` by the ground query, so
    it can be read live: `+0x290` == the flags of the triangle being stood on
    (verified byte-exact against the disk mesh).
    """
    surface_id: int
    material: int
    exclude: int

    @classmethod
    def unpack(cls, word: int):
        return cls(word & 0xFF, (word >> 8) & 0xFF, word >> 16)

    def pack(self) -> int:
        return (self.surface_id & 0xFF) | ((self.material & 0xFF) << 8) | (self.exclude << 16)


#: what each bit of a surface-property table entry does, and where the engine
#: reads it (`game_sub.ovl`). The actor offsets are on the combat entity.
SURFACE_BITS = {
    0x01: ("contact flag",  "0x09C38FEC — sets 0x0002 in the contact record"),
    0x02: ("actor flag",    "0x09C39770 — writes 1 to actor+0x27E"),
    0x10: ("query hit",     "0x09C413F4 — makes the probe 0x09C411A8 answer true"),
    0x20: ("sink",          "0x09C3F154 — surface height = hit - actor+0x2A8; "
                            "0x09C3E8E0 skips the triangle"),
    0x40: ("exclude",       "0x09C3E224 / 0x09C3FE04 — with 0x20, drops the triangle"),
    0x80: ("wade",          "0x09C3F110 — actor+0x410 |= 0x40; actor types "
                            "8/14/26/34/43/44/83 then stand on the SURFACE height"),
}


@dataclass
class HitsChunk:
    index: int
    offset: int          # into the sub-resource blob
    size: int
    grid: tuple          # (nx, nz)        -- header +0x10 / +0x14
    cell: tuple          # (cx, cz) units  -- header +0x08 / +0x0C
    origin: tuple        # (ox, oz) units  -- header +0x18 / +0x1C, (0, 0) in every ship file
    tri_offset: int      # into the sub-resource blob
    tri_count: int

    @property
    def ncells(self) -> int:
        return self.grid[0] * self.grid[1]

    def cell_of(self, x, z):
        """grid index owning a world xz, or None if it falls outside the lattice."""
        ix = int((x - self.origin[0]) // self.cell[0])
        iz = int((z - self.origin[1]) // self.cell[1])
        if not (0 <= ix < self.grid[0] and 0 <= iz < self.grid[1]):
            return None
        return ix * self.grid[1] + iz


@dataclass
class Stage:
    number: int
    blob: bytes
    subs: list

    # ---- sub-resources -------------------------------------------------
    def sub(self, k) -> bytes:
        o, s = self.subs[k]
        return self.blob[o:o + s]

    @property
    def terrain_pmo(self) -> bytes: return self.sub(0)

    @property
    def textures(self) -> bytes: return self.sub(1)

    @property
    def props_pmo(self) -> bytes: return self.sub(2)

    def environment(self) -> dict:
        """sub[3]: fog colour + the clip floats behind it."""
        b = self.sub(3)
        if len(b) < 16:
            return {}
        return {
            "kind": struct.unpack_from("<H", b)[0],
            "fog_rgba": tuple(b[2:6]),
            "floats": struct.unpack_from("<3f", b, 10),
        }

    # ---- collision -----------------------------------------------------
    def hits_chunks(self):
        b = self.sub(5)
        if len(b) < 12:
            return []
        n = struct.unpack_from("<I", b)[0]
        out = []
        for c in range(n):
            o, s = struct.unpack_from("<II", b, 4 + 8 * c)
            if o + s > len(b) or b[o:o + 4] != b"HITS":
                continue
            h = struct.unpack_from("<10I", b, o + 4)
            start = o + h[8] + 8          # offsets are relative to chunk_base + 8
            cnt = max(0, (o + s - start) // TRI_STRIDE)
            out.append(HitsChunk(c, o, s, (h[3], h[4]), (h[1], h[2]), (h[5], h[6]),
                                 start, cnt))
        return out

    def triangles(self, chunk=None):
        """yield (chunk_index, tri_index, flags, (v0,v1,v2), normal, plane_d)."""
        b = self.sub(5)
        for ch in self.hits_chunks():
            if chunk is not None and ch.index != chunk:
                continue
            for t in range(ch.tri_count):
                o = ch.tri_offset + TRI_STRIDE * t
                flags = struct.unpack_from("<I", b, o)[0]
                f = struct.unpack_from("<13f", b, o + 4)
                yield ch.index, t, flags, (f[0:3], f[3:6], f[6:9]), f[9:12], f[12]

    def tri_list(self, chunk=None):
        """[Tri] for one chunk (or the first), ready to hand back to build_hits()."""
        out = []
        for ci, _, flags, v, n, d in self.triangles(chunk):
            out.append(Tri(flags, v[0], v[1], v[2], n, d))
        return out

    def cells(self, chunk: int):
        """[[tri_index, ...]] per grid cell, in cell order — the broadphase, decoded."""
        b = self.sub(5)
        ch = [c for c in self.hits_chunks() if c.index == chunk][0]
        g = list(struct.unpack_from("<%dI" % ch.ncells, b, ch.offset + GRID_OFF))
        g.append(ch.tri_offset - ch.offset - 8)      # the array ends where triangles start
        out = []
        for i in range(ch.ncells):
            lo, hi = ch.offset + 8 + g[i], ch.offset + 8 + g[i + 1]
            words = struct.unpack_from("<%dI" % ((hi - lo) // 4), b, lo)
            assert words[-1] == TERM, "cell %d of chunk %d is unterminated" % (i, chunk)
            out.append([w // TRI_STRIDE for w in words[:-1]])
        return out

    def grid_check(self, chunk: int) -> dict:
        """Is the shipped broadphase a superset of the exact overlap set? (the oracle)

        `missing` must be zero for anything the player can stand on. The shipped files
        only ever miss triangles whose xz projection is a line (`missing_vertical`),
        which no broadphase query can hit anyway.
        """
        ch = [c for c in self.hits_chunks() if c.index == chunk][0]
        tris = self.tri_list(chunk)
        have = [set(c) for c in self.cells(chunk)]
        want = cell_map(tris, ch.grid, ch.cell, ch.origin)
        missing = [(i, t) for i in range(ch.ncells) for t in want[i] - have[i]]
        listed = set().union(*have) if have else set()
        return {
            "entries": sum(len(c) for c in have),
            "missing": len(missing),
            "missing_vertical": sum(1 for _, t in missing if tris[t].vertical()),
            "extra": sum(len(have[i] - want[i]) for i in range(ch.ncells)),
            "unlisted": ch.tri_count - len(listed),
        }

    def surface_ids(self):
        """{(chunk, surface_id): count} — which table entries this stage uses."""
        out = {}
        for ci, _, fl, _, _, _ in self.triangles():
            k = (ci, fl & 0xFF)
            out[k] = out.get(k, 0) + 1
        return out

    def tri_offset(self, chunk: int, tri: int) -> int:
        """byte offset of one triangle record, relative to the START OF THE PAC."""
        base = self.subs[5][0]
        for ch in self.hits_chunks():
            if ch.index == chunk:
                return base + ch.tri_offset + TRI_STRIDE * tri
        raise KeyError(chunk)

    def meshes(self, k=0):
        """Decoded PMO for sub[0] (terrain) or sub[2] (props), in WORLD space.

        Stage PMOs use the same **0x18-stride** mesh table as monster PMOs, so
        `mhfu_model.pmo` reads them unchanged — 487 of the 487 that ship re-encode
        byte-identically. Vertices are in the same world frame as the collision:
        under a standing hunter the visible surface and the HITS floor agree to
        0.43 units, one s16 quantization step.
        """
        from . import pmo as _pmo
        return _pmo.parse(self.sub(k))

    @property
    def slack(self) -> bytes:
        """Bytes after the last sub-resource.

        ⚠️ NOT part of the PAC — `extract_iso.py` slices on sector boundaries, so an
        extracted file carries a tail of whatever followed it on the disc (276 bytes
        of high-entropy data for st098). Carry it through and an unedited repack is
        byte-identical, which is the only honest control for the writer.
        """
        o, sz = self.subs[-1]
        return self.blob[o + sz:]

    def with_sub(self, k, blob, keep_slack=True) -> bytes:
        """A new PAC blob with sub-resource `k` replaced. Only the table moves."""
        subs = [self.sub(i) for i in range(len(self.subs))]
        subs[k] = blob
        out = build_package(subs, align=SUB_ALIGN)
        if not keep_slack:
            return out
        # build_package pads the LAST sub up to `align` too; the source does not,
        # so cut back to the end of the last sub before re-attaching the slack.
        o, sz = subresources(out)[-1]
        return out[:o + sz] + self.slack

    def with_collision(self, chunks) -> bytes:
        """A new PAC blob with sub[5] replaced by these HITS chunk blobs.

        The other sub-resources are copied byte for byte; only the offset table moves.
        Nothing outside a HITS chunk points into it, so this is safe for a disk edit —
        but a LIVE edit does not need it (see the module docstring: relink a pointer).
        """
        subs = [self.sub(k) for k in range(len(self.subs))]
        subs[5] = build_package(chunks, align=4)
        return build_package(subs, align=SUB_ALIGN)

    def verify(self) -> dict:
        """plane-equation self-check: |dot(n,v0) + d| must vanish and |n| == 1."""
        ok = bad = 0
        for _, _, _, v, n, d in self.triangles():
            nl = sum(c * c for c in n) ** 0.5
            resid = abs(sum(a * b for a, b in zip(n, v[0])) + d)
            scale = max(max(abs(c) for c in v[0]), 1.0)
            if abs(nl - 1.0) < 0.02 and resid / scale < 0.02:
                ok += 1
            else:
                bad += 1
        return {"ok": ok, "bad": bad, "frac": ok / (ok + bad) if ok + bad else 0.0}

    def bbox(self, chunk=None):
        lo = [1e30] * 3
        hi = [-1e30] * 3
        for _, _, _, v, _, _ in self.triangles(chunk):
            for p in v:
                for a in range(3):
                    lo[a] = min(lo[a], p[a]); hi[a] = max(hi[a], p[a])
        return tuple(lo), tuple(hi)

    def to_obj(self, chunk=None) -> str:
        """Wavefront OBJ of the collision mesh, one group per HITS chunk."""
        lines, vi, cur = [], 1, None
        lines.append("# MHFU st%03d collision (HITS)" % self.number)
        for ci, ti, flags, v, _, _ in self.triangles(chunk):
            if ci != cur:
                lines.append("g chunk%d" % ci); cur = ci
            for p in v:
                lines.append("v %.4f %.4f %.4f" % p)
            lines.append("f %d %d %d" % (vi, vi + 1, vi + 2))
            vi += 3
        return "\n".join(lines) + "\n"


# ---- writing: triangles, the broadphase, a whole chunk ------------------


@dataclass
class Tri:
    """One 56-byte collision record. `from_verts` derives the normal and the plane."""
    flags: int
    v0: tuple
    v1: tuple
    v2: tuple
    normal: tuple
    plane_d: float

    @classmethod
    def from_verts(cls, v0, v1, v2, flags=0):
        a = [v1[k] - v0[k] for k in range(3)]
        b = [v2[k] - v0[k] for k in range(3)]
        n = (a[1] * b[2] - a[2] * b[1],
             a[2] * b[0] - a[0] * b[2],
             a[0] * b[1] - a[1] * b[0])
        ln = sum(c * c for c in n) ** 0.5
        if ln < 1e-9:
            raise ValueError("degenerate triangle: %r %r %r" % (v0, v1, v2))
        n = tuple(c / ln for c in n)
        return cls(flags, tuple(v0), tuple(v1), tuple(v2), n,
                   -sum(n[k] * v0[k] for k in range(3)))

    def pack(self) -> bytes:
        return struct.pack("<I13f", self.flags, *self.v0, *self.v1, *self.v2,
                           *self.normal, self.plane_d)

    def xz(self):
        return (self.v0[0], self.v0[2]), (self.v1[0], self.v1[2]), (self.v2[0], self.v2[2])

    def vertical(self) -> bool:
        """xz projection is a line — the one case the shipped packer drops from cells."""
        (ax, az), (bx, bz), (cx, cz) = self.xz()
        return abs((bx - ax) * (cz - az) - (cx - ax) * (bz - az)) < 1e-3


def _tri_hits_rect(p, q, r, x0, x1, z0, z1) -> bool:
    """2D separating-axis test: triangle (p, q, r) against an axis-aligned rect."""
    if max(p[0], q[0], r[0]) < x0 or min(p[0], q[0], r[0]) > x1:
        return False
    if max(p[1], q[1], r[1]) < z0 or min(p[1], q[1], r[1]) > z1:
        return False
    corners = ((x0, z0), (x1, z0), (x1, z1), (x0, z1))
    cen = ((p[0] + q[0] + r[0]) / 3.0, (p[1] + q[1] + r[1]) / 3.0)
    for a, b in ((p, q), (q, r), (r, p)):
        nx, nz = b[1] - a[1], a[0] - b[0]
        d = [nx * (c[0] - a[0]) + nz * (c[1] - a[1]) for c in corners]
        inside = nx * (cen[0] - a[0]) + nz * (cen[1] - a[1])
        if (inside >= 0 and max(d) < 0) or (inside < 0 and min(d) > 0):
            return False
    return True


def cells_for_tri(tri: Tri, grid, cell, origin=(0, 0), pad=0.0):
    """Every grid index the triangle overlaps. Conservative: `pad` widens each cell."""
    nx, nz = grid
    cx, cz = cell
    p, q, r = tri.xz()
    lox = min(p[0], q[0], r[0]) - origin[0] - pad
    hix = max(p[0], q[0], r[0]) - origin[0] + pad
    loz = min(p[1], q[1], r[1]) - origin[1] - pad
    hiz = max(p[1], q[1], r[1]) - origin[1] + pad
    out = []
    for ix in range(max(0, int(lox // cx)), min(nx - 1, int(hix // cx)) + 1):
        for iz in range(max(0, int(loz // cz)), min(nz - 1, int(hiz // cz)) + 1):
            if _tri_hits_rect(p, q, r,
                              origin[0] + ix * cx - pad, origin[0] + (ix + 1) * cx + pad,
                              origin[1] + iz * cz - pad, origin[1] + (iz + 1) * cz + pad):
                out.append(ix * nz + iz)
    return out


def cell_map(tris, grid, cell, origin=(0, 0), pad=0.0):
    """[set(tri_index)] per cell — the broadphase a rebuild should emit."""
    out = [set() for _ in range(grid[0] * grid[1])]
    for t, tri in enumerate(tris):
        for i in cells_for_tri(tri, grid, cell, origin, pad):
            out[i].add(t)
    return out


def fit_grid(tris, cell=(501, 501), origin=(0, 0), cap=256):
    """Smallest (nx, nz) whose lattice covers every triangle, clamped to `cap`."""
    hix = max(max(v[0] for v in (t.v0, t.v1, t.v2)) for t in tris) - origin[0]
    hiz = max(max(v[2] for v in (t.v0, t.v1, t.v2)) for t in tris) - origin[1]
    return (max(1, min(cap, int(hix // cell[0]) + 1)),
            max(1, min(cap, int(hiz // cell[1]) + 1)))


def build_hits(tris, grid=None, cell=(501, 501), origin=(0, 0), pad=0.0) -> bytes:
    """Pack triangles into a complete HITS chunk: header, grid, cell lists, records.

    Layout matches the shipped files exactly — grid at +0x28, lists back to back in
    cell order, triangles last, nothing padded. Offsets are written the way the disk
    files write them, so the loader's fixup turns them into the right VAs.
    """
    tris = list(tris)
    if grid is None:
        grid = fit_grid(tris, cell, origin)
    nx, nz = grid
    lists = [sorted(c) for c in cell_map(tris, grid, cell, origin, pad)]

    grid_at = GRID_OFF - 8                       # relative to chunk_base + 8
    off = GRID_OFF + 4 * nx * nz - 8
    heads, body = [], bytearray()
    for c in lists:
        heads.append(off)
        for t in c:
            body += struct.pack("<I", t * TRI_STRIDE)
        body += struct.pack("<I", TERM)
        off += 4 * (len(c) + 1)

    out = bytearray(b"HITS")
    out += struct.pack("<9I", 0, cell[0], cell[1], nx, nz, origin[0], origin[1],
                       grid_at, off)
    out += struct.pack("<%dI" % (nx * nz), *heads)
    out += body
    for t in tris:
        out += t.pack()
    struct.pack_into("<I", out, 4, len(out))      # +0x04 is the chunk's own size
    return bytes(out)


def build_package(blobs, align=SUB_ALIGN) -> bytes:
    """`u32 count` + `(offset, size)[]` + the payloads — a PAC or a nested one."""
    head = 4 + 8 * len(blobs)
    off = (head + align - 1) // align * align if align else head
    table, body, cur = [], bytearray(), off
    for b in blobs:
        # an EMPTY sub-resource is (0, 0) in retail, not (cursor, 0) — five stages
        # ship one, and getting this wrong is the only thing that stopped a repack
        # being byte-identical.
        table.append((cur if b else 0, len(b)))
        body += b
        cur += len(b)
        pad = (-cur) % align if align else 0
        body += b"\x00" * pad
        cur += pad
    out = bytearray(struct.pack("<I", len(blobs)))
    for o, sz in table:
        out += struct.pack("<II", o, sz)
    out += b"\x00" * (off - len(out))
    out += body
    return bytes(out)


def load(data_dir, stage: int) -> Stage:
    blob = read_extracted(data_dir, pac_file(stage))
    return Stage(stage, blob, subresources(blob))


# ---- the area table -----------------------------------------------------
#
# `game_sub.ovl` (file_00075, base 0x09C19000) carries the map table the stage
# loader resolves through:
#
#   0x09CE470C   {u32 record_ptr, u32 flags}[32]   one row per MAP
#   record       u16 stage_numbers[]               [0] = base camp / entry area
#   0x09CE4848   u16 per stage, stride 4           the tag-38 file the stage pulls
#   0x089A9470   u16 pair per stage, stride 4      ambience .bd / .phd file ids (EBOOT)
#
# The row a session is in is `[*(0x09A4F060) + 644]`; the stage it is showing is
# `[*(0x09A4F060) + 648]`.
GAME_SUB_FILE = 75
GAME_SUB_BASE = 0x09C19000
MAP_TABLE_VA = 0x09CE470C
MAP_TABLE_ROWS = 32
STAGE_SND_TABLE_VA = 0x089A9470      # in the EBOOT, not game_sub


def map_table(data_dir):
    """[(row, flags, [stage, ...])] read straight out of game_sub.ovl."""
    d = read_extracted(data_dir, GAME_SUB_FILE)
    end = GAME_SUB_BASE + len(d)

    def u32(va): return struct.unpack_from("<I", d, va - GAME_SUB_BASE)[0]
    def u16(va): return struct.unpack_from("<H", d, va - GAME_SUB_BASE)[0]

    rows = [(i, u32(MAP_TABLE_VA + 8 * i), u32(MAP_TABLE_VA + 8 * i + 4))
            for i in range(MAP_TABLE_ROWS)]
    starts = sorted({p for _, p, _ in rows if GAME_SUB_BASE <= p < end})
    out = []
    for i, p, flags in rows:
        if p not in starts:
            out.append((i, flags, []))
            continue
        j = starts.index(p)
        stop = starts[j + 1] if j + 1 < len(starts) else p + 16
        stages = [u16(p + 2 * k) for k in range((stop - p) // 2)]
        out.append((i, flags, stages))
    return out
