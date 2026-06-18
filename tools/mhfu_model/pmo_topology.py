"""PMO topology-GROW encoder (Phase 5).

The in-place encoder in `pmo.py` reshapes vertices at CONSTANT topology / constant
file size (byte-identical round-trip on all 49 big-mon PACs). Adding or removing
vertices/faces needs a *rebuild* of the PMO's GE-display-list region — that lives
here, kept separate so the proven reshape path is never at risk.

What it does
------------
A monster PMO is, from header field `geBase` (h[12]) to EOF, a packed region of
per-vertex-group blocks, each = [GE display list .. RET] + [vertex buffer] +
[index buffer], all 16-byte aligned. Every vertex-group record in the table at
[h[8], h[9]) (stride 0x10) caches that block's three geBase-relative offsets:
`I3=geoff, I4=vbuf, I5=ibuf` (== the list's VADDR/IADDR args). Tables BEFORE geBase
(mesh tables, vgroup table, material table) keep their positions; only the vgroup
records' I3/I4/I5, the geBase region, and header size h[0] change. So growth is:

  1. parse every vgroup's GE list + vertex buffer + index buffer (byte level);
  2. grow a chosen group (duplicate N verts, optional position shift, append a
     triangle-list PRIM of new faces) — inherits the group's VTYPE / bone / material;
  3. re-lay-out the whole region contiguously (16-byte aligned), patch each list's
     VADDR/IADDR + each vgroup record's I3/I4/I5, bump h[0]; reassemble.

Hard limits (RE-proven): 8-bit indices cap a group at 256 verts; a NEW vgroup has
undefined bone binding (so we only grow WITHIN existing groups). See
docs/PMO_MODEL_FORMAT.md + memory phase5-geometry-anim-re.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

HEADER_OFF = 8
HEADER_FMT = "I4f2H8I"      # h[0..12] at offset 8
# header indices
H_SIZE, H_MESHCNT_A, H_MESHCNT_B = 0, 5, 6
H_MESHTAB_A, H_VGTAB, H_VGTAB_END = 7, 8, 9
H_MESHTAB_B, H_MATTAB, H_GEBASE = 10, 11, 12

ALIGN = 16


def _align(n: int, a: int = ALIGN) -> int:
    return (n + a - 1) & ~(a - 1)


# --------------------------------------------------------------------------- #
# VTYPE layout (just enough to size a vertex + find the position field)
# --------------------------------------------------------------------------- #
_COMP_SIZE = {"b": 1, "B": 1, "h": 2, "H": 2, "f": 4}


@dataclass
class VType:
    word: int
    vsize: int
    index_char: Optional[str]          # 'B'/'H'/'I' struct char, or None (non-indexed)
    pos_off: int                       # byte offset of position within a vertex
    pos_char: str                      # 'b'/'h'/'f'
    bypass: bool                       # through-mode position layout (2bB/2hH/3f)


def decode_vtype(word: int) -> VType:
    """Decode a GE VTYPE (0x12) arg into vertex size + position field location.

    Mirrors the field walk in pmo.run_ge (native struct alignment, no '<')."""
    off = 0

    def field(count: int, char: str) -> int:
        nonlocal off
        a = _COMP_SIZE[char]
        start = (off + a - 1) & ~(a - 1)
        off = start + count * _COMP_SIZE[char]
        return start

    weight = (word >> 9) & 3
    if weight:
        cnt = ((word >> 14) & 7) + 1
        field(cnt, (None, "B", "H", "f")[weight])
    bypass = bool((word >> 23) & 1)
    texture = word & 3
    if texture:
        field(2, (None, "B", "H", "f")[texture])
    color = (word >> 2) & 7
    if color:
        field(1, (None, None, None, None, "H", "H", "H", "I")[color])
    normal = (word >> 5) & 3
    if normal:
        field(3, (None, "b", "h", "f")[normal])
    position = (word >> 7) & 3
    pos_off = pos_char = None
    if position:
        if bypass:
            # 2bB / 2hH / 3f — position spans 3 components; first is the aligned start
            pchar = (None, "b", "h", "f")[position]
            pos_off = field(3, pchar) if position == 3 else _through_pos(off, position)
            pos_char = pchar
        else:
            pchar = (None, "b", "h", "f")[position]
            pos_off = field(3, pchar)
            pos_char = pchar
    if (word >> 18) & 7:
        raise ValueError("morph not supported")
    index_char = (None, "B", "H", "I")[(word >> 11) & 3]
    vsize = struct.calcsize(_vertex_struct(word))
    return VType(word, vsize, index_char, pos_off or 0, pos_char or "b", bypass)


def _through_pos(off, position):
    # through-mode 2bB/2hH: x,y are 2x(b/h), z is the unsigned (B/H). Aligned start.
    a = _COMP_SIZE[(None, "b", "h")[position]]
    return (off + a - 1) & ~(a - 1)


def _vertex_struct(word: int) -> str:
    """Reproduce pmo.run_ge's vertex_format string (for size only)."""
    fmt = ""
    weight = (word >> 9) & 3
    if weight:
        cnt = ((word >> 14) & 7) + 1
        fmt += str(cnt) + (None, "B", "H", "f")[weight]
    bypass = (word >> 23) & 1
    texture = word & 3
    if texture:
        fmt += (None, "2B", "2H", "2f")[texture]
    color = (word >> 2) & 7
    if color:
        fmt += (None, None, None, None, "H", "H", "H", "I")[color]
    normal = (word >> 5) & 3
    if normal:
        fmt += (None, "3b", "3h", "3f")[normal]
    position = (word >> 7) & 3
    if position:
        if bypass:
            fmt += (None, "2bB", "2hH", "3f")[position]
        else:
            fmt += (None, "3b", "3h", "3f")[position]
    return fmt


# --------------------------------------------------------------------------- #
# GE list + buffers per vertex group
# --------------------------------------------------------------------------- #
@dataclass
class VGroup:
    rec_index: int                     # index in the vgroup table
    rec: Tuple[int, int, int, int, int, int]   # 2BH3I record (mat,grp,backref,I3,I4,I5)
    geoff: int                         # original geBase-relative GE-list offset (I3)
    words: List[int]                   # GE command words (incl trailing RET)
    vaddr_widx: int                    # index of the VADDR (0x01) word
    iaddr_widx: Optional[int]          # index of the IADDR (0x02) word
    prims: List[Tuple[int, int, int]]  # (word_index, prim_type, index_count)
    vtype: VType
    vbuf: bytearray                    # raw vertex bytes
    indices: List[int]                 # flat index list (all PRIMs concatenated, in order)
    shared_with: Optional[int] = None  # rec_index whose block this duplicates (dedup)

    @property
    def vcount(self) -> int:
        return len(self.vbuf) // self.vtype.vsize


def _parse_ge_list(blob: bytes, ge_start: int):
    """Parse one GE list -> (words, vaddr_widx, iaddr_widx, prims, vtype, vaddr_rel,
    iaddr_rel). prims = list of (word_index, prim_type, index_count)."""
    words: List[int] = []
    vaddr_widx = iaddr_widx = None
    vaddr_rel = iaddr_rel = None
    prims: List[Tuple[int, int, int]] = []
    vtype = None
    p = ge_start
    while True:
        (cmd,) = struct.unpack_from("<I", blob, p)
        widx = len(words)
        words.append(cmd)
        op = cmd >> 24
        arg = cmd & 0xFFFFFF
        if op == 0x01:                 # VADDR
            vaddr_widx = widx
            vaddr_rel = arg
        elif op == 0x02:               # IADDR
            iaddr_widx = widx
            iaddr_rel = arg
        elif op == 0x04:               # PRIM
            prims.append((widx, (cmd >> 16) & 7, cmd & 0xFFFF))
        elif op == 0x12:               # VTYPE
            vtype = decode_vtype(arg)
        elif op == 0x0B:               # RET
            break
        p += 4
        if p - ge_start > 0x4000:
            raise ValueError("GE list runaway at 0x%X" % ge_start)
    return words, vaddr_widx, iaddr_widx, prims, vtype, vaddr_rel, iaddr_rel


def parse(blob: bytes):
    """Parse a PMO into (header_list, [VGroup]). Header is the decoded h[0..12]."""
    if blob[:8] != b"pmo\x001.0\x00":
        raise ValueError("not a monster PMO 1.0")
    header = list(struct.unpack_from(HEADER_FMT, blob, HEADER_OFF))
    vg_tab, vg_end, ge_base = header[H_VGTAB], header[H_VGTAB_END], header[H_GEBASE]
    nvg = (vg_end - vg_tab) // 0x10
    groups: List[VGroup] = []
    by_geoff = {}                      # dedup shared GE blocks
    for i in range(nvg):
        vo = vg_tab + i * 0x10
        rec = struct.unpack_from("<2BH3I", blob, vo)
        geoff, vbuf_rel, ibuf_rel = rec[3], rec[4], rec[5]
        if geoff in by_geoff:
            g = VGroup(rec_index=i, rec=rec, geoff=geoff, words=[], vaddr_widx=0,
                       iaddr_widx=None, prims=[], vtype=by_geoff[geoff].vtype,
                       vbuf=bytearray(), indices=[], shared_with=by_geoff[geoff].rec_index)
            groups.append(g)
            continue
        ge_start = ge_base + geoff
        words, va, ia, prims, vt, va_rel, ia_rel = _parse_ge_list(blob, ge_start)
        if vt is None or va is None:
            raise ValueError("vgroup %d: no VTYPE/VADDR" % i)
        # index list: PRIMs read sequentially from the index buffer
        indices: List[int] = []
        if ia is not None and vt.index_char:
            ip = ge_base + ibuf_rel
            for _, _ptype, icount in prims:
                chunk = struct.unpack_from("<%d%s" % (icount, vt.index_char), blob, ip)
                indices.extend(chunk)
                ip += icount * _COMP_SIZE[vt.index_char]
        ncount = (max(indices) + 1) if indices else 0
        vbuf = bytearray(blob[ge_base + vbuf_rel: ge_base + vbuf_rel + ncount * vt.vsize])
        g = VGroup(rec_index=i, rec=rec, geoff=geoff, words=words, vaddr_widx=va,
                   iaddr_widx=ia, prims=prims, vtype=vt, vbuf=vbuf, indices=indices)
        by_geoff[geoff] = g
        groups.append(g)
    return header, groups


# --------------------------------------------------------------------------- #
# Grow operation
# --------------------------------------------------------------------------- #
def grow_group(g: VGroup, n_new: int, shift=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0),
               src_vertex: int = 0, spread: float = 0.0):
    """Append `n_new` vertices to group g (copied from `src_vertex`, position-shifted
    by `shift` in MODEL units), and one triangle-list PRIM fanning them into
    (n_new-2) triangles. Inherits VTYPE/weights/material. 8-bit-index safe (raises
    past 256). The list's PRIM count + vbuf + indices are updated; final re-layout is
    done in serialize().

    `spread` (MODEL units, >0): scatter the new verts on a ring of this radius around
    `shift` (varying X/Z, alternating Y) so the fan triangles have AREA and are
    visible — a uniform shift makes degenerate zero-area triangles that never render.
    """
    if g.shared_with is not None:
        raise ValueError("group %d shares a GE block; grow the owner" % g.rec_index)
    if g.vtype.index_char != "B":
        raise ValueError("group %d index fmt != 8-bit (got %r); 16-bit promote TODO"
                         % (g.rec_index, g.vtype.index_char))
    if n_new < 3:
        raise ValueError("need >=3 new verts for a triangle")
    import math
    vt = g.vtype
    base = g.vcount
    if base + n_new > 256:
        raise ValueError("group %d would exceed 256 verts (8-bit index cap): %d+%d"
                         % (g.rec_index, base, n_new))
    if not g.vbuf:
        raise ValueError("group %d has no vertex buffer" % g.rec_index)
    src = bytes(g.vbuf[src_vertex * vt.vsize:(src_vertex + 1) * vt.vsize])

    def q(model_units, axis):
        if not vt.bypass and vt.pos_char in ("b", "h"):
            ptrans = {"b": 0x7f, "h": 0x7fff}[vt.pos_char]
            return int(round(model_units / scale[axis] * ptrans))
        return int(round(model_units))

    lim = {"b": (-128, 127), "h": (-32768, 32767), "f": (None, None)}[vt.pos_char]
    sx, sy, sz = shift
    for k in range(n_new):
        v = bytearray(src)
        if spread > 0.0:
            ang = (2.0 * math.pi * k) / max(1, n_new)
            ox = sx + spread * math.cos(ang)
            oz = sz + spread * math.sin(ang)
            oy = sy + (spread * 0.5 if (k & 1) else -spread * 0.5)
        else:
            ox, oy, oz = sx, sy, sz
        dq = (q(ox, 0), q(oy, 1), q(oz, 2))
        if vt.pos_char in ("b", "h", "f"):
            cur = list(struct.unpack_from("<3%s" % vt.pos_char, v, vt.pos_off))
            nv = []
            for c, d in zip(cur, dq):
                x = c + d
                if lim[0] is not None:
                    x = max(lim[0], min(lim[1], x))
                nv.append(x)
            struct.pack_into("<3%s" % vt.pos_char, v, vt.pos_off, *nv)
        g.vbuf.extend(v)
    # triangle fan over the new verts -> (n_new-2) triangles
    new_idx = list(range(base, base + n_new))
    tri = []
    for k in range(1, n_new - 1):
        tri += [new_idx[0], new_idx[k], new_idx[k + 1]]
    g.indices.extend(tri)
    # add a PRIM word (triangle list, prim_type=3) right before the RET
    ret_widx = len(g.words) - 1
    prim_word = 0x04000000 | (3 << 16) | (len(tri) & 0xFFFF)
    g.words.insert(ret_widx, prim_word)
    g.prims.append((ret_widx, 3, len(tri)))


# --------------------------------------------------------------------------- #
# Serialize
# --------------------------------------------------------------------------- #
def serialize(blob: bytes, header: List[int], groups: List[VGroup]) -> bytes:
    """Rebuild the geBase region from `groups`, patch the vgroup table + header size,
    keep everything before geBase byte-identical. Returns the new PMO bytes."""
    ge_base = header[H_GEBASE]
    vg_tab = header[H_VGTAB]
    out = bytearray(blob[:ge_base])     # header + all tables (unchanged so far)

    region = bytearray()
    placed = {}                         # geoff(orig) -> (new_geoff, new_vbuf, new_ibuf)
    # lay out owner groups first; shared groups reuse the owner's offsets
    for g in groups:
        if g.shared_with is not None:
            continue
        # block = GE list | vbuf | ibuf, each 16-byte aligned, geBase-relative
        geoff = _align(len(region))
        while len(region) < geoff:
            region.append(0)
        # placeholder for words; patch VADDR/IADDR after we know vbuf/ibuf offsets
        words = list(g.words)
        ge_bytes_len = len(words) * 4
        vbuf_off = _align(geoff + ge_bytes_len)
        ibuf_off = _align(vbuf_off + len(g.vbuf))
        # patch VADDR/IADDR args (relative to geoff)
        words[g.vaddr_widx] = 0x01000000 | ((vbuf_off - geoff) & 0xFFFFFF)
        if g.iaddr_widx is not None:
            words[g.iaddr_widx] = 0x02000000 | ((ibuf_off - geoff) & 0xFFFFFF)
        # emit GE list
        for w in words:
            region += struct.pack("<I", w)
        while len(region) < vbuf_off:
            region.append(0)
        region += g.vbuf
        while len(region) < ibuf_off:
            region.append(0)
        if g.vtype.index_char:
            region += struct.pack("<%d%s" % (len(g.indices), g.vtype.index_char),
                                   *g.indices)
        placed[g.geoff] = (geoff, vbuf_off, ibuf_off)

    # patch vgroup table records I3/I4/I5
    for g in groups:
        src = g.geoff if g.shared_with is None else g.geoff
        geoff, vbuf_off, ibuf_off = placed[src]
        vo = vg_tab + g.rec_index * 0x10
        rec = list(g.rec)
        rec[3], rec[4], rec[5] = geoff, vbuf_off, ibuf_off
        struct.pack_into("<2BH3I", out, vo, *rec)

    out += region
    # header size h[0]
    new_size = len(out)
    struct.pack_into("<I", out, HEADER_OFF + 0, new_size)
    return bytes(out)


def roundtrip_region(blob: bytes) -> bytes:
    """Parse + serialize with NO edits — should yield a VALID PMO that re-parses to
    the same vgroup geometry (NOT byte-identical: layout is rebuilt). Used by tests."""
    header, groups = parse(blob)
    return serialize(blob, header, groups)


def grow_pmo(blob: bytes, group: Optional[int], n_new: int,
             shift=(0.0, 200.0, 0.0), spread: float = 0.0) -> bytes:
    """Grow ONE owner vgroup (the first 8-bit-indexed one if `group` is None) by
    `n_new` verts / (n_new-2) faces and return the new (bigger) PMO bytes."""
    header, groups = parse(blob)
    if group is None:
        g = next(x for x in groups if x.shared_with is None and x.vtype.index_char == "B")
    else:
        g = groups[group]
    grow_group(g, n_new, shift=shift, scale=header[2:5], spread=spread)
    return serialize(blob, header, groups)


def grow_pac(pac_in: bytes, group: Optional[int], n_new: int,
             shift=(0.0, 200.0, 0.0), spread: float = 0.0) -> bytes:
    """Grow the PMO sub-resource of a monster PAC and re-flow the container."""
    from . import pac as _pac
    P = _pac.MonsterPac.from_bytes(pac_in)
    sub = next((s for s in P.subs if s.magic == b"pmo\x00"), None)
    if sub is None:
        raise ValueError("no PMO sub-resource in this PAC")
    grown = grow_pmo(sub.data, group, n_new, shift, spread)
    P.subs[sub.index] = _pac.SubResource(sub.index, grown)
    return P.to_bytes()


def _main(argv):
    import argparse
    ap = argparse.ArgumentParser(description="Grow a monster PAC's geometry (Phase 5).")
    ap.add_argument("pac_in")
    ap.add_argument("pac_out")
    ap.add_argument("-g", "--group", type=int, default=None,
                    help="vgroup record index (default: first 8-bit-indexed owner)")
    ap.add_argument("-n", "--verts", type=int, default=6,
                    help="number of vertices to add (>=3; makes n-2 triangles)")
    ap.add_argument("--shift", default="0,200,0",
                    help="model-unit position shift for the new verts (x,y,z)")
    ap.add_argument("--spread", type=float, default=0.0,
                    help="model-unit ring radius to scatter new verts (>0 = visible, "
                         "non-degenerate triangles)")
    a = ap.parse_args(argv)
    shift = tuple(float(x) for x in a.shift.split(","))
    data = open(a.pac_in, "rb").read()
    out = grow_pac(data, a.group, a.verts, shift, a.spread)
    open(a.pac_out, "wb").write(out)
    print("wrote %s (%d -> %d bytes, +%d)" % (a.pac_out, len(data), len(out),
                                              len(out) - len(data)))


if __name__ == "__main__":
    import sys
    _main(sys.argv[1:])
