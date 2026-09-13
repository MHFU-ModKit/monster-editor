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
    pos_trans: float = 1.0             # model-unit -> stored quantum (0x7f / 0x7fff / 1)
    # texture (UV), normal and weight field locations — captured so grown verts get
    # real values (not a copy of vertex0). None when the field is absent in this VTYPE.
    uv_off: Optional[int] = None
    uv_char: Optional[str] = None      # 'B'/'H'/'f'
    uv_trans: float = 1.0
    nrm_off: Optional[int] = None
    nrm_char: Optional[str] = None     # 'b'/'h'/'f'
    nrm_trans: float = 1.0
    wt_off: Optional[int] = None
    wt_char: Optional[str] = None      # 'B'/'H'/'f'
    wt_count: int = 0                  # number of weight (bone-palette) components
    wt_trans: float = 1.0


def decode_vtype(word: int) -> VType:
    """Decode a GE VTYPE (0x12) arg into vertex size + the field locations the grow
    path writes (position, UV, normal, weights).

    Mirrors the field walk + quantization scalars in pmo.run_ge (native struct
    alignment, no '<'). Field order in a vertex: weight, texture(UV), color, normal,
    position."""
    off = 0

    def field(count: int, char: str) -> int:
        nonlocal off
        a = _COMP_SIZE[char]
        start = (off + a - 1) & ~(a - 1)
        off = start + count * _COMP_SIZE[char]
        return start

    bypass = bool((word >> 23) & 1)
    weight = (word >> 9) & 3
    wt_off = wt_char = None
    wt_count = 0
    wt_trans = 1.0
    if weight:
        wt_count = ((word >> 14) & 7) + 1
        wt_char = (None, "B", "H", "f")[weight]
        wt_trans = (None, 0x80, 0x8000, 1)[weight]
        wt_off = field(wt_count, wt_char)
    texture = word & 3
    uv_off = uv_char = None
    uv_trans = 1.0
    if texture:
        uv_char = (None, "B", "H", "f")[texture]
        uv_trans = 1 if bypass else (None, 0x80, 0x8000, 1)[texture]
        uv_off = field(2, uv_char)
    color = (word >> 2) & 7
    if color:
        field(1, (None, None, None, None, "H", "H", "H", "I")[color])
    normal = (word >> 5) & 3
    nrm_off = nrm_char = None
    nrm_trans = 1.0
    if normal:
        nrm_char = (None, "b", "h", "f")[normal]
        nrm_trans = 1 if bypass else (None, 0x7f, 0x7fff, 1)[normal]
        nrm_off = field(3, nrm_char)
    position = (word >> 7) & 3
    pos_off = pos_char = None
    pos_trans = 1.0
    if position:
        if bypass:
            # 2bB / 2hH / 3f — position spans 3 components; first is the aligned start
            pchar = (None, "b", "h", "f")[position]
            pos_off = field(3, pchar) if position == 3 else _through_pos(off, position)
            pos_char = pchar
            pos_trans = 1
        else:
            pchar = (None, "b", "h", "f")[position]
            pos_off = field(3, pchar)
            pos_char = pchar
            pos_trans = (None, 0x7f, 0x7fff, 1)[position]
    if (word >> 18) & 7:
        raise ValueError("morph not supported")
    index_char = (None, "B", "H", "I")[(word >> 11) & 3]
    vsize = struct.calcsize(_vertex_struct(word))
    return VType(word, vsize, index_char, pos_off or 0, pos_char or "b", bypass,
                 pos_trans=pos_trans,
                 uv_off=uv_off, uv_char=uv_char, uv_trans=uv_trans,
                 nrm_off=nrm_off, nrm_char=nrm_char, nrm_trans=nrm_trans,
                 wt_off=wt_off, wt_char=wt_char, wt_count=wt_count, wt_trans=wt_trans)


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
    vtype_widx: Optional[int]          # index of the VTYPE (0x12) word (for promote)
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
    vaddr_widx = iaddr_widx = vtype_widx = None
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
            vtype_widx = widx
        elif op == 0x0B:               # RET
            break
        p += 4
        if p - ge_start > 0x4000:
            raise ValueError("GE list runaway at 0x%X" % ge_start)
    return words, vaddr_widx, iaddr_widx, vtype_widx, prims, vtype, vaddr_rel, iaddr_rel


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
                       iaddr_widx=None, vtype_widx=None, prims=[],
                       vtype=by_geoff[geoff].vtype,
                       vbuf=bytearray(), indices=[], shared_with=by_geoff[geoff].rec_index)
            groups.append(g)
            continue
        ge_start = ge_base + geoff
        words, va, ia, vtw, prims, vt, va_rel, ia_rel = _parse_ge_list(blob, ge_start)
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
                   iaddr_widx=ia, vtype_widx=vtw, prims=prims, vtype=vt, vbuf=vbuf,
                   indices=indices)
        by_geoff[geoff] = g
        groups.append(g)
    return header, groups


# --------------------------------------------------------------------------- #
# Grow operation
# --------------------------------------------------------------------------- #
_LIM = {"b": (-128, 127), "B": (0, 255), "h": (-32768, 32767),
        "H": (0, 65535), "f": (None, None)}


def _pack_comps(v: bytearray, off: int, char: str, values):
    """Quantize-clamp `values` (already in stored units) and pack at `off`."""
    if char == "f":
        struct.pack_into("<%df" % len(values), v, off, *(float(x) for x in values))
        return
    lo, hi = _LIM[char]
    raw = [max(lo, min(hi, int(round(x)))) for x in values]
    struct.pack_into("<%d%s" % (len(values), char), v, off, *raw)


def promote_to_16bit(g: VGroup):
    """Promote a group from 8-bit (`B`) to 16-bit (`H`) vertex indices so it can hold
    >256 verts. Patches the VTYPE word's index field (bits 11-12: 1->2); the vertex
    struct size is unchanged (index format is not part of a vertex). Shared groups
    reuse the owner's GE words + vtype object, so this propagates to them."""
    if g.vtype.index_char == "H":
        return
    if g.vtype.index_char != "B":
        raise ValueError("group %d index fmt %r: only B->H promote supported"
                         % (g.rec_index, g.vtype.index_char))
    if g.vtype_widx is None:
        raise ValueError("group %d has no own VTYPE word (shared?); promote the owner"
                         % g.rec_index)
    w = g.words[g.vtype_widx]
    w = (w & ~(3 << 11)) | (2 << 11)
    g.words[g.vtype_widx] = w
    g.vtype.word = w
    g.vtype.index_char = "H"


def grow_group(g: VGroup, n_new: int, shift=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0),
               src_vertex: int = 0, spread: float = 0.0, weight_slot: int = 0,
               force_16bit: bool = False):
    """Append `n_new` vertices to group g and one triangle-list PRIM fanning them into
    (n_new-2) triangles. New verts inherit the group's VTYPE/material but get REAL
    per-vertex attributes (not a copy of vertex0):

      * position: `src_vertex`'s position + (`shift` + a `spread`-radius ring), MODEL
        units — `spread`>0 gives the fan triangles AREA (a uniform shift makes
        degenerate, invisible triangles);
      * UV: a circular patch in texture space (so the surface samples real texels, not
        one — fixes the dark/untextured look);
      * normal: the outward ring direction (so the dome catches light correctly);
      * bone weights: a CLEAN bind — 1.0 on palette slot `weight_slot` (the group's
        primary bone by default), 0 elsewhere — instead of vertex0's arbitrary blend.

    Index width: 8-bit groups auto-promote to 16-bit past 256 verts (or when
    `force_16bit`); cap is then 65536. Final re-layout is done in serialize()."""
    if g.shared_with is not None:
        raise ValueError("group %d shares a GE block; grow the owner" % g.rec_index)
    if g.vtype.index_char not in ("B", "H"):
        raise ValueError("group %d index fmt %r unsupported" % (g.rec_index,
                                                                g.vtype.index_char))
    if n_new < 3:
        raise ValueError("need >=3 new verts for a triangle")
    if not g.vbuf:
        raise ValueError("group %d has no vertex buffer" % g.rec_index)
    import math
    base = g.vcount
    # index-width / cap handling
    if force_16bit:
        promote_to_16bit(g)
    if base + n_new > 256 and g.vtype.index_char == "B":
        promote_to_16bit(g)
    cap = 256 if g.vtype.index_char == "B" else 65536
    if base + n_new > cap:
        raise ValueError("group %d would exceed %d verts (index cap): %d+%d"
                         % (g.rec_index, cap, base, n_new))
    vt = g.vtype
    src = bytes(g.vbuf[src_vertex * vt.vsize:(src_vertex + 1) * vt.vsize])
    sx, sy, sz = shift

    def qpos(model_units, axis):
        # store = model_units / scale * pos_trans (matches pmo.run_ge inverse)
        if not vt.bypass and vt.pos_char in ("b", "h"):
            return model_units / scale[axis] * vt.pos_trans
        return model_units

    for k in range(n_new):
        v = bytearray(src)
        if spread > 0.0:
            ang = (2.0 * math.pi * k) / max(1, n_new)
            dx, dy, dz = math.cos(ang), (0.5 if (k & 1) else -0.5), math.sin(ang)
            ox, oy, oz = sx + spread * dx, sy + spread * dy, sz + spread * dz
        else:
            dx, dy, dz = 0.0, 0.0, 0.0
            ox, oy, oz = sx, sy, sz
        # position (additive to src vertex)
        if vt.pos_char in ("b", "h", "f"):
            cur = list(struct.unpack_from("<3%s" % vt.pos_char, v, vt.pos_off))
            dq = (qpos(ox, 0), qpos(oy, 1), qpos(oz, 2))
            _pack_comps(v, vt.pos_off, vt.pos_char, [c + d for c, d in zip(cur, dq)])
        # normal = outward ring direction (absolute)
        if vt.nrm_off is not None:
            nl = math.sqrt(dx * dx + dy * dy + dz * dz) or 1.0
            nrm = [dx / nl * vt.nrm_trans, dy / nl * vt.nrm_trans, dz / nl * vt.nrm_trans]
            _pack_comps(v, vt.nrm_off, vt.nrm_char, nrm)
        # UV = circular patch in texture space (absolute)
        if vt.uv_off is not None:
            u = (0.5 + 0.5 * math.cos((2.0 * math.pi * k) / max(1, n_new)))
            wv = (0.5 + 0.5 * math.sin((2.0 * math.pi * k) / max(1, n_new)))
            _pack_comps(v, vt.uv_off, vt.uv_char, [u * vt.uv_trans, wv * vt.uv_trans])
        # weights = clean single-bone bind on slot `weight_slot`
        if vt.wt_off is not None and vt.wt_count:
            slot = max(0, min(vt.wt_count - 1, weight_slot))
            ws = [0.0] * vt.wt_count
            ws[slot] = 1.0 * vt.wt_trans
            _pack_comps(v, vt.wt_off, vt.wt_char, ws)
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


def _prim_slots(g: VGroup):
    """Word indices of the PRIM commands in a GE display list."""
    return [i for i, w in enumerate(g.words) if (w >> 24) == 0x04]


def _insert_prim_after_the_last_one(g: VGroup, prim_word: int) -> int:
    """Put a new PRIM where PRIMs already are — never at the end of the list.

    🔴 The order of a GE list is not decoration. These lists open with ORIGIN_ADDR
    (0x14), which latches the list's own address so that VADDR/IADDR can be written
    relative to it, and most of them CLOSE with OFFSETADDR 0 (0x13), which throws that
    origin away again. Append a PRIM before the RET and it lands on the far side of
    that reset: the vertex pointer resolves to a raw offset near zero, the GPU draws
    nothing at all, and the edit reads as "my geometry never appeared" with a PMO that
    decodes perfectly. Measured in the snowy base camp — 168 triangles, correct
    positions, invisible.
    """
    slots = _prim_slots(g)
    at = (slots[-1] + 1) if slots else max(0, len(g.words) - 1)
    g.words.insert(at, prim_word)
    return at


def grow_group_explicit(g: VGroup, verts, tris, scale=(1.0, 1.0, 1.0),
                        weight_slot: int = 0, force_16bit: bool = False,
                        src_indices=None):
    """Append author-supplied vertices + triangles to group g (the Blender path).

    `verts`  : list of dicts {x,y,z[, i,j,k normal][, u,v]} in ENGINE units / unit
               normals / [0,1] UV (== the importer's decoded fields, inverse-converted
               by the exporter).
    `tris`   : list of (a,b,c) triangle vertex indices in the group's FULL index space
               (existing verts keep indices [0,vcount); new verts get [vcount, ...) in
               `verts` order — so a Blender face may reference old AND new verts).

    `src_indices` : optional per-new-vertex index of an EXISTING vertex to inherit the
               non-authored fields from (vertex colour, above all). Defaults to
               vertex 0 for every new vertex. Stage PMOs carry no normals — their
               VTYPE is (uv16, colour16, pos16, no normal, no weight) — so vertex
               colour *is* the lighting, and inheriting it from the nearest
               neighbour instead of from vertex0 is what keeps a new face from
               reading as a flat bright patch.

    Unlike grow_group (synthetic ring), positions/normals/UVs are written ABSOLUTE
    from `verts`. Auto-promotes to 16-bit indices past 256 verts. Adds ONE
    triangle-list PRIM for the new faces."""
    if g.shared_with is not None:
        raise ValueError("group %d shares a GE block; grow the owner" % g.rec_index)
    if g.vtype.index_char not in ("B", "H"):
        raise ValueError("group %d index fmt %r unsupported" % (g.rec_index,
                                                                g.vtype.index_char))
    if not g.vbuf:
        raise ValueError("group %d has no vertex buffer" % g.rec_index)
    n_new = len(verts)
    if n_new < 1 or not tris:
        raise ValueError("need >=1 new vert and >=1 triangle")
    vt = g.vtype
    base = g.vcount
    total = base + n_new
    if force_16bit:
        promote_to_16bit(g)
    if total > 256 and g.vtype.index_char == "B":
        promote_to_16bit(g)
    cap = 256 if g.vtype.index_char == "B" else 65536
    if total > cap:
        raise ValueError("group %d would exceed %d verts (index cap): %d+%d"
                         % (g.rec_index, cap, base, n_new))
    maxidx = max(i for t in tris for i in t)
    if maxidx >= total:
        raise ValueError("triangle index %d out of range (have %d verts)"
                         % (maxidx, total))
    def _src(k):
        i = 0 if src_indices is None else max(0, min(base - 1, src_indices[k]))
        return bytes(g.vbuf[i * vt.vsize:(i + 1) * vt.vsize])

    def qpos(engine_units, axis):
        if not vt.bypass and vt.pos_char in ("b", "h"):
            return engine_units / scale[axis] * vt.pos_trans
        return engine_units

    for k, vd in enumerate(verts):
        v = bytearray(_src(k))
        if vt.pos_char in ("b", "h", "f"):
            _pack_comps(v, vt.pos_off, vt.pos_char,
                        [qpos(vd["x"], 0), qpos(vd["y"], 1), qpos(vd["z"], 2)])
        if vt.nrm_off is not None and "i" in vd:
            import math
            nx, ny, nz = vd["i"], vd["j"], vd["k"]
            nl = math.sqrt(nx * nx + ny * ny + nz * nz) or 1.0
            _pack_comps(v, vt.nrm_off, vt.nrm_char,
                        [nx / nl * vt.nrm_trans, ny / nl * vt.nrm_trans,
                         nz / nl * vt.nrm_trans])
        if vt.uv_off is not None and "u" in vd:
            _pack_comps(v, vt.uv_off, vt.uv_char,
                        [vd["u"] * vt.uv_trans, vd["v"] * vt.uv_trans])
        if vt.wt_off is not None and vt.wt_count:
            slot = max(0, min(vt.wt_count - 1, weight_slot))
            ws = [0.0] * vt.wt_count
            ws[slot] = 1.0 * vt.wt_trans
            _pack_comps(v, vt.wt_off, vt.wt_char, ws)
        g.vbuf.extend(v)
    flat = []
    for a, b, c in tris:
        flat += [a, b, c]
    g.indices.extend(flat)
    prim_word = 0x04000000 | (3 << 16) | (len(flat) & 0xFFFF)
    at = _insert_prim_after_the_last_one(g, prim_word)
    g.prims.append((at, 3, len(flat)))


def _pad_words_to(g: VGroup, n_words: int) -> None:
    """Pad a GE display list back to `n_words` with NOPs, inserted before the RET.

    Opcode 0x00 is NOP, so the padding draws nothing — but it keeps `len(g.words)`
    constant, and that is the whole point: every downstream offset in the block
    (`vbuf_off = geoff + align(len(words)*4)`, then `ibuf_off`) is derived from the
    word count, so a GE list that shrinks silently moves the vertex and index buffers
    and rewrites the vgroup record's I4/I5. In a file that is fine. In a RESIDENT PAC
    it is the documented instant-garbage failure (docs/STAGE_MAP_FORMAT.md §4b).
    """
    if len(g.words) > n_words:
        raise ValueError("group %d GE list grew to %d words, cannot pad back to %d"
                         % (g.rec_index, len(g.words), n_words))
    at = len(g.words) - 1               # before the RET
    for _ in range(n_words - len(g.words)):
        g.words.insert(at, 0x00000000)


def overwrite_group(g: VGroup, verts, tris, scale=(1.0, 1.0, 1.0),
                    weight_slot: int = 0, src_indices=None) -> None:
    """REPLACE what a vgroup draws, reusing the buffers it already has, IN PLACE.

    The resident-safe sibling of `clear_group` + `grow_group_explicit`. Those two
    change the group's GE-list length and its vertex count, which moves the vertex and
    index buffers inside the block and therefore rewrites I4/I5 in the vgroup table —
    and the engine reads that table while it draws, so the frame corrupts the instant
    the write lands. Here nothing about the layout changes at all:

      * the new vertices are written OVER the first `len(verts)` slots of the existing
        vertex buffer, which keeps its size;
      * the new indices go at the existing index buffer's offset and must fit in it;
      * the GE list is padded back to its original word count with NOPs.

    So `_block_size(g)` is unchanged, `serialize_inplace` reports nothing moved, and
    the vgroup table comes out byte-identical — the edit is only in the geBase region,
    which is load-time. That is the one shape of mesh edit that can sit in a resident
    PAC without being visible as corruption before the area reloads.

    `verts`/`tris`/`src_indices` are as `grow_group_explicit`, except that the triangle
    indices are relative to the NEW vertex list (0..len(verts)-1) and `src_indices`
    defaults to slot k for vertex k — i.e. each new vertex inherits the colour and UV
    of the vertex it overwrites.
    """
    if g.shared_with is not None:
        raise ValueError("group %d shares a GE block; overwrite the owner" % g.rec_index)
    if g.vtype.index_char not in ("B", "H"):
        raise ValueError("group %d index fmt %r unsupported" % (g.rec_index,
                                                                g.vtype.index_char))
    if not g.vbuf:
        raise ValueError("group %d has no vertex buffer" % g.rec_index)
    vt = g.vtype
    n_words = len(g.words)
    have_bytes = len(g.indices) * _COMP_SIZE[vt.index_char]
    if len(verts) > g.vcount:
        raise ValueError("group %d holds %d vertices, cannot take %d without growing "
                         "the buffer" % (g.rec_index, g.vcount, len(verts)))
    need_bytes = len(tris) * 3 * _COMP_SIZE[vt.index_char]
    if need_bytes > have_bytes:
        raise ValueError("group %d index buffer is %d bytes, %d triangles need %d"
                         % (g.rec_index, have_bytes, len(tris), need_bytes))
    maxidx = max(i for t in tris for i in t)
    if maxidx >= len(verts):
        raise ValueError("triangle index %d out of range (%d vertices)"
                         % (maxidx, len(verts)))

    def qpos(engine_units, axis):
        if not vt.bypass and vt.pos_char in ("b", "h"):
            return engine_units / scale[axis] * vt.pos_trans
        return engine_units

    for k, vd in enumerate(verts):
        slot = k if src_indices is None else max(0, min(g.vcount - 1, src_indices[k]))
        v = bytearray(g.vbuf[slot * vt.vsize:(slot + 1) * vt.vsize])
        if vt.pos_char in ("b", "h", "f"):
            _pack_comps(v, vt.pos_off, vt.pos_char,
                        [qpos(vd["x"], 0), qpos(vd["y"], 1), qpos(vd["z"], 2)])
        if vt.uv_off is not None and "u" in vd:
            _pack_comps(v, vt.uv_off, vt.uv_char,
                        [vd["u"] * vt.uv_trans, vd["v"] * vt.uv_trans])
        if vt.wt_off is not None and vt.wt_count:
            ws = [0.0] * vt.wt_count
            ws[max(0, min(vt.wt_count - 1, weight_slot))] = 1.0 * vt.wt_trans
            _pack_comps(v, vt.wt_off, vt.wt_char, ws)
        g.vbuf[k * vt.vsize:(k + 1) * vt.vsize] = v

    g.indices = [i for t in tris for i in t]
    slots = _prim_slots(g)
    if not slots:
        raise ValueError("group %d has no PRIM to replace" % g.rec_index)
    # Overwrite the FIRST PRIM and NOP the others, rather than stripping and
    # re-appending: every other command keeps its position, so the ORIGIN_ADDR /
    # OFFSETADDR bracket still encloses the draw (see _insert_prim_after_the_last_one),
    # and the word count is unchanged for free.
    g.words[slots[0]] = 0x04000000 | (3 << 16) | (len(g.indices) & 0xFFFF)
    for i in slots[1:]:
        g.words[i] = 0x00000000
    g.prims = [(slots[0], 3, len(g.indices))]
    _pad_words_to(g, n_words)


def set_vertex_positions(g: VGroup, items, scale=(1.0, 1.0, 1.0)) -> None:
    """Move vertices that are already there. `items` = [(index, (x, y, z)), ...] in
    ENGINE units.

    The one stage edit that is resident-safe for free: it writes over the position
    field of existing vertices, so no buffer changes length, no offset moves and the
    vgroup table is untouched. It is also the only way to move an object in a stage
    mesh — a stage PMO has no object boundaries below the vertex group, so "the crate"
    is whatever set of vertices you select, and a selection that cuts through a face
    stretches it instead of moving it. Pick the selection so that no face straddles
    the edge (see `stage_tool.py edit`, which reports exactly that count).
    """
    vt = g.vtype
    if vt.pos_char not in ("b", "h", "f"):
        raise ValueError("group %d position format %r unsupported"
                         % (g.rec_index, vt.pos_char))

    def qpos(engine_units, axis):
        if not vt.bypass and vt.pos_char in ("b", "h"):
            return engine_units / scale[axis] * vt.pos_trans
        return engine_units

    for i, (x, y, z) in items:
        if not 0 <= i < g.vcount:
            raise ValueError("vertex %d out of range (group %d has %d)"
                             % (i, g.rec_index, g.vcount))
        v = bytearray(g.vbuf[i * vt.vsize:(i + 1) * vt.vsize])
        _pack_comps(v, vt.pos_off, vt.pos_char,
                    [qpos(x, 0), qpos(y, 1), qpos(z, 2)])
        g.vbuf[i * vt.vsize:(i + 1) * vt.vsize] = v


def sculpt_group(g: VGroup, verts, tris, scale=(1.0, 1.0, 1.0),
                 first: int = 0, collapse: bool = True) -> dict:
    """Re-shape a vertex group into arbitrary geometry WITHOUT touching a PRIM word.

    The safest edit there is, and the only one measured to actually show up in a
    running quest area. `overwrite_group` rewrites the group's PRIM to one triangle
    list; that decodes correctly, survives every offline check, and draws NOTHING in
    game — whatever the engine builds at load, a PRIM this side of it is not enough.
    Vertex data, on the other hand, is honoured: moving a group's vertices moves what
    is on screen (the base camp's supply box, 2026-09-13).

    So this keeps every PRIM exactly as it is and only moves vertices. Each existing
    primitive is handed ONE triangle from `tris`: its first three vertex slots get the
    triangle's corners and any remaining slot is collapsed onto the last corner, which
    makes the rest of that strip degenerate and therefore invisible. Primitives past
    the end of `tris` collapse to a point.

    The budget is one triangle per PRIM, not per index — g9 of st098 has 158 four-vertex
    strips, so 158 triangles. Returns {"prims", "used", "spare"}.

    `first` starts at that PRIM instead of 0, and `collapse=False` leaves every PRIM the
    shape did not need exactly as it was. Together they carve the new object out of a
    CORNER of a big group's triangle budget and leave the rest of it standing — which is
    how you get an opaque-textured object without flattening the scenery that supplies
    the texture.
    """
    vt = g.vtype
    if not g.prims:
        raise ValueError("group %d draws nothing to sculpt" % g.rec_index)

    def qpos(engine_units, axis):
        if not vt.bypass and vt.pos_char in ("b", "h"):
            return engine_units / scale[axis] * vt.pos_trans
        return engine_units

    def put(slot, xyz):
        v = bytearray(g.vbuf[slot * vt.vsize:(slot + 1) * vt.vsize])
        _pack_comps(v, vt.pos_off, vt.pos_char,
                    [qpos(xyz[0], 0), qpos(xyz[1], 1), qpos(xyz[2], 2)])
        g.vbuf[slot * vt.vsize:(slot + 1) * vt.vsize] = v

    at = sum(c for _w, _t, c in g.prims[:first])
    used = 0
    for _widx, _ptype, count in g.prims[first:]:
        slots = g.indices[at:at + count]
        at += count
        if len(slots) < 3:
            continue
        if used >= len(tris) and not collapse:
            break
        if used < len(tris):
            tri = [verts[i] for i in tris[used]]
            pts = [(v["x"], v["y"], v["z"]) for v in tri]
            used += 1
        else:
            pts = [(verts[0]["x"], verts[0]["y"], verts[0]["z"])] * 3
        for k, slot in enumerate(slots):
            put(slot, pts[k] if k < 3 else pts[2])
    return {"prims": len(g.prims) - first, "used": used,
            "spare": len(g.prims) - first - used}


def clear_group(g: VGroup, drop_vertices: bool = True,
                keep_layout: bool = False) -> int:
    """DELETE everything a vertex group draws: drop its PRIM words and its whole index
    list. Returns the index-buffer bytes freed.

    This is the one topology delete that is safe on a stage without re-encoding
    anything. Removing SOME faces from a triangle strip would mean splitting the strip
    (or re-emitting the group as a triangle list, which costs x1.74 on the index buffer
    and can make a group BIGGER); dropping the group whole costs nothing and frees
    space. The vertex buffer stays — nothing reads it with no PRIM left, and keeping it
    means `VADDR` still points somewhere valid.

    `keep_layout=True` pads the GE list back to its original word count with NOPs and
    frees NOTHING. That is the right call for a RESIDENT PAC: a shrinking GE list moves
    the vertex/index buffers and rewrites the vgroup record, which corrupts the frame on
    the spot. See `overwrite_group`.

    The freed bytes are what pays for an `grow_group_explicit` elsewhere when the PMO
    has to come out at its ORIGINAL SIZE — which it does whenever it is going to be
    injected into a resident PAC rather than written to a file.
    """
    if g.shared_with is not None:
        raise ValueError("group %d shares a GE block; clear the owner" % g.rec_index)
    freed = (len(g.indices) * _COMP_SIZE[g.vtype.index_char or "H"]
             + (len(g.vbuf) if drop_vertices else 0))
    g.indices = []
    if keep_layout:
        # Free nothing on purpose: NOP the PRIMs where they stand. The block keeps its
        # exact shape, the vgroup record does not move, and no other command changes
        # position. That is what a resident PAC needs.
        for i in _prim_slots(g):
            g.words[i] = 0x00000000
        g.prims = []
        return 0
    g.words = [w for w in g.words if (w >> 24) != 0x04]
    g.prims = []
    if drop_vertices:
        g.vbuf = bytearray()
    return freed


# --------------------------------------------------------------------------- #
# Serialize
# --------------------------------------------------------------------------- #
def serialize(blob: bytes, header: List[int], groups: List[VGroup],
              pad_to: Optional[int] = None) -> bytes:
    """Rebuild the geBase region from `groups`, patch the vgroup table + header size,
    keep everything before geBase byte-identical. Returns the new PMO bytes.

    `pad_to` zero-pads the result to exactly that many bytes and reports the padded
    length in h[0] (retail already pads to 16 the same way). Use it to hold a PMO at
    its ORIGINAL size after an edit that changed the length — the only way an edited
    mesh can be written over a RESIDENT PAC, where growing sub[0] would run into
    sub[1]. Raises if the edit does not fit."""
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
    # Pad to the 16-byte boundary and count the pad in h[0]. Retail does this on every
    # shipped PMO: without it the region ends 2..14 bytes short and the round trip is
    # off by exactly that tail — which is the whole difference between "re-encodes" and
    # "re-encodes byte for byte". h[0] is the padded size, not the used size.
    while len(out) % ALIGN:
        out.append(0)
    if pad_to is not None:
        if len(out) > pad_to:
            raise ValueError("edited PMO is %d bytes, will not fit %d (free space by "
                             "clear_group()ing a vgroup)" % (len(out), pad_to))
        out += b"\x00" * (pad_to - len(out))
    struct.pack_into("<I", out, HEADER_OFF + 0, len(out))
    return bytes(out)


def _block_size(g: VGroup) -> int:
    """Bytes one vgroup's [GE list | vertex buffer | index buffer] block occupies."""
    vbuf_off = _align(len(g.words) * 4)
    ibuf_off = _align(vbuf_off + len(g.vbuf))
    isz = (len(g.indices) * _COMP_SIZE[g.vtype.index_char]) if g.vtype.index_char else 0
    return ibuf_off + isz


def serialize_inplace(blob: bytes, header: List[int], groups: List[VGroup]):
    """Re-emit the geBase region KEEPING every block that still fits at its ORIGINAL
    offset, and return (bytes, moved_rec_indices). Same length as `blob`, always.

    `serialize` re-lays the whole region contiguously, which is right for a file but
    wrong for a RESIDENT PAC: clearing one vgroup shifts every block after it, so the
    byte diff against the original is most of the mesh. Editing a running game means
    writing that diff over the debugger, repeatedly, to hold it across an area load —
    75 % of a 280 KB sub-resource is not something you can hold at 20 ms.

    Here a block that got smaller (`clear_group`) stays put and leaves a HOLE, a block
    that is unchanged stays put byte-for-byte, and only a block that GREW is relocated,
    into the first hole it fits. Clear one group and grow a small one and the diff is
    a couple of KB.

    🔴 The returned `moved` list must be EMPTY for anything written into a resident PAC.
    The geBase region is load-time, but the engine reads the vgroup TABLE while drawing:
    14 bytes of changed I3/I4/I5 corrupt the frame the instant they land (measured in
    Pokke village, and it reverts exactly when the original bytes go back). Relocation is
    fine in a file and fine once the area reloads — not on a PAC that is on screen. To
    avoid it entirely, `clear_group` a vgroup and grow THAT SAME vgroup: its block keeps
    its offset and is now mostly empty.
    """
    ge_base = header[H_GEBASE]
    region_len = len(blob) - ge_base
    owners = sorted((g for g in groups if g.shared_with is None), key=lambda g: g.geoff)
    extents = {}
    for i, g in enumerate(owners):
        end = owners[i + 1].geoff if i + 1 < len(owners) else region_len
        extents[g.rec_index] = (g.geoff, end)

    placed, moved, holes = {}, [], []
    for g in owners:
        lo, hi = extents[g.rec_index]
        need = _block_size(g)
        if need <= hi - lo:
            placed[g.rec_index] = lo
            if hi - (lo + _align(need)) > 0:
                holes.append([lo + _align(need), hi])
        else:
            moved.append(g)
            holes.append([lo, hi])
    holes.sort()
    merged = []
    for h in holes:                     # coalesce adjacent holes
        if merged and merged[-1][1] == h[0]:
            merged[-1][1] = h[1]
        else:
            merged.append(list(h))
    for g in moved:
        need = _block_size(g)
        for h in merged:
            start = _align(h[0])
            if start + need <= h[1]:
                placed[g.rec_index] = start
                h[0] = start + need
                break
        else:
            raise ValueError("vgroup %d needs %d bytes and no hole is that big — "
                             "clear_group() another group to make room"
                             % (g.rec_index, need))

    region = bytearray(blob[ge_base:])
    for g in owners:
        geoff = placed[g.rec_index]
        words = list(g.words)
        vbuf_off = geoff + _align(len(words) * 4)
        ibuf_off = _align(vbuf_off + len(g.vbuf))
        words[g.vaddr_widx] = 0x01000000 | ((vbuf_off - geoff) & 0xFFFFFF)
        if g.iaddr_widx is not None:
            words[g.iaddr_widx] = 0x02000000 | ((ibuf_off - geoff) & 0xFFFFFF)
        region[geoff:geoff + len(words) * 4] = struct.pack("<%dI" % len(words), *words)
        region[vbuf_off:vbuf_off + len(g.vbuf)] = g.vbuf
        if g.vtype.index_char and g.indices:
            packed = struct.pack("<%d%s" % (len(g.indices), g.vtype.index_char),
                                 *g.indices)
            region[ibuf_off:ibuf_off + len(packed)] = packed

    out = bytearray(blob[:ge_base])
    for g in groups:
        rec = list(g.rec)
        geoff = placed[g.rec_index if g.shared_with is None else g.shared_with]
        rec[3] = geoff
        rec[4] = geoff + _align(len(g.words) * 4)
        rec[5] = _align(rec[4] + len(g.vbuf))
        struct.pack_into("<2BH3I", out, header[H_VGTAB] + g.rec_index * 0x10, *rec)
    out += region
    assert len(out) == len(blob), (len(out), len(blob))
    struct.pack_into("<I", out, HEADER_OFF + 0, len(out))
    return bytes(out), [g.rec_index for g in moved]


def roundtrip_region(blob: bytes) -> bytes:
    """Parse + serialize with NO edits. This is BYTE-IDENTICAL on everything shipped —
    all 282 stage terrain/props PMOs and the monster set — because the re-layout here
    reproduces retail's exactly. Treat any diff as a bug in the encoder, not as
    expected churn. Used by tests."""
    header, groups = parse(blob)
    return serialize(blob, header, groups)


def vgroup_bone_table(blob: bytes):
    """Return [(rec_index, is_owner, vcount, index_char, weight_components)] for every
    vgroup. The vgroup's draw-order index == its primary bind (bone) index, so this is
    the lookup a modder uses to pick the target group for a given body part / bone."""
    _hdr, groups = parse(blob)
    out = []
    for g in groups:
        out.append((g.rec_index, g.shared_with is None, g.vcount,
                    g.vtype.index_char, g.vtype.wt_count))
    return out


def _resolve_group(groups, group, bone):
    """Pick the owner VGroup to grow. `group` = explicit vgroup index; `bone` = bind
    index (== vgroup draw order). If both None, the first 8-bit-indexed owner."""
    if group is not None:
        g = groups[group]
        if g.shared_with is not None:
            raise ValueError("vgroup %d shares a GE block; pick its owner %d"
                             % (group, g.shared_with))
        return g
    if bone is not None:
        if bone < len(groups) and groups[bone].shared_with is None:
            return groups[bone]
        # nearest owner at/after the bone index
        for g in groups[bone:] + groups[:bone]:
            if g.shared_with is None:
                return g
        raise ValueError("no owner vgroup for bone %d" % bone)
    return next(x for x in groups if x.shared_with is None)


def grow_pmo(blob: bytes, group: Optional[int], n_new: int,
             shift=(0.0, 200.0, 0.0), spread: float = 0.0, weight_slot: int = 0,
             bone: Optional[int] = None, force_16bit: bool = False) -> bytes:
    """Grow ONE owner vgroup by `n_new` verts / (n_new-2) faces; return the new
    (bigger) PMO bytes. Group is chosen by `group` (vgroup index) or `bone` (bind
    index); default = first owner. New verts bind 100% to palette slot `weight_slot`."""
    header, groups = parse(blob)
    g = _resolve_group(groups, group, bone)
    grow_group(g, n_new, shift=shift, scale=header[2:5], spread=spread,
               weight_slot=weight_slot, force_16bit=force_16bit)
    return serialize(blob, header, groups)


def grow_pac(pac_in: bytes, group: Optional[int], n_new: int,
             shift=(0.0, 200.0, 0.0), spread: float = 0.0, weight_slot: int = 0,
             bone: Optional[int] = None, force_16bit: bool = False) -> bytes:
    """Grow the PMO sub-resource of a monster PAC and re-flow the container."""
    from . import pac as _pac
    P = _pac.MonsterPac.from_bytes(pac_in)
    sub = next((s for s in P.subs if s.magic == b"pmo\x00"), None)
    if sub is None:
        raise ValueError("no PMO sub-resource in this PAC")
    grown = grow_pmo(sub.data, group, n_new, shift, spread, weight_slot, bone, force_16bit)
    P.subs[sub.index] = _pac.SubResource(sub.index, grown)
    return P.to_bytes()


def _main(argv):
    import argparse
    ap = argparse.ArgumentParser(description="Grow a monster PAC's geometry (Phase 5).")
    ap.add_argument("pac_in")
    ap.add_argument("pac_out", nargs="?", help="(omit with --list)")
    ap.add_argument("-g", "--group", type=int, default=None,
                    help="vgroup record index (default: first owner)")
    ap.add_argument("--bone", type=int, default=None,
                    help="bind/bone index (== vgroup draw order) to attach to; "
                         "alternative to -g")
    ap.add_argument("-n", "--verts", type=int, default=6,
                    help="number of vertices to add (>=3; makes n-2 triangles)")
    ap.add_argument("--shift", default="0,200,0",
                    help="model-unit position shift for the new verts (x,y,z)")
    ap.add_argument("--spread", type=float, default=0.0,
                    help="model-unit ring radius to scatter new verts (>0 = visible, "
                         "non-degenerate triangles)")
    ap.add_argument("--weight-slot", type=int, default=0,
                    help="bone-palette slot the new verts bind 100%% to (default 0 = "
                         "the group's primary bone)")
    ap.add_argument("--force-16bit", action="store_true",
                    help="promote the group to 16-bit indices even below 256 verts")
    ap.add_argument("--list", action="store_true",
                    help="just print the vgroup->bone table and exit")
    a = ap.parse_args(argv)
    data = open(a.pac_in, "rb").read()
    if a.list:
        from . import pac as _pac
        sub = next(s for s in _pac.MonsterPac.from_bytes(data).subs
                   if s.magic == b"pmo\x00")
        print("vgroup  owner  verts  idx  wcomps")
        for ri, own, vc, ic, wc in vgroup_bone_table(sub.data):
            print("%6d  %5s  %5d  %3s  %5d" % (ri, "Y" if own else "-", vc, ic, wc))
        return
    shift = tuple(float(x) for x in a.shift.split(","))
    out = grow_pac(data, a.group, a.verts, shift, a.spread, a.weight_slot,
                   a.bone, a.force_16bit)
    open(a.pac_out, "wb").write(out)
    print("wrote %s (%d -> %d bytes, +%d)" % (a.pac_out, len(data), len(out),
                                              len(out) - len(data)))


if __name__ == "__main__":
    import sys
    _main(sys.argv[1:])
