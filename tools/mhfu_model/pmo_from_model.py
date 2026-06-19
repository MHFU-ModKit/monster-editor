"""Build a MHFU 1.0 PMO binary from a neutral Model (v102 or any source).

Used for the MHP3rd→MHFU port path: `pmo_p3rd.parse` returns a Model with
fully-decoded float x/y/z/u/v/i/j/k/weights; this module re-encodes that into
a byte-identical-to-engine MHFU 1.0 PMO.

Layout (from RE of file_06134.bin / docs/PMO_MODEL_FORMAT.md):

  offset 0x00  magic 'pmo\x00'
  offset 0x04  version '1.0\x00'
  offset 0x08  header: struct '<I4f2H8I'  (14 words, indices h[0..13])
                 h[0]  = total PMO size
                 h[1..3] = global_scale (sx,sy,sz) = max per-axis across ALL groups
                 h[4]  = h[2] (compat copy of sy)
                 h[5]  = nmesh (one record per group, stride 0x20)
                 h[6]  = nvg_total (== nmesh for a 1-to-1 layout)
                 h[7]  = mesh_tab offset  (= 0x40 always, fixed after magic+ver+hdr)
                 h[8]  = vgroup_tab offset (= mesh_tab + nmesh*0x20)
                 h[9]  = vgroup_tab_end    (= vgroup_tab + nvg*0x10)
                 h[10] = mesh_tab_B (second mesh table; = h[9] padded; set = h[9])
                 h[11] = mat_tab offset    (after mesh_tab_B; each record 0x10)
                 h[12] = ge_base offset    (start of GE region; after mat_tab)
                 h[13] = 0

  mesh records (stride 0x20):   4f + 2H + 6I
    [0..2]  local_scale (1.0,1.0,1.0) — global scale used, not per-mesh
    [3]     0.0
    [4]     mat_base (texture palette start index; 0-indexed into mat table)
    [5]     vg_count (== 1 for simple 1:1 mesh:vg layout; but stored as 0 here,
                       actual count encoded in GE list count field)
    [6]     vg_start (index into vgroup table for this mesh's first vgroup)
    [7..11] padding zeros

  vgroup records (stride 0x10):  2B + H + 3I
    [0]  mat_pal_idx (index into mat table; same as mesh's mat_base)
    [1]  unk = 2 (observed constant)
    [2]  boneref = group draw order (bone index)
    [3]  ge_off = offset of this group's GE list from ge_base
    [4]  vbuf_off = offset of vertex buffer from ge_base
    [5]  ibuf_off = offset of index buffer from ge_base (== vbuf_off + vbuf_size)

  material records (stride 0x10):  4I
    [0]  0xFFFFFFFF
    [1]  0x7F7F7F (color, always this value)
    [2]  tex_index (0-indexed texture; 0 for mat_idx=0, etc.)
    [3]  0 (unk)

  GE region per group (all 16-byte aligned):
    VTYPE  (0x12 << 24 | vtype_word)
    VADDR  (0x01 << 24 | relative_vbuf_off)     [relative to GE list start]
    IADDR  (0x02 << 24 | relative_ibuf_off)     [relative to GE list start]
    PRIM   (0x04 << 24 | (3 << 16) | index_count)  [prim=3 = triangle list]
    RET    (0x0B << 24)
    <padding to 16-byte alignment>
    <vertex buffer>
    <padding to 16-byte alignment>
    <index buffer>
    <padding to 16-byte alignment>

Vertex format (VTYPE = weights + tex + nrm + pos + index):
  Big monsters use 1 or 2 bone weights (u8 each), UV (u16×2), normals (s8×3),
  positions (s16×3), 8-bit indices.  The exact weight count comes from the vertex
  data (count non-zero weights per vertex, max across the group).  We use:
    VTYPE = weight_count<<14 | weight_type<<9 | tex_type<<0 | nrm_type<<5 | pos_type<<7 | idx_type<<11
  with:
    weight_type = 1 (u8, trans=0x80), tex_type = 2 (u16, trans=0x8000),
    nrm_type    = 1 (s8,  trans=0x7f), pos_type  = 2 (s16, trans=0x7fff),
    idx_type    = 1 (u8) or 2 (u16 if >256 verts)
"""
from __future__ import annotations

import struct
from typing import List, Tuple

from .model import MeshGroup, Model

ALIGN = 16
PMO_MAGIC   = b"pmo\x00"
PMO_VERSION = b"1.0\x00"

_HEADER_FMT    = "<I4f2H8I"   # 14 elements
_HEADER_SIZE   = struct.calcsize(_HEADER_FMT)   # 56 = 0x38
_MESH_STRIDE   = 0x20
_VG_STRIDE     = 0x10
_MAT_STRIDE    = 0x10
_MESH_TAB_OFF  = 0x40        # always: 8 (magic) + 8 (hdr offset) + 56 (hdr) = 0x48 but observed is 0x40
# Actually: 8 (magic+ver) + 56 (hdr struct) = 64 = 0x40.  ✓

# GE command opcodes (upper byte)
_GE_VTYPE = 0x12
_GE_VADDR = 0x01
_GE_IADDR = 0x02
_GE_PRIM  = 0x04    # prim type in bits [18:16]; 3=triangle_list, 4=triangle_strip
_GE_RET   = 0x0B

# VTYPE field encodings
_WT_U8   = 1       # weight: u8, trans=128
_TEX_U16 = 2       # UV: u16, trans=32768
_NRM_S8  = 1       # normal: s8, trans=127
_POS_S16 = 2       # position: s16, trans=32767
_IDX_U8  = 1       # 8-bit index
_IDX_U16 = 2       # 16-bit index


def _align(n: int, a: int = ALIGN) -> int:
    return (n + a - 1) & ~(a - 1)


def _pad(buf: bytearray, a: int = ALIGN) -> None:
    n = _align(len(buf), a) - len(buf)
    buf.extend(b"\x00" * n)


def _weight_count(g: MeshGroup) -> int:
    """Number of bone weights per vertex for this group (1 or 2)."""
    max_wt = 1
    for v in g.vertices:
        ws = v.get("weights", [])
        # count weights that are non-negligible
        cnt = sum(1 for w in ws if w > 1e-5)
        if cnt > max_wt:
            max_wt = cnt
    return max(1, max_wt)


def _group_scale(g: MeshGroup) -> Tuple[float, float, float]:
    """Per-axis max-abs scale for quantizing positions to s16."""
    mx = my = mz = 1.0
    for v in g.vertices:
        ax, ay, az = abs(v["x"]), abs(v["y"]), abs(v["z"])
        if ax > mx: mx = ax
        if ay > my: my = ay
        if az > mz: mz = az
    return (mx, my, mz)


def _build_vtype(wt_count: int, use_u16_idx: bool) -> int:
    """Build a VTYPE word for the given weight count and index width."""
    idx_type = _IDX_U16 if use_u16_idx else _IDX_U8
    # weight_count field = actual_count - 1 (0..7)
    wt_cnt_field = (wt_count - 1) & 7
    vtype = (
        (_WT_U8  << 9) |
        (wt_cnt_field << 14) |
        (_TEX_U16 << 0) |
        (_NRM_S8  << 5) |
        (_POS_S16 << 7) |
        (idx_type << 11)
    )
    return vtype


def _encode_vertex(v: dict, sx: float, sy: float, sz: float,
                   wt_count: int) -> bytes:
    """Encode one vertex into the MHFU 1.0 binary format:
       weights(u8×wt_count) + UV(u16×2) + normals(s8×3) + positions(s16×3)
    """
    out = bytearray()
    ws = v.get("weights", [1.0])
    # pad / truncate to wt_count
    while len(ws) < wt_count:
        ws = list(ws) + [0.0]
    ws = list(ws[:wt_count])
    # normalize so they sum to 1.0 (avoid overflow)
    total = sum(ws)
    if total > 0:
        ws = [w / total for w in ws]
    else:
        ws = [1.0] + [0.0] * (wt_count - 1)
    for w in ws:
        q = max(0, min(255, round(w * 128)))
        out += struct.pack("B", q)
    # Native-alignment padding: after wt_count u8 weights, align to u16 (2-byte boundary)
    # run_ge uses struct.calcsize WITHOUT '<' (native alignment) so padding IS inserted.
    cur = len(out)
    if cur % 2 != 0:
        out += b"\x00"
    # UV: u16 in [0,1] range (trans=32768; Blender already flips V)
    u = max(0.0, min(1.0, v.get("u", 0.0)))
    vv = max(0.0, min(1.0, v.get("v", 0.0)))
    out += struct.pack("<2H", round(u * 32768), round(vv * 32768))
    # normals: s8, trans=127
    i = max(-1.0, min(1.0, v.get("i", 0.0)))
    j = max(-1.0, min(1.0, v.get("j", 0.0)))
    k = max(-1.0, min(1.0, v.get("k", 0.0)))
    out += struct.pack("<3b", round(i * 127), round(j * 127), round(k * 127))
    # Native-alignment padding: after 3 s8 normals, align to s16 (2-byte boundary)
    cur = len(out)
    if cur % 2 != 0:
        out += b"\x00"
    # positions: s16, quantized by per-axis scale
    # guard against zero scale
    dx = 32767.0 / sx if sx > 1e-6 else 0.0
    dy = 32767.0 / sy if sy > 1e-6 else 0.0
    dz = 32767.0 / sz if sz > 1e-6 else 0.0
    px = max(-32768, min(32767, round(v["x"] * dx)))
    py = max(-32768, min(32767, round(v["y"] * dy)))
    pz = max(-32768, min(32767, round(v["z"] * dz)))
    out += struct.pack("<3h", px, py, pz)
    return bytes(out)


_GE_BASE   = 0x14  # BASE command (ct=0x14)
_GE_UNK10  = 0x10  # unknown 0x10 command (present in all real PMOs)
_GE_FFACE  = 0x9B  # face order (0 = CW)


def _build_ge_block(g: MeshGroup,
                    sx: float, sy: float, sz: float) -> Tuple[bytes, int, int]:
    """Build the GE display list + vertex buffer + index buffer for one group.

    Matches the exact command sequence observed in MHFU 1.0 PMOs:
      BASE(0x14,0) + 0x10(0) + IADDR + VADDR + VTYPE + FFACE(0) +
      one PRIM(triangle_list, 3) per triangle + RET

    sx/sy/sz must be the GLOBAL per-axis scale stored in the PMO header
    (h[2], h[3], h[4]) because the decoder uses those same values for all
    groups — using per-group scale would produce wrong positions.

    Returns (block_bytes, vbuf_rel, ibuf_rel) where rel offsets are relative
    to the GE list start (the BASE command), for use in VADDR/IADDR arguments
    and the vgroup record I4/I5 fields.
    """
    wt_count = _weight_count(g)
    n_verts  = len(g.vertices)
    use_u16  = n_verts > 255
    vtype    = _build_vtype(wt_count, use_u16)

    # Encode vertex buffer using the GLOBAL scale (matches decoder expectation)
    vbuf = bytearray()
    for v in g.vertices:
        vbuf += _encode_vertex(v, sx, sy, sz, wt_count)

    # Encode index buffer: one u8/u16 per vertex reference (3 per triangle)
    idx_char = "<H" if use_u16 else "<B"
    ibuf = bytearray()
    for f in g.faces:
        ibuf += struct.pack(idx_char, f["v1"])
        ibuf += struct.pack(idx_char, f["v2"])
        ibuf += struct.pack(idx_char, f["v3"])

    n_triangles = len(g.faces)

    # GE command list size:
    #   2 header cmds (BASE, 0x10) + IADDR + VADDR + VTYPE + FFACE +
    #   n_triangles PRIM cmds + RET = (7 + n_triangles) × 4 bytes
    #   (BASE + unk10 + IADDR + VADDR + VTYPE + FFACE = 6, then n_triangles PRIM, then 1 RET)
    ge_list_raw  = (7 + n_triangles) * 4
    ge_list_size = _align(ge_list_raw)

    # VADDR/IADDR args are offsets FROM the GE list start.
    vbuf_rel = ge_list_size
    vbuf_size = _align(len(vbuf))
    ibuf_rel  = vbuf_rel + vbuf_size

    # Build GE list
    ge = bytearray()
    ge += struct.pack("<I", (_GE_BASE  << 24) | 0)    # BASE 0
    ge += struct.pack("<I", (_GE_UNK10 << 24) | 0)    # 0x10 0
    ge += struct.pack("<I", (_GE_IADDR << 24) | ibuf_rel)
    ge += struct.pack("<I", (_GE_VADDR << 24) | vbuf_rel)
    ge += struct.pack("<I", (_GE_VTYPE << 24) | vtype)
    ge += struct.pack("<I", (_GE_FFACE << 24) | 0)    # FFACE 0 (CW winding)
    # One PRIM per triangle (triangle_strip = prim_type 4, 3 indices = 1 triangle)
    # This matches the real MHFU 1.0 PMO format (prim_type=4 observed in file_06134).
    for _ in range(n_triangles):
        ge += struct.pack("<I", (_GE_PRIM << 24) | (4 << 16) | 3)
    ge += struct.pack("<I", (_GE_RET << 24))
    _pad(ge)   # align to 16 bytes

    # Vertex buffer (padded)
    vb_section = bytearray(vbuf)
    _pad(vb_section)

    # Index buffer (padded)
    ib_section = bytearray(ibuf)
    _pad(ib_section)

    block = bytes(ge) + bytes(vb_section) + bytes(ib_section)
    return block, vbuf_rel, ibuf_rel


def encode_from_model(model: Model) -> bytes:
    """Re-encode a neutral Model into a MHFU 1.0 PMO binary.

    Works on any Model whose mesh_groups have decoded float x/y/z/u/v/i/j/k
    vertex data — produced by pmo.parse(), pmo_p3rd.parse(), or the Blender
    importer round-trip.  The output is a self-contained PMO binary (no companion
    file) suitable for embedding in an MHFU 1.0 PAC.

    Scale encoding:
      Global header scale = per-axis maximum across ALL groups (for compat);
      each group's VTYPE positions are quantized by the per-group max (tighter,
      reduces quantization error for small body parts).
    """
    groups = [g for g in model.mesh_groups if g.vertices and g.faces]
    if not groups:
        raise ValueError("Model has no non-empty mesh groups")

    n = len(groups)

    # Compute unique materials (texture indices used across groups)
    used_mats = sorted(set(g.material for g in groups))
    mat_to_slot = {m: i for i, m in enumerate(used_mats)}

    # Layout offsets
    mesh_tab  = _MESH_TAB_OFF                    # 0x40
    vg_tab    = mesh_tab  + n * _MESH_STRIDE     # after all mesh records
    vg_end    = vg_tab    + n * _VG_STRIDE       # end of vgroup table
    mat_tab   = _align(vg_end)                   # padded
    n_mats    = len(used_mats)
    ge_base   = _align(mat_tab + n_mats * _MAT_STRIDE)

    # Global scale = per-axis max-abs across ALL groups.
    # CRITICAL: the decoder reads model.scale = tuple(header[2:5]) and uses
    #   scale[0] for X, scale[1] for Y, scale[2] for Z.
    # So header layout must be: h[2]=gsx, h[3]=gsy, h[4]=gsz.
    # h[1] = max(gsx, gsy, gsz) (observed pattern in native MHFU PMOs).
    gsx = gsy = gsz = 1.0
    for g in groups:
        sx, sy, sz = _group_scale(g)
        if sx > gsx: gsx = sx
        if sy > gsy: gsy = sy
        if sz > gsz: gsz = sz

    # Build GE blocks per group — pass GLOBAL scale so positions quantize
    # with the same values the decoder will use to dequantize.
    ge_blocks    = []   # bytes per group
    ge_offsets   = []   # cumulative ge_base-relative offset per group
    vbuf_rels    = []   # vbuf_rel within each group's block
    ibuf_rels    = []   # ibuf_rel within each group's block
    ge_cursor    = 0
    for g in groups:
        block, vbuf_rel, ibuf_rel = _build_ge_block(g, gsx, gsy, gsz)
        ge_blocks.append(block)
        ge_offsets.append(ge_cursor)
        vbuf_rels.append(vbuf_rel)
        ibuf_rels.append(ibuf_rel)
        ge_cursor += len(block)

    total_size = ge_base + ge_cursor

    # --- Assemble binary ---
    out = bytearray()

    # 1. Magic + version
    out += PMO_MAGIC + PMO_VERSION

    # 2. Header (h[0..14]) — fmt '<I4f2H8I' = 1+4+2+8 = 15 items
    #    Decoder: model.scale = tuple(header[2:5]) = (h[2], h[3], h[4])
    #             run_ge: x *= scale[0]=h[2], y *= scale[1]=h[3], z *= scale[2]=h[4]
    #    Therefore: h[2]=gsx, h[3]=gsy, h[4]=gsz, h[1]=max of all three.
    g_max = max(gsx, gsy, gsz)
    out += struct.pack(_HEADER_FMT,
        total_size,         # I   h[0]  total size
        g_max,              # f   h[1]  max of all axes (observed in native PMOs)
        gsx,                # f   h[2]  x-axis decode scale → model.scale[0]
        gsy,                # f   h[3]  y-axis decode scale → model.scale[1]
        gsz,                # f   h[4]  z-axis decode scale → model.scale[2]
        n,                  # H   h[5]  nmesh
        n,                  # H   h[6]  nvg_total
        mesh_tab,           # I   h[7]  mesh table offset
        vg_tab,             # I   h[8]  vgroup table offset
        vg_end,             # I   h[9]  vgroup table end
        vg_end,             # I   h[10] mesh_tab_B
        mat_tab,            # I   h[11] material table offset
        ge_base,            # I   h[12] GE region base
        0,                  # I   h[13] zero
        0,                  # I   h[14] zero
    )

    # Pad to mesh_tab (0x40)
    while len(out) < mesh_tab:
        out += b"\x00"
    assert len(out) == mesh_tab, "header overflowed mesh_tab"

    # 3. Mesh records (stride 0x20 = 2f+2I+4H+2I, as expected by pmo._walk)
    #    _walk reads: mh = struct.unpack_from("2f2I4H2I", blob, m)
    #    then: mat_base=mh[5], vg_count=mh[6], vg_start=mh[7]
    #    So: mh[4]=pad_u16, mh[5]=mat_base(u16), mh[6]=vg_count(u16), mh[7]=vg_start(u16)
    #
    #    IMPORTANT: mat_base MUST be 0.  The engine computes:
    #      material = mat_tab + (mat_base + vg[0]) * 16
    #    where vg[0] = mat_pal_idx from the vgroup record.  We set vg[0] = mat_slot
    #    (the index into the used_mats list), so mat_base must be 0 to avoid
    #    doubling the offset.  Native Tigrex (file_06134) confirms: mat_base always 0,
    #    vgroup mat_pal_idx = 0..7 directly.
    for gi, g in enumerate(groups):
        rec = struct.pack("<2f2I4H2I",  # 2f(8)+2I(8)+4H(8)+2I(8) = 32 = 0x20
            1.0, 1.0,      # mh[0..1]: 2 floats (local scale)
            0, 0,           # mh[2..3]: 2 u32 (padding)
            0,              # mh[4]: u16 padding
            0,              # mh[5]: mat_base = 0 (vg[0] is the direct mat row index)
            1,              # mh[6]: vg_count (u16) = 1 group per mesh
            gi,             # mh[7]: vg_start (u16) = index into vgroup table
            0, 0,           # mh[8..9]: 2 u32 padding
        )
        out += rec
    assert len(out) == vg_tab, "mesh table overrun: %d != %d" % (len(out), vg_tab)

    # 4. Vgroup records (stride 0x10): one per group
    for gi, g in enumerate(groups):
        mat_slot = mat_to_slot[g.material]
        ge_off  = ge_offsets[gi]
        # vbuf and ibuf offsets are ABSOLUTE from ge_base (= ge_off + rel)
        vbuf_abs = ge_off + vbuf_rels[gi]
        ibuf_abs = ge_off + ibuf_rels[gi]
        rec = struct.pack("<2BH3I",
            mat_slot,   # mat_pal_idx
            2,          # unk (observed constant = 2)
            gi,         # boneref = draw order = group index
            ge_off,     # ge_off (GE list start, relative to ge_base)
            vbuf_abs,   # I4 = vertex buffer offset from ge_base
            ibuf_abs,   # I5 = index buffer offset from ge_base
        )
        out += rec
    assert len(out) == vg_end

    # Pad to mat_tab
    while len(out) < mat_tab:
        out += b"\x00"

    # 5. Material records (stride 0x10): one per unique texture
    for tex_idx in used_mats:
        rec = struct.pack("<4I",
            0xFFFFFFFF,    # [0] sentinel
            0x7F7F7F,      # [1] color (observed constant)
            tex_idx,       # [2] texture index
            0,             # [3] zero
        )
        out += rec

    # Pad to ge_base
    while len(out) < ge_base:
        out += b"\x00"

    # 6. GE blocks (already aligned)
    for block in ge_blocks:
        out += block

    assert len(out) == total_size, (
        "size mismatch: built %d, expected %d" % (len(out), total_size))

    return bytes(out)
