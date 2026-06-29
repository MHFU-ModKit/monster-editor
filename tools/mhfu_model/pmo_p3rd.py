"""MHP3rd PMO v102 parser -> mhfu_model.model.Model.

MHP3rd (gen-3) big-monster PMO differs from MHFU in two ways:
  1. Version word is b'102\\x00' (not b'1.0\\x00').
  2. Per-mesh scale lives in the mesh-table record (not a header global).
  3. For in-quest/high-quality PACs the GE display list is in a COMPANION file
     (the byte immediately after the PAC: file_NNNN+1.bin).  The PMO header
     field `ge_base` equals the PMO blob size in that case, so we detect it by
     `ge_base >= len(pmo_blob)` and redirect GE reads into `geo_blob`.

API (as contracted with blender):
    parse(blob, geo_blob=None) -> Model
        blob      - raw PMO v102 bytes (from the PAC sub that starts with b'pmo\\x00')
        geo_blob  - companion GE file bytes, or None for self-contained blobs
                    (lobby-style PACs where ge_base < len(blob))

The returned Model uses the same dataclasses as mhfu_model.model so the Blender
importer / MHFU exporter pipeline is unchanged.  Per-mesh scale is baked into
vertex x/y/z positions (the Blender layer does not need to apply it again).

The `Model.raw` field holds the original v102 blob.  Until a full v102->1.0
re-encoder exists the exporter embeds these bytes unchanged (preview/scaffold path).
"""
from __future__ import annotations

import io
import struct
from typing import List, Optional

from .model import MeshGroup, Model
from .pmo import run_ge  # validated GE-display-list walker from the MHFU path

PMO_MAGIC = b"pmo\x00"
P3RD_VER = b"102\x00"


# --------------------------------------------------------------------------- #
# v102 GE-display-list walker (faithful port of AsteriskAmpersand's PMO-Importer
# pmo_parse.build_prim — fixes the strip over-expansion of the MHFU run_ge: v102
# vertices are addressed THROUGH the index buffer and DEDUPED by address, not read
# flat/sequentially). See docs + /tmp/pmoimp/struct/pmo_parse.py.
# --------------------------------------------------------------------------- #
# VTYPE bit fields (GE command 0x12). NOTE weightCount is 4 bits (14..18) and
# there is a `bypass` bit (23) selecting normalized(U)/raw(R) value scaling — the
# MHFU walker used a 3-bit weightCount and no bypass, which mis-sized v102 verts.
_VT = {
    "uv": (0, 2), "color": (2, 4), "colorUse": (4, 5), "normal": (5, 7),
    "pos": (7, 9), "weight": (9, 11), "index": (11, 13),
    "wcount": (14, 18), "morph": (18, 21), "bypass": (23, 24),
}


def _bits(cmd, lo, hi):
    return (cmd >> lo) & ((1 << (hi - lo)) - 1)


def _vtype_layout(cmd):
    """Decode a VTYPE command -> (stride, field descriptors, index_size).

    Field byte sizes mirror PMO-Importer's construct structs exactly (incl. the
    int16 pad on 8-bit UV and the trailing w on 8/16-bit normals)."""
    f = {k: _bits(cmd, lo, hi) for k, (lo, hi) in _VT.items()}
    if f["morph"]:
        raise ValueError("morph class not supported")
    wcls, wcnt = f["weight"], f["wcount"] + 1
    wsz = [0, 1, 2, 4][wcls] * wcnt
    wpad = (-([0, 1, 2, 4][wcls] * wcnt)) % 2
    uvsz = [0, 4, 4, 8][f["uv"]]                 # 8-bit uv = 1+1+2pad
    csz = 0 if not f["colorUse"] else [2, 2, 2, 4][f["color"]]
    nsz = [0, 4, 8, 12][f["normal"]]             # 8/16-bit normal carry a w
    psz = [0, 3, 6, 12][f["pos"]]
    stride = wsz + wpad + uvsz + csz + nsz + psz
    isz = [0, 1, 2, 4][f["index"]]
    return stride, f, isz


def _read_vertex(data, off, f, scale):
    """Decode one vertex at ``off`` per the VTYPE field descriptors ``f``.
    Returns a dict with x/y/z (scaled), u/v, i/j/k (normal) and weights[]."""
    wcls, wcnt = f["weight"], f["wcount"] + 1
    bypass = f["bypass"]
    o = off
    weights = []
    if wcls:
        wfmt = {1: "B", 2: "H", 3: "f"}[wcls]
        wnorm = {1: 0x80, 2: 0x8000, 3: 1}[wcls]
        for _ in range(wcnt):
            weights.append(struct.unpack_from("<" + wfmt, data, o)[0] / wnorm)
            o += {1: 1, 2: 2, 3: 4}[wcls]
        o += (-([0, 1, 2, 4][wcls] * wcnt)) % 2          # weight pad
    u = v = 0.0
    if f["uv"]:
        if f["uv"] == 1:
            ru, rv = struct.unpack_from("<BB", data, o); o += 4   # +2 pad
            n = 1 if bypass else 0x80
        elif f["uv"] == 2:
            ru, rv = struct.unpack_from("<HH", data, o); o += 4
            n = 1 if bypass else 0x8000
        else:
            ru, rv = struct.unpack_from("<ff", data, o); o += 8; n = 1
        u, v = ru / n, rv / n
    if f["colorUse"]:
        o += [2, 2, 2, 4][f["color"]]
    nx = ny = nz = 0.0
    if f["normal"]:
        if f["normal"] == 1:
            nx, ny, nz, _w = struct.unpack_from("<bbbb", data, o); o += 4
            nn = 1 if bypass else 0x7F
        elif f["normal"] == 2:
            nx, ny, nz, _w = struct.unpack_from("<hhhh", data, o); o += 8
            nn = 1 if bypass else 0x7FFF
        else:
            nx, ny, nz = struct.unpack_from("<fff", data, o); o += 12; nn = 1
        nx, ny, nz = nx / nn, ny / nn, nz / nn
    px = py = pz = 0.0
    if f["pos"]:
        if f["pos"] == 1:
            px, py, pz = struct.unpack_from("<bbb", data, o); o += 3
            pn = 1 if bypass else 0x7F
        elif f["pos"] == 2:
            px, py, pz = struct.unpack_from("<hhh", data, o); o += 6
            pn = 1 if bypass else 0x7FFF
        else:
            px, py, pz = struct.unpack_from("<fff", data, o); o += 12; pn = 1
        px, py, pz = px / pn, py / pn, pz / pn
    return {"x": px * scale[0], "y": py * scale[1], "z": pz * scale[2],
            "u": u, "v": v, "i": nx, "j": ny, "k": nz, "weights": weights}


def run_ge_v102(data, base, scale):
    """Walk the v102 GE display list for one vgroup starting at byte ``base``.

    Returns (vertices, faces) — vertices deduped by source address (so a tristrip's
    shared verts are one vertex), faces as {v1,v2,v3} dicts. ``data`` is the whole
    GE source bytes; ``base`` the vgroup's mesh-data offset."""
    pos = base
    vaddr = iaddr = None
    stride = isz = 0
    fdesc = None
    face_order = 0
    verts = []
    faces = []
    addr2idx = {}
    while pos + 4 <= len(data):
        cmd = struct.unpack_from("<I", data, pos)[0]; pos += 4
        ct = cmd >> 24
        low = cmd & 0xFFFFFF
        if ct == 0x10:                                   # BASE (high addr bits)
            pass
        elif ct == 0x14 or ct == 0x13:                   # ORIGIN / OFFSET
            pass
        elif ct == 0x01:                                 # VADDR (relative to base)
            vaddr = base + low
        elif ct == 0x02:                                 # IADDR (relative to base)
            iaddr = base + low
        elif ct == 0x12:                                 # VTYPE
            stride, fdesc, isz = _vtype_layout(cmd)
        elif ct == 0x9B:                                 # FFACE (cull order)
            face_order = low & 1
        elif ct == 0x04:                                 # PRIM
            count = cmd & 0xFFFF
            ptype = (cmd >> 16) & 7
            # index list
            if iaddr is not None and isz:
                ifmt = {1: "B", 2: "H", 4: "I"}[isz]
                idxs = list(struct.unpack_from("<%d%s" % (count, ifmt), data, iaddr))
                iaddr += count * isz
            else:
                idxs = list(range(count))
            rng = range(0, count, 3) if ptype == 3 else range(count - 2)
            signum = face_order
            for k in rng:
                tri = (idxs[k + signum], idxs[k + 1 - signum], idxs[k + 2])
                signum ^= 1
                out = []
                for vi in tri:
                    addr = vaddr + vi * stride
                    di = addr2idx.get(addr)
                    if di is None:
                        di = len(verts)
                        addr2idx[addr] = di
                        verts.append(_read_vertex(data, addr, fdesc, scale))
                    out.append(di)
                if out[0] != out[1] and out[1] != out[2] and out[0] != out[2]:
                    faces.append({"v1": out[0], "v2": out[1], "v3": out[2]})
        elif ct == 0x0B:                                 # RET
            break
        # ignore other GPU/bool state commands
    return verts, faces


def _read_header(blob: bytes):
    """Parse the 14-element v102 PMO fixed header (offsets 0..0x38).

    Layout from tools/mhff/psp/pmo.py convert_mh3_pmo::
        struct.unpack('<I4f2H8I', blob[8:8+0x38])  ->  14 values
        [0]  total_size (header + geo)
        [1..3]  global scale x,y,z
        [4]  ? (padding)
        [5]  nmesh
        [6]  total vgroup count
        [7]  mesh table offset
        [8]  vgroup table offset
        [9..10]  unknown
        [11] material table offset
        [12] ge_base  (== pmo blob size for companion-file monsters)
        [13] unknown
    """
    if len(blob) < 0x40:
        raise ValueError("PMO blob too short (%d bytes)" % len(blob))
    return struct.unpack_from("<I4f2H8I", blob, 8)


def parse(blob: bytes, geo_blob: Optional[bytes] = None) -> Model:
    """Parse a MHP3rd PMO v102 blob -> Model.

    Parameters
    ----------
    blob:     Raw PMO v102 bytes (starts with b'pmo\\x00' b'102\\x00').
    geo_blob: Raw companion GE display-list file bytes, or None.
              Required for in-quest high-quality PACs where ge_base >= len(blob).
              Pass None for lobby-style self-contained PMOs.

    Returns
    -------
    Model with MeshGroups populated (x/y/z vertices; u/v UVs; i/j/k normals
    where present).  Per-mesh scale is baked into vertex positions.
    """
    if len(blob) < 8:
        raise ValueError("blob too short for PMO magic/version")
    if blob[:4] != PMO_MAGIC:
        raise ValueError("not a PMO blob (magic=%r)" % blob[:4])
    ver = blob[4:8]
    if ver != P3RD_VER:
        raise ValueError("expected PMO v102, got %r" % ver)

    hdr = _read_header(blob)
    global_scale = hdr[1:4]   # (sx, sy, sz) float
    nmesh      = hdr[5]
    nvg_total  = hdr[6]
    mesh_tab   = hdr[7]
    vg_tab     = hdr[8]
    mat_tab    = hdr[11]
    ge_base    = hdr[12]

    # Decide where GE reads come from.
    # For companion-file PACs: ge_base == len(blob) and companion is geo_blob.
    # For self-contained lobby PACs: ge_base < len(blob), reads stay in blob.
    uses_companion = ge_base >= len(blob)
    if uses_companion:
        if geo_blob is None:
            # Return a Model with no geometry rather than crashing; callers that
            # need geometry must supply geo_blob.
            return Model(
                magic=PMO_MAGIC, version=P3RD_VER,
                mesh_groups=[], raw=blob, scale=global_scale,
                stride=0x30, edited=False,
            )
        ge_source = geo_blob
        ge_offset = 0   # vgroup ge_off is absolute into geo_blob
    else:
        ge_source = blob
        ge_offset = ge_base  # vgroup ge_off is relative to ge_base inside blob

    geo_buf = io.BytesIO(ge_source)

    # --- bone palette (header field 10) -> authentic per-vertex blend skinning ----
    # v102 uses the SAME palette model as MHFU (decoded + zero-error validated
    # 2026-06-29 on file_05248; see docs/PMO_MODEL_FORMAT.md "skinning"): the vgroup
    # record `2BH3I` carries bc=vg[1] (boneCount) + cum=vg[2] (cumulativeBoneCount);
    # header field 10 points at a `Weight{slot:u8,bone:u8}[]` array; the engine keeps
    # a running `aux[slot]=bone` and each vertex's weightCount fractions blend
    # `aux[0..wc-1]`. Resolving it here lets the porter ship the monster's REAL skin
    # (source-skeleton bone indices) instead of a nearest-bone guess. NOTE: header
    # field 6 (nvg) is unreliable (e.g. 24 vs real 88); the true vgroup count is the
    # max (vg_start+vg_count) over the mesh table.
    pal_off = hdr[10]
    eff_palettes = None
    if pal_off and pal_off < len(blob):
        nvg_real = 0
        for i in range(nmesh):
            mr = mesh_tab + i * 0x30
            if mr + 0x30 > len(blob):
                break
            mh = struct.unpack_from("<8f2I4H", blob, mr)
            nvg_real = max(nvg_real, mh[13] + mh[12])   # vg_start + vg_count
        vg_records = []
        for k in range(nvg_real):
            off = vg_tab + k * 0x10
            if off + 0x10 > len(blob):
                break
            vg_records.append(struct.unpack_from("<2BH3I", blob, off))
        if vg_records:
            pal_len = vg_records[-1][2] + vg_records[-1][1]
            if pal_off + pal_len * 2 <= len(blob):
                patches = [struct.unpack_from("<2B", blob, pal_off + p * 2)
                           for p in range(pal_len)]
                from .pmo_skin import _resolve_running_palette
                eff_palettes = _resolve_running_palette(patches, vg_records)

    groups: List[MeshGroup] = []
    draw_order = 0

    for i in range(nmesh):
        mesh_rec_off = mesh_tab + i * 0x30
        if mesh_rec_off + 0x30 > len(blob):
            break
        # Mesh record: 8 floats then 2 u32 then 4 u16
        mh = struct.unpack_from("<8f2I4H", blob, mesh_rec_off)
        # Per-mesh scale overrides the global header scale for this record.
        mesh_scale = mh[0:3]   # (sx, sy, sz)
        cum_mat    = mh[11]    # u16 cumulativeMaterialCount (material base for this mesh)
        vg_count   = mh[12]    # u16
        vg_start   = mh[13]    # u16

        for j in range(vg_count):
            vg_rec_idx = vg_start + j
            vg_off = vg_tab + vg_rec_idx * 0x10
            if vg_off + 0x10 > len(blob):
                break

            # Vgroup record: mat(u8) unk(u8) boneref(u16) ge_off(u32) I4(u32) I5(u32)
            vg = struct.unpack_from("<2BH3I", blob, vg_off)
            mat_pal_idx = vg[0]
            boneref     = vg[2]   # REAL skeleton bone this rigid group binds to
            ge_rel      = vg[3]

            # Resolve texture/material index from the material table.
            # The material-data index is the mesh's cumulativeMaterialCount + the
            # vgroup's per-mesh materialOffset (vg[0]); there is no materialRemap
            # in these monster PMOs (header field 9 == 0). Omitting cum_mat was a
            # bug that collapsed every group onto materialData[0..2] (tex 9/17).
            tex_idx = 0
            mat_idx = cum_mat + mat_pal_idx
            mat_off = mat_tab + mat_idx * 16
            if mat_off + 16 <= len(blob):
                tex_idx = struct.unpack_from("<4I", blob, mat_off)[2]

            # Seek to the GE display list for this vgroup.
            if uses_companion:
                ge_abs = ge_rel          # absolute offset into geo_blob
            else:
                ge_abs = ge_offset + ge_rel   # absolute offset into blob
            if ge_abs >= len(ge_source):
                continue

            try:
                verts, faces = run_ge_v102(ge_source, ge_abs, mesh_scale)
            except (ValueError, struct.error, IndexError):
                continue

            # Attach each vertex's authentic (source_bone, weight) influences from
            # the resolved palette for this vgroup record (parallel to vertex weights).
            if eff_palettes is not None and vg_rec_idx < len(eff_palettes):
                pal = eff_palettes[vg_rec_idx]
                for v in verts:
                    w = v.get("weights")
                    if w:
                        v["influences"] = [(pal[k] if k < len(pal) else -1, w[k])
                                           for k in range(len(w))]
                    elif pal:
                        v["influences"] = [(pal[0], 1.0)]

            g = MeshGroup(
                index=draw_order,
                material=tex_idx,
                mesh_record=i,
                vertex_count=len(verts),
                face_count=len(faces),
                scale=mesh_scale,
                vertices=verts,
                faces=faces,
                enc=None,       # no in-place re-encode (v102 passthrough)
                vg_rec=vg_rec_idx,
                boneref=boneref,
            )
            groups.append(g)
            draw_order += 1

    return Model(
        magic=PMO_MAGIC,
        version=P3RD_VER,
        mesh_groups=groups,
        raw=blob,
        scale=global_scale,
        stride=0x30,
        edited=False,
    )
