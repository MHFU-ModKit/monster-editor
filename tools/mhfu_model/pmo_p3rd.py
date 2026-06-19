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
            ge_rel      = vg[3]

            # Resolve texture/material index from the material table.
            tex_idx = 0
            mat_off = mat_tab + mat_pal_idx * 16
            if mat_off + 16 <= len(blob):
                tex_idx = struct.unpack_from("<4I", blob, mat_off)[2]

            # Seek to the GE display list for this vgroup.
            if uses_companion:
                ge_abs = ge_rel          # absolute offset into geo_blob
            else:
                ge_abs = ge_offset + ge_rel   # absolute offset into blob
            if ge_abs >= len(ge_source):
                continue

            geo_buf.seek(ge_abs)
            try:
                verts_raw, faces_raw, enc = run_ge(geo_buf, mesh_scale)
            except (ValueError, struct.error):
                continue

            # run_ge returns a sparse list (None slots for out-of-order indices);
            # filter them out so the MeshGroup has a dense vertex list.
            verts = [v for v in verts_raw if v is not None]
            faces = list(faces_raw)

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
