"""Provisional MHP3rd PMO v102 parser -> mhfu_model.model.Model.

This is the SCAFFOLD that backs the Blender MHP3rd import path until
`asset` delivers the final `pmo_p3rd.py`.  It uses the same GE-display-list
walker as the MHFU pmo.py but drives it through the MHP3rd mesh table
(0x30-byte records, per-mesh scale in the record header, different header
field positions for v102).

The v102 PMO header layout (from mhff/psp/pmo.py convert_mh3_pmo):
    struct.unpack('I4f2H8I', 0x38 bytes)
    [0]  magic / version word  (= b'pmo\\x00' / '102\\x00')
    [1..4]  scale x,y,z + ?
    [5]  mesh count
    [6]  ?
    [7]  mesh table offset
    [8]  vertex-group table offset
    [9..10]  ?
    [11] material table offset
    [12] GE-list base offset

Mesh record (0x30 bytes, at mesh_tab + i*0x30):
    struct.unpack('8f2I4H', 0x30)
    [0..2]  per-mesh scale x,y,z
    [12]  vertex-group count (u16)
    [13]  vertex-group start index (u16)

Vertex-group record (0x10 bytes, at vg_tab + (start+j)*0x10):
    struct.unpack('2BH3I', 0x10)
    [0]  material palette index
    [3]  GE-list offset relative to ge_base

Material record (16 bytes at mat_tab + mat_idx*16):
    struct.unpack('4I', 16)
    [2]  texture/material index

Replaces this with pmo_p3rd.py once asset ships that module; until then
this is the live fallback the Blender addon uses.
"""
from __future__ import annotations

import array
import io
import struct
from typing import List

from .model import MeshGroup, Model
from .pmo import run_ge      # re-use the validated GE walker from the MHFU path

PMO_MAGIC = b"pmo\x00"
P3RD_VER  = b"102\x00"


def _read_header(buf: io.BytesIO):
    """Read and return the 14-element v102 PMO header tuple."""
    buf.seek(0)
    hdr = struct.unpack("<I4f2H8I", buf.read(0x38))
    return hdr


def parse(blob: bytes) -> Model:
    """Parse a MHP3rd PMO v102 blob -> Model.

    Returns a Model with MeshGroups populated (x/y/z vertices, UVs, normals).
    The model is NOT editable (model.edited stays False); the exporter converts
    it to MHFU 1.0 via a fresh encode, which is currently a raw passthrough —
    see the export note below.

    EXPORT NOTE: pmo_p3rd writes the Model.raw bytes unchanged.  The FULL
    conversion (v102 -> MHFU 1.0) is the job of `pmo_p3rd.py` from asset.
    Until that lands, export will embed the original v102 blob byte-for-byte
    (the engine won't render it — this is a scaffold/preview path only).
    """
    if blob[:4] != PMO_MAGIC:
        raise ValueError("not a PMO blob (magic=%r)" % blob[:4])
    ver = blob[4:8]
    if ver != P3RD_VER:
        raise ValueError("expected PMO v102, got %r" % ver)

    buf = io.BytesIO(blob)
    hdr = _read_header(buf)
    # header field positions (from convert_mh3_pmo in mhff/psp/pmo.py):
    global_scale = hdr[1:4]          # x, y, z  (float)
    nmesh      = hdr[5]
    mesh_tab   = hdr[7]
    vg_tab     = hdr[8]
    mat_tab    = hdr[11]
    ge_base    = hdr[12]

    groups: List[MeshGroup] = []
    draw_order = 0

    for i in range(nmesh):
        buf.seek(mesh_tab + i * 0x30)
        if buf.tell() + 0x30 > len(blob):
            break
        mh = struct.unpack("<8f2I4H", buf.read(0x30))
        mesh_scale = mh[0:3]          # per-mesh scale overrides global for this record
        vg_count   = mh[12]
        vg_start   = mh[13]

        for j in range(vg_count):
            vg_off = vg_tab + (vg_start + j) * 0x10
            if vg_off + 0x10 > len(blob):
                break
            buf.seek(vg_off)
            vg = struct.unpack("<2BH3I", buf.read(0x10))
            mat_pal_idx = vg[0]
            ge_rel      = vg[3]

            # resolve material / texture index
            mat_off = mat_tab + mat_pal_idx * 16
            tex_idx = 0
            if mat_off + 16 <= len(blob):
                buf.seek(mat_off)
                mat = struct.unpack("<4I", buf.read(16))
                tex_idx = mat[2]

            ge_abs = ge_base + ge_rel
            if ge_abs >= len(blob):
                continue
            buf.seek(ge_abs)
            verts_raw, faces_raw = run_ge(buf, mesh_scale)

            # run_ge returns None-padded vertex list; strip Nones
            verts = [v for v in verts_raw if v is not None]
            faces = [f for f in faces_raw]

            # scale the vertex positions by the per-mesh scale
            # (run_ge already multiplied by scale passed in, so don't double-apply)

            g = MeshGroup(
                index=draw_order,
                material=tex_idx,
                mesh_record=i,
                vertex_count=len(verts),
                face_count=len(faces),
                scale=mesh_scale,
                vertices=verts,
                faces=faces,
                enc=None,      # no in-place encode (v102 passthrough only)
                vg_rec=draw_order,
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
