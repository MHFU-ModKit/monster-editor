"""PMO geometry (PAC sub-1) -> data model.

Decodes the MHFU/MHP2G *monster* PMO (magic 'pmo\\x00' ver '1.0\\x00', 0x18-stride
mesh table) into MeshGroups carrying full vertex/face geometry. GE-display-list
walk (`run_ge`) is ported from tools/mhff/psp/pmo.py; the monster mesh-table walk
mirrors `convert_mhfu_monster_meshes` there. See docs/PMO_MODEL_FORMAT.md.

Monster geometry is rigid-skinned: vertex groups carry no per-vertex blend weights
(`weights` will be absent/zero); each mesh group binds to one bone by draw order.

`encode()` is a lossless passthrough until the Phase 3 PMO encoder lands.
"""
from __future__ import annotations

import array
import io
import struct
from typing import List

from .model import MeshGroup, Model


def run_ge(buf: io.BytesIO, scale):
    """Walk one GE display list -> (vertices, faces). Ported from mhff pmo.py."""
    file_address = buf.tell()
    index_offset = 0
    vertices: List[dict] = []
    faces: List[dict] = []
    vertex_address = index_address = vertex_format = None
    position_trans = normal_trans = color_trans = texture_trans = weight_trans = None
    index_format = face_order = None
    while True:
        command = array.array("I", buf.read(4))[0]
        ct = command >> 24
        if ct == 0x00:                                   # NOP
            pass
        elif ct == 0x01:                                 # VADDR
            if vertex_address is not None:
                index_offset = len(vertices)
            vertex_address = file_address + (command & 0xffffff)
        elif ct == 0x02:                                 # IADDR
            index_address = file_address + (command & 0xffffff)
        elif ct == 0x04:                                 # PRIM
            primative_type = (command >> 16) & 7
            index_count = command & 0xffff
            command_address = buf.tell()
            index = range(len(vertices) - index_offset,
                          len(vertices) + index_count - index_offset)
            if index_format is not None:
                index = array.array(index_format)
                buf.seek(index_address)
                index.fromfile(buf, index_count)
                index_address = buf.tell()
            vertex_size = struct.calcsize(vertex_format)
            for i in index:
                buf.seek(vertex_address + vertex_size * i)
                raw_vertex = list(struct.unpack(vertex_format, buf.read(vertex_size)))
                vertex = {}
                vertex["z"] = (raw_vertex.pop() / position_trans) * scale[2]
                vertex["y"] = (raw_vertex.pop() / position_trans) * scale[1]
                vertex["x"] = (raw_vertex.pop() / position_trans) * scale[0]
                if normal_trans is not None:
                    vertex["k"] = raw_vertex.pop() / normal_trans
                    vertex["j"] = raw_vertex.pop() / normal_trans
                    vertex["i"] = raw_vertex.pop() / normal_trans
                if color_trans is not None:
                    raw_vertex.pop()
                if texture_trans is not None:
                    vertex["v"] = raw_vertex.pop() / texture_trans
                    vertex["u"] = raw_vertex.pop() / texture_trans
                if weight_trans is not None:
                    vertex["weights"] = [w / weight_trans for w in raw_vertex]
                if len(vertices) <= (i + index_offset):
                    vertices.extend([None] * (i + index_offset + 1 - len(vertices)))
                vertices[i + index_offset] = vertex
            buf.seek(command_address)
            r = range(index_count - 2)
            if primative_type == 3:
                r = range(0, index_count, 3)
            for i in r:
                face = {"v3": index[i + 2] + index_offset}
                if ((i + face_order) % 2) or ((primative_type == 3) and face_order):
                    face["v2"] = index[i] + index_offset
                    face["v1"] = index[i + 1] + index_offset
                else:
                    face["v1"] = index[i] + index_offset
                    face["v2"] = index[i + 1] + index_offset
                faces.append(face)
        elif ct == 0x0b:                                 # RET
            break
        elif ct == 0x10:                                 # BASE
            pass
        elif ct == 0x12:                                 # VTYPE
            vertex_format = ""
            weight = (command >> 9) & 3
            if weight != 0:
                count = ((command >> 14) & 7) + 1
                vertex_format += str(count) + (None, "B", "H", "f")[weight]
                weight_trans = (None, 0x80, 0x8000, 1)[weight]
            bypass_transform = (command >> 23) & 1
            texture = command & 3
            if texture != 0:
                vertex_format += (None, "2B", "2H", "2f")[texture]
                texture_trans = 1 if bypass_transform else (None, 0x80, 0x8000, 1)[texture]
            color = (command >> 2) & 7
            if color != 0:
                vertex_format += (None, None, None, None, "H", "H", "H", "I")[color]
                color_trans = (None, None, None, None,
                               "rgb565", "rgba5", "rgba4", "rgba8")[color]
            normal = (command >> 5) & 3
            if normal != 0:
                vertex_format += (None, "3b", "3h", "3f")[normal]
                normal_trans = 1 if bypass_transform else (None, 0x7f, 0x7fff, 1)[normal]
            position = (command >> 7) & 3
            if position != 0:
                if bypass_transform:
                    vertex_format += (None, "2bB", "2hH", "3f")[position]
                    position_trans = 1
                else:
                    vertex_format += (None, "3b", "3h", "3f")[position]
                    position_trans = (None, 0x7f, 0x7fff, 1)[position]
            index_format = (None, "B", "H", "I")[(command >> 11) & 3]
            if (command >> 18) & 7 > 0:
                raise ValueError("Can not handle morphing")
        elif ct in (0x13, 0x14):                         # offset/origin BASE
            pass
        elif ct == 0x9b:                                 # FFACE
            face_order = command & 1
        else:
            raise ValueError("Unknown GE command: 0x{:02X}".format(ct))
    return vertices, faces


def _walk(blob, header, scale, stride):
    """Walk the mesh table with a given record stride; return (groups, score).

    Two MHFU PMO mesh-table strides exist:
      * 0x20 (MH2/legacy, e.g. Tigrex): record '2f2I4H2I'; vg_count=mh[6],
        vg_start=mh[7], material base index mh[5] (material at mat_tab +
        (mh[5]+vg[0])*16).
      * 0x18 (small-monster, e.g. file_06139): vg_count=u16@+0x10,
        vg_start=u16@+0x12, no material base (material at mat_tab + vg[0]*16).
    The vertex-group record ('2BH3I') and GE display list are common to both.
    `score` = total decoded vertices; a layout that overruns returns (None, -1)
    so the caller can pick the stride that actually fits this file.
    """
    fsz = len(blob)
    nmesh, mesh_tab, vg_tab, mat_tab, ge_base = (
        header[5], header[7], header[8], header[11], header[12])
    if mesh_tab + nmesh * stride > fsz:
        return None, -1
    buf = io.BytesIO(blob)
    groups: List[MeshGroup] = []
    total = 0
    try:
        for i in range(nmesh):
            m = mesh_tab + i * stride
            if stride == 0x20:
                mh = struct.unpack_from("2f2I4H2I", blob, m)
                mat_base, vg_count, vg_start = mh[5], mh[6], mh[7]
            else:
                vg_count, vg_start = struct.unpack_from("2H", blob, m + 0x10)
                mat_base = 0
            verts: List[dict] = []
            faces: List[dict] = []
            material = 0
            for j in range(vg_count):
                vo = vg_tab + (vg_start + j) * 0x10
                if vo + 0x10 > fsz:
                    return None, -1
                vg = struct.unpack_from("2BH3I", blob, vo)
                mo = mat_tab + (mat_base + vg[0]) * 16
                if mo + 16 <= fsz:
                    material = struct.unpack_from("4I", blob, mo)[2]
                ge = ge_base + vg[3]
                if ge >= fsz:
                    return None, -1
                buf.seek(ge)
                gv, gf = run_ge(buf, scale)
                base = len(verts)
                verts.extend(gv)
                for f in gf:
                    faces.append({"v1": f["v1"] + base, "v2": f["v2"] + base,
                                  "v3": f["v3"] + base})
            groups.append(MeshGroup(
                index=i, material=material, vertex_count=len(verts),
                face_count=len(faces), scale=scale, vertices=verts, faces=faces,
            ))
            total += len(verts)
    except (struct.error, ValueError, IndexError):
        return None, -1
    return groups, total


def parse(blob: bytes) -> Model:
    """Decode a big-monster PMO into MeshGroups.

    The header struct lives at offset 8 (after the 8-byte 'pmo\\x00','1.0\\x00'
    magic). The mesh-table stride varies per file (0x20 legacy vs 0x18 small-mon),
    so both walks are attempted and the one that fully validates with the most
    decoded geometry wins. Geometry decode is best-effort: if neither layout fits
    (a non-monster or as-yet-unknown variant) `mesh_groups` is left empty rather
    than raising, so the PAC round-trip (raw passthrough) is never blocked.
    """
    type_, version = struct.unpack_from("4s4s", blob, 0)
    if type_ != b"pmo\x00":
        raise ValueError("not a PMO (magic=%r)" % type_)
    model = Model(magic=type_, version=version, raw=blob)
    if 8 + 0x38 > len(blob):
        return model
    header = struct.unpack_from("I4f2H8I", blob, 8)
    scale = header[2:5]
    best, best_score = None, 0
    for stride in (0x20, 0x18):
        groups, score = _walk(blob, header, scale, stride)
        if groups is not None and score > best_score:
            best, best_score = groups, score
    if best is not None:
        model.mesh_groups = best
    return model


def encode(model: Model) -> bytes:
    """Lossless passthrough (Phase 0). Phase 3 will rebuild the GE lists."""
    return model.raw
