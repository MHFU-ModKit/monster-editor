# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2013 Seth VanHeulen (mhff, https://github.com/svanheulen/mhff)
# Copyright 2026 sp00ktober
"""PMO geometry (PAC sub-1) <-> data model.

Decodes the MHFU/MHP2G *monster* PMO (magic 'pmo\\x00' ver '1.0\\x00') into
MeshGroups carrying full vertex/face geometry. The GE-display-list walk (`run_ge`)
is ported from tools/mhff/psp/pmo.py; the monster mesh-table walk mirrors
`convert_mhfu_monster_meshes` there. See docs/PMO_MODEL_FORMAT.md.

Geometry is rigid- or 2-bone-blend-skinned (VTYPE may set weight bits — big
monsters like Tigrex do; small monsters don't). Each mesh group binds to one bone
by draw order.

Encoder (Phase 3): `encode()` returns the source bytes unchanged unless
`model.edited` is set, in which case it re-encodes each group's vertex POSITIONS
(and normals/UVs) back into the existing vertex buffers **in place** — preserving
the VTYPE, indices, weights, colors, all header/mesh/material tables, and the file
layout. This is byte-identical for an unedited model and correct for vertex moves
(reshaping). It deliberately does NOT change topology (vertex/face counts): adding
or removing geometry needs a from-scratch GE-list rebuild (the documented stretch),
and is reported as an error rather than silently corrupting the file.
"""
from __future__ import annotations

import array
import io
import struct
from typing import List, Optional

from .model import MeshGroup, Model

# python struct char + byte size for a PSP component encoding
_COMP = {"b": ("b", 1), "B": ("B", 1), "h": ("h", 2), "H": ("H", 2), "f": ("f", 4)}


def run_ge(buf: io.BytesIO, scale):
    """Walk one GE display list -> (vertices, faces, enc).

    `enc` describes the (single) vertex buffer for in-place re-encoding, or carries
    `plain=False` when the list isn't a simple single-VADDR group (then the encoder
    leaves it untouched). Ported verbatim from mhff pmo.py with field-offset capture
    added.
    """
    file_address = buf.tell()
    index_offset = 0
    vertices: List[dict] = []
    faces: List[dict] = []
    vertex_address = index_address = vertex_format = None
    position_trans = normal_trans = color_trans = texture_trans = weight_trans = None
    index_format = face_order = None
    # in-place re-encode descriptor (positions/normals/uv field layout)
    enc = {"plain": True, "vaddr": None, "vsize": None,
           "pos": None, "nrm": None, "tex": None}
    vaddr_count = 0
    while True:
        command = array.array("I", buf.read(4))[0]
        ct = command >> 24
        if ct == 0x00:                                   # NOP
            pass
        elif ct == 0x01:                                 # VADDR
            if vertex_address is not None:
                index_offset = len(vertices)
            vertex_address = file_address + (command & 0xffffff)
            vaddr_count += 1
            enc["vaddr"] = vertex_address
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
            enc["vsize"] = vertex_size
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
                # A triangle LIST does not alternate winding — only a strip/fan does.
                # The old parity term made every other triangle of a list come out
                # mirrored. Retail never showed it (all 479 shipped lists are a single
                # triangle, so i is always 0), but the topology-grow path emits
                # multi-triangle lists and tripped it on the first quad.
                swap = face_order if primative_type == 3 else (i + face_order) % 2
                if swap:
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
            # Field byte offsets must match the parser's NATIVE struct alignment
            # (it unpacks `vertex_format` without a '<', so e.g. a `3h` position is
            # padded to a 2-byte boundary). Track an aligned running offset.
            off = 0

            def _field(count, char):           # -> aligned start offset; advances off
                nonlocal off
                a = _COMP[char][1]             # native alignment == component size
                start = (off + a - 1) & ~(a - 1)
                off = start + count * _COMP[char][1]
                return start

            weight = (command >> 9) & 3
            if weight != 0:
                count = ((command >> 14) & 7) + 1
                wc = (None, "B", "H", "f")[weight]
                vertex_format += str(count) + wc
                weight_trans = (None, 0x80, 0x8000, 1)[weight]
                _field(count, wc)
            bypass_transform = (command >> 23) & 1
            texture = command & 3
            if texture != 0:
                tc = (None, "B", "H", "f")[texture]
                vertex_format += (None, "2B", "2H", "2f")[texture]
                texture_trans = 1 if bypass_transform else (None, 0x80, 0x8000, 1)[texture]
                enc["tex"] = (_field(2, tc), tc, texture_trans)
            color = (command >> 2) & 7
            if color != 0:
                cfmt = (None, None, None, None, "H", "H", "H", "I")[color]
                vertex_format += cfmt
                color_trans = (None, None, None, None,
                               "rgb565", "rgba5", "rgba4", "rgba8")[color]
                _field(1, cfmt)
            normal = (command >> 5) & 3
            if normal != 0:
                nc = (None, "b", "h", "f")[normal]
                vertex_format += (None, "3b", "3h", "3f")[normal]
                normal_trans = 1 if bypass_transform else (None, 0x7f, 0x7fff, 1)[normal]
                enc["nrm"] = (_field(3, nc), nc, normal_trans)
            position = (command >> 7) & 3
            if position != 0:
                if bypass_transform:
                    vertex_format += (None, "2bB", "2hH", "3f")[position]
                    position_trans = 1
                    enc["plain"] = False     # bypass layout — leave it to passthrough
                else:
                    pc = (None, "b", "h", "f")[position]
                    vertex_format += (None, "3b", "3h", "3f")[position]
                    position_trans = (None, 0x7f, 0x7fff, 1)[position]
                    enc["pos"] = (_field(3, pc), pc, position_trans)
            index_format = (None, "B", "H", "I")[(command >> 11) & 3]
            if (command >> 18) & 7 > 0:
                raise ValueError("Can not handle morphing")
        elif ct in (0x13, 0x14):                         # offset/origin BASE
            pass
        elif ct == 0x9b:                                 # FFACE
            face_order = command & 1
        else:
            raise ValueError("Unknown GE command: 0x{:02X}".format(ct))
    if vaddr_count != 1 or enc["pos"] is None:
        enc["plain"] = False
    enc["vcount"] = len(vertices)
    return vertices, faces, enc


def _walk(blob, header, scale, stride):
    """Walk the mesh table with a given record stride; return (groups, score).

    Strides: 0x20 (legacy/MH2 — Tigrex & most big monsters: mat base mh[5],
    vg_count mh[6], vg_start mh[7]) and 0x18 (small-mon: vg_count/start u16 @+0x10).
    `score` = total decoded vertices; an overrun returns (None, -1).
    """
    fsz = len(blob)
    nmesh, mesh_tab, vg_tab, mat_tab, ge_base = (
        header[5], header[7], header[8], header[11], header[12])
    if mesh_tab + nmesh * stride > fsz:
        return None, -1, set()
    # 🔴 For the 0x18-stride layout the material is NOT `mat_tab[vg[0]]`. `vg[0]` is the
    # group's ordinal within its mesh (0..count-1, always), and the real binding is a
    # u8 PER VGROUP at header[9] (`vg_end`, the bytes between the vgroup table and the
    # material table): `material_index = mat_map[vgroup_index]`, with `header[6]` the
    # material count it indexes. Verified 487/487 stage PMOs (every byte < header[6],
    # zero padding to `mat_tab`); only 78 of them are the identity map the old lookup
    # assumed, and 409 drew some group with the wrong texture. docs/PMO_MODEL_FORMAT.md.
    mat_map_off, nmat = header[9], header[6]
    buf = io.BytesIO(blob)
    groups: List[MeshGroup] = []
    total = 0
    draw = 0                       # global vertex-group draw order == bind index
    seen_vg = set()                # vgroup indices the mesh table referenced
    try:
        for i in range(nmesh):
            m = mesh_tab + i * stride
            if stride == 0x20:
                mh = struct.unpack_from("2f2I4H2I", blob, m)
                mat_base, vg_count, vg_start = mh[5], mh[6], mh[7]
            else:
                vg_count, vg_start = struct.unpack_from("2H", blob, m + 0x10)
                mat_base = None                       # -> the per-vgroup map
            for j in range(vg_count):
                vgi = vg_start + j
                vo = vg_tab + vgi * 0x10
                if vo + 0x10 > fsz:
                    return None, -1, set()
                vg = struct.unpack_from("2BH3I", blob, vo)
                material = 0
                if mat_base is None:
                    mi = blob[mat_map_off + vgi] if mat_map_off + vgi < fsz else nmat
                    mo = mat_tab + mi * 16 if mi < nmat else fsz
                else:
                    mo = mat_tab + (mat_base + vg[0]) * 16
                if mo + 16 <= fsz:
                    material = struct.unpack_from("4I", blob, mo)[2]
                ge = ge_base + vg[3]
                if ge >= fsz:
                    return None, -1, set()
                buf.seek(ge)
                gv, gf, enc = run_ge(buf, scale)
                g = MeshGroup(
                    index=draw, material=material, mesh_record=i,
                    vertex_count=len(gv), face_count=len(gf), scale=scale,
                    vertices=gv, faces=gf, vg_rec=vgi,
                )
                g.enc = enc                       # private re-encode descriptor
                groups.append(g)
                total += len(gv)
                draw += 1
                seen_vg.add(vgi)
    except (struct.error, ValueError, IndexError):
        return None, -1, set()
    return groups, total, seen_vg


def _append_unreferenced_vgroups(blob, header, scale, groups, seen_vg):
    """Some big-monster PMOs (e.g. the native-quest Tigrex file_06185) split their
    geometry into TWO mesh sets: header[5] meshes index only a SUBSET of the vgroup
    table (the extremities), while the body lives in vgroups the header[5] table
    never references (a 2nd set described by header[6] + tables at header[9]/[10]).
    The full vgroup table spans [header[8], header[9]); enumerate it directly and
    append any vgroup the mesh-table walk missed, so the WHOLE model decodes.
    Geometry-complete + round-trip-safe (each appended group carries its own enc);
    material is best-effort (the vgroup's own material byte via mat_tab)."""
    fsz = len(blob)
    vg_tab, t9, mat_tab, ge_base = header[8], header[9], header[11], header[12]
    if not (vg_tab < t9 <= fsz):
        return groups
    nvg = (t9 - vg_tab) // 0x10
    buf = io.BytesIO(blob)
    draw = max((g.index for g in groups), default=-1) + 1
    for vgi in range(nvg):
        if vgi in seen_vg:
            continue
        vo = vg_tab + vgi * 0x10
        if vo + 0x10 > fsz:
            break
        try:
            vg = struct.unpack_from("2BH3I", blob, vo)
            ge = ge_base + vg[3]
            if not (ge_base <= ge < fsz):
                continue
            buf.seek(ge)
            gv, gf, enc = run_ge(buf, scale)
        except (struct.error, ValueError, IndexError):
            continue
        if not gv:
            continue
        material = 0
        mo = mat_tab + vg[0] * 16
        if mo + 16 <= fsz:
            material = struct.unpack_from("4I", blob, mo)[2]
        g = MeshGroup(index=draw, material=material, mesh_record=-1,
                      vertex_count=len(gv), face_count=len(gf), scale=scale,
                      vertices=gv, faces=gf, vg_rec=vgi)
        g.enc = enc
        groups.append(g)
        draw += 1
    return groups


def _attach_influences(blob: bytes, header, groups) -> int:
    """Give every vertex its AUTHENTIC ``(bone, weight)`` list from the PMO's own
    bone palette. Returns the number of groups resolved (0 = nothing to attach).

    🔴 Draw order is NOT the bind index for a native MHFU monster. Each vgroup
    record carries ``(bone_count, cumulative_bone_count)`` into the skeleton's
    ``Weight[]`` patch list at header[10]; replaying that list in vgroup-table
    order yields a per-vgroup palette, and a vertex's VTYPE weights index INTO
    that palette. `pmo_p3rd` has always done this for MHP3rd; the MHFU walk never
    did, so every consumer fell back to "group N is welded to bone N" — which for
    file_06185 welds 166 of 214 groups onto one bone. That is the single reason
    the offline Blender render could never be trusted against the game.

    Best-effort: a PMO without a palette (small monsters, no weight bits) is left
    alone and the caller keeps its rigid-by-draw-order fallback.
    """
    from .pmo_skin import _resolve_running_palette
    vg_tab, t9, skel_off = header[8], header[9], header[10]
    if not (0 < vg_tab < t9 <= len(blob)) or not skel_off:
        return 0
    nvg = (t9 - vg_tab) // 0x10
    try:
        recs = [struct.unpack_from("<2BH3I", blob, vg_tab + i * 0x10)
                for i in range(nvg)]
        pal_len = (recs[-1][2] + recs[-1][1]) if recs else 0
        if not pal_len or skel_off + pal_len * 2 > len(blob):
            return 0
        patches = [struct.unpack_from("<2B", blob, skel_off + i * 2)
                   for i in range(pal_len)]
        palettes = _resolve_running_palette(patches, recs)
    except (struct.error, IndexError):
        return 0
    done = 0
    for g in groups:
        if not (0 <= g.vg_rec < len(palettes)):
            continue
        pal = palettes[g.vg_rec]
        if not pal:
            continue
        g.boneref = pal[0]
        for v in g.vertices:
            w = v.get("weights")
            if w:
                v["influences"] = [(pal[k] if k < len(pal) else -1, w[k])
                                   for k in range(len(w))]
            else:
                v["influences"] = [(pal[0], 1.0)]
        done += 1
    return done


def parse(blob: bytes) -> Model:
    """Decode a big-monster PMO into MeshGroups (best-effort geometry).

    Header struct at offset 8 (`I4f2H8I`). Mesh-table stride varies (0x20/0x18) —
    both walks are tried and the valid one with the most decoded geometry wins; the
    winning stride is recorded on the model for the encoder.

    Vertices additionally carry ``influences`` — the REAL skin — whenever the file
    has a bone palette (`_attach_influences`).
    """
    type_, version = struct.unpack_from("4s4s", blob, 0)
    if type_ != b"pmo\x00":
        raise ValueError("not a PMO (magic=%r)" % type_)
    model = Model(magic=type_, version=version, raw=blob)
    if 8 + 0x38 > len(blob):
        return model
    header = struct.unpack_from("I4f2H8I", blob, 8)
    model.scale = tuple(header[2:5])
    best, best_score, best_stride, best_seen = None, 0, None, set()
    for stride in (0x20, 0x18):
        groups, score, seen = _walk(blob, header, model.scale, stride)
        if groups is not None and score > best_score:
            best, best_score, best_stride, best_seen = groups, score, stride, seen
    if best is not None:
        # recover any 2nd-set vgroups the mesh table didn't reference (split-mesh
        # monsters like file_06185 — see _append_unreferenced_vgroups).
        best = _append_unreferenced_vgroups(blob, header, model.scale, best, best_seen)
        _attach_influences(blob, header, best)
        model.mesh_groups = best
        model.stride = best_stride
    return model


# --------------------------------------------------------------------------- #
# Encoder
# --------------------------------------------------------------------------- #
def _quant(value, char):
    """Quantize a float to one VTYPE component (clamped to its integer range)."""
    if char == "f":
        return float(value)
    q = int(round(value))
    lim = {"b": (-128, 127), "B": (0, 255),
           "h": (-32768, 32767), "H": (0, 65535)}[char]
    return max(lim[0], min(lim[1], q))


def _rewrite_group(out: bytearray, g: MeshGroup, scale):
    """Re-encode g's vertex positions/normals/UVs into `out` in place.

    Returns True if rewritten, False if the group isn't in-place-encodable
    (multi-VADDR / bypass layout — caller decides whether that's an error)."""
    enc = getattr(g, "enc", None)
    if not enc or not enc.get("plain"):
        return False
    vaddr, vsize = enc["vaddr"], enc["vsize"]
    if vaddr is None or vsize is None:
        return False
    if len(g.vertices) != enc.get("vcount", len(g.vertices)):
        raise ValueError(
            "mesh group %d: vertex count changed (%d -> %d) — topology edits need a "
            "GE-list rebuild (not yet supported); reshape with the same vertex count"
            % (g.index, enc["vcount"], len(g.vertices)))
    poff, pchar, ptrans = enc["pos"]
    for slot, v in enumerate(g.vertices):
        if v is None:
            continue
        base = vaddr + slot * vsize
        rx = _quant(v["x"] / scale[0] * ptrans, pchar)
        ry = _quant(v["y"] / scale[1] * ptrans, pchar)
        rz = _quant(v["z"] / scale[2] * ptrans, pchar)
        struct.pack_into("<3%s" % pchar, out, base + poff, rx, ry, rz)
        if enc.get("nrm") and ("i" in v):
            noff, nchar, ntrans = enc["nrm"]
            struct.pack_into("<3%s" % nchar, out, base + noff,
                             _quant(v["i"] * ntrans, nchar),
                             _quant(v["j"] * ntrans, nchar),
                             _quant(v["k"] * ntrans, nchar))
        if enc.get("tex") and ("u" in v):
            toff, tchar, ttrans = enc["tex"]
            struct.pack_into("<2%s" % tchar, out, base + toff,
                             _quant(v["u"] * ttrans, tchar),
                             _quant(v["v"] * ttrans, tchar))
    return True


def encode(model: Model) -> bytes:
    """Serialize a PMO from the data model.

    Unedited (`model.edited` false) -> the source bytes verbatim (byte-identical
    round-trip). Edited -> re-encode every in-place-capable group's vertex data
    into a copy of the source, preserving structure/weights/colors/indices/tables.
    A group whose vertex count changed, or that can't be re-encoded in place while
    edited, raises a clear error (topology rebuild is the documented Phase-3
    stretch; the Blender exporter never edits geometry, so the safe path is the
    default).
    """
    raw = model.raw or b""
    if not getattr(model, "edited", False) or not raw:
        return raw
    scale = getattr(model, "scale", None) or (
        model.mesh_groups[0].scale if model.mesh_groups else (1.0, 1.0, 1.0))
    out = bytearray(raw)
    for g in model.mesh_groups:
        if not _rewrite_group(out, g, scale):
            enc = getattr(g, "enc", None)
            raise ValueError(
                "mesh group %d cannot be re-encoded in place (%s); geometry edits "
                "on this group need a GE-list rebuild (not yet supported)"
                % (g.index, "no descriptor" if not enc else "multi-VADDR/bypass"))
    return bytes(out)
