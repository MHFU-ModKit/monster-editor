"""MHFU/MHP2G monster PMO *blend skinning* reader + encoder.

This is the skinning-aware companion to `pmo.py` (which handles byte-identical
reshape) and `pmo_topology.py` (which grows geometry). It decodes — and re-emits —
the **native multi-bone blend skinning** every MHFU big monster actually uses.

The skinning model (decoded from native `file_06185` + AsteriskAmpersand's
PMO-Importer, the authority — see docs/PMO_MODEL_FORMAT.md "skinning"):

* The vgroup record `2BH3I` =
  `(materialOffset:u8, boneCount:u8, cumulativeBoneCount:u16, meshOffset, vertexOffset, indexOffset)`.
  (NOTE: long mislabeled `vg[1]=unk`, `vg[2]=boneref` — that was WRONG; vg[2] is a
  *cumulative count*, not a bone.)
* The PMO header field 10 (`skeletonOffset`) points at the **bone palette** =
  an array of `Weight{slot:u8, bone:u8}`.
* The engine walks vgroups in index order maintaining a RUNNING palette
  `aux[slot] = bone`. Each vgroup consumes its `boneCount` Weight entries (at
  `palette[cumulativeBoneCount : +boneCount]`), patching `aux`. `boneCount == 0`
  reuses the prior `aux` (state persists across vgroups).
* Each vertex carries `weightCount` per-vertex weights (VTYPE weight field, 8-bit
  `/0x80`); weight[k] blends `aux[k]` (the bone currently in slot k). So a vertex's
  bone influences = `[(aux[k], weight[k]) for k in range(weightCount)]`.

`bone` indexes the 0xC0000000 skeleton joints (see skeleton.py).
"""
from __future__ import annotations

import io
import struct
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from . import pmo as _pmo


# --------------------------------------------------------------------------- #
# data model
# --------------------------------------------------------------------------- #
@dataclass
class SkinVGroup:
    """One vgroup (tristrip/submesh) with resolved blend skinning."""
    index: int                       # vgroup index (== engine processing order)
    bone_count: int                  # this vgroup's palette-patch count
    cum_bone_count: int              # offset into the global palette array
    palette: List[int]               # effective aux[slot]->bone after this vgroup
    material: int
    mesh_offset: int
    vertex_offset: int
    index_offset: int
    vertices: List[dict]             # from run_ge: x/y/z (+ i/j/k, u/v, weights)
    faces: List[dict]
    # per-vertex resolved (bone, weight) influences, parallel to `vertices`
    influences: List[List[Tuple[int, float]]] = field(default_factory=list)


@dataclass
class SkinModel:
    magic: bytes
    version: bytes
    scale: Tuple[float, float, float]
    vgroups: List[SkinVGroup]
    palette_raw: List[Tuple[int, int]]      # the raw skeleton (slot,bone) patch list
    header: tuple
    raw: bytes


# --------------------------------------------------------------------------- #
# header / table helpers
# --------------------------------------------------------------------------- #
_HDR = "I4f2H8I"          # at file offset 8


def _mesh_header_stride(version: bytes) -> int:
    # v1.0 (FU): uvScale[2]f + unkn[8]B + 4H  = 24
    # 102 (P3rd): scale[4]f + uvScale[2]f + (uvOffset[2]f+unkn[2]i) + 4H = 48
    return 48 if version == b"102\x00" else 24


def _submesh_count(blob: bytes, mesh_hdr_off: int, mesh_count: int,
                   stride: int) -> int:
    """Total vgroups = sum of each mesh header's subMeshCount (the last 2 u16 of
    each header are subMeshCount, cumulativeSubmeshCount)."""
    total = 0
    for m in range(mesh_count):
        # subMeshCount is the 3rd of the trailing four u16
        sub = struct.unpack_from("<H", blob, mesh_hdr_off + m * stride + stride - 4)[0]
        total += sub
    return total


def _resolve_running_palette(patches: List[Tuple[int, int]],
                             vg_records: List[tuple]) -> List[List[int]]:
    """Replay the running aux palette in vgroup index order.

    `patches` = the skeleton Weight[] list (slot, bone). `vg_records[i]` =
    (..., bone_count, cum_bone_count, ...). Returns, per vgroup, the effective
    `aux` as a dense `[bone for slot 0..maxslot]` list (slots never set -> -1).
    """
    aux: dict = {}
    out: List[List[int]] = []
    for rec in vg_records:
        bone_count, cum = rec[1], rec[2]
        for k in range(bone_count):
            slot, bone = patches[cum + k]
            aux[slot] = bone
        if aux:
            dense = [aux.get(s, -1) for s in range(max(aux) + 1)]
        else:
            dense = []
        out.append(dense)
    return out


# --------------------------------------------------------------------------- #
# reader
# --------------------------------------------------------------------------- #
def read(blob: bytes) -> SkinModel:
    """Decode a monster PMO into vgroups with resolved per-vertex blend skinning."""
    magic, version = struct.unpack_from("4s4s", blob, 0)
    if magic != b"pmo\x00":
        raise ValueError("not a PMO (magic=%r)" % magic)
    header = struct.unpack_from(_HDR, blob, 8)
    scale = tuple(header[2:5])
    mesh_count = header[5]
    mesh_hdr_off = header[7]
    vg_hdr_off = header[8]
    skel_off = header[10]
    mesh_data_off = header[12]

    stride = _mesh_header_stride(version)
    nvg = _submesh_count(blob, mesh_hdr_off, mesh_count, stride)

    vg_records = [struct.unpack_from("<2BH3I", blob, vg_hdr_off + i * 16)
                  for i in range(nvg)]

    # palette length = last vgroup's cumulative + its boneCount
    pal_len = (vg_records[-1][2] + vg_records[-1][1]) if vg_records else 0
    patches = [struct.unpack_from("<2B", blob, skel_off + i * 2)
               for i in range(pal_len)]
    eff_palettes = _resolve_running_palette(patches, vg_records)

    buf = io.BytesIO(blob)
    vgroups: List[SkinVGroup] = []
    for i, rec in enumerate(vg_records):
        material_off, bone_count, cum, mesh_o, vtx_o, idx_o = rec
        buf.seek(mesh_data_off + mesh_o)
        verts, faces, _enc = _pmo.run_ge(buf, scale)
        pal = eff_palettes[i]
        infl: List[List[Tuple[int, float]]] = []
        for v in verts:
            w = v.get("weights") if v else None
            if w:
                vi = [(pal[k] if k < len(pal) else -1, w[k])
                      for k in range(len(w))]
            else:
                # no per-vertex weights -> rigid on slot 0 (palette[0]) weight 1
                vi = [(pal[0] if pal else -1, 1.0)]
            infl.append(vi)
        vgroups.append(SkinVGroup(
            index=i, bone_count=bone_count, cum_bone_count=cum, palette=pal,
            material=material_off, mesh_offset=mesh_o, vertex_offset=vtx_o,
            index_offset=idx_o, vertices=verts, faces=faces, influences=infl))

    return SkinModel(magic=magic, version=version, scale=scale, vgroups=vgroups,
                     palette_raw=patches, header=header, raw=blob)


# --------------------------------------------------------------------------- #
# encoder
# --------------------------------------------------------------------------- #
def _align(n: int, a: int) -> int:
    return (n + a - 1) & ~(a - 1)


def _clamp(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


def _vtype_word(nw: int, idx16: bool) -> int:
    """VTYPE GE command for the native monster vertex format:
    8-bit weight (nw), 16-bit UV, 8-bit normal, 16-bit position, 8/16-bit index."""
    bits = 0x0b22                       # tex=2,nrm=1,pos=2,wt=1,idx=1 (base)
    if idx16:
        bits = (bits & ~(3 << 11)) | (2 << 11)   # idx=2 (16-bit)
    bits |= (nw - 1) << 14              # weightCount
    return 0x12000000 | bits


def _vertex_fmt(nw: int) -> str:
    # SAME order/components the reader (pmo.run_ge) builds; native alignment.
    return "%dB2H3b3h" % nw


def _vgroup_palette(vg: SkinVGroup) -> List[int]:
    """Distinct bones influencing this vgroup, in first-seen order (== slot order)."""
    seen: List[int] = []
    for infl in vg.influences:
        for b, w in infl:
            if w != 0 and b >= 0 and b not in seen:
                seen.append(b)
    if not seen:
        seen = [0]
    return seen


def _encode_block(vg: SkinVGroup, scale) -> Tuple[bytes, int, List[Tuple[int, int]]]:
    """Serialize one vgroup's [GE list | vertex buffer | index buffer] block.

    Returns (block_bytes, vertex_count, palette_patches) where palette_patches is
    the list of (slot, bone) for the skeleton section."""
    palette = _vgroup_palette(vg)
    if len(palette) > 8:
        raise ValueError("vgroup %d needs %d bones (>8) — split required"
                         % (vg.index, len(palette)))
    nw = len(palette)
    bone_to_slot = {b: i for i, b in enumerate(palette)}
    verts = vg.vertices
    nverts = len(verts)
    idx16 = nverts > 256
    vfmt = _vertex_fmt(nw)
    vsize = struct.calcsize(vfmt)

    # vertex buffer
    vbuf = bytearray()
    for vi, v in enumerate(verts):
        wraw = [0] * nw
        for b, w in vg.influences[vi]:
            if b in bone_to_slot and w != 0:
                wraw[bone_to_slot[b]] += int(round(w * 0x80))
        wraw = [_clamp(x, 0, 255) for x in wraw]
        u = v.get("u", 0.0); vv = v.get("v", 0.0)
        ur = _clamp(int(round(u * 0x8000)), 0, 0xFFFF)
        vr = _clamp(int(round(vv * 0x8000)), 0, 0xFFFF)
        ni = _clamp(int(round(v.get("i", 0.0) * 0x7f)), -128, 127)
        nj = _clamp(int(round(v.get("j", 0.0) * 0x7f)), -128, 127)
        nk = _clamp(int(round(v.get("k", 1.0) * 0x7f)), -128, 127)
        px = _clamp(int(round(v["x"] / scale[0] * 0x7fff)), -32768, 32767)
        py = _clamp(int(round(v["y"] / scale[1] * 0x7fff)), -32768, 32767)
        pz = _clamp(int(round(v["z"] / scale[2] * 0x7fff)), -32768, 32767)
        vbuf += struct.pack(vfmt, *wraw, ur, vr, ni, nj, nk, px, py, pz)

    # index buffer: per-face triangle strips (prim=4, count=3) -> flat indices
    ichar = "H" if idx16 else "B"
    ibuf = bytearray()
    for f in vg.faces:
        ibuf += struct.pack("<3%s" % ichar, f["v1"], f["v2"], f["v3"])

    # GE command list (native sequence)
    nfaces = len(vg.faces)
    cmds = [0x14000000, 0x10000000]            # ORIGIN, BASE
    # placeholders for IADDR/VADDR (need offsets first); insert after we know sizes
    # layout: [cmds][vbuf @ vaddr][ibuf @ iaddr], vaddr/iaddr relative to block start
    n_prim = nfaces
    # command count: ORIGIN,BASE,IADDR,VADDR,VTYPE,FFACE, n_prim*PRIM, OFFADDR,RET
    cmd_count = 6 + n_prim + 2
    cmds_size = cmd_count * 4
    vaddr = _align(cmds_size, 16)
    iaddr = _align(vaddr + len(vbuf), 4)
    words = [0x14000000, 0x10000000,
             0x02000000 | (iaddr & 0xFFFFFF),
             0x01000000 | (vaddr & 0xFFFFFF),
             _vtype_word(nw, idx16),
             0x9b000000]
    for _ in range(n_prim):
        words.append(0x04000000 | (4 << 16) | 3)   # prim=4 (strip), count=3
    words.append(0x13000000)                        # OFFADDR
    words.append(0x0b000000)                         # RET
    block = bytearray()
    for w in words:
        block += struct.pack("<I", w)
    block += b"\x00" * (vaddr - len(block))
    block += vbuf
    block += b"\x00" * (iaddr - len(block))
    block += ibuf
    patches = [(s, palette[s]) for s in range(nw)]
    return bytes(block), nverts, patches


def encode(sm: SkinModel) -> bytes:
    """Re-serialize a SkinModel to a native MHFU monster PMO with blend skinning.

    Preserves the source's mesh-header / material-remap / material-data tables
    verbatim (so material/texture binding is unchanged); rebuilds the vgroup
    headers, the skeleton bone palette, and the mesh data (GE lists + vertex /
    index buffers) from the resolved geometry + per-vertex influences.
    Each vgroup re-declares its own full palette (slots 0..N-1).
    """
    h = sm.header
    version = sm.version
    scale = sm.scale
    mesh_count = h[5]
    mat_count = h[6]
    raw = sm.raw
    mesh_hdr_off = h[7]
    vg_hdr_off = h[8]
    mat_remap_off = h[9]
    skel_off = h[10]
    mat_data_off = h[11]
    mesh_data_off = h[12]

    stride = _mesh_header_stride(version)
    # preserved table slices
    mesh_hdr_blob = raw[mesh_hdr_off:mesh_hdr_off + mesh_count * stride]
    # material remap count = last mesh's cumMat + matCount
    last = struct.unpack_from("<H", raw, mesh_hdr_off + (mesh_count - 1) * stride + stride - 8)[0]
    last_cum = struct.unpack_from("<H", raw, mesh_hdr_off + (mesh_count - 1) * stride + stride - 6)[0]
    remap_count = last + last_cum
    mat_remap_blob = raw[mat_remap_off:mat_remap_off + remap_count]
    mat_data_blob = raw[mat_data_off:mat_data_off + mat_count * 16]

    clip = struct.unpack_from("<f", raw, 8 + 4)[0]
    return _assemble(version, scale, clip, mesh_count, mat_count,
                     mesh_hdr_blob, mat_remap_blob, mat_data_blob, sm.vgroups)


def _assemble(version, scale, clip, mesh_count, mat_count,
              mesh_hdr_blob, mat_remap_blob, mat_data_blob, vgroups) -> bytes:
    """Lay out a complete monster PMO from the (preserved or generated) tables +
    the vgroup geometry. Rebuilds vgroup headers, the skeleton bone palette and the
    mesh data (GE lists + vertex / index buffers); each vgroup re-declares its full
    palette (slots 0..N-1)."""
    blocks = []
    vg_records = []
    palette: List[Tuple[int, int]] = []
    cum_bone = 0
    cur = 0                              # running offset within mesh data
    for vg in vgroups:
        block, _nv, patches = _encode_block(vg, scale)
        block_off = cur
        blocks.append(block)
        cur = _align(cur + len(block), 16)
        bone_count = len(patches)
        iaddr = struct.unpack_from("<I", block, 8)[0] & 0xFFFFFF
        vaddr = struct.unpack_from("<I", block, 12)[0] & 0xFFFFFF
        vg_records.append((vg.material & 0xFF, bone_count, cum_bone,
                           block_off, block_off + vaddr, block_off + iaddr))
        for s, b in patches:
            palette.append((s, b))
        cum_bone += bone_count
    mesh_data = bytearray()
    for block in blocks:
        mesh_data += block
        mesh_data += b"\x00" * (_align(len(mesh_data), 16) - len(mesh_data))

    out = bytearray()
    out += b"pmo\x00" + version
    hdr_fixed_off = len(out)
    out += b"\x00" * 48
    out += b"\x00" * (_align(len(out), 16) - len(out))   # -> 0x40

    o_mesh_hdr = len(out); out += mesh_hdr_blob
    out += b"\x00" * (_align(len(out), 16) - len(out))
    o_vg_hdr = len(out)
    for rec in vg_records:
        out += struct.pack("<2BH3I", *rec)
    out += b"\x00" * (_align(len(out), 16) - len(out))
    o_mat_remap = len(out); out += mat_remap_blob
    out += b"\x00" * (_align(len(out), 16) - len(out))
    o_skel = len(out)
    for s, b in palette:
        out += struct.pack("<2B", s, b)
    out += b"\x00" * (_align(len(out), 16) - len(out))
    o_mat_data = len(out); out += mat_data_blob
    out += b"\x00" * (_align(len(out), 16) - len(out))
    o_mesh_data = len(out); out += mesh_data

    struct.pack_into("<I4f2H8I", out, hdr_fixed_off,
                     len(out), clip, scale[0], scale[1], scale[2],
                     mesh_count, mat_count,
                     o_mesh_hdr, o_vg_hdr, o_mat_remap, o_skel, o_mat_data, o_mesh_data,
                     0, 0)
    return bytes(out)


def build(scale, vgroups, materials, version=b"1.0\x00", clip=0.0) -> bytes:
    """Build a native MHFU monster PMO FROM SCRATCH (no source PMO to preserve).

    `vgroups`  - list of SkinVGroup (vertices/faces/influences + `.material` =
                 material index 0..len(materials)-1).
    `materials`- list of dicts {`texID`:int, optional `rgba`,`shadow`,`unkn`}.
    Generates a single mesh header containing all vgroups as submeshes, an identity
    material remap, and the material-data table; then assembles via `_assemble`.
    Use this for ported monsters whose geometry comes from another game.
    """
    nvg = len(vgroups)
    nmat = max(1, len(materials))
    # one MeshHeader (v1.0 stride 24): uvScale[2]=1, unkn[8]=0, matCount, cumMat, subMesh, cumSub
    mesh_hdr = struct.pack("<2f8B4H", 1.0, 1.0, *([0] * 8), nmat, 0, nvg, 0)
    if nmat > 256:
        raise ValueError("too many materials (%d > 256)" % nmat)
    mat_remap = bytes(range(nmat))                         # identity
    mat_data = bytearray()
    for m in (materials or [{"texID": 0}]):
        rgba = tuple(m.get("rgba", (255, 255, 255, 255)))
        shadow = tuple(m.get("shadow", (0, 0, 0, 0)))
        mat_data += struct.pack("<4B4BiI", *rgba, *shadow, int(m["texID"]), 0)
    return _assemble(version, scale, clip, 1, nmat,
                     mesh_hdr, mat_remap, bytes(mat_data), vgroups)


# --------------------------------------------------------------------------- #
# auto-skinning (derive blend weights from geometry + skeleton)
# --------------------------------------------------------------------------- #
def _tree_neighborhoods(parents, hops):
    """For each bone, the set of bones within ``hops`` tree edges (parent/child).

    ``parents[i]`` = parent index of bone i (-1 = root). Returns ``list[set[int]]``
    where entry i = {i} plus every bone reachable from i in <= hops steps along the
    undirected skeleton tree. Used to keep a vertex's blend bones anatomically
    connected (a tail vertex blends only adjacent tail joints, never a euclidean-
    near leg bone) — fixes cross-region scramble (e.g. the Brute tail).
    """
    n = len(parents)
    adj = [set() for _ in range(n)]
    for i, p in enumerate(parents):
        if 0 <= p < n:
            adj[i].add(p)
            adj[p].add(i)
    nbr = []
    for i in range(n):
        seen = {i}
        frontier = {i}
        for _ in range(hops):
            nxt = set()
            for f in frontier:
                nxt |= adj[f]
            nxt -= seen
            seen |= nxt
            frontier = nxt
            if not frontier:
                break
        nbr.append(seen)
    return nbr


def auto_skin(mesh_groups, bone_world, materials_of=None, nb=3, max_pal=8,
              parents=None, hops=2):
    """Derive smooth blend skinning for a mesh that has NO source weights.

    For a ported monster whose source model is rigid-piece (no per-vertex weights
    — e.g. MHP3rd monsters), weight each vertex to its ``nb`` nearest skeleton
    bones (inverse-distance), instead of one rigid bone. This removes the
    rigid-binding splay at extreme animation poses (e.g. the Brute's wings during
    a roar). Per-vgroup palette is capped at ``max_pal`` (PSP's 8-matrix limit):
    the most-influential bones are kept and weights re-normalised over them.

    Chain-aware mode (``parents`` given): each vertex is first bound to its single
    nearest bone (the *primary*), then its blend partners are chosen ONLY from the
    primary's tree-neighborhood (bones within ``hops`` skeleton edges). This keeps
    the blend anatomically connected — a tail vertex blends adjacent tail joints,
    not a leg bone that merely happens to be euclidean-near — fixing the tail
    scramble + any cross-region bleed. Without ``parents`` it falls back to the
    plain nearest-``nb``-anywhere behavior (backward compatible).

    Parameters
    ----------
    mesh_groups : list of model.MeshGroup (vertices with x/y/z[/u/v/i/j/k], faces).
    bone_world  : list[(x,y,z)] of each bone's BIND-WORLD position (same space as
                  the verts). Compute via bone_match.bind_world_positions.
    materials_of: callable(group)->material index, or None (-> 0).
    parents     : optional list[int] of each bone's parent index (-1 = root) for
                  chain-aware weighting. None -> nearest-anywhere.
    hops        : tree radius for the chain-aware candidate set (default 2).
    Returns a list of SkinVGroup ready for ``build``/``encode``.
    """
    import math
    bw = list(bone_world)
    nbr = _tree_neighborhoods(parents, hops) if parents is not None else None
    out = []
    for g in mesh_groups:
        infl = []
        for v in g.vertices:
            ds = [((v["x"] - b[0]) ** 2 + (v["y"] - b[1]) ** 2 + (v["z"] - b[2]) ** 2, i)
                  for i, b in enumerate(bw)]
            ds.sort()
            if nbr is not None:
                primary = ds[0][1]                 # single nearest bone
                cand = nbr[primary]
                near = [(d, i) for d, i in ds if i in cand][:nb]
            else:
                near = ds[:nb]
            ws = [(i, 1.0 / (math.sqrt(d) + 1e-3)) for d, i in near]
            s = sum(w for _, w in ws) or 1.0
            infl.append([(i, w / s) for i, w in ws])
        # per-vgroup palette = most-influential bones, capped
        acc = {}
        for vi in infl:
            for b, w in vi:
                acc[b] = acc.get(b, 0.0) + w
        palette = [b for b, _ in sorted(acc.items(), key=lambda x: -x[1])[:max_pal]]
        pset = set(palette)
        infl2 = []
        for vi in infl:
            kept = [(b, w) for b, w in vi if b in pset]
            if not kept:
                kept = [(palette[0], 1.0)]
            s = sum(w for _, w in kept) or 1.0
            infl2.append([(b, w / s) for b, w in kept])
        out.append(SkinVGroup(
            index=len(out), bone_count=0, cum_bone_count=0, palette=palette,
            material=(materials_of(g) if materials_of else 0),
            mesh_offset=0, vertex_offset=0, index_offset=0,
            vertices=g.vertices, faces=g.faces, influences=infl2))
    return out


# --------------------------------------------------------------------------- #
# high-level: re-skin geometry onto a native frame's skeleton, splice into frame
# --------------------------------------------------------------------------- #
def _pac_subs(blob):
    n = struct.unpack_from("<I", blob, 0)[0]
    return [struct.unpack_from("<II", blob, 4 + i * 8) for i in range(n)]


def skeleton_tree(skel_blob):
    """(parents, local_pos) from a 0xC0000000 skeleton, scanning section magics
    (header/stride-robust; file_06185 has a 0x20 header, not 0x1C)."""
    magics = (0x40000001, 0x40000002)
    offs = [i for i in range(0, len(skel_blob) - 4, 4)
            if struct.unpack_from("<I", skel_blob, i)[0] in magics]
    parents, local = [], []
    for so in offs:
        _idx, par, _ch, _sib = struct.unpack_from("<4i", skel_blob, so + 0xC)
        parents.append(par)
        local.append(struct.unpack_from("<3f", skel_blob, so + 0x3C))
    return parents, local


def frame_skeleton(frame_blob, skel_sub_index=0):
    """(parents, local_pos, bind_world) for the frame PAC's skeleton sub."""
    from .bone_match import bind_world_positions
    so, ss = _pac_subs(frame_blob)[skel_sub_index]
    parents, local = skeleton_tree(frame_blob[so:so + ss])
    return parents, local, bind_world_positions(parents, local)


def splice_pmo_into_frame(frame_blob, pmo_bytes, pmo_sub_index=1):
    """Overwrite the frame PAC's PMO sub with ``pmo_bytes`` (zero-padded to the slot),
    keeping every other sub (skeleton/textures/animation/secondary) byte-identical.
    Returns a same-size PAC (the proven in-place inject path). Raises if the PMO
    exceeds the slot."""
    po, psz = _pac_subs(frame_blob)[pmo_sub_index]
    if len(pmo_bytes) > psz:
        raise ValueError("PMO %d B > frame slot %d B (lower nb / split vgroups, or "
                         "use the relocate inject path)" % (len(pmo_bytes), psz))
    out = bytearray(frame_blob)
    out[po:po + psz] = pmo_bytes + b"\x00" * (psz - len(pmo_bytes))
    return bytes(out)


def splice_skinned_pmo(frame_blob, geo_model, nb=3, hops=1,
                       pmo_sub_index=1, skel_sub_index=0):
    """Re-skin a geometry PMO onto the frame's skeleton and splice it into the frame.

    Shared core of the offline CLI (tools/build_brute_pac.py) and the Blender
    exporter, so both reproduce byte-for-byte:

      1. parse the frame's skeleton sub -> bone bind-world + parents,
      2. chain-aware blend-skin ``geo_model``'s geometry against it (auto_skin, hops),
      3. encode a native MHFU monster PMO reusing ``geo_model``'s OWN mesh-header /
         material tables + scale (so the Brute's textures/material binding survive —
         NOT the frame's native tables), and
      4. overwrite the frame's PMO sub (zero-padded), keeping all other subs identical.

    ``geo_model`` is a SkinModel (from ``read`` of the source Brute PMO) — only its
    geometry + tables are used; its skinning is discarded and re-derived. Returns
    ``(pac_bytes, stats_dict)``.
    """
    parents, _local, bw = frame_skeleton(frame_blob, skel_sub_index)
    vgs = auto_skin(geo_model.vgroups, bw, materials_of=lambda g: g.material,
                    nb=nb, max_pal=8, parents=parents, hops=hops)
    geo_model.vgroups = vgs
    pmo = encode(geo_model)                       # uses geo_model's tables, not frame's
    out = splice_pmo_into_frame(frame_blob, pmo, pmo_sub_index)
    stats = dict(bones=len(parents), vgroups=len(vgs),
                 verts=sum(len(g.vertices) for g in vgs),
                 pmo_bytes=len(pmo), slot=_pac_subs(frame_blob)[pmo_sub_index][1],
                 total=len(out),
                 avg_pal=sum(len(g.palette) for g in vgs) / max(1, len(vgs)))
    return out, stats


# --------------------------------------------------------------------------- #
# CLI / quick validation
# --------------------------------------------------------------------------- #
def _summary(sm: SkinModel) -> str:
    nverts = sum(len(g.vertices) for g in sm.vgroups)
    wc_hist: dict = {}
    bad_bone = 0
    bad_sum = 0
    maxbone = -1
    for g in sm.vgroups:
        for vi in g.influences:
            wc = len([1 for b, w in vi if w != 0])
            wc_hist[wc] = wc_hist.get(wc, 0) + 1
            s = sum(w for b, w in vi)
            if abs(s - 1.0) > 0.05 and s != 0:
                bad_sum += 1
            for b, w in vi:
                if w != 0:
                    if b < 0:
                        bad_bone += 1
                    maxbone = max(maxbone, b)
    return ("ver=%r vgroups=%d verts=%d palette_entries=%d\n"
            "  influence_count_hist=%s\n"
            "  max_bone_ref=%d  unresolved(-1)_weights=%d  weight_sum!=1=%d"
            % (sm.version, len(sm.vgroups), nverts, len(sm.palette_raw),
               wc_hist, maxbone, bad_bone, bad_sum))


if __name__ == "__main__":
    import sys
    path = sys.argv[1]
    with open(path, "rb") as f:
        blob = f.read()
    # if it's a PAC, pull sub starting with pmo\0
    if blob[:4] != b"pmo\x00":
        n = struct.unpack_from("<I", blob, 0)[0]
        off = 4
        for _ in range(n):
            o, s = struct.unpack_from("<II", blob, off)
            off += 8
            if blob[o:o + 4] == b"pmo\x00":
                blob = blob[o:o + s]
                break
    sm = read(blob)
    print(_summary(sm))
