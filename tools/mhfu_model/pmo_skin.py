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


def weld_seams(vgroups, bone_world, eps: float = 1.5,
               min_bonedist: float = 150.0, max_bonedist: float = 300.0):
    """Fix skinning-tear HOLES by welding seam vertices to a single shared bone.

    A tear = vertices COINCIDENT in bind pose (so no gap at rest) that live in
    DIFFERENT vgroups skinned to DIFFERENT, FAR-APART bones: when the skeleton is
    posed they separate and open a hole (the red backface in-game). This finds each
    such coincident cluster (spanning >1 vgroup, with bone-world spread >
    ``min_bonedist``) and re-binds ALL its verts to ONE shared bone (the most-weighted
    across the cluster) at weight 1.0 — identical influences ⇒ they transform together
    ⇒ no gap, in EVERY pose (pose-independent, unlike a render check). Modifies
    ``vgroups[*].influences`` in place; returns the number of clusters welded.

    The shared bone is the cluster's dominant bone, so it is already in the affected
    vgroups' palettes in the common case (no palette growth past the 8-bone cap).
    Welding to a single bone makes the thin seam strip rigid (a faint crease at worst)
    — far better than a hole. Tune ``min_bonedist`` up to weld only the worst tears.
    """
    from collections import Counter

    def d(a, b):
        return ((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2) ** 0.5

    items = []                                   # (gi, vi, pos)
    for gi, vg in enumerate(vgroups):
        for vi, v in enumerate(vg.vertices):
            if v is not None:
                items.append((gi, vi, (v["x"], v["y"], v["z"])))
    # current palette (distinct bones) per vgroup — weld must not push any past 8
    palettes = []
    for vg in vgroups:
        s = set()
        for infl in vg.influences:
            for (b, w) in infl:
                if w != 0 and b >= 0:
                    s.add(b)
        palettes.append(s)
    cell = max(eps, 2.0)
    grid = {}
    for k, (_gi, _vi, p) in enumerate(items):
        grid.setdefault((round(p[0]/cell), round(p[1]/cell), round(p[2]/cell)), []).append(k)

    visited = set()
    welded = 0
    for k, (gi, vi, p) in enumerate(items):
        if k in visited:
            continue
        kx, ky, kz = round(p[0]/cell), round(p[1]/cell), round(p[2]/cell)
        cluster = [k]
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for j in grid.get((kx+dx, ky+dy, kz+dz), []):
                        if j != k and j not in visited and d(items[j][2], p) <= eps:
                            cluster.append(j)
        if len({items[c][0] for c in cluster}) < 2:
            continue
        wsum = Counter()
        doms = []                                # dominant bone of each cluster vert
        for c in cluster:
            g2, v2, _ = items[c]
            infl = vgroups[g2].influences[v2]
            for (b, w) in infl:
                if w > 0.01 and 0 <= b < len(bone_world):
                    wsum[b] += w
            if infl:
                db = max(infl, key=lambda bw_: bw_[1])[0]
                if 0 <= db < len(bone_world):
                    doms.append(db)
        if len(wsum) < 2 or len(set(doms)) < 2:
            continue
        # TEAR MAGNITUDE = max distance between the cluster verts' DOMINANT bones (the
        # actual separation when posed) — NOT the union of all blended bones (a chest
        # vert that also lightly blends a far bone must not be judged a wing tear).
        spread = max(d(bone_world[a], bone_world[b]) for a in doms for b in doms)
        # BAND: weld only mid-range tears. Below min = adjacent bones (no visible gap).
        # Above max = a legitimately stretchy membrane spanning far bones (e.g. the wing
        # root, dominant-dist ~333) that the engine poses gently and does NOT visibly
        # tear — welding it to one bone over-stiffens the membrane (the v54 upper-wing
        # distortion). So skip those; only weld the genuine chest/hip tears (~150-300).
        if not (min_bonedist <= spread <= max_bonedist):
            continue
        affected = {items[c][0] for c in cluster}
        # Weld to ONE shared bone: the most-weighted that fits every affected vgroup's
        # 8-bone palette cap (prefer one already present -> no growth). Identical single
        # bone on all cluster verts ⇒ they transform together ⇒ no gap, in every pose.
        wb = None
        for cand, _w in wsum.most_common():
            if all(cand in palettes[g] or len(palettes[g]) < 8 for g in affected):
                wb = cand
                break
        if wb is None:
            continue
        for c in cluster:
            g2, v2, _ = items[c]
            vgroups[g2].influences[v2] = [(wb, 1.0)]
            palettes[g2].add(wb)
            visited.add(c)
        welded += 1
    return welded


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


def _seg_dist2(p, a, b):
    """Squared distance from point p to the line SEGMENT a-b."""
    ax, ay, az = a; bx, by, bz = b; px, py, pz = p
    dx, dy, dz = bx - ax, by - ay, bz - az
    L2 = dx * dx + dy * dy + dz * dz
    if L2 < 1e-9:
        return (px - ax) ** 2 + (py - ay) ** 2 + (pz - az) ** 2
    t = ((px - ax) * dx + (py - ay) * dy + (pz - az) * dz) / L2
    t = 0.0 if t < 0 else (1.0 if t > 1 else t)
    cx, cy, cz = ax + t * dx, ay + t * dy, az + t * dz
    return (px - cx) ** 2 + (py - cy) ** 2 + (pz - cz) ** 2


def auto_skin(mesh_groups, bone_world, materials_of=None, nb=3, max_pal=8,
              parents=None, hops=2, region_lock=True, region_hops=3, exclude=None,
              segment=True):
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
    from collections import Counter
    bw = list(bone_world)
    excl = set(exclude or ())          # host joints that won't be animated (rest/dead):
    #                                    never bind geometry to them or it pins to an
    #                                    un-rotating joint and tears (the Brute tail).
    nbr = _tree_neighborhoods(parents, hops) if parents is not None else None
    # region neighborhoods (wider) for the per-vgroup region lock
    rnbr = _tree_neighborhoods(parents, region_hops) if parents is not None else None
    # bone "segments" (joint -> its parent's joint) for segment-distance weighting:
    # a chest vertex is far from the WING joints but close to the SPINE segment it lies
    # along, so segment distance keeps it on the spine and off the wings (vs joint
    # distance, which pulls chest verts toward the euclidean-near wing-root joint ->
    # 679u stretch flaps). Root bones have no segment -> fall back to the point.
    seg = None
    if segment and parents is not None:
        seg = []
        for i, b in enumerate(bw):
            p = parents[i]
            seg.append(bw[p] if (0 <= p < len(bw)) else b)
    out = []
    for g in mesh_groups:
        # per-vertex sorted distances to every bone (excluded bones removed)
        allds = []
        for v in g.vertices:
            pt = (v["x"], v["y"], v["z"])
            if seg is not None:
                ds = [(_seg_dist2(pt, bw[i], seg[i]), i)
                      for i in range(len(bw)) if i not in excl]
            else:
                ds = [((v["x"] - b[0]) ** 2 + (v["y"] - b[1]) ** 2 + (v["z"] - b[2]) ** 2, i)
                      for i, b in enumerate(bw) if i not in excl]
            ds.sort()
            allds.append(ds)
        # REGION LOCK: a source vgroup is one rigid body PART, so confine ALL its verts
        # to the dominant bone's tree-neighborhood. Without this, tail-base verts whose
        # euclidean-nearest bone is a WING root (the parts overlap in space) bind to wing
        # bones and fly off under wing animation (the Brute tail shards). The dominant
        # bone = the majority per-vertex nearest; its region_hops neighborhood spans the
        # whole part (e.g. all 5 tail joints) but never reaches another limb (that needs
        # crossing the spine fork, > region_hops away).
        region = None
        if nbr is not None and region_lock and allds:
            votes = Counter(ds[0][1] for ds in allds)
            dominant = votes.most_common(1)[0][0]
            region = rnbr[dominant]
        infl = []
        for ds in allds:
            if nbr is not None:
                if region is not None:
                    cand = region
                    near = [(d, i) for d, i in ds if i in cand][:nb]
                    if not near:                      # vert outside the region: nearest in-region
                        near = [(ds[0])]
                else:
                    cand = nbr[ds[0][1]]
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
# reference weight transfer (same-family port: copy a native monster's skinning)
# --------------------------------------------------------------------------- #
def _ref_triangle_soup(ref: "SkinModel"):
    """Flatten a fully-skinned reference SkinModel into parallel arrays:
    (A, B, C) triangle corner positions + (iA, iB, iC) per-corner influence dicts.
    Degenerate (zero-area) triangles are dropped. Returns numpy arrays + lists."""
    import numpy as np
    A, B, C, iA, iB, iC = [], [], [], [], [], []
    for vg in ref.vgroups:
        verts = vg.vertices
        infl = vg.influences
        nv = len(verts)
        for f in vg.faces:
            i1, i2, i3 = f["v1"], f["v2"], f["v3"]
            if i1 >= nv or i2 >= nv or i3 >= nv:
                continue
            va, vb, vc = verts[i1], verts[i2], verts[i3]
            if va is None or vb is None or vc is None:
                continue
            pa = (va["x"], va["y"], va["z"])
            pb = (vb["x"], vb["y"], vb["z"])
            pc = (vc["x"], vc["y"], vc["z"])
            # skip degenerate triangles (no surface to sample)
            e1 = (pb[0]-pa[0], pb[1]-pa[1], pb[2]-pa[2])
            e2 = (pc[0]-pa[0], pc[1]-pa[1], pc[2]-pa[2])
            cx = e1[1]*e2[2] - e1[2]*e2[1]
            cy = e1[2]*e2[0] - e1[0]*e2[2]
            cz = e1[0]*e2[1] - e1[1]*e2[0]
            if cx*cx + cy*cy + cz*cz < 1e-6:
                continue
            A.append(pa); B.append(pb); C.append(pc)
            iA.append(infl[i1] if i1 < len(infl) else [])
            iB.append(infl[i2] if i2 < len(infl) else [])
            iC.append(infl[i3] if i3 < len(infl) else [])
    if not A:
        raise ValueError("reference mesh has no non-degenerate triangles")
    return (np.asarray(A, dtype=np.float64), np.asarray(B, dtype=np.float64),
            np.asarray(C, dtype=np.float64), iA, iB, iC)


def _closest_bary_all(P, A, B, C):
    """Barycentric coords of the closest point on every triangle (A,B,C) to point P.

    Vectorized Ericson ClosestPtPointTriangle (Real-Time Collision Detection),
    applied to all T triangles at once. Returns bary (T,3) = (u,v,w) such that the
    closest point on triangle t is u*A[t] + v*B[t] + w*C[t]."""
    import numpy as np
    ab = B - A; ac = C - A
    ap = P[None, :] - A
    d1 = (ab * ap).sum(1); d2 = (ac * ap).sum(1)
    bp = P[None, :] - B
    d3 = (ab * bp).sum(1); d4 = (ac * bp).sum(1)
    cp = P[None, :] - C
    d5 = (ab * cp).sum(1); d6 = (ac * cp).sum(1)

    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4

    # face region (default): denom = va+vb+vc
    denom = va + vb + vc
    sden = np.where(np.abs(denom) > 1e-20, denom, 1.0)
    fv = vb / sden; fw = vc / sden
    bary = np.stack([1.0 - fv - fw, fv, fw], axis=1)

    def _set(mask, u, v, w):
        m = mask[:, None]
        cand = np.stack([np.broadcast_to(u, va.shape),
                         np.broadcast_to(v, va.shape),
                         np.broadcast_to(w, va.shape)], axis=1)
        return np.where(m, cand, bary)

    # apply in REVERSE priority (lowest first) so highest-priority region wins
    # edge BC: w=(d4-d3)/((d4-d3)+(d5-d6)), bary (0,1-w,w)
    den_bc = (d4 - d3) + (d5 - d6)
    wbc = (d4 - d3) / np.where(den_bc != 0, den_bc, 1.0)
    m_bc = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
    bary = _set(m_bc, 0.0, 1.0 - wbc, wbc)
    # edge AC: w=d2/(d2-d6), bary (1-w,0,w)
    wac = d2 / np.where((d2 - d6) != 0, (d2 - d6), 1.0)
    m_ac = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    bary = _set(m_ac, 1.0 - wac, 0.0, wac)
    # vertex C: (0,0,1)
    m_c = (d6 >= 0) & (d5 <= d6)
    bary = _set(m_c, 0.0, 0.0, 1.0)
    # edge AB: v=d1/(d1-d3), bary (1-v,v,0)
    vab = d1 / np.where((d1 - d3) != 0, (d1 - d3), 1.0)
    m_ab = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    bary = _set(m_ab, 1.0 - vab, vab, 0.0)
    # vertex B: (0,1,0)
    m_b = (d3 >= 0) & (d4 <= d3)
    bary = _set(m_b, 0.0, 1.0, 0.0)
    # vertex A: (1,0,0)
    m_a = (d1 <= 0) & (d2 <= 0)
    bary = _set(m_a, 1.0, 0.0, 0.0)
    return bary


def _live_ancestor(bone, parents, dead):
    """Walk up the parent chain until a non-dead joint (a joint the retargeted anim
    actually drives). Returns the live ancestor, or the original bone if none found."""
    if dead is None or bone not in dead:
        return bone
    seen = set()
    b = bone
    while 0 <= b < len(parents) and b in dead and b not in seen:
        seen.add(b)
        b = parents[b]
    return b if (0 <= b < len(parents) and b not in dead) else bone


def transfer_weights_from_reference(mesh_groups, ref, materials_of=None,
                                    max_pal=8, parents=None, dead=None,
                                    smooth_passes=0):
    """Transfer blend skinning onto rigid-piece geometry from a fully-skinned
    REFERENCE monster sharing the SAME model space + skeleton.

    The principled replacement for ``auto_skin`` (nearest-bone guess) + ``weld_seams``
    (single-bone patch) on **same-family** ports: the native MHFU monster (e.g. the
    Tigrex ``file_06185`` sub1 for a Brute) is already perfectly skinned to the exact
    host rig, so it is the ground-truth oracle. For each target vertex we find the
    closest point on the reference mesh SURFACE (min over all reference triangles) and
    barycentric-blend the three reference vertices' (bone, weight) influences there.
    The result is smooth + anatomically correct by construction — no seam tears (the
    reference doesn't tear) and no rigid spikes.

    ``dead`` / ``parents``: host joints the retargeted animation will NOT drive
    (unmatched in the bone map) are reassigned to their nearest LIVE ancestor, so no
    vertex pins to an un-rotating joint (preserves the Brute-tail fix). Per-vgroup
    palette is capped at ``max_pal`` (PSP 8-matrix limit); weights re-normalized.

    Returns a list of SkinVGroup ready for ``build``/``encode``.
    """
    import numpy as np
    A, B, C, iA, iB, iC = _ref_triangle_soup(ref)
    parents = list(parents) if parents is not None else None
    deadset = set(dead) if dead else None

    out: List[SkinVGroup] = []
    for g in mesh_groups:
        infl: List[List[Tuple[int, float]]] = []
        for v in g.vertices:
            P = np.array([v["x"], v["y"], v["z"]], dtype=np.float64)
            bary = _closest_bary_all(P, A, B, C)
            cp = (bary[:, 0:1] * A + bary[:, 1:2] * B + bary[:, 2:3] * C)
            d2 = ((P[None, :] - cp) ** 2).sum(1)
            t = int(np.argmin(d2))
            u, vv, w = float(bary[t, 0]), float(bary[t, 1]), float(bary[t, 2])
            acc: dict = {}
            for src, bw in ((iA[t], u), (iB[t], vv), (iC[t], w)):
                if bw <= 0:
                    continue
                for (b, wt) in src:
                    if wt == 0 or b < 0:
                        continue
                    rb = _live_ancestor(b, parents, deadset) if deadset else b
                    acc[rb] = acc.get(rb, 0.0) + bw * wt
            if not acc:
                # closest triangle had no influence (shouldn't happen) -> nearest corner
                acc = {0: 1.0}
            s = sum(acc.values()) or 1.0
            infl.append([(b, wv / s) for b, wv in acc.items()])

        # per-vgroup palette cap (most-influential bones), then per-vertex renormalize
        accg: dict = {}
        for vi in infl:
            for b, wt in vi:
                accg[b] = accg.get(b, 0.0) + wt
        palette = [b for b, _ in sorted(accg.items(), key=lambda x: -x[1])[:max_pal]]
        pset = set(palette)
        infl2 = []
        for vi in infl:
            kept = [(b, wt) for b, wt in vi if b in pset]
            if not kept:
                kept = [(palette[0], 1.0)]
            s = sum(wt for _, wt in kept) or 1.0
            infl2.append([(b, wt / s) for b, wt in kept])
        out.append(SkinVGroup(
            index=len(out), bone_count=0, cum_bone_count=0, palette=palette,
            material=(materials_of(g) if materials_of else 0),
            mesh_offset=0, vertex_offset=0, index_offset=0,
            vertices=g.vertices, faces=g.faces, influences=infl2))
    return out


# --------------------------------------------------------------------------- #
# authentic source skinning (port the monster's OWN blend weights)
# --------------------------------------------------------------------------- #
def from_source_influences(mesh_groups, bone_remap=None, materials_of=None,
                           max_pal=8):
    """Build SkinVGroups from the source mesh's OWN per-vertex (bone, weight)
    influences — the authentic skin, NOT a nearest-bone guess.

    Each vertex dict must carry ``influences`` = ``[(source_bone, weight), ...]`` as
    resolved from the source PMO's bone palette (e.g. ``pmo_p3rd.parse``, which reads
    the v102 ``Weight{slot,bone}[]`` palette). This is the principled path when the
    source rig is SHIPPED (source-skeleton mode): the source bone indices map onto the
    output rig 1:1 (offset by the leading-origin pad), so no oracle and no guess.

    Parameters
    ----------
    mesh_groups : list of model.MeshGroup; each vertex has ``influences`` (and the
                  usual x/y/z, u/v, i/j/k). Groups/verts WITHOUT influences fall back
                  to a single bone (palette[0]) — they should not occur on a fully
                  skinned source.
    bone_remap  : callable(src_bone) -> out_bone, or None for identity. Returning a
                  value < 0 drops that influence (its weight redistributes over the
                  vertex's remaining influences). Use to shift source bone indices onto
                  the output rig (source-skeleton: src i -> i + lead_pad; retarget:
                  invert the host<-source bone map).
    Returns a list of SkinVGroup ready for ``build`` / ``encode``.
    """
    def rm(b):
        return b if bone_remap is None else bone_remap(b)
    out: List[SkinVGroup] = []
    for g in mesh_groups:
        infl: List[List[Tuple[int, float]]] = []
        for v in g.vertices:
            src = v.get("influences") or []
            acc: dict = {}
            for (b, w) in src:
                if w == 0 or b is None or b < 0:
                    continue
                ob = rm(b)
                if ob is None or ob < 0:
                    continue
                acc[ob] = acc.get(ob, 0.0) + w
            if not acc:
                acc = {0: 1.0}
            s = sum(acc.values()) or 1.0
            infl.append([(b, w / s) for b, w in acc.items()])
        # per-vgroup palette cap (PSP 8-matrix limit): keep the most-influential bones
        accg: dict = {}
        for vi in infl:
            for b, w in vi:
                accg[b] = accg.get(b, 0.0) + w
        palette = [b for b, _ in sorted(accg.items(), key=lambda x: -x[1])[:max_pal]]
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
