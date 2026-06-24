"""Generalized MHP3rd big-monster -> MHFU port backend.

Turns an MHP3rd big-monster (model + skeleton + textures, plus its raw moveset) into
an injectable MHFU big-monster PAC by splicing onto an MHFU host frame (e.g. the Tigrex
PAC ``file_06185``):

  sub0  host skeleton (0xC0000000)         -- reused from the frame (the MHP3rd big
                                              monsters share the MHFU Tigrex rig)
  sub1  PMO  = MHP3rd geometry, chain-aware blend-skinned onto the host skeleton
  sub2  TMH  = the monster's OWN textures (MHP3rd .TMH == MHFU .TMH format)
  sub3  anim = the monster's OWN moveset, retargeted to the host + in-game 3-stream
  sub4+ secondary subs reused from the frame verbatim

This is the single backend behind both the CLI (`tools/build_p3rd_port.py`) and the
Blender addon ("Port P3rd Monster"), so a modder reproduces the exact same `.bin`.
Generalized: works for any MHP3rd big monster whose rig is close to the host's (the
Tigrex-family — Tigrex/Brute Tigrex/Barioth/Nargacuga/… — all do); pass the right
source files (see docs/MHP3RD_FILE_MONSTER_MAP.md).
"""
from __future__ import annotations

import os
import struct
from typing import Optional

from . import pmo_p3rd as _p3rd
from . import pmo_skin as _skin
from . import anim as _flatanim
from . import anim_ingame as _ig
from . import bone_match as _bm
from . import skeleton_p3rd as _skp


# --------------------------------------------------------------------------- #
def _pac_subs(blob):
    n = struct.unpack_from("<I", blob, 0)[0]
    return [struct.unpack_from("<II", blob, 4 + i * 8) for i in range(n)]


def _find_sub(blob, magic, which=0):
    for o, s in _pac_subs(blob):
        if blob[o:o + 4] == magic:
            if which == 0:
                return o, s
            which -= 1
    return None


def _p3rd_main_pmo(blob):
    """Return (offset, size, mesh_index) of the LARGEST-geometry pmo sub (the main
    model, not the low-detail/shadow one). MHP3rd model PACs carry 2 pmo subs."""
    best = None
    for i, (o, s) in enumerate(_pac_subs(blob)):
        if blob[o:o + 4] == b"pmo\x00":
            hdr = struct.unpack_from("<I4f2H8I", blob, o + 8)
            nmesh = hdr[5]
            if best is None or nmesh > best[3]:
                best = (o, s, i, nmesh)
    return best


# --------------------------------------------------------------------------- #
def port_monster(model_pac: bytes, frame_pac: bytes,
                 geo_companion: Optional[bytes] = None,
                 anim_blob: Optional[bytes] = None,
                 nb: int = 3, hops: int = 1,
                 host_count: int = 45, split=None,
                 keep_anim_size: bool = False, ground_lift: float = 0.0,
                 weld: bool = True, weld_min_bonedist: float = 150.0):
    """Port an MHP3rd big monster onto an MHFU host frame. Returns (pac_bytes, info).

    Parameters
    ----------
    model_pac     : the MHP3rd model+skel+TMH PAC bytes (e.g. file_05248).
    frame_pac     : the MHFU host PAC bytes (e.g. file_06185 = Tigrex) — supplies the
                    0xC0000000 skeleton (sub0) the engine drives + the secondary subs.
    geo_companion : the MHP3rd GE-list companion bytes (file_<model+1>) when the PMO's
                    geometry is external (ge_base >= pmo size). None = self-contained.
    anim_blob     : the MHP3rd raw moveset bytes (file_<model+2>) to retarget; None =
                    keep the frame's native animation.
    host_count    : the host overlay slot's animated-bone count (Tigrex = 45).
    Result fits-in-place when possible; otherwise the caller uses the relocate inject.
    """
    subs = _pac_subs(frame_pac)
    info = {}

    # --- geometry: parse MHP3rd v102 PMO (fixed walker) ---
    mp = _p3rd_main_pmo(model_pac)
    if mp is None:
        raise ValueError("no PMO sub in MHP3rd model PAC")
    mo, ms, mi, nmesh = mp
    model = _p3rd.parse(model_pac[mo:mo + ms], geo_blob=geo_companion)
    if not model.mesh_groups:
        raise ValueError("MHP3rd geometry parsed empty (need the GE companion file?)")
    info["src_groups"] = len(model.mesh_groups)
    info["src_verts"] = sum(len(g.vertices) for g in model.mesh_groups)

    parents, _local, bw = _skin.frame_skeleton(frame_pac)

    # --- cross-rig bone correspondence (computed FIRST so skinning can avoid the
    #     host joints that the moveset won't drive). When the host chain is LONGER
    #     than the source (Tigrex tail 5 joints vs Brute 4) the extra joint is left
    #     UNMATCHED (rest pose) — and geometry must NOT bind to it, else it pins to
    #     an un-rotating joint and tears (the tail shards). We do NOT fill the gap
    #     (filling makes a parent+child share one source -> FK compounds rotation ->
    #     the tip whips out). ---
    bone_map = None
    if anim_blob is not None:
        src_skel_sub = _find_sub(model_pac, b"\x00\x00\x00\x80")
        if src_skel_sub:
            try:
                ssk = _skp.parse(model_pac[src_skel_sub[0]:src_skel_sub[0] + src_skel_sub[1]])
                sp = [b.parent for b in ssk.bones]
                sl = [tuple(b.bind_pos) for b in ssk.bones]
                bone_map = _bm.match_skeletons(sp, sl, parents, _local)
            except Exception:
                bone_map = None
    # host joints the anim leaves at rest (unmatched within the animated count) ->
    # exclude from skinning so no geometry pins to an un-rotating joint.
    dead = set()
    if bone_map is not None:
        dead = {d for d in range(host_count) if bone_map.get(d) is None}
    info["dead_joints"] = sorted(dead)

    # --- skin geometry onto the host (frame) skeleton, chain-aware, skipping dead joints ---
    vgs = _skin.auto_skin(model.mesh_groups, bw,
                          materials_of=lambda g: g.material,
                          nb=nb, max_pal=8, parents=parents, hops=hops, exclude=dead)
    # Weld skinning-tear seams: coincident cross-vgroup verts skinned to far-apart
    # bones separate when posed and open HOLES (the chest/wing-root red gaps). Re-bind
    # each such cluster to one shared bone so they can't split. See pmo_skin.weld_seams.
    if weld:
        info["welded_seams"] = _skin.weld_seams(vgs, bw, min_bonedist=weld_min_bonedist)
    # materials = one per distinct texID the groups reference (identity material table)
    texids = sorted({g.material for g in model.mesh_groups})
    tex_to_idx = {t: i for i, t in enumerate(texids)}
    for vg, g in zip(vgs, model.mesh_groups):
        vg.material = tex_to_idx[g.material]
    materials = [{"texID": t} for t in texids]
    pmo = _skin.build(model.scale, vgs, materials)
    info["pmo_bytes"] = len(pmo)
    info["materials"] = texids

    # --- assemble the output PAC: frame subs, swapping in our PMO + Brute TMH (+anim) ---
    brute_tmh = _find_sub(model_pac, b".TMH")
    tmh_bytes = model_pac[brute_tmh[0]:brute_tmh[0] + brute_tmh[1]] if brute_tmh else None

    new = bytearray(frame_pac)
    new = bytearray(_replace_sub(bytes(new), 1, pmo))            # PMO sub
    if tmh_bytes is not None:
        new = bytearray(_replace_sub(bytes(new), 2, tmh_bytes))  # his own TMH
        info["tmh_bytes"] = len(tmh_bytes)

    # --- animation: retarget the monster's moveset to the host (unmatched -> rest) ---
    if anim_blob is not None:
        flat = _flatanim.parse_p3rd(anim_blob)
        info["anim_clips"] = len(flat.animations)
        # GROUND LIFT: a swap-spawned monster is NOT terrain-placed — the engine pins
        # its world Y at 0 (verified live), so the body (root + pelvis-lift ~300) sits
        # mostly below the snow floor (~270). Baking +ground_lift units into the pelvis
        # locY channel raises the whole body+feet onto the ground (the relocate inject
        # can't rely on a per-frame Y write — the engine resets it every frame).
        if ground_lift:
            _apply_ground_lift(flat, ground_lift)
            info["ground_lift"] = ground_lift
        out, ainfo = _ig.swap_anim_to_realmotion(
            bytes(new), flat, anim_index=3, skel_index=0,
            host_count=host_count, split=split,
            keep_size=keep_anim_size, bone_map=bone_map)
        new = bytearray(out)
        info["anim"] = ainfo
        info["bone_map_matched"] = (sum(1 for v in bone_map.values() if v is not None)
                                    if bone_map else None)

    info["total"] = len(new)
    info["native_total"] = len(frame_pac)
    info["needs_relocate"] = len(new) != len(frame_pac)
    return bytes(new), info


def _apply_ground_lift(flat, lift_units):
    """Add ``lift_units`` (world units) to the pelvis bone's locY across every clip.

    The pelvis = the bone whose locY channel carries the body height (the big ~300-unit
    value; the root above it stays ~0). Loc quant = /16, so we add lift_units*16 raw.
    Shifts the whole body+legs up uniformly (the relative bob is preserved)."""
    LOCY = 0x80
    raw = int(round(lift_units * 16))
    # identify the pelvis bone index = the one with the largest mean |locY| over clips
    import collections
    score = collections.Counter()
    for a in flat.animations:
        for bi, t in enumerate(a.tracks):
            for c in t.channels:
                if (c.type & 0xFFF) == LOCY and c.keyframes:
                    score[bi] += sum(abs(k.value) for k in c.keyframes) / len(c.keyframes)
    if not score:
        return
    pelvis = score.most_common(1)[0][0]
    for a in flat.animations:
        if pelvis < len(a.tracks):
            for c in a.tracks[pelvis].channels:
                if (c.type & 0xFFF) == LOCY:
                    for k in c.keyframes:
                        k.value = max(-32768, min(32767, k.value + raw))


def _replace_sub(pac: bytes, idx: int, data: bytes) -> bytes:
    """Rebuild a PAC replacing sub ``idx`` with ``data`` (re-laying offsets, 16-aligned).
    Keeps every other sub verbatim. Works whether the new sub is larger or smaller."""
    subs = _pac_subs(pac)
    blobs = []
    for i, (o, s) in enumerate(subs):
        blobs.append(data if i == idx else pac[o:o + s])
    n = len(blobs)
    # header: count + (off,size) table, then 16-aligned sub data (match native layout)
    table_end = 4 + n * 8
    cur = (table_end + 0xF) & ~0xF
    if subs:
        cur = subs[0][0]   # preserve the frame's first-sub offset (0x40 for big-mon)
    out = bytearray(4 + n * 8)
    struct.pack_into("<I", out, 0, n)
    offs = []
    for b in blobs:
        if len(out) < cur:
            out += b"\x00" * (cur - len(out))
        offs.append(len(out))
        out += b
        cur = (len(out) + 0xF) & ~0xF
    for i, (off, b) in enumerate(zip(offs, blobs)):
        struct.pack_into("<II", out, 4 + i * 8, off, len(b))
    return bytes(out)
