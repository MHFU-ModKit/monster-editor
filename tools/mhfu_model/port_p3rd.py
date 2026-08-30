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
from . import p3rd_anim_map as _p3am
from . import skeleton as _sk


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
def _source_bone_remap(source_skeleton: bool, lead_pad: int, bone_map,
                       newpos=None):
    """Return callable(src_bone) -> out_bone mapping the SOURCE skeleton's bone
    indices onto the OUTPUT rig's joint indices, for the authentic-skin path.

    source_skeleton: the output ships the source rig with ``lead_pad`` origin bones
    prepended (skeleton.p3rd_to_mhfu), so src bone i -> i + lead_pad (1:1).
    retarget: the output rig is the HOST; invert the host<-source bone_map
    (host_joint -> source_track) to source -> host_joint. A source bone with no host
    correspondence returns -1 (dropped; its weight redistributes over the rest)."""
    if source_skeleton:
        if newpos is not None:
            return lambda b: newpos.get(b, -1)
        pad = lead_pad or 0
        return (lambda b: b + pad) if pad else (lambda b: b)
    if bone_map:
        inv: dict = {}
        for host_j, src_t in bone_map.items():
            if src_t is not None and src_t not in inv:
                inv[src_t] = host_j
        return lambda b: inv.get(b, -1)
    return lambda b: b


# --------------------------------------------------------------------------- #

def _p3am_records(anim_blob):
    """Bone records per clip in an MHP3rd moveset (0 if unknown). All streams agree."""
    if not anim_blob:
        return 0
    try:
        pk = _flatanim.parse_p3rd(anim_blob, stream=0)
        counts = {len(a.tracks) for a in pk.animations}
        return counts.pop() if len(counts) == 1 else 0
    except Exception:
        return 0


def port_monster(model_pac: bytes, frame_pac: bytes,
                 geo_companion: Optional[bytes] = None,
                 anim_blob: Optional[bytes] = None,
                 nb: int = 3, hops: int = 1,
                 host_count: int = 45, split=None,
                 keep_anim_size: bool = False, ground_lift: float = 0.0,
                 weld: bool = True, weld_min_bonedist: float = 150.0,
                 skin: str = "auto", source_skeleton: bool = False,
                 src_animated: Optional[int] = None,
                 em_id: Optional[int] = None,
                 anim_bone_offset: Optional[int] = None,
                 anim_skip: Optional[list] = None,
                 reparent_orphans: bool = True,
                 reweight_undriven: bool = False,
                 drop_joints=None):
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
    skin          : "auto" = nearest-bone blend (auto_skin) + optional seam weld
                    (no native reference needed — the no-similar-monster path).
                    "transfer" = copy the host frame's OWN native skinning (frame
                    sub1) onto the source geometry via closest-surface weight
                    transfer — the principled SAME-FAMILY path (Brute<-Tigrex): the
                    native monster is the perfect oracle, so NO guess + NO weld
                    (drops the rigid-spike/hole whack-a-mole). Falls back to "auto"
                    if the frame has no usable PMO reference.
                    "source" = ship the monster's OWN authentic skin — pair each
                    vertex's source bone-palette (bone, weight) influences (parsed
                    from the v102 palette by pmo_p3rd) with the output rig. The
                    principled path when the source rig is shipped (source_skeleton):
                    bone indices map 1:1 (+lead_pad), so NO guess and NO oracle. In
                    source_skeleton mode "auto" auto-upgrades to "source" when the
                    source carries weights; falls back to "auto" if it doesn't.
    source_skeleton : when True, ship the monster's OWN skeleton (no down-rig). sub0 is
                    rebuilt from the MHP3rd 0x80000000 skeleton into the MHFU 0x10C
                    format (`skeleton.p3rd_to_mhfu`, stream-ids assigned), the geometry
                    is skinned to the SOURCE rig, and the anim plays 1:1 (no bone_map /
                    host_count). Use for monsters whose shape differs from the host
                    (the joint count is data-driven, so no FK overrun). Default False =
                    the proven retarget-onto-host path.
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

    src_skel_sub = _find_sub(model_pac, b"\x00\x00\x00\x80")
    src_skel_blob = (model_pac[src_skel_sub[0]:src_skel_sub[0] + src_skel_sub[1]]
                     if src_skel_sub else None)

    lead_pad = 0
    newpos = None
    bone_order = None
    if source_skeleton:
        # --- SOURCE-SKELETON path: rig = the monster's OWN skeleton (no down-rig). ---
        if src_skel_blob is None:
            raise ValueError("source_skeleton mode needs a 0x80000000 skeleton sub")
        from .bone_match import bind_world_positions
        ssk = _skp.parse(src_skel_blob)
        parents = [b.parent for b in ssk.bones]
        _local = [tuple(b.bind_pos) for b in ssk.bones]
        bw = bind_world_positions(parents, _local)

        # ADOPT ORPHAN ROOT CHAINS. An MHP3rd rig can carry a SECOND root chain
        # (bone with parent == -1 that is not bone 0). The Zinogre's is bones 46-50:
        # the **severed tail**, the carvable object dropped when the tail is cut. It is
        # authored on its own root because once severed it lies in world space.
        # ⚠️ The native MHFU Tigrex has the same thing (bones 45-47) and leaves it
        # UNPARENTED, so adoption is a deviation from native, not a fix. It exists only
        # because an MHP3rd rig's origin is at the hip (~435 units up) rather than the
        # ground, so an unadopted chain floats at flank height instead of lying under
        # the monster. Neither placement is right until the severed-tail datum is
        # worked out; adoption at least keeps it moving with the body.
        reparent = {}
        if reparent_orphans:
            host = 1 if len(parents) > 1 else 0
            for i, pp in enumerate(parents):
                if i != 0 and (pp is None or pp < 0):
                    reparent[i] = host
            if reparent:
                parents = [host if (i != 0 and (pp is None or pp < 0)) else pp
                           for i, pp in enumerate(parents)]
                info["adopted_orphan_roots"] = sorted(reparent)

        # The native big-mon OVERLAY hardcodes the hip/ground joint index (Tigrex = the
        # tail of a 3-bone leading-origin chain = joint 2). A source skeleton with a
        # SHORTER leading-origin chain lands its hip at a lower joint -> the overlay's
        # lift misses it -> the body sinks. Pad the leading origin chain to match the
        # host's count so the hip aligns; shift the anim + skinning by the same amount.
        def _lead_origin(world, eps=1.0):
            c = 0
            for w in world:
                if (w[0] * w[0] + w[1] * w[1] + w[2] * w[2]) ** 0.5 < eps:
                    c += 1
                else:
                    break
            return c
        hp, hl, hbw = _skin.frame_skeleton(frame_pac)
        lead_pad = max(0, _lead_origin(hbw) - _lead_origin(bw))
        info["lead_pad"] = lead_pad

        # --- stream partition, READ OFF THE BONE TREE (not the Tigrex formula) ---
        # MHFU's FK walks the rig in 3 streams and each must be a contiguous run of
        # bone indices. Native rigs are authored body/head/tail so the trailing
        # slice happens to be right; an MHP3rd rig is not (the Zinogre's head sits
        # at bones 18..23, mid-order). Derive the real subtrees and REORDER the rig
        # so the partition is contiguous; the same permutation then drives the skin
        # and the anim, so nothing else has to know.
        # 🔴 +0x1C IS NOT AN ANIMATED-BONE COUNT. On most MHP3rd rigs it reads
        # 0x40000001 (a section magic) and the range test below rejects it; on the
        # Zinogre it happens to read 46, which IS in range, so it was trusted — and
        # 46 is wrong. The moveset drives 37 bones, and the rig has 51.
        #
        # The constraint that actually matters is the FK invariant (agent_memory_map
        # "Bone-count rule"):
        #
        #     skeleton bone_count - 1 == anim 3-stream partition total == entity+0x1A4
        #
        # A partition SHORTER than the walk leaves the trailing joints with no bone
        # section, and a joint with no section keeps a ZEROED matrix — every vertex
        # riding it collapses to the origin and the mesh stretches to meet it. The
        # Zinogre shipped with partition 46 against a walk of 51 and 4% of his mesh
        # (the neck/jaw chain, bones 47-50) smeared to the world origin.
        #
        # So the partition must span EVERY bone. Bones the moveset does not drive are
        # not a problem in themselves: from_flat_anim gives them rest_bone(), which
        # poses them at bind RELATIVE TO THEIR PARENT, so they are carried along by
        # the body instead of collapsing. Pass ``src_animated`` only to override.
        if src_animated is None:
            # Span exactly the DRIVEN range. Not every bone: partition == FK walk
            # hangs the joint builder (agent_memory_map "Bone-count rule"). Not the
            # +0x1C word either: it is not an animated count.
            _n = _p3am_records(anim_blob)
            if _n:
                _b2r = _p3am.for_monster(em_id if em_id is not None else -1, _n,
                                         len(ssk.bones), offset=anim_bone_offset,
                                         skip=anim_skip)
                src_animated = (max(_b2r) + 1) if _b2r else len(ssk.bones)
            else:
                src_animated = len(ssk.bones)
        if not (0 < src_animated <= len(ssk.bones)):
            src_animated = len(ssk.bones)
        split, order = _sk.derive_stream_partition(parents, bw, src_animated)
        split = [split[0] + lead_pad] + list(split[1:])
        bone_order = order
        info["stream_split"] = split
        info["bone_reorder"] = (order != list(range(len(ssk.bones))))

        # source bone i -> output joint index (permutation + leading-origin pad)
        newpos = {old_i: new_i + lead_pad for new_i, old_i in enumerate(order)}
        parents = _sk.reorder_bones(parents, order)
        bw = [bw[old_i] for old_i in order]
        if lead_pad:
            parents = [(-1 if j == 0 else j - 1) for j in range(lead_pad)] + \
                      [(p + lead_pad if p >= 0 else lead_pad - 1) for p in parents]
            bw = [(0.0, 0.0, 0.0)] * lead_pad + list(bw)
        # 🔴 MHP3rd ANIM RECORDS ARE NOT POSITIONAL. This used to be
        #     bone_map = {newpos[i]: i}
        # i.e. "source bone i is driven by track i", which is what the Blender
        # importer for these files explicitly is NOT: it walks the skeleton from a
        # per-monster `bone_offset`, stepping over a per-monster skip list
        # (`p3rd_anim_map`). On the Zinogre the positional read put 37 records on
        # bones 0..36, leaving the TAIL and part of the jaw with no data at all —
        # they froze at bind while the body moved, and the skin between them
        # stretched. The same 37 records, mapped correctly, cover bones 1..46
        # INCLUDING the tail. → docs/ANIMATION_FORMAT.md, p3rd_anim_map.
        n_rec = _p3am_records(anim_blob)
        b2r = None
        if n_rec:
            b2r = _p3am.for_monster(em_id if em_id is not None else -1, n_rec,
                                    len(ssk.bones), offset=anim_bone_offset,
                                    skip=anim_skip)
            info["anim_bone_offset"] = (anim_bone_offset
                                        if anim_bone_offset is not None
                                        else _p3am.BONE_OFFSET.get(em_id,
                                                                   _p3am.DEFAULT_BONE_OFFSET))
            info["anim_driven_bones"] = len(b2r)
            info["anim_max_driven_bone"] = max(b2r) if b2r else None
        _src_bw_for_reweight = list(bw)          # SOURCE indices; bw is permuted below
        _r2b_for_lift = ({r: b for b, r in b2r.items()} if b2r else None)
        _src_parents_for_lift = [x.parent for x in ssk.bones]
        # source bone -> RECORD (not track index); unmapped bones stay at rest.
        bone_map = ({newpos[i]: (b2r.get(i) if b2r is not None else i)
                     for i in range(len(ssk.bones))}
                    if (b2r is not None or lead_pad or info["bone_reorder"]) else None)
        dead = set()
        info["mode"] = "source_skeleton"
        info["src_bones"] = len(ssk.bones)
    else:
        parents, _local, bw = _skin.frame_skeleton(frame_pac)
        info["mode"] = "retarget_to_host"
        # --- cross-rig bone correspondence (computed FIRST so skinning can avoid the
        #     host joints that the moveset won't drive). When the host chain is LONGER
        #     than the source (Tigrex tail 5 joints vs Brute 4) the extra joint is left
        #     UNMATCHED (rest pose) — and geometry must NOT bind to it, else it pins to
        #     an un-rotating joint and tears (the tail shards). We do NOT fill the gap
        #     (filling makes a parent+child share one source -> FK compounds rotation ->
        #     the tip whips out). ---
        bone_map = None
        if anim_blob is not None and src_skel_blob is not None:
            try:
                ssk = _skp.parse(src_skel_blob)
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

    # --- drop auxiliary geometry (opt-in, and usually WRONG) ---
    # ⚠️ The orphan-root chain on an MHP3rd rig — the Zinogre's bones 46-50, 165 verts —
    # is the **SEVERED TAIL**: the carvable object the game drops when the tail is cut.
    # Both the Zinogre and the native MHFU Tigrex (its own is bones 45-47, 150 verts)
    # have a cuttable tail and both carry one. It is authored on its own root because
    # once severed it lies in world space, not on the animal. That is why it is not
    # animated, why nothing else is parented to it, and why hiding it leaves no hole.
    # 🔴 So do NOT drop it by default — dropping it removes a real gameplay object.
    # This flag stays for the case where a port genuinely carries geometry the target
    # engine cannot place. Check first with blender_mhfu/render_port_views.py
    # (MHFU_VIEW_ONLY=1 to see it, =2 to see the model without it).
    if drop_joints:
        _drop = set(drop_joints)
        _kept, _lost = [], 0
        for g in model.mesh_groups:
            _tot = {}
            for v in g.vertices:
                for bb, w in (v.get("influences") or ()):
                    if w > 1e-4 and bb >= 0:
                        _tot[bb] = _tot.get(bb, 0.0) + w
            _dom = max(_tot, key=_tot.get) if _tot else -1
            if _dom in _drop:
                _lost += len(g.vertices)
            else:
                _kept.append(g)
        info["dropped_joints"] = sorted(_drop)
        info["dropped_groups"] = len(model.mesh_groups) - len(_kept)
        info["dropped_verts"] = _lost
        model.mesh_groups = _kept

    # --- skin geometry onto the output skeleton ---
    # AUTHENTIC-SKIN path: when the source carries its OWN per-vertex blend weights
    # (v102 bone palette, parsed by pmo_p3rd) and we ship the source rig, prefer them
    # over any guess/transfer. Auto-upgrade the default in source-skeleton mode so a
    # plain `--source-skeleton` build ships the real skin.
    # 🔴 SUPERSEDED — DO NOT ENABLE. Kept only so the experiment is reproducible.
    # The premise was that the undriven geometry past the last record is BODY
    # geometry ("bones 48/49/50 = the middle of his back") and needs folding onto a
    # driven neighbour. That premise is false. Rendered in isolation those joints are
    # a flat auxiliary PLATE on the rig's second root chain, and hiding them leaves
    # no hole anywhere on the monster (blender_mhfu/render_port_views.py,
    # MHFU_VIEW_ONLY). Re-weighting therefore tears one contiguous decoration across
    # the head, spine, hip, tail and legs — in game, a large moving spike over a
    # slab that does not move. Use `drop_joints` instead.
    # ⚠️ It also silently mangles a REAL undriven body part if one ever exists, for
    # the same reason: nearest-driven-bone is per vertex, so a patch does not stay
    # together. Any future version must choose ONE target per source bone.
    if reweight_undriven and source_skeleton and b2r:
        import math as _m
        _bw = locals().get("_src_bw_for_reweight") or []
        _skipset = set(anim_skip if anim_skip is not None
                       else _p3am.SKIPPED_BONES.get(em_id, []))
        # Candidates are driven bones that actually OWN geometry. Structural bones
        # (the origin chain) are nearest to any torso vertex by Euclidean distance
        # while deforming nothing, so including them just re-creates the rigid patch
        # one bone over — 58 of the Zinogre's 165 vertices went to the rig root
        # before this filter.
        _owns = {}
        for g in model.mesh_groups:
            for v in g.vertices:
                for bb, w in (v.get("influences") or []):
                    if w:
                        _owns[bb] = _owns.get(bb, 0) + 1
        undriven = {b for b in range(len(_bw)) if b not in b2r and b not in _skipset}
        driven = [b for b in b2r
                  if b < len(_bw) and _owns.get(b, 0) >= 8 and b not in undriven]
        moved = 0
        chose = {}
        if driven and undriven:
            # 🔴 PER VERTEX, not per bone. Picking the driven bone nearest the BONE
            # sends the whole patch to whatever is closest to that bone's ORIGIN,
            # which for a chain rooted at (0,0,0) is the rig root — so the lower-back
            # geometry (centroid z -78) was being folded onto bone 1 and would have
            # stayed just as rigid. Each vertex goes to the driven bone nearest to
            # ITSELF, so the patch deforms with whatever is actually next to it.
            for g in model.mesh_groups:
                for v in g.vertices:
                    infl = v.get("influences")
                    if not infl or not any(bb in undriven for bb, _w in infl):
                        continue
                    vp = (v["x"], v["y"], v["z"])
                    acc = {}
                    for bb, w in infl:
                        tb = bb
                        if bb in undriven:
                            tb = min(driven, key=lambda d: _m.dist(_bw[d], vp))
                            chose[bb] = chose.get(bb, {})
                            chose[bb][tb] = chose[bb].get(tb, 0) + 1
                        acc[tb] = acc.get(tb, 0.0) + w
                    v["influences"] = sorted(acc.items())
                    moved += 1
            info["reweighted_verts"] = moved
            info["reweight_map"] = {b: sorted(c.items(), key=lambda t: -t[1])
                                    for b, c in sorted(chose.items())}

    has_src_infl = any(v.get("influences") for g in model.mesh_groups for v in g.vertices)
    if source_skeleton and skin == "auto" and has_src_infl:
        skin = "source"
    info["skin_mode"] = skin
    ref_pmo = _find_sub(frame_pac, b"pmo\x00", which=0) if skin == "transfer" else None
    if skin == "source" and has_src_infl:
        # SHIP THE MONSTER'S OWN SKIN: pair each vertex's source-palette (bone, weight)
        # influences with the output rig (source-skeleton: src bone i -> i + lead_pad;
        # retarget: invert the host<-source bone map). No nearest-bone guess, no oracle.
        remap = _source_bone_remap(source_skeleton, lead_pad, bone_map,
                                   newpos=newpos)
        vgs = _skin.from_source_influences(
            model.mesh_groups, bone_remap=remap,
            materials_of=lambda g: g.material, max_pal=8)
        info["skin_mode"] = "source"
        info["source_weight_verts"] = sum(
            1 for g in model.mesh_groups for v in g.vertices if v.get("influences"))
    elif skin == "transfer" and ref_pmo is not None:
        # SAME-FAMILY: transfer the native monster's own (perfect) skinning from the
        # frame's PMO sub onto the source geometry by closest-surface barycentric
        # weight transfer. No nearest-bone guess, no seam weld -> no spikes/holes.
        ref = _skin.read(frame_pac[ref_pmo[0]:ref_pmo[0] + ref_pmo[1]])
        vgs = _skin.transfer_weights_from_reference(
            model.mesh_groups, ref, materials_of=lambda g: g.material,
            max_pal=8, parents=parents, dead=dead)
        info["skin_mode"] = "transfer"
    else:
        if skin == "source":
            info["skin_mode"] = "auto(fallback:no-source-weights)"
        elif skin == "transfer":
            info["skin_mode"] = "auto(fallback:no-ref-pmo)"
        # NO-REFERENCE path: nearest-bone blend, chain-aware, skipping dead joints.
        vgs = _skin.auto_skin(model.mesh_groups, bw,
                              materials_of=lambda g: g.material,
                              nb=nb, max_pal=8, parents=parents, hops=hops, exclude=dead)
        # Weld skinning-tear seams: coincident cross-vgroup verts skinned to far-apart
        # bones separate when posed and open HOLES (the chest/wing-root red gaps). Re-bind
        # each such cluster to one shared bone so they can't split. See weld_seams.
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
    if source_skeleton:
        conv = _sk.p3rd_to_mhfu(src_skel_blob, lead_pad=lead_pad,
                                split=split, order=bone_order,
                                src_animated=src_animated,
                                reparent=reparent)               # own rig, stream-ordered
        new = bytearray(_replace_sub(bytes(new), 0, conv))
        info["skeleton_bytes"] = len(conv)

    # --- animation: retarget the monster's moveset to the host (unmatched -> rest) ---
    if anim_blob is not None:
        flat = _flatanim.parse_p3rd(anim_blob)
        info["anim_clips"] = len(flat.animations)
        # GROUND LIFT: the port inherits the SOURCE GAME'S vertical datum.
        # ⛔ The rationale that used to sit here — "a swap-spawned monster is NOT
        # terrain-placed, the engine pins its world Y at 0" — is RETRACTED with the
        # rest of the terrain-registration theory (memory `brute-terrain-sink-re`);
        # the engine grounds the entity correctly. The real reason a cross-game port
        # needs this is a DATUM mismatch: MHFU's convention is "the rest pose puts
        # the feet at the skeleton origin", and a source rig authored against another
        # game's convention does not. Measured with `tools/port_rest_floor.py`:
        #   native Tigrex   lowest bone -297.9 + pelvis lift 300.6 = +2.7  (on the floor)
        #   Zinogre port    lowest bone -389.0 + pelvis lift 226.4 = -162.6 (sunk)
        # so `--ground-lift 165.3` and the port lands at +2.7 too. The native reading
        # +2.7 is what makes the method trustworthy — it is the control.
        # Baking it into the PELVIS locY channel (not a per-frame write) is what makes
        # it hold in every pose and while moving.
        if ground_lift:
            _apply_ground_lift(flat, ground_lift,
                               bone_of_record=locals().get("_r2b_for_lift"),
                               parents=locals().get("_src_parents_for_lift"))
            info["ground_lift"] = ground_lift
        out, ainfo = _ig.swap_anim_to_realmotion(
            bytes(new), flat, anim_index=3, skel_index=0,
            host_count=(None if source_skeleton else host_count),
            split=split,
            keep_size=keep_anim_size, bone_map=bone_map)
        new = bytearray(out)
        info["anim"] = ainfo
        info["bone_map_matched"] = (sum(1 for v in bone_map.values() if v is not None)
                                    if bone_map else None)

    info["total"] = len(new)
    info["native_total"] = len(frame_pac)
    info["needs_relocate"] = len(new) != len(frame_pac)
    return bytes(new), info


def _apply_ground_lift(flat, lift_units, bone_of_record=None, parents=None):
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
    # 🔴 PICK BY TOPOLOGY, NOT BY MAGNITUDE. "Biggest locY" finds the bone that
    # carries the body height — which is only the right lever if it is also an
    # ANCESTOR OF THE WHOLE MONSTER. The Zinogre's rig branches at bone 1 into a
    # front half (bone 2: spine/head/forelimbs) and a rear half (bone 25: hips/
    # hind legs/tail). Its biggest-locY record maps to bone 2, so lifting it
    # raised the front of the body and left the tail and hind legs on the floor —
    # a monster doing a permanent handstand. Lift the ROOT-MOST bone that has a
    # locY channel instead: its subtree is everything.
    pelvis = score.most_common(1)[0][0]
    if bone_of_record and parents:
        def depth(b):
            d = 0
            while 0 <= b < len(parents) and parents[b] >= 0 and d < 64:
                b = parents[b]; d += 1
            return d
        cands = [r for r in score if r in bone_of_record]
        if cands:
            pelvis = min(cands, key=lambda r: (depth(bone_of_record[r]), r))
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
