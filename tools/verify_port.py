#!/usr/bin/env python3
"""Structural audit of a ported big-monster PAC, offline.

Checks the invariants the ENGINE enforces — the ones whose violation reads in-game
as a crash, a collapsed mesh or a limb animating with the wrong clip — against the
MHP3rd source the port was built from. Every check is one the emulator would
otherwise have to answer at the cost of a cold boot.

    python tools/verify_port.py tmp/zinogre_streamfix.bin \
        --model workspace/extracted_mhp3/data_files/file_05339.bin \
        --geo   workspace/extracted_mhp3/data_files/file_05340.bin
"""
import argparse
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mhfu_model import pmo_p3rd as P3
from mhfu_model import pmo_skin as SK
from mhfu_model import skeleton_p3rd as SKP
from mhfu_model import anim_ingame as IG
from mhfu_model.bone_match import bind_world_positions

SECTION_MAGIC = 0x40000001


def _subs(blob):
    n = struct.unpack_from("<I", blob, 0)[0]
    return [struct.unpack_from("<II", blob, 4 + i * 8) for i in range(n)]


def _sub(blob, magic, pick=None):
    """First sub with ``magic``, or the winner under ``pick``.

    ``pick="size"``  — the biggest sub. Right for a PORT's output: our encoder puts
                       every vgroup in ONE mesh record, so its mesh count (1) loses
                       to the low-detail model's (3).
    ``pick="mesh"``  — the most mesh records. Right for an MHP3rd SOURCE, and the
                       same rule ``port_p3rd._p3rd_main_pmo`` selects with."""
    best = best_k = None
    for o, s in _subs(blob):
        if blob[o:o + 4] != magic:
            continue
        if pick is None:
            return blob[o:o + s]
        k = s if pick == "size" else struct.unpack_from("<I4f2H8I", blob, o + 8)[5]
        if best is None or k > best_k:
            best, best_k = blob[o:o + s], k
    return best


def _skel_fields(blob):
    """(parents, bind_world, stream_ids, bone_count, animated) of a 0xC0000000 sub."""
    sk = SKP.parse(blob)
    par = [b.parent for b in sk.bones]
    bw = bind_world_positions(par, [tuple(b.bind_pos) for b in sk.bones])
    sids, o = [], len(sk.header)
    for b in sk.bones:
        sids.append(struct.unpack_from("<H", blob, o + 0x50)[0])
        o += b.section_size
    animated = struct.unpack_from("<I", blob, 0x1C)[0]
    if not (0 < animated <= len(par)):
        animated = len(par)
    return par, bw, sids, struct.unpack_from("<I", blob, 4)[0], animated


def _runs(sids, animated):
    out, cur, n = [], sids[0], 0
    for s in sids[:animated]:
        if s == cur:
            n += 1
        else:
            out.append((cur, n)); cur, n = s, 1
    out.append((cur, n))
    return out


def check(pac, model_pac=None, geo=None):
    res = []                                   # (ok, name, detail)

    def ck(ok, name, detail=""):
        res.append((bool(ok), name, detail))
        return ok

    skel = _sub(pac, b"\x00\x00\x00\xc0")
    if skel is None:
        ck(False, "skeleton sub present", "no 0xC0000000 sub")
        return res
    par, bw, sids, bone_count, animated = _skel_fields(skel)
    n = len(par)

    ck(bone_count == n + 1, "skeleton bone_count == sections + 1",
       "bone_count=%d sections=%d (engine sets entity+0x1A4 = bone_count-1)"
       % (bone_count, n))
    ck(all(par[i] < i for i in range(n) if par[i] >= 0),
       "every parent precedes its child",
       "violations: %s" % [i for i in range(n) if 0 <= i <= n and par[i] >= i])

    # --- stream partition: contiguous runs, each a complete subtree -------------
    runs = _runs(sids, animated)
    ck(len({s for s, _ in runs}) == len(runs), "stream ids form contiguous runs",
       "runs=%s" % runs)
    kids = {}
    for i, p in enumerate(par):
        kids.setdefault(p, []).append(i)
    a, subtree_ok, detail = 0, True, []
    for si, (sid, cnt) in enumerate(runs):
        rng = set(range(a, a + cnt))
        outside = [i for i in rng if par[i] >= 0 and par[i] not in rng]
        # An appendage stream is exactly one subtree, so exactly one bone hangs off
        # a lower stream: its root. Stream 0 is the remainder and may legitimately
        # hold several root chains (2 of MHP3rd's 212 in-quest rigs do).
        if si and len(outside) > 1:
            subtree_ok = False
            detail.append("stream %d has %d roots %s" % (sid, len(outside), outside[:6]))
        back = [c for i in rng for c in kids.get(i, ()) if c < a]
        if back:                       # a child in an EARLIER stream breaks the walk
            subtree_ok = False
            detail.append("stream %d has children behind it %s" % (sid, back[:6]))
        a += cnt
    ck(subtree_ok, "each stream is one complete subtree", "; ".join(detail))

    # --- animation: partition must agree with the skeleton ----------------------
    anim = None
    for o, s in _subs(pac):
        if s > 64 and o + s <= len(pac):
            try:
                cand = IG.parse_ingame(pac[o:o + s])
            except Exception:
                continue
            if cand.streams and any(st.clips for st in cand.streams):
                anim = cand
                break
    if anim is not None:
        occupied = [(i, st) for i, st in enumerate(anim.streams) if st.clips]
        ck(True, "anim sub parses", "streams=%d slots=%s"
           % (len(anim.streams), [len(st.clips) for st in anim.streams]))
        # Each populated stream's blocks must carry exactly the bones the
        # skeleton assigns to that stream. A mismatch walks the FK joint array off
        # the end of a section — the v20/v21 crash class.
        want = [c for _sid, c in runs]
        detail, agree = [], True
        for k, (_i, st) in enumerate(occupied):
            got = sorted({len(bl.bones) for bl in st.clips.values()})
            if k < len(want) and got and got != [want[k]]:
                agree = False
                detail.append("stream %d carries %s bones, partition says %d"
                              % (k, got, want[k]))
        ck(agree, "anim bone counts match the skeleton partition", "; ".join(detail))

        # 🔴 THE FK INVARIANT (docs/agent_memory_map.md "Bone-count rule"):
        #     skeleton bone_count - 1  ==  anim partition total  ==  entity+0x1A4
        # Every check above is INTERNAL consistency — the partition against itself,
        # the skin against the source — so a build whose partition simply does not
        # SPAN the rig passes all of them. The shipped Zinogre did: it walked 51
        # joints with a 46-bone partition, leaving joints 46-50 (4% of the mesh,
        # the neck/jaw chain) with no bone section at all. Nothing here noticed.
        # ⚠️ The native Tigrex "violates" this by 3 and renders fine, so a shortfall
        # is a WARNING about unmanaged joints, not a proven crash — but it should
        # never be silent, and a shortfall that carries real geometry is a defect.
        part_total = sum(max(len(bl.bones) for bl in st.clips.values())
                         for _i, st in occupied)
        fk_walk = bone_count - 1
        # ⛔ RETRACTED 2026-08-29 AS A REQUIREMENT — it is a REPORT, never a gate.
        # This was added believing the bone-count rule demanded partition == walk.
        # It does not: the NATIVE Tigrex runs partition 45 against a walk of 48 and
        # is perfectly fine, so trailing joints WITHOUT a bone section are the normal
        # authored shape, not a defect. Worse, a Zinogre rebuilt to satisfy it
        # (partition 51 == walk 51) HUNG the game on the quest-load screen, while the
        # 46 build spawns and fights. Equality appears to be the one value the joint
        # builder cannot take. Report the shortfall and how much mesh rides it; do
        # NOT fail on it, and do not "fix" a build by growing the partition.
        ck(True, "anim partition vs FK walk (report only)",
           "partition %d vs FK walk %d (skeleton bone_count %d - 1)%s"
           % (part_total, fk_walk, bone_count,
              "" if part_total == fk_walk
              else " -> joints %d..%d get NO bone section"
                   % (part_total, fk_walk - 1)))

        # --- THE FORK RULE, measured ------------------------------------------
        # 🔴 A LOCATION channel translates its joint's whole SUBTREE, so every joint
        # carrying one must sit at or ABOVE the body fork (the first joint with more
        # than one child, where the rig splits front from rear). Land one below it and
        # it lifts half the animal; the waist geometry is left to span the gap.
        # ⚠️ Do NOT gate on the structure — the NATIVE Tigrex puts a loc channel on a
        # TAIL joint (43, displacing it up to 1873 u) and is fine, so "loc below the
        # fork" alone false-fails the control. Gate on the CONSEQUENCE instead: how
        # far geometry is actually stretched ACROSS the fork. Swept over every clip:
        #     native Tigrex        0 u
        #     Zinogre offset 1   208 u   <- the visible "L-shaped back"
        #     Zinogre offset 0    52 u
        # Raw magnitude will not do it either: native peaks at 239 u overall, HIGHER
        # than the broken Zinogre's 208 — but native's is an elbow and the Zinogre's
        # was the waist, every single clip.
        from mhfu_model import convert as _C
        from mhfu_model import stretch as _ST
        from mhfu_model.p3rd_anim_map import body_fork, loc_below_fork
        loc_joints = set()
        flat = IG.to_flat_anim(anim)
        for _a in flat.animations:
            for _j, _tr in enumerate(_a.tracks):
                for _ch in _tr.channels:
                    if _C.channel_kind(_ch.type)[0] == "loc" and _ch.keyframes:
                        loc_joints.add(_j)
        below = loc_below_fork(par, loc_joints) if loc_joints else []
        ck(True, "location channels vs the body fork (report only)",
           "fork = joint %d; loc on %s%s"
           % (body_fork(par), sorted(loc_joints),
              "" if not below else "; %s are BELOW it -> see the tear check" % below))

    else:
        ck(False, "anim sub parses", "no parsable in-game anim sub")

    # --- geometry: palette bones must exist and be animated ---------------------
    pmo = _sub(pac, b"pmo\x00", pick="size")
    if pmo is None:
        ck(False, "PMO sub present")
        return res
    sm = SK.read(pmo)
    pal = {b for vg in sm.vgroups for b in vg.palette}
    ck(max(pal) < n, "every skinned bone exists in the skeleton",
       "max palette bone=%d, skeleton has %d" % (max(pal), n))
    dead = sorted(b for b in pal if b >= animated)
    if dead:
        nv = sum(1 for vg in sm.vgroups for vi in vg.influences
                 if any(b in dead and w for b, w in vi))
        ck(nv * 4 < sum(len(vg.vertices) for vg in sm.vgroups),
           "little geometry on a NON-ANIMATED joint",
           "%d verts ride bones %s (>= animated %d) and cannot move"
           % (nv, dead[:8], animated))
    else:
        ck(True, "little geometry on a NON-ANIMATED joint")

    # --- does the clip TEAR the mesh across the body fork? ----------------------
    # The one check here that measures the shipped animation against the shipped
    # geometry instead of checking a table against a table. See the fork note above.
    if anim is not None:
        import mhfu_model as _MM
        _mm = _MM.load_pac_bytes(pac)
        tear, where = _ST.cross_fork_tear(_mm.model, _mm.skeleton,
                                          _MM.anim_ingame.to_flat_anim(anim).animations)
        ck(tear < 120.0, "clips do not tear the mesh across the body fork",
           "worst %.0fu%s" % (tear, "" if where is None else
                              " (slot %d frame %d, joints %d<->%d)" % where))

    # --- against the source ------------------------------------------------------
    if model_pac is not None:
        ssk_blob = _sub(model_pac, b"\x00\x00\x00\x80")
        ssk = SKP.parse(ssk_blob)
        spar = [b.parent for b in ssk.bones]
        sbw = bind_world_positions(spar, [tuple(b.bind_pos) for b in ssk.bones])

        # Rebuild the source->output bone correspondence WITHOUT trusting the
        # porter: match on bind position plus descendant count. Position alone is
        # ambiguous exactly where it matters — a leading-origin pad adds
        # placeholder joints at (0,0,0), indistinguishable from the source's own
        # origin chain, and a one-off there shifts the whole skin.
        def _desc(parents):
            kid = {}
            for i, pp in enumerate(parents):
                kid.setdefault(pp, []).append(i)
            out = [0] * len(parents)
            for i in reversed(range(len(parents))):
                out[i] = 1 + sum(out[c] for c in kid.get(i, ()))
            return out
        sdesc, odesc = _desc(spar), _desc(par)

        # ⚠️ Keyed on BIND POSITION ONLY. It used to include the descendant count,
        # which made it fire on any deliberate topology change: adopting the
        # Zinogre's orphan root chain (bones 46-50) under bone 1 adds 5 descendants
        # to bones 0 and 1, and the check then reported bones 0/1 "missing" while
        # all 51 were present and every bind position was unchanged. A survival
        # check must test SURVIVAL, or the next session "fixes" the port to satisfy
        # it. Descendant drift is reported below instead of failing.
        def key(p):
            return (round(p[0], 2), round(p[1], 2), round(p[2], 2))
        out_pos = {}
        for i, p in enumerate(bw):
            out_pos.setdefault(key(p), []).append(i)
        missing = [i for i, p in enumerate(sbw) if key(p) not in out_pos]
        _reparented = sum(1 for i, p in enumerate(sbw)
                          if key(p) in out_pos and i < len(sdesc)
                          and sdesc[i] not in [odesc[j] for j in out_pos[key(p)]])
        ck(not missing, "every SOURCE bone survives in the output rig"
           + (" (%d re-parented)" % _reparented if _reparented else ""),
           "%d missing, e.g. %s" % (len(missing), missing[:8]))

        perm = {}
        used = set()
        for i, p in enumerate(sbw):
            for c in out_pos.get(key(p), ()):
                if c not in used:
                    perm[i] = c
                    used.add(c)
                    break
        if geo is not None:
            src = P3.parse(_sub(model_pac, b"pmo\x00", pick="mesh"), geo_blob=geo)
            got = [{b: w for b, w in vi if w} for vg in sm.vgroups for vi in vg.influences]
            want = []
            for g in src.mesh_groups:
                for v in g.vertices:
                    acc = {}
                    for b, w in (v.get("influences") or []):
                        if w and b in perm:
                            acc[perm[b]] = acc.get(perm[b], 0.0) + w
                    t = sum(acc.values()) or 1.0
                    want.append({b: w / t for b, w in acc.items()})
            ck(len(got) == len(want), "vertex count preserved",
               "output %d, source %d" % (len(got), len(want)))
            bad = sum(1 for a2, b2 in zip(want, got) if set(a2) != set(b2))
            ck(not bad, "every vertex keeps its AUTHENTIC bone set",
               "%d of %d vertices differ" % (bad, len(want)))
            err = max((abs(a2[k] - b2[k]) for a2, b2 in zip(want, got)
                       for k in a2 if k in b2), default=0.0)
            ck(err <= 1 / 128.0, "weights within the u8 quantisation step",
               "max error %.6f" % err)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pac")
    ap.add_argument("--model", help="the MHP3rd source PAC, to cross-check against")
    ap.add_argument("--geo", help="the MHP3rd GE companion")
    a = ap.parse_args()
    res = check(open(a.pac, "rb").read(),
                open(a.model, "rb").read() if a.model else None,
                open(a.geo, "rb").read() if a.geo else None)
    for ok, name, detail in res:
        print("  %s  %-48s %s" % ("PASS" if ok else "FAIL", name,
                                  "" if ok else detail))
    bad = sum(1 for ok, _, _ in res if not ok)
    print("\n%d checks, %d failed" % (len(res), bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
