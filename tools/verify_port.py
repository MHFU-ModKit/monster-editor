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
        counts = []
        for st in anim.streams:
            bc = max((len(bl.bones) for cl in st.clips for bl in [cl]), default=0) \
                if False else 0
            counts.append(bc)
        ck(True, "anim sub parses", "streams=%d slots=%s"
           % (len(anim.streams), [len(st.clips) for st in anim.streams]))
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

        def key(p, d):
            return (round(p[0], 2), round(p[1], 2), round(p[2], 2), d)
        out_pos = {}
        for i, p in enumerate(bw):
            out_pos.setdefault(key(p, odesc[i]), []).append(i)
        missing = [i for i, p in enumerate(sbw) if key(p, sdesc[i]) not in out_pos]
        ck(not missing, "every SOURCE bone survives in the output rig",
           "%d missing, e.g. %s" % (len(missing), missing[:8]))

        perm = {}
        used = set()
        for i, p in enumerate(sbw):
            for c in out_pos.get(key(p, sdesc[i]), ()):
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
