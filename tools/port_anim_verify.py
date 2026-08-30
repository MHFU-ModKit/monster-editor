"""Does a BUILT port actually play its source moveset? Compare joint-for-joint.

The offline render tells you a port *looks* right. This tells you the motion is the
source's motion, as a number: pose the SOURCE clips on the source rig (through the
MHP3rd record→bone map) and the BUILT PAC's clips on its own rig, then compare every
joint at the clip's midpoint. A faithful port matches to ~0.

🔴 COMPARE UNDER THE PORTER'S PERMUTATION, NOT BY INDEX. `port_p3rd` REORDERS bones
(the Zinogre: source 18–23 → built 33–38, source 24–38 → built 18–32), so an
index-wise diff reports 570–830-unit differences on exactly that band while every
other joint matches perfectly. That reads like a broken limb chain and is an artefact
of the comparison. The permutation is recovered here by tree isomorphism — same shape,
same bind offsets — walking down from the root.

⚠️ The porter's ground lift is a constant Y offset on every joint; it is removed
before comparing, and reported, so it cannot mask a real difference.

⚠️ Needs `mathutils`, so it runs INSIDE the Blender container:

    ./blender_mhfu/blender-docker.sh --background \
        --python tools/port_anim_verify.py -- <built.bin> <em_id> <model_pac_id>

e.g. `-- tmp/zinogre_v10.bin 40 5339`. Model pac N => geometry N+1, moveset N+2.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "blender_mhfu")))
sys.path.insert(0, _HERE)

import fk_bake                                              # noqa: E402
from mhfu_model import anim_ingame as AI                    # noqa: E402
from mhfu_model import load_pac, load_pac_p3rd              # noqa: E402
from mhfu_model import p3rd_anim_map as _p3am               # noqa: E402
from mhfu_model.pac import MonsterPac                       # noqa: E402

_argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
PORT = _argv[0] if _argv else "tmp/zinogre_v10.bin"
EM = int(_argv[1]) if len(_argv) > 1 else 40
MON = int(_argv[2]) if len(_argv) > 2 else 5339
ROOT = os.environ.get("MHFU_P3RD_ROOT", "workspace/extracted_mhp3/data_files")
TOL = float(os.environ.get("MHFU_VERIFY_TOL", "1.0"))


def _children(skel):
    out = {}
    for b in skel.bones:
        out.setdefault(b.parent, []).append(b.index)
    return out


def bone_permutation(src_skel, port_skel):
    """source bone -> built bone, by tree isomorphism.

    ⚠️ The built rig is NOT always the source rig permuted — `port_p3rd` can insert a
    LEAD PAD of origin bones at the root (the Brute builds 47 bones from 46, the
    Zinogre 51 from 51). Anchoring blindly at built bone 0 then matches almost
    nothing: on the Brute it matched 2 of 46 and the caller went on to "compare" two
    joints and report a 1872-unit failure — a false alarm far worse than no check.
    So try every node down the built root's single-child chain and keep the best.
    """
    sb = {b.index: b for b in src_skel.bones}
    pb = {b.index: b for b in port_skel.bones}
    sk, pk = _children(src_skel), _children(port_skel)

    def isomorphism(start):
        perm = {}

        def walk(si, bi):
            perm[si] = bi
            used = set()
            for s in sk.get(si, []):
                best, bd = None, 1e9
                for b in pk.get(bi, []):
                    if b in used:
                        continue
                    d = max(abs(sb[s].bind_pos[k] - pb[b].bind_pos[k]) for k in range(3))
                    if d < bd:
                        best, bd = b, d
                # A near-exact bind offset under the same parent is the match. A loose
                # one means the chains genuinely differ — leave it unmatched rather
                # than inventing a correspondence the comparison would call fine.
                if best is not None and bd < 0.5:
                    used.add(best)
                    walk(s, best)
        walk(0, start)
        return perm

    best, pad = {}, 0
    node = 0
    for depth in range(8):
        cand = isomorphism(node)
        if len(cand) > len(best):
            best, pad = cand, depth
        kids = pk.get(node, [])
        if len(kids) != 1:                  # past the pad once the rig branches
            break
        node = kids[0]
    return best, pad


def main() -> int:
    src = load_pac_p3rd(os.path.join(ROOT, "file_%05d.bin" % MON),
                        geo_path=os.path.join(ROOT, "file_%05d.bin" % (MON + 1)),
                        anim_path=os.path.join(ROOT, "file_%05d.bin" % (MON + 2)))
    port = load_pac(PORT)
    ig = AI.parse_ingame(MonsterPac.from_bytes(open(PORT, "rb").read()).subs[3].data)
    built = {a.slot: a for a in AI.to_flat_anim(ig).animations}

    perm, pad = bone_permutation(src.skeleton, port.skeleton)
    n_src = len(src.skeleton.bones)
    unmatched = sorted(set(b.index for b in src.skeleton.bones) - set(perm))
    print("bone permutation: %d/%d source bones matched (lead pad %d); unmatched %s"
          % (len(perm), n_src, pad, unmatched or "none"))
    # 🔴 REFUSE TO JUDGE on a bad permutation. Every number below is meaningless if
    # the two rigs were not lined up, and a confident "the motion DIFFERS" from a
    # 2-joint comparison is worse than saying nothing.
    if len(perm) < 0.8 * n_src:
        print("❌ could not line the two rigs up (%d%% matched) — NOT a verdict on the "
              "port. Check that the source model pac id is right and that the build "
              "really came from it." % round(100.0 * len(perm) / max(1, n_src)))
        return 2

    nrec = max((len(a.tracks) for a in src.anim.animations), default=0)
    r2b = {r: b for b, r in
           _p3am.for_monster(EM, nrec, len(src.skeleton.bones)).items()}
    print("em%03d: %d records -> joints %d..%d (offset=%s)\n"
          % (EM, nrec, min(r2b.values()), max(r2b.values()),
             _p3am.BONE_OFFSET.get(EM, _p3am.DEFAULT_BONE_OFFSET)))

    worst, checked, missing, partial = 0.0, 0, [], []
    print("slot  joints   median     p90      max   lift")
    for a in sorted(src.anim.animations, key=lambda x: x.slot):
        if a.slot not in built:
            missing.append(a.slot)
            continue
        ss = fk_bake.sample(a, r2b)
        frames = fk_bake.key_frames(ss)
        if len(frames) < 2:
            continue
        mid = frames[len(frames) // 2]
        sp = fk_bake.sample(built[a.slot], None)
        # ⚠️ NOT every port slot holds a whole-rig clip. The host fills some slots
        # only in streams 2/4 — a 9-track head-and-neck clip played over an idle
        # body (the Tigrex's 24/25) — and the porter correctly leaves those alone.
        # Diffing a whole-rig source clip against one is meaningless, and reporting
        # it as "the motion differs" is a false alarm that hides real ones.
        n_src = sum(1 for v in ss.values() if v["rot"] or v["loc"])
        n_prt = sum(1 for v in sp.values() if v["rot"] or v["loc"])
        if n_prt * 2 < n_src:
            partial.append((a.slot, n_prt, n_src))
            continue
        ws = fk_bake.engine_world(src.skeleton, ss, mid)
        wb = fk_bake.engine_world(port.skeleton, sp, mid)
        # the lift is read at the SOURCE root's image, not built bone 0 — with a lead
        # pad those are different bones and bone 0 is an undriven origin.
        lift = wb[perm[0]].translation.y - ws[0].translation.y
        ds = []
        for s, b in perm.items():
            if s in ws and b in wb:
                v = ws[s].translation.copy()
                v.y += lift
                ds.append((v - wb[b].translation).length)
        if not ds:
            continue
        ds.sort()
        checked += 1
        worst = max(worst, ds[-1])
        print("%4d %6d %8.2f %7.2f %8.2f %6.1f"
              % (a.slot, len(ds), ds[len(ds) // 2], ds[int(len(ds) * 0.9)],
                 ds[-1], lift))

    print()
    if missing:
        print("⚠️  %d source clip(s) absent from the port: %s"
              % (len(missing), ", ".join(str(m) for m in missing)))
    if partial:
        print("⚠️  %d slot(s) hold the HOST's own PARTIAL clip, not a whole-rig one "
              "(not compared): %s"
              % (len(partial), ", ".join("%d (%d of %d joints)" % t for t in partial)))
    print("%d clip(s) compared, worst joint deviation %.2f units (tolerance %.2f)"
          % (checked, worst, TOL))
    if not checked:
        print("❌ nothing compared — wrong em id, model pac id, or port file?")
        return 2
    print("✅ the port plays its source moveset" if worst <= TOL else
          "❌ the port's motion DIFFERS from its source")
    return 0 if worst <= TOL else 1


sys.exit(main())
