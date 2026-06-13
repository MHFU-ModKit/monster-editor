"""CLI: `python -m mhfu_model.dump <file_0XXXX.bin>` — model/skeleton/anim summary.

Phase 0 acceptance tool. Run from the repo's `tools/` dir or with PYTHONPATH=tools.
"""
from __future__ import annotations

import sys

from . import load_pac
from .anim import CHANNEL_BITS


def _names(mask):
    return ",".join(n for bit, n in CHANNEL_BITS if mask & bit) or "(none)"


def main(argv):
    if not argv:
        print(__doc__)
        return 1
    path = argv[0]
    mm = load_pac(path)
    pac = mm.pac
    print("PAC %s — %d sub-resources" % (path, len(pac.subs)))
    for s in pac.subs:
        print("  [%d] %-8s size=0x%06x" % (s.index, pac.role(s), len(s.data)))

    # byte-identity check
    same = pac.to_bytes() == pac.raw
    print("  round-trip: %s" % ("byte-identical OK" if same else "MISMATCH!"))

    if mm.skeleton:
        sk = mm.skeleton
        print("\nSKELETON: %d bones (parsed %d), roots=%s"
              % (sk.bone_count, len(sk.bones), sk.roots()))
        for b in sk.bones[:6]:
            print("  bone %2d parent=%-3d child=%-3d sib=%-3d  pos=(%.1f %.1f %.1f)"
                  % (b.index, b.parent, b.child, b.sibling, *b.bind_pos))
        if len(sk.bones) > 6:
            print("  ... (%d more)" % (len(sk.bones) - 6))

    if mm.model:
        md = mm.model
        tv = sum(g.vertex_count for g in md.mesh_groups)
        tf = sum(g.face_count for g in md.mesh_groups)
        print("\nMODEL: ver %r, %d mesh groups, %d verts, %d faces"
              % (md.version, len(md.mesh_groups), tv, tf))

    if mm.anim:
        an = mm.anim
        print("\nANIM: magic=0x%x, %d slots, %d populated"
              % (an.magic, an.slot_count, len(an.animations)))
        for a in an.animations[:4]:
            kf = sum(len(c.keyframes) for t in a.tracks for c in t.channels)
            print("  slot %2d: %d bone-tracks, loop=%d loop_start=%g, %d keyframes"
                  % (a.slot, a.bone_count, a.loop, a.loop_start, kf))
        if len(an.animations) > 4:
            print("  ... (%d more anims)" % (len(an.animations) - 4))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
