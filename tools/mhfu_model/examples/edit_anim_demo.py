"""Phase 3 demo / HITL artifact: edit a Tigrex animation and repack the PAC.

Produces an engine-valid `file_06134.bin` whose animations are visibly distorted
(every rotation keyframe amplified), so the change is obvious in-game. The library
validates the result before writing. Load the output in PPSSPP (FUComplete file
replacer, or repack into DATA.BIN) and watch Tigrex move — this is the one HITL
acceptance step the automated tests can't cover.

Run:
    PYTHONPATH=tools python tools/mhfu_model/examples/edit_anim_demo.py \
        workspace/extracted/data_files/file_06134.bin \
        workspace/modified/file_06134.bin [amplify=1.5]
"""
from __future__ import annotations

import os
import sys

from mhfu_model import load_pac, repack
from mhfu_model import constraints as K
from mhfu_model.convert import channel_kind


def amplify_rotations(mm, factor):
    """Scale every rotation keyframe value by `factor` (clamped to s16)."""
    n = 0
    for a in mm.anim.animations:
        for tr in a.tracks:
            for ch in tr.channels:
                if channel_kind(ch.type)[0] != "rot":
                    continue
                for kf in ch.keyframes:
                    kf.value = max(-32768, min(32767, int(round(kf.value * factor))))
                    n += 1
    return n


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    src, dst = argv[0], argv[1]
    factor = float(argv[2]) if len(argv) > 2 else 1.5

    mm = load_pac(src)
    K.register_template("tigrex", mm.skeleton)
    n = amplify_rotations(mm, factor)
    rep = K.validate(mm.model, mm.skeleton, mm.anim, target_species="tigrex")
    if not rep.ok:
        print("VALIDATION FAILED:\n%s" % rep)
        return 1

    data = repack(mm)
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    with open(dst, "wb") as f:
        f.write(data)
    print("edited %d rotation keyframes (x%.2f); validated OK (%d warnings)"
          % (n, factor, len(rep.warnings)))
    print("wrote %s (%d bytes)" % (dst, len(data)))
    # confirm the change survives a re-decode
    back = load_pac(dst)
    print("re-decodes: %d anims, %d bones"
          % (len(back.anim.animations), len(back.skeleton.bones)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
