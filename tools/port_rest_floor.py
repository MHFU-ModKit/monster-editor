#!/usr/bin/env python3
"""Where does a monster's REST POSE put its feet, relative to the skeleton origin?

MHFU's convention is "feet at the skeleton origin": the engine grounds the entity
and draws the rig from there, so a build whose rest pose puts its lowest bone at
Y != 0 is drawn that far off the floor. A cross-game port inherits the SOURCE
game's datum instead, and nothing in the structural checks notices.

The estimate is deliberately crude — lowest bone in bind pose, plus the pelvis
locY at frame 0 of the idle clip — and it is trustworthy for exactly one reason:
run on the NATIVE it answers +2.6, i.e. on the floor. A method that lands the
control within 3 units of zero is measuring the right thing.

    python tools/port_rest_floor.py <pac> [<pac> ...]

Feed the number back as `build_p3rd_port.py --ground-lift <native - port>`.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mhfu_model
from mhfu_model import anim_ingame as ig, bone_match, convert
from mhfu_model.pac import MonsterPac

LOCY = 0x080


def rest_floor(path, idle_slot=1):
    mm = mhfu_model.load_pac(str(path))
    w = convert.bone_world_positions(mm.skeleton)
    low_bone = min(w.items(), key=lambda kv: kv[1][1])
    a = ig.parse_ingame(MonsterPac.from_bytes(Path(path).read_bytes()).find("anim").data)
    blk = next((st.clips[idle_slot] for st in a.streams if idle_slot in st.clips), None)
    # the pelvis = the bone carrying the big locY; the root above it stays ~0
    best = 0.0
    for bn in (blk.bones if blk else []):
        for ch in bn.channels:
            if (ch.ctype & 0xFFF) == LOCY and ch.keyframes:
                v = ch.keyframes[0].value / 16.0
                if abs(v) > abs(best):
                    best = v
    return {"path": Path(path).name, "low_bone": low_bone[0],
            "low_y": low_bone[1][1], "pelvis_lift": best,
            "rest_floor": low_bone[1][1] + best}


if __name__ == "__main__":
    rows = [rest_floor(p) for p in sys.argv[1:]]
    print(f"{'pac':<30} {'low bone':>9} {'bind Y':>9} {'pelvis lift':>12} "
          f"{'REST FLOOR':>11}")
    for r in rows:
        print(f"{r['path']:<30} {r['low_bone']:>9} {r['low_y']:>9.1f} "
              f"{r['pelvis_lift']:>12.1f} {r['rest_floor']:>11.1f}")
    if len(rows) >= 2:
        print(f"\nlift needed to match {rows[0]['path']}: "
              f"{rows[0]['rest_floor'] - rows[-1]['rest_floor']:+.1f} units "
              f"(--ground-lift)")
