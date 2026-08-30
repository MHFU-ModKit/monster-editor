#!/usr/bin/env python3
"""Where does a monster's REST POSE put its feet, relative to the skeleton origin?

MHFU's convention is "feet at the skeleton origin": the engine grounds the entity
and draws the rig from there, so a build whose rest pose puts its lowest bone at
Y != 0 is drawn that far off the floor. A cross-game port inherits the SOURCE
game's datum instead, and nothing in the structural checks notices.

The estimate is deliberately crude — lowest bone in bind pose, plus every locY
along its ancestor chain at frame 0 of the idle clip — and it is trustworthy for
exactly one reason:
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
    """Bind-pose floor of the LOWEST bone, with the idle clip's root motion applied.

    🔴 SUM THE WHOLE ANCESTOR CHAIN. This used to take the single largest-magnitude
    locY anywhere in the clip and call it "the pelvis lift". That is right only while
    exactly one joint carries the lift. The moment `--ground-lift` lands on a
    DIFFERENT joint from the source's own root motion — which is what happens once
    the anim record->bone map is correct, because records 0 and 1 are two separate
    location nodes — the two split (165.3 on joint 1 + 226.4 on joint 2) and reading
    only the larger under-reports the lift by the whole of the other one. It reported
    the Zinogre 162 units SUNK when the chain total was byte-identical to the build
    that measured correct.
    """
    mm = mhfu_model.load_pac(str(path))
    w = convert.bone_world_positions(mm.skeleton)
    low_bone = min(w.items(), key=lambda kv: kv[1][1])
    parents = {b.index: b.parent for b in mm.skeleton.bones}
    bind_y = {b.index: b.bind_pos[1] for b in mm.skeleton.bones}

    flat = ig.to_flat_anim(ig.parse_ingame(
        MonsterPac.from_bytes(Path(path).read_bytes()).find("anim").data))
    clip = next((a for a in flat.animations if a.slot == idle_slot), None)
    locy = {}
    for j, tr in enumerate(clip.tracks if clip else []):
        for ch in tr.channels:
            if (ch.type & 0xFFF) == LOCY and ch.keyframes:
                locy[j] = ch.keyframes[0].value / 16.0

    # walk low bone -> root, replacing each joint's bind Y with its channel Y
    lift, j, seen = 0.0, low_bone[0], set()
    while j is not None and j >= 0 and j not in seen:
        seen.add(j)
        if j in locy:
            lift += locy[j] - bind_y.get(j, 0.0)
        j = parents.get(j, -1)

    return {"path": Path(path).name, "low_bone": low_bone[0],
            "low_y": low_bone[1][1], "pelvis_lift": lift,
            "rest_floor": low_bone[1][1] + lift}


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
