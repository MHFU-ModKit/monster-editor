#!/usr/bin/env python3
"""Phase-4 A1 visible proof: inflate the Tigrex's non-root bones via bind-scale.

The skeleton sub routes through the proven get_subresource hook (type=skeleton),
so this edit applies at quest load -> the live Tigrex renders visibly deformed.
We scale bones[1:] (NOT bone 0) so the sub's first 64 bytes (header + bone 0)
stay byte-identical to the original -> the PRX content gate still matches it.

Usage:
    PYTHONPATH=tools python tools/mhfu_model/examples/edit_skeleton_demo.py \
        [src.bin] [out.bin] [factor]
Then stage with:  python -m mhfu_model.inject <out.bin>
"""
import sys
from mhfu_model import load_pac, repack
from mhfu_model.constraints import validate, register_template


def main(argv):
    src = argv[1] if len(argv) > 1 else "workspace/extracted/data_files/file_06134.bin"
    out = argv[2] if len(argv) > 2 else "workspace/modified/file_06134.bin"
    factor = float(argv[3]) if len(argv) > 3 else 1.5

    mm = load_pac(src)
    sk = mm.skeleton
    n = 0
    for b in sk.bones[1:]:                     # skip bone 0 -> keep the 64-byte head intact
        sx, sy, sz = b.bind_scale
        b.bind_scale = (sx * factor, sy * factor, sz * factor)
        px, py, pz = b.bind_pos               # also displace bones (rigid mesh follows)
        b.bind_pos = (px, py + 120.0, pz)
        n += 1

    # ALSO edit PMO geometry (definitely visible, routes through pmoCompile via
    # get_subresource). Scale every vertex position ~1.2x -> chunkier Tigrex.
    # Header/tables (first 64 B) are untouched, so the content gate still matches.
    vtx = 0
    if mm.model and mm.model.mesh_groups:
        mm.model.edited = True
        for g in mm.model.mesh_groups:
            for v in g.vertices:
                if not v:
                    continue
                for ax in ("x", "y", "z"):
                    if ax in v:
                        v[ax] *= 1.2
                vtx += 1
    print(f"scaled {vtx} PMO vertices x1.2")
    mm.skeleton.edited = getattr(mm.skeleton, "edited", False)
    print(f"scaled {n} non-root bones x{factor} (bone 0 untouched)")

    register_template("tigrex", load_pac(src).skeleton)
    rep = validate(mm.model, mm.skeleton, mm.anim.animations if mm.anim else [],
                   target_species="tigrex")
    print("validate:", "OK" if rep else "ERRORS", f"({len(rep.warnings)} warnings)")
    if not rep:
        for r in rep.errors:
            print("  ", r)
        return 2

    data = repack(mm)
    with open(out, "wb") as f:
        f.write(data)
    src_sz = len(open(src, "rb").read())
    print(f"wrote {out} ({len(data)} bytes; source {src_sz}) — "
          f"{'SAME SIZE (gate-safe)' if len(data) == src_sz else 'SIZE CHANGED!'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
