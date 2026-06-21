#!/usr/bin/env python3
"""Reproducible Brute-Tigrex MHFU monster-PAC builder.

Rebuilds the live-injectable Brute PAC from:
  * a SOURCE geometry PMO  (the Brute mesh in MHFU model-space, any PAC whose
    sub1 is a `pmo\\x00` blob — e.g. a previous brute_tigrex_vNN build), and
  * the NATIVE Tigrex frame (`file_06185.bin`): its sub0 skeleton drives the
    skinning, and its sub table / secondary subs / textures / animation are
    reused verbatim so the result is byte-for-byte the engine's expected layout
    (same 1216512 B -> same-size in-place inject path, no relocate).

The skinning is re-derived from scratch (chain-aware blend, pmo_skin.auto_skin)
so the result does not inherit the source's binding — only its geometry. This is
the offline/CLI twin of the Blender exporter's `export_monster_pac` path; both
call the same library, so a modder can reproduce this from Blender.

Usage:
    python tools/build_brute_pac.py \
        --geometry tmp/brute_tigrex_v44_autoblend.bin \
        --frame    workspace/extracted/data_files/file_06185.bin \
        --out      tmp/brute_tigrex_v45_chainskin.bin \
        [--nb 3] [--hops 2] [--render out.png]
"""
from __future__ import annotations

import argparse
import struct
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from mhfu_model import pmo_skin as PS


def find_pmo_sub(blob: bytes):
    n = struct.unpack_from("<I", blob, 0)[0]
    for i in range(n):
        o, s = struct.unpack_from("<II", blob, 4 + i * 8)
        if blob[o:o + 4] == b"pmo\x00":
            return o, s
    raise SystemExit("no PMO sub-resource in source")


# --------------------------------------------------------------------------- #
# build
# --------------------------------------------------------------------------- #
def build(geometry_path: str, frame_path: str, nb: int = 3, hops: int = 1):
    """Read the Brute geometry from a source PMO and re-skin it onto the native
    frame's skeleton. Thin CLI wrapper over pmo_skin.splice_skinned_pmo (the same
    library call the Blender exporter uses), so both paths are byte-identical."""
    geo_blob = open(geometry_path, "rb").read()
    frame = open(frame_path, "rb").read()
    go, gs = find_pmo_sub(geo_blob)
    geo_model = PS.read(geo_blob[go:go + gs])    # Brute geometry + its OWN tables
    return PS.splice_skinned_pmo(frame, geo_model, nb=nb, hops=hops)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry", required=True, help="source PAC (sub1 = Brute PMO geometry)")
    ap.add_argument("--frame", default="workspace/extracted/data_files/file_06185.bin",
                    help="native Tigrex PAC (skeleton + layout + secondary subs)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--nb", type=int, default=3, help="bones blended per vertex")
    ap.add_argument("--hops", type=int, default=2, help="tree radius for chain-aware blend")
    ap.add_argument("--render", help="optional offline EEVEE PNG via blender_mhfu/render_check.py")
    a = ap.parse_args()

    pac, st = build(a.geometry, a.frame, nb=a.nb, hops=a.hops)
    open(a.out, "wb").write(pac)
    print("wrote %s" % a.out)
    print("  skeleton bones=%d  vgroups=%d  verts=%d  avg_palette=%.1f"
          % (st["bones"], st["vgroups"], st["verts"], st["avg_pal"]))
    print("  pmo=%d B  slot=%d B  total=%d B%s"
          % (st["pmo_bytes"], st["slot"], st["total"],
             "  (size-matched)" if st["total"] == os.path.getsize(a.frame) else "  (!! size mismatch)"))
    if a.render:
        import subprocess
        rc = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "..", "blender_mhfu", "render_check.py")
        subprocess.run(["Blender", "--background", "--python", rc, "--",
                        a.out, a.render], check=False)


if __name__ == "__main__":
    main()
