#!/usr/bin/env python3
"""CLI: port an MHP3rd big monster -> injectable MHFU PAC (port_p3rd backend).

Example (Brute Tigrex -> Tigrex host):
    python tools/build_p3rd_port.py \
        --model workspace/extracted_mhp3/data_files/file_05248.bin \
        --geo   workspace/extracted_mhp3/data_files/file_05249.bin \
        --anim  workspace/extracted_mhp3/data_files/file_05250.bin \
        --frame workspace/extracted/data_files/file_06185.bin \
        --out   tmp/brute_tigrex_v47_authentic.bin
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mhfu_model import port_p3rd as PORT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="MHP3rd model+skel+TMH PAC")
    ap.add_argument("--geo", help="MHP3rd GE-list companion (model+1)")
    ap.add_argument("--anim", help="MHP3rd raw moveset (model+2)")
    ap.add_argument("--frame", default="workspace/extracted/data_files/file_06185.bin",
                    help="MHFU host PAC (Tigrex)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--nb", type=int, default=3)
    ap.add_argument("--hops", type=int, default=1)
    ap.add_argument("--ground-lift", type=float, default=0.0,
                    help="raise the body N world-units (swap-spawn isn't terrain-placed)")
    ap.add_argument("--skin", choices=("auto", "transfer", "source"), default="auto",
                    help="auto = nearest-bone blend + seam weld (no native ref needed); "
                         "transfer = copy the host frame's own native skinning onto the "
                         "geometry (same-family path, e.g. Brute<-Tigrex; no weld); "
                         "source = ship the monster's OWN authentic skin from its v102 "
                         "bone palette (best with --source-skeleton; auto-selected there)")
    ap.add_argument("--source-skeleton", action="store_true",
                    help="ship the monster's OWN skeleton (no lossy down-rig to the host "
                         "rig) — for shapes that differ from the host; anim plays 1:1")
    a = ap.parse_args()

    model = open(a.model, "rb").read()
    frame = open(a.frame, "rb").read()
    geo = open(a.geo, "rb").read() if a.geo else None
    anim = open(a.anim, "rb").read() if a.anim else None

    pac, info = PORT.port_monster(model, frame, geo_companion=geo, anim_blob=anim,
                                  nb=a.nb, hops=a.hops, ground_lift=a.ground_lift,
                                  skin=a.skin, source_skeleton=a.source_skeleton)
    open(a.out, "wb").write(pac)
    print("wrote %s" % a.out)
    for k in ("mode", "src_bones", "lead_pad", "stream_split", "bone_reorder",
              "skeleton_bytes", "src_groups", "src_verts",
              "skin_mode", "welded_seams", "pmo_bytes",
              "tmh_bytes", "materials", "anim_clips", "bone_map_matched", "anim",
              "total", "native_total", "needs_relocate"):
        if k in info:
            print("  %-16s %s" % (k, info[k]))


if __name__ == "__main__":
    main()
