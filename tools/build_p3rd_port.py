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
                    help="raise the rest pose N units so the feet land on the "
                         "skeleton origin, MHFU's datum. Get the number from "
                         "tools/port_rest_floor.py (native reads +2.7). NOT the "
                         "retracted terrain-placement theory — see port_p3rd.py.")
    ap.add_argument("--skin", choices=("auto", "transfer", "source"), default="auto",
                    help="auto = nearest-bone blend + seam weld (no native ref needed); "
                         "transfer = copy the host frame's own native skinning onto the "
                         "geometry (same-family path, e.g. Brute<-Tigrex; no weld); "
                         "source = ship the monster's OWN authentic skin from its v102 "
                         "bone palette (best with --source-skeleton; auto-selected there)")
    ap.add_argument("--animated", type=int, default=None,
                    help="override the animated-bone count (= the anim 3-stream "
                         "partition total). Default: EVERY bone, which is what the "
                         "FK invariant wants (skeleton bone_count-1 == partition "
                         "total == entity+0x1A4). A partition shorter than the walk "
                         "leaves the trailing joints with a zeroed matrix and their "
                         "mesh collapses to the origin.")
    ap.add_argument("--em-id", type=int, default=None,
                    help="MHP3rd monster id (Zinogre = 40). Selects the animation "
                         "bone-offset + skipped-bone list from p3rd_anim_map, which "
                         "is REQUIRED for correct motion: P3rd anim records are not "
                         "positional.")
    ap.add_argument("--anim-bone-offset", type=int, default=None,
                    help="override the anim bone offset (measure it with "
                         "p3rd_anim_map.score_offsets, do not guess)")
    ap.add_argument("--reweight-undriven", action="store_true",
                    help="🔴 SUPERSEDED, do not use. Folds vertices riding a "
                         "record-less bone onto the nearest driven bone — which "
                         "tears the auxiliary plate across the whole body. Use "
                         "--drop-joints. Kept so the experiment is reproducible.")
    ap.add_argument("--drop-joints", default=None,
                    help="⚠️ comma list of SOURCE joints whose mesh groups are dropped. "
                         "The orphan-root chain is usually the SEVERED TAIL (a real "
                         "carvable object — the native Tigrex has one too), so do NOT "
                         "drop it. For geometry the target engine genuinely cannot "
                         "place. Inspect with render_port_views.py MHFU_VIEW_ONLY=1/2.")
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
                                  skin=a.skin, source_skeleton=a.source_skeleton,
                                  src_animated=a.animated, em_id=a.em_id,
                                  anim_bone_offset=a.anim_bone_offset,
                                  reweight_undriven=a.reweight_undriven,
                                  drop_joints=([int(x) for x in a.drop_joints.split(",")]
                                               if a.drop_joints else None))
    open(a.out, "wb").write(pac)
    print("wrote %s" % a.out)
    for k in ("mode", "src_bones", "lead_pad", "stream_split", "bone_reorder",
              "anim_bone_offset", "anim_driven_bones", "anim_max_driven_bone",
              "adopted_orphan_roots", "dropped_joints", "dropped_groups",
              "dropped_verts", "reweighted_verts", "reweight_map",
              "skeleton_bytes", "src_groups", "src_verts",
              "skin_mode", "welded_seams", "pmo_bytes",
              "tmh_bytes", "materials", "anim_clips", "bone_map_matched", "anim",
              "total", "native_total", "needs_relocate"):
        if k in info:
            print("  %-16s %s" % (k, info[k]))


if __name__ == "__main__":
    main()
