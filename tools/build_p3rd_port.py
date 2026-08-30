#!/usr/bin/env python3
"""CLI: port an MHP3rd big monster -> injectable MHFU PAC (port_p3rd backend).

Example (Brute Tigrex -> Tigrex host):
    python tools/build_p3rd_port.py \
        --model workspace/extracted_mhp3/data_files/file_05248.bin \
        --geo   workspace/extracted_mhp3/data_files/file_05249.bin \
        --anim  workspace/extracted_mhp3/data_files/file_05250.bin \
        --frame workspace/extracted/data_files/file_06185.bin \
        --out   tmp/brute_tigrex_v47_authentic.bin

Or, with the flag soup in a manifest (`ports/*.toml`, mhfu_monster_editor.manifest) —
the same build, byte for byte, and a file the app and the runtime read too:

    python tools/build_p3rd_port.py --manifest ports/brute_tigrex.toml \\
        --out tmp/brute_tigrex_em058.bin

The manifest supplies a DEFAULT for every flag it knows; anything you also type on the
command line still wins, so `--manifest ports/zinogre.toml --skin auto` is a one-off
variant of a recorded build rather than a new flag soup.
"""
import argparse
import os
import re
import sys

_TOOLS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _TOOLS)
sys.path.insert(0, os.path.dirname(_TOOLS))          # repo root: mhfu_monster_editor
from mhfu_model import p3rd_anim_map as _p3am
from mhfu_model import port_p3rd as PORT


def _parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", help="ports/<name>.toml — supplies a default for "
                                       "every flag below (mhfu_monster_editor)")
    ap.add_argument("--data-root", default="workspace",
                    help="extract root the manifest's file ids resolve against "
                         "(<root>/extracted, <root>/extracted_mhp3)")
    ap.add_argument("--model", help="MHP3rd model+skel+TMH PAC")
    ap.add_argument("--geo", help="MHP3rd GE-list companion (model+1)")
    ap.add_argument("--anim", help="MHP3rd raw moveset (model+2)")
    ap.add_argument("--frame", default="workspace/extracted/data_files/file_06185.bin",
                    help="MHFU host PAC (Tigrex)")
    ap.add_argument("--out", help="output PAC. Falls back to tmp/<manifest port.pac>")
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
                    help="MHP3rd monster id (Zinogre = 40, Brute Tigrex = 58). Selects "
                         "the animation bone-offset + skipped-bone list from "
                         "p3rd_anim_map, which is REQUIRED for correct motion: P3rd "
                         "anim records are not positional. DEFAULT: resolved from the "
                         "--model file number. Pass a NEGATIVE value to opt out and "
                         "build unmapped (reproduces pre-2026-08-30 output).")
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
    return ap


def _explicit_dests(argv):
    """Which options the caller actually TYPED, as argparse dests.

    A manifest may only fill in what was left out, and "left out" cannot be inferred
    from the parsed value — `--ground-lift 0.0` and no flag at all both come back 0.0.
    So parse a second time against a copy of the same parser whose every default is
    None: anything not None was typed. This keeps one parser definition and does not
    care about `--flag=value`, abbreviations or ordering.
    """
    probe = _parser()
    for act in probe._actions:
        act.default = None
        act.required = False
    ns, _ = probe.parse_known_args(argv)
    return {k for k, v in vars(ns).items() if v is not None}


def _apply_manifest(ap, a, argv):
    """Fill every flag the caller did NOT type from the manifest. Explicit flags win."""
    from mhfu_monster_editor import manifest as MF

    man = MF.load(a.manifest)
    typed = _explicit_dests(argv)
    filled = []
    for dest, value in man.build_args(a.data_root).items():
        if dest in typed:
            continue
        setattr(a, dest, value)
        filled.append(dest)
    if a.out is None:
        # `port.pac` is a BARE FILENAME, and a built port is Capcom data spliced from
        # two games (docs/ASSETS.md category D). Dropping it in the CWD puts an
        # unignored 750 KB PAC at the repo root, one `git add -A` from being committed.
        # `tmp/` is the gitignored home ASSETS.md registers for it.
        a.out = os.path.join("tmp", man.pac)
        os.makedirs("tmp", exist_ok=True)
    print("[port] manifest %s -> %s (host species %d, %s em%03d); manifest set %s"
          % (a.manifest, man.name, man.host_species, man.source.game,
             man.source.em_id if man.source.em_id is not None else -1,
             ", ".join(sorted(filled)) or "nothing"))
    return man


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = _parser()
    a = ap.parse_args(argv)
    if a.manifest:
        _apply_manifest(ap, a, argv)
    for req in ("model", "out"):
        if getattr(a, req) is None:
            ap.error("--%s is required (or give a --manifest that supplies it)" % req)

    model = open(a.model, "rb").read()
    frame = open(a.frame, "rb").read()
    geo = open(a.geo, "rb").read() if a.geo else None
    anim = open(a.anim, "rb").read() if a.anim else None

    # 🔴 RESOLVE THE em ID FROM THE MODEL PAC BY DEFAULT. Without this the builder
    # took DEFAULT_BONE_OFFSET (2) for any monster the caller did not name, while
    # `render_anim_clips` looked the same monster up and used its MEASURED offset —
    # so a port could BUILD at one bone map and RENDER at another with nothing said.
    # That is how the Brute shipped at offset 2 (his fork is bone 1, so both his
    # location records landed on the front branch and tore his middle) while looking
    # acceptable, because `auto_skin` was smearing the bad map into something
    # plausible. The measured map is the correct default; guessing is the exception.
    # Escape hatch: pass a NEGATIVE --em-id to force the old unmapped behaviour, or
    # --anim-bone-offset to set the number outright.
    if a.em_id is None and a.model:
        m = re.search(r"file_(\d+)", os.path.basename(a.model))
        found = _p3am.em_for_model_pac(int(m.group(1))) if m else -1
        if found >= 0:
            a.em_id = found
            print("[port] %s -> em%03d: bone offset %d, %d skipped bone(s) "
                  "(p3rd_anim_map)"
                  % (os.path.basename(a.model), found,
                     _p3am.BONE_OFFSET.get(found, _p3am.DEFAULT_BONE_OFFSET),
                     len(_p3am.SKIPPED_BONES.get(found, []))))
        else:
            print("⚠️  %s is not in p3rd_anim_map.EM_BY_MODEL_PAC — building at the "
                  "DEFAULT bone offset %d, which is very probably WRONG. Read the "
                  "species name from the `MWo3` header at offset 32 of the overlay "
                  "two files below the model pac, pin the offset with the FORK RULE, "
                  "and add a row. -> docs/ANIMATION_FORMAT.md"
                  % (os.path.basename(a.model), _p3am.DEFAULT_BONE_OFFSET),
                  file=sys.stderr)
    if a.em_id is not None and a.em_id < 0:
        a.em_id = None                      # explicit opt-out of the lookup

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
