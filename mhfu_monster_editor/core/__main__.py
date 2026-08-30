"""What is in this PAC? — the one command that answers it.

    python -m mhfu_monster_editor.core workspace/extracted/data_files/file_06185.bin
    python -m mhfu_monster_editor.core ports/zinogre.toml --pac tmp/zinogre_v10.bin
    python -m mhfu_monster_editor.core ports/zinogre.toml --side source
    python -m mhfu_monster_editor.core <pac> --clips
    python -m mhfu_monster_editor.core <pac> --slot 7 --frame 41
    python -m mhfu_monster_editor.core <pac> --bench

Prints the bind skeleton, the mesh groups, the textures and the clip table, plus
anything the loader had to guess or drop. `--slot` poses and reports what moved, which
is the cheap sanity check that a clip is really driving joints and not just present.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from .scene import SceneError, open_scene


def _pose_report(sc, slot, frame, top) -> None:
    clip = sc.clip(slot)
    at = clip.frames // 2 if frame is None else frame
    p = sc.pose(clip, at)
    moved = np.linalg.norm(p.joints - sc.rig.bind_joints, axis=1)
    order = np.argsort(-moved)
    print("\nclip %s  frame %g/%d  (%d tracks -> %d driven joints, loop=%s%s)"
          % (clip.name, at, clip.frames, clip.tracks, len(clip.driven), clip.loop,
             "" if clip.whole_rig else ", PARTIAL"))
    print("  joints displaced from bind: %d of %d, max %.1f u, median %.1f u"
          % (int((moved > 1e-6).sum()), len(moved), moved.max(),
             float(np.median(moved[moved > 1e-6])) if (moved > 1e-6).any() else 0.0))
    print("  furthest: " + ", ".join("j%d %+.0fu" % (j, moved[j]) for j in order[:top]))
    verts = p.skin(sc.merged)
    d = np.linalg.norm(verts - sc.merged.positions, axis=1)
    print("  vertices displaced: %d of %d, max %.1f u"
          % (int((d > 1e-6).sum()), len(d), d.max()))


def _bench(sc, slot) -> None:
    """Time the whole model through both paths — the same clip, frame and geometry."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
    import mhfu_model
    from mhfu_model import stretch as ST

    clip = sc.clip(slot)
    at = clip.frames // 2
    sc.pose(clip, at).skin(sc.merged)              # warm the curve compile + the merge
    n = 50
    t0 = time.perf_counter()
    for i in range(n):
        sc.pose(clip, at + (i % 3)).skin(sc.merged)
    fast = (time.perf_counter() - t0) / n * 1000

    raw = mhfu_model.load_pac(str(sc.path)).model.mesh_groups
    samples = ST.sample(clip._anim)
    t0 = time.perf_counter()
    for i in range(3):
        d = ST.deform_matrices(sc.skeleton, samples, at + (i % 3))
        for g in raw:
            ST.skin_group(g, d)
    slow = (time.perf_counter() - t0) / 3 * 1000

    print("\nwhole model, %d vertices, clip %s frame %d" % (sc.n_vertices, clip.name, at))
    print("  numpy  (core.pose)              %8.3f ms/frame  (%.0f fps)"
          % (fast, 1000.0 / fast))
    print("  scalar (mhfu_model.stretch)     %8.3f ms/frame   <- the reference" % slow)
    print("  speedup                         %8.1fx" % (slow / fast))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m mhfu_monster_editor.core")
    ap.add_argument("target", help="a model PAC, or a ports/*.toml manifest")
    ap.add_argument("--pac", help="the built PAC, when target is a manifest")
    ap.add_argument("--root", default="workspace", help="game-extract root")
    ap.add_argument("--side", choices=("port", "source"), default="port",
                    help="manifest only: the BUILT port, or its MHP3rd donor")
    ap.add_argument("--em-id", type=int, help="override the MHP3rd donor's em id")
    ap.add_argument("--clips", action="store_true", help="list every clip")
    ap.add_argument("--groups", action="store_true", help="list every mesh group")
    ap.add_argument("--slot", type=int, help="pose this clip and report what moved")
    ap.add_argument("--frame", type=float, help="frame for --slot (default: mid-clip)")
    ap.add_argument("--top", type=int, default=6)
    ap.add_argument("--bench", action="store_true",
                    help="time the numpy path against the scalar reference")
    a = ap.parse_args(argv)

    kw = {}
    if Path(a.target).suffix == ".toml":
        kw = {"root": a.root, "side": a.side}
        if a.pac:
            kw["pac"] = a.pac
    elif a.em_id is not None:
        kw = {"em_id": a.em_id}
    try:
        sc = open_scene(a.target, **kw)
    except (SceneError, FileNotFoundError) as e:
        print("%s" % e, file=sys.stderr)
        return 2
    print(sc.summary())

    if a.groups:
        print("\n  idx  verts  tris  mat  tex  vg_rec  maxinfl")
        for g in sc.groups:
            print("  %4d %6d %5d %4d %4s %7d %8d"
                  % (g.index, g.n_vertices, g.n_faces, g.material,
                     "-" if g.texture is None else g.texture, g.vg_rec,
                     g.skin.max_influences))
    if a.clips:
        print("\n  slot  frames  loop  tracks  driven  name")
        for c in sc.clips:
            print("  %4d %7d %5s %7d %7d  %s%s"
                  % (c.slot, c.frames, c.loop, c.tracks, len(c.driven), c.name,
                     "" if c.whole_rig else "   [partial]"))
    if a.slot is not None:
        _pose_report(sc, a.slot, a.frame, a.top)
    if a.bench:
        _bench(sc, a.slot if a.slot is not None else sc.clips[0].slot)
    return 0


if __name__ == "__main__":
    sys.exit(main())
