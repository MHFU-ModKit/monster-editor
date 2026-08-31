"""``python -m mhfu_monster_editor <pac|port.toml>`` — the app, or one headless PNG.

    # interactive (macOS, and any Linux/Windows box with a display)
    python -m mhfu_monster_editor workspace/extracted/data_files/file_06185.bin
    python -m mhfu_monster_editor ports/zinogre.toml --side source

    # no display anywhere in sight
    python -m mhfu_monster_editor <pac> --headless out.png --view side --size 1600x1200

    # "what IS that patch of geometry?" — `render_port_views.py`'s HILITE/ONLY pair
    python -m mhfu_monster_editor <pac> --headless head.png --hilite 3,4,5,6 --only tagged
    python -m mhfu_monster_editor <pac> --headless rest.png --hilite 3,4,5,6 --only rest

Both paths open the same :class:`~mhfu_monster_editor.core.scene.Scene` and draw it
through the same :class:`~mhfu_monster_editor.render.viewport.Viewport`; the only
difference is whether there is a window in front of it.

`python -m mhfu_monster_editor.core` is still the *text* report on a PAC — this is the
picture. They take the same target argument on purpose.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

from .render.camera import VIEWS
from .render.mesh import MODES


def _size(text: str) -> Tuple[int, int]:
    try:
        w, h = text.lower().replace("*", "x").split("x")
        size = (int(w), int(h))
    except ValueError:
        raise argparse.ArgumentTypeError("size must look like 1280x800, not %r" % text)
    if min(size) < 16:
        raise argparse.ArgumentTypeError("size %dx%d is too small to render" % size)
    return size


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m mhfu_monster_editor",
        description="The monster editor: a PAC or a port.toml, on screen or as a PNG.")
    p.add_argument("target", help="a monster PAC, or a ports/*.toml manifest")
    p.add_argument("--headless", metavar="OUT.PNG",
                   help="render one frame to this file and exit; needs no display")
    p.add_argument("--size", type=_size, default=(1440, 900), metavar="WxH",
                   help="window size, or headless image size (default 1440x900)")
    p.add_argument("--view", default="three", choices=sorted(VIEWS),
                   help="starting camera angle (default three)")
    p.add_argument("--samples", type=int, default=4, metavar="N",
                   help="MSAA samples, clamped to what the driver allows (default 4)")
    p.add_argument("--no-grid", action="store_true", help="hide the ground plane")

    v = p.add_argument_group("what to draw (issue #6)")
    v.add_argument("--slot", type=int, metavar="N",
                   help="pose with this clip slot instead of the chosen default")
    v.add_argument("--frame", type=float, metavar="F",
                   help="frame within the clip (default its midpoint)")
    v.add_argument("--bind", action="store_true",
                   help="the BIND pose. ⚠️ it cannot validate skinning — at bind every "
                        "joint is at bind and any weighting looks correct")
    v.add_argument("--shading", default="textured", choices=list(MODES),
                   help="textured (default), flat, or a colour per vertex group")
    v.add_argument("--wireframe", action="store_true")
    v.add_argument("--no-bones", action="store_true", help="hide the joint overlay")
    v.add_argument("--hilite", metavar="J,J,...",
                   help="paint these joints red — `MHFU_VIEW_HILITE`")
    v.add_argument("--only", choices=("tagged", "rest"),
                   help="draw only --hilite's joints, or only everything else "
                        "— `MHFU_VIEW_ONLY` 1 and 2")

    g = p.add_argument_group("manifest targets")
    g.add_argument("--pac", help="the built PAC for a port.toml (default tmp/<pac>)")
    g.add_argument("--side", default="port", choices=("port", "source"),
                   help="'port' is the built MHFU PAC, 'source' the MHP3rd donor")
    g.add_argument("--root", default="workspace",
                   help="where the extracts live (default workspace)")
    return p


def open_target(args):
    """The scene, with the manifest keywords only when the target IS a manifest."""
    from .core import open_scene

    kw = {}
    if Path(args.target).suffix == ".toml":
        kw = {"side": args.side, "root": args.root}
        if args.pac:
            kw["pac"] = args.pac
    elif args.pac or args.side != "port":
        raise SystemExit("--pac/--side only mean anything for a port.toml target")
    return open_scene(args.target, **kw)


def apply_startup(vp, args) -> None:
    """Put the parsed flags onto a viewport — the SAME function for both paths.

    The window and ``--headless`` differ only in whether something is looking at the
    result, so the flags cannot be interpreted twice: `--hilite 3,4,5 --only tagged`
    has to mean one thing. The app calls this on its first frame, `--headless` calls
    it from `render_to_file`'s configure hook.
    """
    vp.show_ground = not args.no_grid
    vp.show_skeleton = not args.no_bones
    vp.wireframe = args.wireframe
    vp.mesh.mode = MODES.index(args.shading)
    if args.hilite:
        vp.tag_joints(int(x) for x in args.hilite.replace(" ", "").split(",") if x)
    if args.only:
        vp.mesh.isolate = 1 if args.only == "tagged" else 2

    # --bind / --slot / --frame over the viewport's own chosen default pose.
    if args.bind:
        vp.set_pose(None)
    elif args.slot is not None or args.frame is not None:
        clip = vp.clip if args.slot is None else vp.scene.clip(args.slot)
        frame = clip.frames * 0.5 if args.frame is None else args.frame
        vp.set_pose(clip, frame)

    # re-frame AFTER the pose: a different clip puts the animal somewhere else.
    vp.camera.frame(vp.bounds).look(args.view)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        scene = open_target(args)
    except Exception as e:
        print("%s: %s" % (type(e).__name__, e), file=sys.stderr)
        return 2
    print(scene.summary(), file=sys.stderr)

    from .render.context import ContextError

    try:
        if args.headless:
            from .render.viewport import render_to_file

            out = render_to_file(scene, args.headless, size=args.size,
                                 view=args.view, samples=args.samples,
                                 configure=lambda vp: apply_startup(vp, args))
            print("wrote %s  (%dx%d)" % (out, *args.size))
            return 0

        from .ui import EditorApp

        app = EditorApp(scene, size=args.size, view=args.view, startup=args)
        app.run()
        return 0
    except ContextError as e:
        print("\n%s" % e, file=sys.stderr)
        return 3
    except ImportError as e:
        print("\nthe editor's UI needs imgui-bundle:\n"
              "    pip install -r mhfu_monster_editor/requirements.txt\n  (%s)" % e,
              file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
