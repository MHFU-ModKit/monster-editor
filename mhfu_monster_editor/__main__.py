"""``python -m mhfu_monster_editor <pac|port.toml>`` — the app, or one headless PNG.

    # interactive (macOS, and any Linux/Windows box with a display)
    python -m mhfu_monster_editor workspace/extracted/data_files/file_06185.bin
    python -m mhfu_monster_editor ports/zinogre.toml --side source

    # no display anywhere in sight
    python -m mhfu_monster_editor <pac> --headless out.png --view side --size 1600x1200

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

            def configure(vp):
                vp.show_ground = not args.no_grid

            out = render_to_file(scene, args.headless, size=args.size,
                                 view=args.view, samples=args.samples,
                                 configure=configure)
            print("wrote %s  (%dx%d)" % (out, *args.size))
            return 0

        from .ui import EditorApp

        app = EditorApp(scene, size=args.size, view=args.view)
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
