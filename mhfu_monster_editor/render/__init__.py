"""`mhfu_monster_editor.render` — GL that does not care whether a window exists.

The middle of the three layers issue #5 asks for: `core` (no GL) -> **`render`** (GL,
with or without a window) -> `ui` (imgui). Everything draws into an offscreen
:class:`~.target.Target`, so the app's viewport panel and a headless regression PNG are
the same pixels out of the same code.

  * :mod:`.camera` — :class:`~.camera.OrbitCamera`, :class:`~.camera.Bounds` and the
    4x4s. **Pure numpy**: importable, and testable, with no graphics driver at all.
  * :mod:`.context` — the two doors to a GL context, headless or attached, and a
    diagnosis rather than a stack trace when there is none.
  * :mod:`.target` — the offscreen framebuffer, MSAA resolve, RGBA read, PNG out.
  * :mod:`.shaders` — the GLSL, cached per context.
  * :mod:`.overlay` — ground, axes, bounds box, point cloud.
  * :mod:`.viewport` — :class:`~.viewport.Viewport`: one frame, and
    :func:`~.viewport.render_to_file` for the one-shot.

🔴 World space is engine space — X flank, **Y up, +Z the nose** — and there is no
conversion matrix anywhere in this package. :mod:`.camera` records how that was
measured.
"""
from .camera import (UP, VIEWS, Bounds, OrbitCamera, gl_bytes, look_at, perspective)
from .context import ContextError, attached, describe, headless
from .target import Target, write_png
from .viewport import BACKGROUND, Viewport, render_to_file, scene_bounds

__all__ = ["BACKGROUND", "Bounds", "ContextError", "OrbitCamera", "Target", "UP",
           "VIEWS", "Viewport", "attached", "describe", "gl_bytes", "headless",
           "look_at", "perspective", "render_to_file", "scene_bounds", "write_png"]
