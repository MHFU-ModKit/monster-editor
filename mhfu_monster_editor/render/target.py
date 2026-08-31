"""The offscreen surface everything draws into — window or not.

The viewport in the app is **not** drawn to the back buffer. It is drawn to a
framebuffer object and then shown as an ``imgui.image``, for one reason worth stating
plainly: it makes the windowed path and the ``--headless`` path the same path. A
regression PNG (issue #13) is then a picture of the viewport, not of something that
merely resembles it.

Two subtleties this hides:

1. **Multisampling cannot be read back.** A multisample framebuffer has no single
   colour per pixel, so :meth:`Target.read` resolves through a plain one first. That
   blit is also what an ``imgui.image`` needs, since it samples an ordinary texture.
2. 🔴 **GL's row 0 is the BOTTOM.** Everything else in this repo — `TextureImage.rgba`
   from a decoded TMH, Pillow, a PNG on disk — puts row 0 at the top. :meth:`read`
   flips, so a saved image is upright; :attr:`Target.uv` carries the same flip for
   `imgui.image`, which samples the texture directly and never goes through `read`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from .context import max_samples

#: default MSAA. 4x is free enough on every GPU this will meet and it is the
#: difference between a bone overlay that reads and one that crawls with aliasing.
SAMPLES = 4


class Target:
    """A colour+depth framebuffer of a given size, resizable, readable as RGBA8."""

    def __init__(self, ctx, size: Tuple[int, int], samples: int = SAMPLES) -> None:
        self.ctx = ctx
        self._wanted_samples = samples
        self.samples = max_samples(ctx, samples)
        self.size = (0, 0)
        self._fbo = None            # what we draw into (multisample when samples > 0)
        self._resolve = None        # single-sample; also the texture imgui shows
        self._tex = None
        self.resize(size)

    # ---- lifecycle ---------------------------------------------------- #
    def resize(self, size: Tuple[int, int]) -> bool:
        """Rebuild at ``size`` if it changed. Returns whether anything was rebuilt.

        Clamped to at least 1x1: a docked imgui panel dragged shut reports 0, and a
        zero-sized framebuffer is a GL error rather than an empty picture.
        """
        w = max(1, int(size[0]))
        h = max(1, int(size[1]))
        if (w, h) == self.size:
            return False
        self.release()
        self.size = (w, h)
        self._tex = self.ctx.texture((w, h), 4)
        self._tex.repeat_x = self._tex.repeat_y = False
        self._resolve = self.ctx.framebuffer(color_attachments=[self._tex])
        if self.samples:
            self._fbo = self.ctx.framebuffer(
                color_attachments=[self.ctx.renderbuffer((w, h), 4,
                                                         samples=self.samples)],
                depth_attachment=self.ctx.depth_renderbuffer((w, h),
                                                             samples=self.samples))
        else:
            self._fbo = self.ctx.framebuffer(
                color_attachments=[self._tex],
                depth_attachment=self.ctx.depth_renderbuffer((w, h)))
        return True

    def release(self) -> None:
        for obj in (self._fbo, self._resolve, self._tex):
            if obj is None:
                continue
            # a non-multisample Target shares its texture between fbo and resolve;
            # releasing the same object twice is harmless but noisy on some drivers.
            try:
                obj.release()
            except Exception:                                # pragma: no cover
                pass
        self._fbo = self._resolve = self._tex = None
        self.size = (0, 0)

    def __enter__(self) -> "Target":
        return self

    def __exit__(self, *exc) -> None:
        self.release()

    # ---- drawing ------------------------------------------------------ #
    @property
    def aspect(self) -> float:
        return self.size[0] / float(self.size[1])

    def clear(self, rgba=(0.10, 0.11, 0.13, 1.0)) -> None:
        self._fbo.clear(*rgba, depth=1.0)

    def use(self) -> None:
        """Make this the draw target and set the viewport to its full extent."""
        self._fbo.use()
        self.ctx.viewport = (0, 0, self.size[0], self.size[1])

    def resolve(self) -> None:
        """Collapse the multisample buffer into the readable texture. No-op at 0x."""
        if self.samples:
            self.ctx.copy_framebuffer(self._resolve, self._fbo)

    # ---- output ------------------------------------------------------- #
    @property
    def texture(self):
        """The single-sample colour texture. Call :meth:`resolve` first."""
        return self._tex

    @property
    def gl_texture_id(self) -> int:
        """The raw GL name, for ``imgui.image``."""
        return self._tex.glo

    #: ``(uv0, uv1)`` that flip GL's bottom-up texture for `imgui.image`.
    uv: Tuple[Tuple[float, float], Tuple[float, float]] = ((0.0, 1.0), (1.0, 0.0))

    def read(self, *, flip: bool = True) -> np.ndarray:
        """``(h, w, 4)`` uint8 with row 0 at the TOP — the repo's convention."""
        self.resolve()
        raw = self._resolve.read(components=4, alignment=1)
        w, h = self.size
        img = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 4)
        return img[::-1].copy() if flip else img.copy()

    def save(self, path, *, flip: bool = True) -> Path:
        """Write a PNG. Pillow if it is there, else a hand-rolled encoder."""
        path = Path(path)
        if path.parent and not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        write_png(path, self.read(flip=flip))
        return path

    def __repr__(self) -> str:
        return "<Target %dx%d msaa=%dx>" % (self.size[0], self.size[1], self.samples)


# --------------------------------------------------------------------------- #
# PNG
# --------------------------------------------------------------------------- #
def write_png(path, rgba: np.ndarray) -> None:
    """Write ``(h, w, 4)`` uint8 to ``path``.

    Pillow is already a dependency of `tools/` and is used when present, but the
    fallback exists so that ``--headless out.png`` — the deliverable of issue #5 —
    cannot fail on a missing *image* library when the *graphics* stack is fine.
    zlib and struct are stdlib.
    """
    rgba = np.ascontiguousarray(rgba, dtype=np.uint8)
    if rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError("write_png wants (h, w, 4) uint8, got %r" % (rgba.shape,))
    try:
        from PIL import Image
    except ImportError:
        _write_png_stdlib(path, rgba)
        return
    Image.fromarray(rgba, "RGBA").save(str(path))


def _write_png_stdlib(path, rgba: np.ndarray) -> None:
    import struct
    import zlib
    h, w, _ = rgba.shape
    # each scanline is prefixed with filter type 0 (None).
    rows = np.concatenate([np.zeros((h, 1), dtype=np.uint8), rgba.reshape(h, w * 4)],
                          axis=1)

    def chunk(tag: bytes, body: bytes) -> bytes:
        return (struct.pack(">I", len(body)) + tag + body
                + struct.pack(">I", zlib.crc32(tag + body) & 0xFFFFFFFF))

    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(rows.tobytes(), 6))
           + chunk(b"IEND", b""))
    Path(path).write_bytes(png)
