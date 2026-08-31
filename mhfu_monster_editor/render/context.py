"""Getting a GL context — the one place that knows whether a window exists.

Two doors, and everything downstream of them is identical:

* :func:`headless` makes a **standalone** context out of nothing. No window, no
  display, no compositor. This is what ``--headless out.png`` and any render test use.
* :func:`attached` binds moderngl to a context somebody else already made current —
  in practice `hello_imgui`'s.

Keeping both behind one module is what makes the layering in issue #5 real: `render/`
never asks whether it is on screen, so a regression image and a live viewport come out
of the same code.

Portability, measured rather than hoped
---------------------------------------
* **macOS** — a standalone context reports ``4.1 Metal``. 4.1 is the ceiling Apple ever
  shipped and it is deprecated, but core 3.3 is a subset of it, so asking for 330 is
  honest and gets satisfied. There is no compatibility profile: **every** draw needs a
  VAO and a shader, which is why nothing here uses immediate mode.
* **Linux** — moderngl reaches for EGL and then GLX. A headless box needs *some*
  libGL/libEGL; Mesa's llvmpipe is enough and needs no GPU. A box with neither cannot
  render at all, and :func:`headless` says exactly that instead of raising moderngl's
  bare ``Cannot create context``. ⚠️ The project's headless Linux server is such a box (only
  `libvulkan` + `/dev/dri`), which is why the server path is a container — issue #13.
* **Windows** — WGL, via the installed display driver.
"""
from __future__ import annotations

import sys
from typing import Optional

#: the floor. 3.3 core is the common subset of macOS 4.1, Mesa and every live Windows
#: driver; nothing in this package uses anything above it.
GL_VERSION = 330


class ContextError(RuntimeError):
    """No GL context could be made, with the reason a human can act on."""


def _import_moderngl():
    try:
        import moderngl
    except ImportError as e:                                # pragma: no cover
        raise ContextError(
            "moderngl is not installed. The editor's render and ui layers are optional "
            "extras — the manifest and core layers do not need them:\n"
            "    pip install -r mhfu_monster_editor/requirements.txt") from e
    return moderngl


def headless(version: int = GL_VERSION):
    """A standalone context, with a diagnosis instead of a stack trace on failure."""
    moderngl = _import_moderngl()
    try:
        return moderngl.create_standalone_context(require=version)
    except Exception as e:
        raise ContextError(_diagnose(e, version)) from e


def attached(version: int = GL_VERSION):
    """Bind to the context the caller has already made current.

    The window toolkit owns creation; we only want moderngl's view of it. Called from
    inside `hello_imgui`'s frame callback, where the context is guaranteed current —
    call it before the runner starts and there is nothing to attach to.
    """
    moderngl = _import_moderngl()
    try:
        return moderngl.create_context(require=version)
    except Exception as e:
        raise ContextError(
            "could not attach to the window's GL context (wanted %s core). If this "
            "fired before the first frame, the context was not current yet — attach "
            "from inside the frame callback.\n  %s" % (_pretty(version), e)) from e


def _pretty(version: int) -> str:
    return "%d.%d" % divmod(version, 100)[0:2] if version >= 100 else str(version)


def _diagnose(exc: Exception, version: int) -> str:
    v = "%d.%d" % (version // 100, (version % 100) // 10)
    head = "no standalone GL %s context on this machine (%s)" % (v, exc)
    if sys.platform == "darwin":
        return head + (
            "\n  macOS tops out at GL 4.1 and supplies it through Metal; a failure "
            "here usually means the process has no window server session at all "
            "(a bare ssh login, or a daemon).")
    if sys.platform == "win32":                             # pragma: no cover
        return head + ("\n  Windows renders through the display driver's WGL. A "
                       "session with no driver — some CI images, some RDP setups — "
                       "has no GL at all.")
    return head + (
        "\n  On Linux moderngl needs EGL or GLX. A headless box still needs a GL "
        "library; Mesa's software rasteriser is enough and needs no GPU:"
        "\n      apt install libegl1 libgl1 libglx-mesa0   # Debian/Ubuntu"
        "\n      dnf install mesa-libEGL mesa-libGL        # Fedora/RHEL"
        "\n  If the host has only libvulkan and /dev/dri — a headless server — "
        "there is no GL path at all and the render must run in a container (issue #13).")


def describe(ctx) -> str:
    """One line naming what we actually got. Worth logging: 'Metal' vs 'llvmpipe'
    explains a timing difference or a rasterisation difference on sight."""
    info = ctx.info
    return "GL %s | %s | %s" % (info.get("GL_VERSION", "?"),
                                info.get("GL_RENDERER", "?"),
                                info.get("GL_VENDOR", "?"))


def max_samples(ctx, wanted: int) -> int:
    """``wanted`` clamped to what the driver will actually give, never below 0.

    Asking for more samples than ``GL_MAX_SAMPLES`` is an incomplete-framebuffer error
    at bind time, a long way from the line that asked.
    """
    if wanted <= 1:
        return 0
    try:
        cap = int(ctx.info.get("GL_MAX_SAMPLES", 0))
    except (TypeError, ValueError):                          # pragma: no cover
        cap = 0
    return max(0, min(int(wanted), cap))


def gl_error_text(ctx) -> Optional[str]:                     # pragma: no cover
    """Drain the GL error queue, or None. For a 'nothing drew' investigation."""
    codes = []
    for _ in range(8):
        e = ctx.error
        if e == "GL_NO_ERROR":
            break
        codes.append(e)
    return ", ".join(codes) or None
