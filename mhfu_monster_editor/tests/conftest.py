"""The one fixture the GL tests share.

test_render_headless.py, test_render_mesh.py and test_render_playback.py each have a `main()`
that runs them as a script and hands the GL tests a context explicitly, so their signatures
are `def test_x(ctx)`. Under pytest that parameter is a fixture request; this is the fixture.
It is module-scoped (one context per file), and it skips the whole GL part on a box with no
driver — the same outcome the script mode prints as SKIP.
"""

import pytest


@pytest.fixture(scope="module")
def ctx():
    try:
        from mhfu_monster_editor.render.context import ContextError, headless
    except ImportError as e:                    # moderngl is not installed
        pytest.skip("moderngl is not installed (%s)" % e)
    try:
        return headless()
    except ContextError as e:
        pytest.skip("no GL on this machine: %s" % e)
