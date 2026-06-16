"""Phase 4 guard: the host inject path (mhfu_model.inject).

Acceptance (host half — the in-game effect is a HITL step): fileId parsing,
atomic memstick write, and that an engine-invalid edit is BLOCKED before any
bytes are written.

Run: PYTHONPATH=tools python tools/mhfu_model/tests/test_inject.py
"""
from __future__ import annotations

import os
import tempfile

from mhfu_model import load_pac
from mhfu_model import inject as I
from mhfu_model.model import Keyframe

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                    "workspace", "extracted", "data_files")
TIGREX = os.path.join(DATA, "file_06134.bin")


def test_file_id_roundtrip():
    assert I.file_id_from_name("any/dir/file_06134.bin") == 6134
    assert I.file_id_from_name("file_06060.bin") == 6060
    assert I.inject_filename(6134) == "file_06134.bin"
    try:
        I.file_id_from_name("tigrex.bin")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for un-numbered name")


def test_atomic_write_no_temp_leftover():
    with tempfile.TemporaryDirectory() as d:
        payload = b"\xde\xad\xbe\xef" * 64
        path = I.write_inject_bytes(payload, 6134, inject_dir=d)
        assert os.path.basename(path) == "file_06134.bin"
        with open(path, "rb") as f:
            assert f.read() == payload
        # the .tmp staging file must be gone (atomic os.replace)
        assert not os.path.exists(path + ".tmp")
        assert sorted(os.listdir(d)) == ["file_06134.bin"]


def test_emit_clean_model_writes():
    mm = load_pac(TIGREX)
    with tempfile.TemporaryDirectory() as d:
        path = I.emit_model(mm, file_id=6134, inject_dir=d, validate=True)
        # layout-preserving: same size as the source PAC (so the PRX signature
        # scan matches the loaded buffer and the in-place overwrite is safe).
        assert os.path.getsize(path) == os.path.getsize(TIGREX)


def test_invalid_edit_blocked_before_write():
    mm = load_pac(TIGREX)
    # push a rotation keyframe out of s16 range -> S16_OVERFLOW error
    ch = next(c for a in mm.anim.animations for tr in a.tracks
              for c in tr.channels if c.keyframes)
    ch.keyframes[0] = Keyframe(value=70000, frame=0, ease_in=0, ease_out=0)
    with tempfile.TemporaryDirectory() as d:
        try:
            I.emit_model(mm, file_id=6134, inject_dir=d, validate=True)
        except ValueError:
            pass
        else:
            raise AssertionError("expected emit_model to block an invalid edit")
        # nothing written
        assert os.listdir(d) == []


if __name__ == "__main__":
    test_file_id_roundtrip()
    test_atomic_write_no_temp_leftover()
    test_emit_clean_model_writes()
    test_invalid_edit_blocked_before_write()
    print("OK — inject: fileId parse, atomic memstick write, invalid edit blocked")
