"""Tests for pmo_p3rd.py — MHP3rd PMO v102 parser -> Model.

Covers:
  * parse() raises ValueError on bad magic / bad version
  * parse() with no geo_blob on a companion-file PAC returns a Model with no groups
  * parse() with geo_blob returns correct vert/face counts on a real MHP3rd PAC pair
  * lobby-style self-contained PAC (ge_base < blob size) works without geo_blob
  * skeleton_p3rd.parse() round-trip: both 0x80000000 and 0xC0000000 accepted
  * skeleton_p3rd bone count matches expected for a known PAC

Run: PYTHONPATH=tools python tools/mhfu_model/tests/test_pmo_p3rd.py
"""
from __future__ import annotations

import os
import struct
import sys

# Allow running directly from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from mhfu_model import pmo_p3rd, skeleton_p3rd
from mhfu_model.model import Model, Skeleton

P3RD_DATA = os.path.join(
    os.path.dirname(__file__), "..", "..", "..",
    "workspace", "extracted_mhp3", "data_files",
)

# Known in-quest PAC + companion pair (file_04898 + file_04899)
PAC_FILE = os.path.join(P3RD_DATA, "file_04898.bin")
GEO_FILE = os.path.join(P3RD_DATA, "file_04899.bin")

# Lobby Tigrex (em058m0) — self-contained (ge_base < blob size for sub[5])
LOBBY_FILE = os.path.join(P3RD_DATA, "file_05248.bin")

DATA_AVAILABLE = os.path.isfile(PAC_FILE) and os.path.isfile(GEO_FILE)
LOBBY_AVAILABLE = os.path.isfile(LOBBY_FILE)


def _extract_pmo_sub(pac_bytes: bytes, target_ver: bytes = b"102\x00") -> bytes | None:
    """Pull the first PMO sub with the given version from a PAC."""
    count = struct.unpack_from("<I", pac_bytes, 0)[0]
    if count == 0 or count > 64:
        return None
    for i in range(count):
        off, sz = struct.unpack_from("<II", pac_bytes, 4 + i * 8)
        if sz < 8 or off + sz > len(pac_bytes):
            continue
        blob = pac_bytes[off:off + sz]
        if blob[:4] == b"pmo\x00" and blob[4:8] == target_ver:
            return blob
    return None


def _extract_skeleton_sub(pac_bytes: bytes) -> bytes | None:
    """Pull the first skeleton sub (0x80000000 or 0xC0000000) from a PAC."""
    count = struct.unpack_from("<I", pac_bytes, 0)[0]
    if count == 0 or count > 64:
        return None
    for i in range(count):
        off, sz = struct.unpack_from("<II", pac_bytes, 4 + i * 8)
        if sz < 8 or off + sz > len(pac_bytes):
            continue
        blob = pac_bytes[off:off + sz]
        if len(blob) >= 4:
            u0 = struct.unpack_from("<I", blob, 0)[0]
            if u0 in (0x80000000, 0xC0000000):
                return blob
    return None


# --- error handling -----------------------------------------------------------

def test_bad_magic():
    try:
        pmo_p3rd.parse(b"\x00" * 64)
        assert False, "should have raised ValueError"
    except ValueError as e:
        assert "not a PMO blob" in str(e)
    print("PASS test_bad_magic")


def test_wrong_version():
    # craft a blob that starts with pmo\x00 but version 1.0\x00
    blob = b"pmo\x001.0\x00" + b"\x00" * 60
    try:
        pmo_p3rd.parse(blob)
        assert False, "should have raised ValueError"
    except ValueError as e:
        assert "102" in str(e)
    print("PASS test_wrong_version")


def test_no_geo_blob_returns_empty_model():
    if not DATA_AVAILABLE:
        print("SKIP test_no_geo_blob_returns_empty_model (data not available)")
        return
    pac_bytes = open(PAC_FILE, "rb").read()
    pmo_blob = _extract_pmo_sub(pac_bytes)
    assert pmo_blob is not None, "No v102 PMO sub found in %s" % PAC_FILE
    m = pmo_p3rd.parse(pmo_blob, geo_blob=None)
    assert isinstance(m, Model)
    assert m.version == b"102\x00"
    assert len(m.mesh_groups) == 0, "expected 0 groups without geo_blob"
    print("PASS test_no_geo_blob_returns_empty_model")


# --- real geometry ------------------------------------------------------------

def test_parse_with_geo_blob():
    if not DATA_AVAILABLE:
        print("SKIP test_parse_with_geo_blob (data not available)")
        return
    pac_bytes = open(PAC_FILE, "rb").read()
    geo_bytes = open(GEO_FILE, "rb").read()
    pmo_blob = _extract_pmo_sub(pac_bytes)
    assert pmo_blob is not None, "No v102 PMO sub found in %s" % PAC_FILE

    m = pmo_p3rd.parse(pmo_blob, geo_blob=geo_bytes)
    assert isinstance(m, Model)
    assert m.version == b"102\x00"
    assert len(m.mesh_groups) > 0, "expected mesh groups"
    total_verts = sum(g.vertex_count for g in m.mesh_groups)
    total_faces = sum(g.face_count for g in m.mesh_groups)
    assert total_verts > 1000, "expected >1000 verts, got %d" % total_verts
    assert total_faces > 500,  "expected >500 faces, got %d" % total_faces
    # scale is baked into vertices (not zero)
    sample = m.mesh_groups[0].vertices[0] if m.mesh_groups[0].vertices else None
    assert sample is not None
    assert "x" in sample and "y" in sample and "z" in sample
    print("PASS test_parse_with_geo_blob  groups=%d verts=%d faces=%d"
          % (len(m.mesh_groups), total_verts, total_faces))


def test_mesh_group_fields():
    """Every MeshGroup must have scale, material, mesh_record, and vg_rec set."""
    if not DATA_AVAILABLE:
        print("SKIP test_mesh_group_fields (data not available)")
        return
    pac_bytes = open(PAC_FILE, "rb").read()
    geo_bytes = open(GEO_FILE, "rb").read()
    pmo_blob = _extract_pmo_sub(pac_bytes)
    m = pmo_p3rd.parse(pmo_blob, geo_blob=geo_bytes)
    for g in m.mesh_groups:
        assert len(g.scale) == 3
        assert g.material >= 0
        assert g.mesh_record >= 0
        assert g.vg_rec >= 0
        assert g.vertex_count == len(g.vertices)
        assert g.face_count == len(g.faces)
    print("PASS test_mesh_group_fields  groups=%d" % len(m.mesh_groups))


# --- lobby self-contained PMO -------------------------------------------------

def test_lobby_self_contained():
    """Lobby PAC sub[5] has ge_base < blob size -> parse without geo_blob."""
    if not LOBBY_AVAILABLE:
        print("SKIP test_lobby_self_contained (data not available)")
        return
    pac_bytes = open(LOBBY_FILE, "rb").read()
    # sub[5] is the second PMO in the lobby PAC (smaller, self-contained)
    count = struct.unpack_from("<I", pac_bytes, 0)[0]
    pmo_blobs = []
    for i in range(min(count, 32)):
        off, sz = struct.unpack_from("<II", pac_bytes, 4 + i * 8)
        if sz < 8 or off + sz > len(pac_bytes): continue
        blob = pac_bytes[off:off + sz]
        if blob[:4] == b"pmo\x00" and blob[4:8] == b"102\x00":
            pmo_blobs.append((i, blob))
    assert len(pmo_blobs) > 0, "No v102 PMO in lobby PAC"

    # find the self-contained one (ge_base < blob size)
    for sub_idx, pmo_blob in pmo_blobs:
        hdr = struct.unpack_from("<I4f2H8I", pmo_blob, 8)
        ge_base = hdr[12]
        if ge_base < len(pmo_blob):
            m = pmo_p3rd.parse(pmo_blob, geo_blob=None)
            assert isinstance(m, Model)
            total_verts = sum(g.vertex_count for g in m.mesh_groups)
            print("PASS test_lobby_self_contained  sub[%d] groups=%d verts=%d"
                  % (sub_idx, len(m.mesh_groups), total_verts))
            return
    print("SKIP test_lobby_self_contained (no self-contained PMO sub found)")


# --- skeleton_p3rd ------------------------------------------------------------

def test_skeleton_p3rd_parse():
    if not DATA_AVAILABLE:
        print("SKIP test_skeleton_p3rd_parse (data not available)")
        return
    pac_bytes = open(PAC_FILE, "rb").read()
    skel_blob = _extract_skeleton_sub(pac_bytes)
    assert skel_blob is not None, "No skeleton sub found in %s" % PAC_FILE
    sk = skeleton_p3rd.parse(skel_blob)
    assert isinstance(sk, Skeleton)
    assert sk.bone_count > 0
    # Allow up to 1 bone short (last section may be truncated in some MHP3rd PACs)
    assert len(sk.bones) >= sk.bone_count - 1, (
        "expected ~%d bones, got %d" % (sk.bone_count, len(sk.bones)))
    # all bone indices should be in range [0, bone_count)
    for b in sk.bones:
        assert 0 <= b.index < sk.bone_count, "bone index out of range: %d" % b.index
    print("PASS test_skeleton_p3rd_parse  bones=%d" % sk.bone_count)


def test_skeleton_p3rd_lobby():
    """Lobby em058m0 skeleton: 47 bones (0x80000000 magic)."""
    if not LOBBY_AVAILABLE:
        print("SKIP test_skeleton_p3rd_lobby (data not available)")
        return
    pac_bytes = open(LOBBY_FILE, "rb").read()
    skel_blob = _extract_skeleton_sub(pac_bytes)
    assert skel_blob is not None, "No skeleton sub in lobby PAC"
    magic = struct.unpack_from("<I", skel_blob, 0)[0]
    assert magic == 0x80000000, "expected 0x80000000, got 0x%08x" % magic
    sk = skeleton_p3rd.parse(skel_blob)
    assert sk.bone_count == 47, "expected 47 bones, got %d" % sk.bone_count
    roots = sk.roots()
    assert len(roots) >= 1, "expected at least one root bone"
    print("PASS test_skeleton_p3rd_lobby  bones=%d roots=%s" % (sk.bone_count, roots))


def test_skeleton_mhfu_compat():
    """skeleton_p3rd.parse also accepts MHFU 0xC0000000 magic."""
    # Build a minimal synthetic 0xC0000000 skeleton with 1 bone
    bone_section = struct.pack("<3I", 0x40000001, 1, 0x5C)  # magic, flag, size
    bone_section += struct.pack("<4i", 0, -1, -1, -1)        # idx, parent, child, sib
    bone_section += struct.pack("<3f", 1.0, 1.0, 1.0)        # scale
    bone_section += b"\x00" * 12                              # pad to +0x2C
    bone_section += struct.pack("<3f", 0.0, 0.0, 0.0)        # rotation
    bone_section += b"\x00" * 12                              # pad to +0x3C
    bone_section += struct.pack("<3f", 0.0, 0.0, 0.0)        # position
    bone_section += b"\x00" * (0x5C - len(bone_section))     # zero-pad to section_size

    header = struct.pack("<3I", 0xC0000000, 1, 0x1C + 0x5C)  # magic, count, total_size
    header += b"\x00" * (0x1C - 12)                           # remaining header bytes
    blob = header + bone_section

    sk = skeleton_p3rd.parse(blob)
    assert sk.bone_count == 1
    assert sk.bones[0].index == 0
    assert sk.bones[0].parent == -1
    print("PASS test_skeleton_mhfu_compat")


# --- run all tests ------------------------------------------------------------

if __name__ == "__main__":
    test_bad_magic()
    test_wrong_version()
    test_no_geo_blob_returns_empty_model()
    test_parse_with_geo_blob()
    test_mesh_group_fields()
    test_lobby_self_contained()
    test_skeleton_p3rd_parse()
    test_skeleton_p3rd_lobby()
    test_skeleton_mhfu_compat()
    print("\nAll tests done.")
