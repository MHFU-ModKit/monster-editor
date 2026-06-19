"""mhfu_model — MHFU big-monster model/skeleton/animation library.

Single source of truth for the on-disk format AND the engine's constraints, shared
by the Blender addon, the CLI, and the live PRX injector. Phase 0 = read side +
byte-exact container; encoders + validators land in later phases.

Quick use:
    from mhfu_model import load_pac
    mm = load_pac("file_06134.bin")
    mm.skeleton.bone_count          # 25 (Tigrex)
    mm.anim.animations[0].tracks    # per-bone keyframe tracks
    mm.pac.to_bytes() == original   # byte-identical round-trip

MHP3rd (gen-3) PACs have a 0x80000000 skeleton magic and PMO v102.  Use
load_pac_p3rd() / parse_pac_p3rd() for those files.  The returned MonsterModel
is populated with the same dataclasses so all downstream code (Blender addon,
validators, etc.) works without modification.
"""
from __future__ import annotations

import struct

from . import anim as _anim
from . import pmo as _pmo
from . import skeleton as _skeleton
from .model import (Animation, AnimationPack, Bone, BoneTrack, Channel,
                    Keyframe, MeshGroup, Model, MonsterModel, Skeleton, Texture)
from .pac import MonsterPac, SubResource

__all__ = [
    "load_pac", "parse_pac", "parse_pac_p3rd", "load_pac_p3rd",
    "MonsterPac", "SubResource",
    "MonsterModel", "Skeleton", "Bone", "Model", "MeshGroup",
    "AnimationPack", "Animation", "BoneTrack", "Channel", "Keyframe", "Texture",
    "repack",
]

# Magic constants for skeleton sub-resource detection.
_SKEL_MAGIC_MHFU = 0xC0000000
_SKEL_MAGIC_P3RD = 0x80000000


def _detect_game(pac: MonsterPac) -> str:
    """Return 'mhfu' or 'p3rd' based on the skeleton magic in any sub.

    MonsterPac.role() only recognises 0xC0000000 as 'skeleton', so we must
    scan all subs for the 0x80000000 (MHP3rd) magic directly.
    """
    for sub in pac.subs:
        if sub.empty or len(sub.data) < 4:
            continue
        magic = struct.unpack_from("<I", sub.data, 0)[0]
        if magic == _SKEL_MAGIC_P3RD:
            return "p3rd"
        if magic == _SKEL_MAGIC_MHFU:
            return "mhfu"
    return "mhfu"


def parse_pac(data: bytes) -> MonsterModel:
    """Parse raw PAC bytes into a decoded MonsterModel (first set: slots 0-3).

    Auto-detects MHFU (0xC0000000 skeleton) vs MHP3rd (0x80000000 skeleton)
    and routes to the appropriate sub-parsers.  For MHP3rd PACs the provisional
    v102 PMO parser is used until asset delivers pmo_p3rd.py.
    """
    pac = MonsterPac.from_bytes(data)
    game = _detect_game(pac)
    if game == "p3rd":
        return _parse_pac_p3rd_from_pac(pac)
    return _parse_pac_mhfu_from_pac(pac)


def _parse_pac_mhfu_from_pac(pac: MonsterPac) -> MonsterModel:
    """Internal: parse an already-opened MHFU PAC."""
    mm = MonsterModel(pac=pac)
    sk = pac.find("skeleton")
    md = pac.find("model")
    tx = pac.find("texture")
    an = pac.find("anim")
    if sk:
        mm.skeleton = _skeleton.parse(sk.data)
        # A converted p3rd skeleton (magic flipped to 0xC0000000 but p3rd section
        # layout) will only yield 1 bone from the MHFU parser — fall back to the
        # p3rd parser which handles both section magic variants.
        if mm.skeleton and len(mm.skeleton.bones) <= 1 and mm.skeleton.bone_count > 1:
            from . import skeleton_p3rd as _skel_p3rd
            mm.skeleton = _skel_p3rd.parse(sk.data)
    if md:
        mm.model = _pmo.parse(md.data)
    if tx:
        mm.texture = Texture(raw=tx.data)
    if an:
        mm.anim = _anim.parse(an.data)
    return mm


def _parse_pac_p3rd_from_pac(pac: MonsterPac,
                             geo_blob: "Optional[bytes]" = None,
                             anim_blob: "Optional[bytes]" = None) -> MonsterModel:
    """Internal: parse an already-opened MHP3rd (gen-3) PAC.

    Skeleton uses skeleton_p3rd.parse() (0x80000000 path).
    PMO uses pmo_p3rd.parse() (v102 full decode).
    Animation: NOT in the model PAC for gen-3 — pass anim_blob from the separate
    emNNN animation file (e.g. file_05413.bin for em058 Tigrex) via load_pac_p3rd().
    anim.parse_p3rd() is used when anim_blob is provided.

    geo_blob: raw bytes of companion GE file (file_NNNN+1.bin) for in-quest PACs.
    anim_blob: raw bytes of the separate animation file (em/animation/emNNN.bin).
    """
    from . import skeleton_p3rd as _skel_p3rd
    from . import pmo_p3rd as _pmo_p3rd

    mm = MonsterModel(pac=pac)
    sk = _find_p3rd_skeleton(pac)
    md = _find_p3rd_pmo(pac)
    tx = pac.find("texture")

    if sk:
        mm.skeleton = _skel_p3rd.parse(sk.data)
    if md:
        try:
            mm.model = _pmo_p3rd.parse(md.data, geo_blob=geo_blob)
        except Exception as exc:
            print("[mhfu p3rd] PMO v102 parse failed: %s — geometry unavailable" % exc)
    if tx:
        mm.texture = Texture(raw=tx.data)
    # Animation: prefer explicit anim_blob; fall back to the PAC's own anim sub
    # (lobby PACs carry animation in sub[7] with a gen-3 header layout).
    _anim_blob = anim_blob
    if _anim_blob is None:
        an_sub = _find_p3rd_anim(pac)
        if an_sub:
            _anim_blob = an_sub.data
    if _anim_blob:
        try:
            mm.anim = _anim.parse_p3rd(_anim_blob)
        except Exception as exc:
            print("[mhfu p3rd] anim parse_p3rd failed: %s — animations unavailable" % exc)

    return mm


def _find_p3rd_skeleton(pac: MonsterPac):
    """Find the skeleton sub (0x80000000 magic) in a gen-3 PAC."""
    for sub in pac.subs:
        if len(sub.data) >= 4:
            magic = struct.unpack_from("<I", sub.data, 0)[0]
            if magic == _SKEL_MAGIC_P3RD:
                return sub
    return None


def _find_p3rd_pmo(pac: MonsterPac):
    """Find the first PMO v102 sub in a gen-3 PAC (sub[0] in observed PACs)."""
    for sub in pac.subs:
        if sub.data[:4] == b"pmo\x00" and sub.data[4:8] == b"102\x00":
            return sub
    return None


def _find_p3rd_anim(pac: MonsterPac):
    """Best-effort: find an anim sub in a gen-3 PAC.

    Gen-3 lobby PACs (8-sub layout) store animation in sub[7]:
      header = anim_count(u32) + total_size(u32) + slot_count(u32) + offset_table[].
    This is distinct from the MHFU anim magic (0x64/0x18) — we detect it by
    checking that sub[7] exists, has a plausible total_size == len(blob), and a
    non-zero slot_count with the offset table starting at offset 12.

    Returns the sub or None.
    """
    # Try MHFU-style magic first (0x64 + 0x18) — some gen-3 PACs may carry one.
    for sub in pac.subs:
        if sub.data and sub.data[0:1] == b"\x64":
            try:
                if struct.unpack_from("<I", sub.data, 4)[0] == 0x18:
                    return sub
            except struct.error:
                pass
    # Gen-3 lobby layout: last sub (index 7) with total_size == len(blob).
    if pac.subs and len(pac.subs) == 8:
        sub = pac.subs[7]
        if sub.data and len(sub.data) >= 12:
            try:
                _anim_count, total_sz, slot_count = struct.unpack_from("<3I", sub.data, 0)
                if total_sz == len(sub.data) and 0 < slot_count < 512:
                    return sub
            except struct.error:
                pass
    return None


def parse_pac_p3rd(data: bytes, geo_blob: "Optional[bytes]" = None,
                   anim_blob: "Optional[bytes]" = None) -> MonsterModel:
    """Force-parse raw PAC bytes as an MHP3rd (gen-3) PAC.

    Use when you know the file is MHP3rd and don't want auto-detection.

    geo_blob: raw companion GE file bytes (file_NNNN+1.bin) for in-quest PACs.
    anim_blob: raw bytes of the separate animation file (emNNN.bin); decoded via
               anim.parse_p3rd().
    """
    pac = MonsterPac.from_bytes(data)
    return _parse_pac_p3rd_from_pac(pac, geo_blob=geo_blob, anim_blob=anim_blob)


def load_pac(path: str) -> MonsterModel:
    """Load a monster PAC from `path` (MHFU or MHP3rd auto-detected).

    For MHP3rd in-quest PACs the companion GE file (file_NNNN+1.bin) is
    probed automatically from the same directory when present.
    """
    import os as _os
    with open(path, "rb") as f:
        data = f.read()
    pac = MonsterPac.from_bytes(data)
    game = _detect_game(pac)
    if game != "p3rd":
        return _parse_pac_mhfu_from_pac(pac)
    # MHP3rd: probe for companion GE file automatically.
    geo_blob = None
    base = _os.path.splitext(path)[0]
    stem = _os.path.basename(base)
    if stem.startswith("file_"):
        try:
            n = int(stem[5:])
            companion = _os.path.join(_os.path.dirname(path) or ".",
                                      "file_%05d.bin" % (n + 1))
            if _os.path.isfile(companion):
                with open(companion, "rb") as f:
                    geo_blob = f.read()
                print("[mhfu] p3rd companion GE: %s" % companion)
        except ValueError:
            pass
    return _parse_pac_p3rd_from_pac(pac, geo_blob=geo_blob)


def load_pac_p3rd(path: str, geo_path: "Optional[str]" = None,
                  anim_path: "Optional[str]" = None) -> MonsterModel:
    """Load an MHP3rd PAC directly (no auto-detect).

    geo_path: path to the companion GE file (file_NNNN+1.bin).  If None,
    the function probes for a sibling file_NNNN+1.bin automatically.

    anim_path: path to the separate animation file (e.g. file_05413.bin for
    em058 Tigrex, file_05252.bin for em023 Black Tigrex).  If provided,
    decoded via anim.parse_p3rd() and attached to mm.anim.
    """
    import os
    with open(path, "rb") as f:
        data = f.read()
    geo_blob = None
    if geo_path:
        with open(geo_path, "rb") as f:
            geo_blob = f.read()
    else:
        # Auto-probe companion: file_NNNNN.bin -> file_(NNNNN+1).bin
        base = os.path.splitext(path)[0]
        stem = os.path.basename(base)
        if stem.startswith("file_"):
            try:
                n = int(stem[5:])
                companion = os.path.join(os.path.dirname(path) or ".",
                                         "file_%05d.bin" % (n + 1))
                if os.path.isfile(companion):
                    with open(companion, "rb") as f:
                        geo_blob = f.read()
                    print("[mhfu p3rd] companion GE: %s" % companion)
            except ValueError:
                pass
    anim_blob = None
    if anim_path:
        with open(anim_path, "rb") as f:
            anim_blob = f.read()
    return parse_pac_p3rd(data, geo_blob=geo_blob, anim_blob=anim_blob)


def repack(mm: MonsterModel) -> bytes:
    """Re-encode the decoded sub-resources back into a PAC (Phase 3 write-back).

    Re-encodes the FIRST skeleton / model / anim sub from the decoded data model
    and leaves every other sub-resource (textures, a monster's second model set,
    unknown subs) byte-identical. Untouched + unedited assets reproduce the source
    file exactly (the encoders are byte-exact); edited assets re-flow offsets via
    `MonsterPac.to_bytes`. Texture (TMH) write-back is deferred to a later phase.

    MHP3rd (v102) model sub: the source bytes are v102-format which the MHFU engine
    cannot parse. On the first model sub encounter, if `mm.model.version == b"102\\x00"`
    we run the full v102->MHFU 1.0 conversion via `pmo_from_model.encode_from_model`
    (which builds a self-contained MHFU 1.0 GE list from the decoded float geometry).
    The skeleton magic is already flipped to 0xC0000000 by `skeleton.encode`.
    """
    pac = mm.pac
    done = {"skeleton": False, "model": False, "anim": False}
    new_subs = []
    for sub in pac.subs:
        r = pac.role(sub)
        data = sub.data
        if r == "skeleton" and mm.skeleton and not done["skeleton"]:
            data = _skeleton.encode(mm.skeleton); done["skeleton"] = True
        elif r == "model" and mm.model and not done["model"]:
            # MHP3rd v102 PMO: the decoded float geometry must be re-encoded as
            # MHFU 1.0 (self-contained GE list) rather than returning the v102 bytes.
            if getattr(mm.model, "version", None) == b"102\x00":
                from . import pmo_from_model as _pfm
                data = _pfm.encode_from_model(mm.model)
            else:
                data = _pmo.encode(mm.model)
            done["model"] = True
        elif r == "anim" and mm.anim and not done["anim"]:
            data = _anim.encode(mm.anim); done["anim"] = True
        new_subs.append(SubResource(sub.index, data))
    return MonsterPac(subs=new_subs, tail=pac.tail).to_bytes()
