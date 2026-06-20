"""Neutral, engine-agnostic in-memory data model for MHFU big-monster assets.

These dataclasses are the boundary every consumer talks to (Blender addon, CLI,
live PRX injector). Parsers populate them; encoders (Phase 3) consume them. Each
sub-resource also retains its original raw bytes so that, until a real encoder
exists, `encode()` is a lossless passthrough — which is what makes the Phase 0
byte-identical round-trip test a genuine guard for the container writer.

Quantization (engine units; see docs/ANIMATION_FORMAT.md):
    rotation 4096 = 90 deg, location 16 = 1.0, scale 256 = 1.0.
The data model keeps RAW s16 keyframe values + bind-pose floats as stored; the
Blender layer dequantizes. Keeping raw avoids lossy float churn in round-trips.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

Vec3 = Tuple[float, float, float]


# --------------------------------------------------------------------------- #
# Skeleton (PAC sub-resource 0, 0xC0000000 blob)
# --------------------------------------------------------------------------- #
@dataclass
class Bone:
    index: int
    parent: int          # bone index, -1 = none
    child: int           # first child bone index, -1 = none
    sibling: int         # next sibling bone index, -1 = none
    bind_scale: Vec3
    bind_rot: Vec3       # euler, engine units (see docs); decompose-from-matrix
    bind_pos: Vec3
    flag: int = 1
    section_size: int = 0x10C
    raw: bytes = b""     # full bone section bytes (lossless re-emit of matrix/aux)


@dataclass
class Skeleton:
    bone_count: int
    total_size: int
    bones: List[Bone] = field(default_factory=list)
    header: bytes = b""  # the 0x1C header bytes (magic/count/size/4 unknowns)
    raw: bytes = b""     # whole blob, for lossless passthrough

    def roots(self) -> List[int]:
        return [b.index for b in self.bones if b.parent == -1]


# --------------------------------------------------------------------------- #
# Animation (PAC sub-resource 3, P3rd-style pack)
# --------------------------------------------------------------------------- #
@dataclass
class Keyframe:
    value: int          # raw s16
    frame: int          # raw s16
    ease_in: int        # raw s16
    ease_out: int       # raw s16


@dataclass
class Channel:
    type: int                          # channel tag (e.g. 0x80000000|bit)
    keyframes: List[Keyframe] = field(default_factory=list)


@dataclass
class BoneTrack:
    tag: int                           # full u32 bone-record tag (FLAG | mask)
    channels: List[Channel] = field(default_factory=list)

    @property
    def mask(self) -> int:             # channel bitmask (low 16 of bone tag)
        return self.tag & 0xFFFF

    @property
    def channel_names(self) -> List[str]:
        from .anim import CHANNEL_BITS
        return [n for bit, n in CHANNEL_BITS if self.mask & bit]


@dataclass
class Animation:
    slot: int                          # offset-table slot index
    tag: int
    bone_count: int
    loop: int
    loop_start: float
    tracks: List[BoneTrack] = field(default_factory=list)
    raw: bytes = b""                   # the anim block bytes


@dataclass
class AnimationPack:
    magic: int
    slot_count: int
    animations: List[Animation] = field(default_factory=list)   # populated slots only
    header: bytes = b""
    raw: bytes = b""                   # whole sub-3 blob, lossless passthrough

    def by_slot(self):
        return {a.slot: a for a in self.animations}


# --------------------------------------------------------------------------- #
# Model / geometry (PAC sub-resource 1, PMO)
# --------------------------------------------------------------------------- #
@dataclass
class MeshGroup:
    """One PMO mesh group == one rigid-bound segment (draw-order ↔ bone index).

    `vertices` are dicts {x,y,z, optional u,v, i,j,k (normal), weights}; `faces`
    are dicts {v1,v2,v3} (0-based indices into `vertices`). These are decoded
    geometry for the Blender importer; `vertex_count`/`face_count` mirror lengths
    for cheap summaries.
    """
    index: int                 # global draw order across all vertex groups == bind index
    material: int
    mesh_record: int = 0       # which PMO mesh record this vertex group belongs to
    vertex_count: int = 0
    face_count: int = 0
    scale: Vec3 = (1.0, 1.0, 1.0)
    vertices: List[dict] = field(default_factory=list)
    faces: List[dict] = field(default_factory=list)
    enc: Optional[dict] = None  # private PMO re-encode descriptor (vertex buffer layout)
    vg_rec: int = -1            # source vgroup-table index (== pmo_topology rec_index);
    #                            draw order != table index for split-mesh monsters
    boneref: int = -1           # vgroup record vg[2]: the skeleton bone this rigid group
    #                            binds to (REAL skinning, not draw order). -1 = unknown.
    #                            The MHFU engine transforms the group's verts by
    #                            joint[boneref]; a wrong/OOB value collapses the mesh.


@dataclass
class Model:
    magic: bytes
    version: bytes
    mesh_groups: List[MeshGroup] = field(default_factory=list)
    raw: bytes = b""                   # whole PMO blob, lossless passthrough
    scale: Vec3 = (1.0, 1.0, 1.0)      # header global scale (encoder re-quantizes with it)
    stride: int = 0                    # winning mesh-table stride (0x20/0x18)
    edited: bool = False               # set by an editor -> encode() re-emits vertex data
    # ADDED geometry pending a topology grow (the Blender add path); each entry =
    # {"vg_rec": int, "verts": [ {x,y,z,i,j,k,u,v} ], "tris": [(a,b,c)] }. The
    # in-place pmo.encode ignores this; the exporter applies it via pmo_topology
    # AFTER repack (a separate, size-changing pass).
    additions: List[dict] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Texture (PAC sub-resource 2, TMH) — opaque for now
# --------------------------------------------------------------------------- #
@dataclass
class Texture:
    raw: bytes = b""


# --------------------------------------------------------------------------- #
# Whole monster (one model PAC) — possibly TWO sets (e.g. file_06111 slots 4-6)
# --------------------------------------------------------------------------- #
@dataclass
class MonsterModel:
    """Decoded view of one big-monster model PAC.

    `sets` groups the sub-resources into one or more (skeleton, model, texture,
    anim) bundles. Most monsters have a single set in slots [0..3]; a few carry a
    second variant set in slots [4..6] (no second anim). `pac` is the underlying
    container, the source of truth for byte-identical re-emit.
    """
    pac: "object"                      # mhfu_model.pac.MonsterPac (avoid import cycle)
    skeleton: Optional[Skeleton] = None
    model: Optional[Model] = None
    texture: Optional[Texture] = None
    anim: Optional[AnimationPack] = None
    extra: List[object] = field(default_factory=list)   # secondary sets / unparsed
