"""A monster PAC assembled into ONE render-agnostic object — no GL, no UI, no Blender.

Everything here already existed as library code in `tools/mhfu_model/`; what did not
exist was a single thing you could hand to a viewer, to the Blender addon or to a test
and say *"draw this at frame 41"*. That assembly is all this module is.

    from mhfu_monster_editor.core import open_scene
    sc = open_scene("workspace/extracted/data_files/file_06185.bin")
    sc.rig.parents                      # bind skeleton, (n,) int32, -1 = root
    sc.groups[0].uvs, .texture          # mesh groups, UVs, texture binding
    sc.textures[0].rgba                 # decoded RGBA8, (h, w, 4) uint8
    sc.clips                            # slot / frames / loop
    sc.pose(7, 41).joints               # (n,3) world joint positions
    sc.pose(7, 41).skin(sc.groups[0].skin)     # (v,3) deformed vertices

Both games load through the same door: MHFU PMO 1.0 (a native monster, or a BUILT
port, which is MHFU-format) and MHP3rd v102 (a donor, model + companion geometry +
its separate moveset file).

Three things this gets right that are easy to get wrong
-------------------------------------------------------
1. 🔴 **An MHP3rd source moveset is not positional.** ``bone = record + offset +
   skipped``, and the offset comes from `mhfu_model.p3rd_anim_map` — the same measured
   map the porter reads. It is never re-derived here and never hardcoded: a port built
   at one bone map and drawn at another is what hid the Brute's mismapping for months
   (`docs/BRUTE_TIGREX_PORT.md`). An unmapped donor says so in :attr:`Scene.notes`.
2. 🔴 **An MHFU clip's tracks are not always joints 0..N either.** The in-game format
   splits a clip's joints across up to three parallel streams, and a few slots
   (24 and 25 on every Tigrex-framed pack seen so far) exist in ONE stream only — a
   head-and-neck clip meant to play over an idle body. Concatenated blindly its 9
   tracks land on joints 0..8 and the monster folds in half. The scene derives each
   stream's joint base from the partition widths, so a partial clip poses the joints
   it actually owns and leaves the rest at bind.
3. **`material` is the PMO's texture index**, resolved against the decoded TMH set the
   way `blender_mhfu/importer.py` resolves it. An index with no image is reported in
   :attr:`Scene.notes` rather than silently falling back.

The maths is in :mod:`.pose` and is pinned against `tools/mhfu_model/stretch.py`.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ..manifest import PortManifest
from ..manifest import load as load_manifest
from ._paths import tools_on_path
from .pose import Curves, Pose, Rig, SkinBinding, sample

tools_on_path()

import mhfu_model                                    # noqa: E402
from mhfu_model import anim_ingame as AI             # noqa: E402
from mhfu_model import p3rd_anim_map as P3AM         # noqa: E402
from mhfu_model import skeleton_p3rd as SKP          # noqa: E402
from mhfu_model.pac import MonsterPac                # noqa: E402
from mhfu_model.tmh import decode_tmh                # noqa: E402

MHFU = "mhfu"
MHP3RD = "mhp3rd"

#: the in-game animation streams that carry a big monster's joint partition, in joint
#: order. `anim_ingame.to_flat_anim`'s own default — main, sub1, sub3.
BODY_STREAMS = (0, 2, 4)


class SceneError(RuntimeError):
    """The PAC cannot be assembled into a scene at all."""


# --------------------------------------------------------------------------- #
# records
# --------------------------------------------------------------------------- #
@dataclass
class TextureImage:
    """One decoded TMH image. ``rgba`` is ``(h, w, 4)`` uint8, row 0 = TOP."""
    index: int
    width: int
    height: int
    rgba: np.ndarray


@dataclass
class MeshGroup:
    """One PMO vertex group: the unit that carries a material and a skin palette."""
    index: int
    material: int                       # the PMO's texture index
    texture: Optional[int]              # position in `Scene.textures`, None = unresolved
    mesh_record: int
    vg_rec: int
    #: ⚠️ passed through from the parser and it means DIFFERENT things per game — the
    #: bound joint for MHFU, `vg[2]` (a cumulative palette offset) for MHP3rd v102. Use
    #: :attr:`skin` for anything that has to be right; this is a breadcrumb.
    boneref: int
    positions: np.ndarray               # (v,3) bind-pose vertices
    triangles: np.ndarray               # (f,3) int32
    skin: SkinBinding
    uvs: Optional[np.ndarray] = None    # (v,2)
    normals: Optional[np.ndarray] = None  # (v,3)
    dropped_faces: int = 0

    @property
    def n_vertices(self) -> int:
        return len(self.positions)

    @property
    def n_faces(self) -> int:
        return len(self.triangles)


@dataclass
class Clip:
    """One animation slot. ``slot`` IS the executor ``a1`` (root `CLAUDE.md` §2.7)."""
    slot: int
    frames: int                         # last keyframe; the clip's authored length - 1
    loop: bool
    tracks: int
    #: track -> skeleton joint. Identity for a whole-rig MHFU clip; the p3rd map for a
    #: donor; the stream partition's own base for a partial MHFU clip.
    track_to_joint: Dict[int, int] = field(default_factory=dict)
    #: joints this clip actually drives
    driven: Tuple[int, ...] = ()
    #: names this clip carries in the port manifest, if one was supplied
    names: Tuple[str, ...] = ()
    #: False when the clip lives in only some of the joint-partition streams
    whole_rig: bool = True
    _anim: object = field(default=None, repr=False, compare=False)
    _curves: Optional[Curves] = field(default=None, repr=False, compare=False)

    @property
    def name(self) -> str:
        return self.names[0] if self.names else "slot_%02d" % self.slot


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def detect_game(pac: MonsterPac) -> str:
    """``"mhfu"`` or ``"mhp3rd"`` from the skeleton magic. Defaults to MHFU."""
    for sub in pac.subs:
        if len(sub.data) >= 4:
            magic = struct.unpack_from("<I", sub.data, 0)[0]
            if magic == SKP.MAGIC_P3RD:
                return MHP3RD
            if magic == SKP.MAGIC_MHFU:
                return MHFU
    return MHFU


def _array(vertices, keys) -> Optional[np.ndarray]:
    """``(v, len(keys))`` float array, or None when any vertex lacks a component."""
    if not vertices or any(k not in vertices[0] for k in keys):
        return None
    try:
        return np.array([[v[k] for k in keys] for v in vertices], dtype=np.float64)
    except KeyError:
        return None


def _triangles(group) -> Tuple[np.ndarray, int]:
    n = len(group.vertices)
    tris = [(f["v1"], f["v2"], f["v3"]) for f in group.faces]
    good = [t for t in tris if max(t) < n and min(t) >= 0]
    arr = np.array(good, dtype=np.int32) if good else np.zeros((0, 3), dtype=np.int32)
    return arr, len(tris) - len(good)


def stream_joint_bases(ig) -> Dict[int, int]:
    """``{stream index: first joint it owns}`` from the partition widths.

    A stream's block always holds the same number of bones on every clip (verified on
    the native Tigrex 31/9/5, the built Zinogre 33/6/7 and the built Brute 31/9/4), and
    `anim_ingame.to_flat_anim` documents that concatenating the populated streams in
    stream order IS joint order. So the base of stream *k* is the total width of the
    populated streams before it — which is what makes a clip that exists in only one
    stream placeable.
    """
    bases: Dict[int, int] = {}
    base = 0
    for si in BODY_STREAMS:
        if si >= len(ig.streams) or not ig.streams[si].clips:
            continue
        widths = {len(b.bones) for b in ig.streams[si].clips.values()}
        if len(widths) != 1:
            raise SceneError("stream %d has clips of %d different bone counts %s — the "
                             "joint partition is not constant, so no joint base can be "
                             "derived" % (si, len(widths), sorted(widths)))
        bases[si] = base
        base += widths.pop()
    return bases


# --------------------------------------------------------------------------- #
# the scene
# --------------------------------------------------------------------------- #
class Scene:
    """Bind skeleton + skinned geometry + textures + clips, and `pose()`.

    Construct through :func:`open_scene`, :meth:`from_pac` or :meth:`from_manifest`.
    """

    def __init__(self, name: str, game: str, skeleton, rig: Rig,
                 groups: List[MeshGroup], textures: List[TextureImage],
                 clips: List[Clip], *, path: Optional[Path] = None,
                 manifest: Optional[PortManifest] = None,
                 record_to_bone: Optional[Dict[int, int]] = None,
                 notes: Optional[List[str]] = None) -> None:
        self.name = name
        self.game = game
        self.skeleton = skeleton
        self.rig = rig
        self.groups = groups
        self.textures = textures
        self.clips = clips
        self.path = Path(path) if path else None
        self.manifest = manifest
        #: the donor's animation record -> joint map, or None for a positional pack
        self.record_to_bone = record_to_bone
        self.notes: List[str] = list(notes or [])
        self._by_slot = {c.slot: c for c in clips}
        self._by_name = {n: c for c in clips for n in c.names}
        self._merged: Optional[SkinBinding] = None
        self._ranges: Optional[List[Tuple[int, int]]] = None

    # ---- construction ------------------------------------------------- #
    @classmethod
    def from_pac(cls, path, *, geo=None, anim=None, em_id: Optional[int] = None,
                 bone_offset: Optional[int] = None,
                 skip_bones: Optional[Sequence[int]] = None,
                 name: Optional[str] = None,
                 manifest: Optional[PortManifest] = None) -> "Scene":
        """Open a model PAC. The donor's companions are probed when not given.

        `geo` / `anim` are only meaningful for an MHP3rd PAC: its geometry lives in the
        next file id and its moveset in the one after (`file_05339/40/41` = the
        Zinogre). `em_id` / `bone_offset` / `skip_bones` override what
        `p3rd_anim_map` says; leave them unset unless you have measured otherwise.
        """
        path = Path(path)
        data = path.read_bytes()
        pac = MonsterPac.from_bytes(data)
        game = detect_game(pac)
        notes: List[str] = []
        if game == MHP3RD:
            geo, anim = _probe_companions(path, geo, anim, notes)
            mm = mhfu_model.load_pac_p3rd(str(path),
                                          geo_path=str(geo) if geo else None,
                                          anim_path=str(anim) if anim else None)
            if em_id is None:
                em_id = _em_from_path(path)
        else:
            mm = mhfu_model.load_pac_bytes(data)
        return cls._assemble(name or path.stem, game, mm, pac, notes,
                             em_id=em_id, bone_offset=bone_offset,
                             skip_bones=skip_bones, path=path, manifest=manifest)

    @classmethod
    def from_manifest(cls, m: Union[PortManifest, str, Path], *, pac=None,
                      root="workspace", side: str = "port") -> "Scene":
        """Open the port a manifest describes — its BUILT PAC, or its donor.

        ``side="port"`` (default) opens the built MHFU PAC: ``pac``, else
        ``tmp/<port.pac>`` (`docs/ASSETS.md` §D is where the porter writes it).
        ``side="source"`` opens the MHP3rd donor named by ``[source]``, with the
        record→bone map its ``em_id`` selects. A manifest's ``[clips]`` names are
        attached either way, and cross-checked by :meth:`clip_mismatches`.
        """
        if not isinstance(m, PortManifest):
            m = load_manifest(m)
        if side == "source":
            src = m.source_paths(root)
            return cls.from_pac(src["model"], geo=src["geo"], anim=src["anim"],
                                em_id=m.source.em_id,
                                bone_offset=m.build.bone_offset,
                                skip_bones=m.build.skip_bones,
                                name="%s (source)" % m.name, manifest=m)
        if side != "port":
            raise ValueError("side must be 'port' or 'source', not %r" % side)
        target = Path(pac) if pac else Path("tmp") / m.pac
        if not target.exists():
            raise SceneError(
                "%s: the built PAC is not there. It is game data and is never "
                "committed — build it with\n  python tools/build_p3rd_port.py "
                "--manifest %s --out %s" % (target, m.path or "ports/%s.toml" % m.name,
                                            target))
        return cls.from_pac(target, name=m.name, manifest=m)

    # ---- assembly ----------------------------------------------------- #
    @classmethod
    def _assemble(cls, name, game, mm, pac, notes, *, em_id=None, bone_offset=None,
                  skip_bones=None, path=None, manifest=None) -> "Scene":
        if mm.skeleton is None or not mm.skeleton.bones:
            raise SceneError("%s: no skeleton sub-resource — not a big-monster PAC" % name)
        if mm.model is None or not mm.model.mesh_groups:
            raise SceneError("%s: the PMO decoded to no geometry" % name)
        rig = Rig.from_skeleton(mm.skeleton)

        textures = _textures(mm, notes)
        groups = _groups(mm.model, rig.n_bones, textures, notes)
        if game == MHFU:
            clips, r2b = _mhfu_clips(pac, rig, notes)
        else:
            clips, r2b = _p3rd_clips(mm, rig, em_id, bone_offset, skip_bones, notes)
        sc = cls(name, game, mm.skeleton, rig, groups, textures, clips,
                 path=path, manifest=manifest, record_to_bone=r2b, notes=notes)
        if manifest is not None:
            sc._attach_manifest_names(manifest)
        return sc

    def attach_manifest(self, m: PortManifest) -> None:
        """Re-bind the scene to a manifest — after the editor has written one back.

        The PAC is unchanged; only the NAMES are, so nothing is re-read. Public
        because issue #8 lets a session rename a clip and save, and the clip list has
        to show the new name without reopening the file.
        """
        self.manifest = m
        self._attach_manifest_names(m)

    def clip_table(self) -> Dict[int, Tuple[int, bool]]:
        """``{slot: (frames, loop)}`` — this PAC's clip fingerprints.

        The same table :func:`mhfu_monster_editor.clips.clip_table` reads straight out
        of a file, from a scene that is already open.
        """
        return {c.slot: (c.frames, c.loop) for c in self.clips}

    def _attach_manifest_names(self, m: PortManifest) -> None:
        by_slot: Dict[int, List[str]] = {}
        for c in m.clips.values():
            by_slot.setdefault(c.slot, []).append(c.name)
        for clip in self.clips:
            clip.names = tuple(sorted(by_slot.get(clip.slot, ())))
        self._by_name = {n: c for c in self.clips for n in c.names}

    # ---- lookup ------------------------------------------------------- #
    def clip(self, key: Union[int, str, Clip]) -> Clip:
        """A clip by slot number, by manifest name, or passed straight through."""
        if isinstance(key, Clip):
            return key
        if isinstance(key, str):
            if key in self._by_name:
                return self._by_name[key]
            raise KeyError("no clip named %r (manifest names: %s)"
                           % (key, ", ".join(sorted(self._by_name)) or "none"))
        if key in self._by_slot:
            return self._by_slot[key]
        raise KeyError("no clip in slot %d (slots: %s)"
                       % (key, ", ".join(str(c.slot) for c in self.clips)))

    def curves(self, key) -> Curves:
        """The clip's channels, compiled once and cached on it."""
        c = self.clip(key)
        if c._curves is None:
            c._curves = Curves(sample(c._anim, c.track_to_joint or None),
                               self.rig.n_bones, self.rig.bind_local)
        return c._curves

    # ---- posing ------------------------------------------------------- #
    def pose(self, key, frame: float = 0.0) -> Pose:
        """The rig at ``frame`` of a clip. Frames outside the clip clamp per channel."""
        c = self.clip(key)
        rot, loc = self.curves(c).eval(float(frame))
        return Pose(rig=self.rig, frame=float(frame), world=self.rig.world(rot, loc),
                    slot=c.slot)

    def bind_pose(self) -> Pose:
        return Pose(rig=self.rig, frame=0.0, world=self.rig.bind_world, slot=None)

    # ---- geometry ----------------------------------------------------- #
    @property
    def n_vertices(self) -> int:
        return sum(g.n_vertices for g in self.groups)

    @property
    def merged(self) -> SkinBinding:
        """Every group's vertices in ONE binding — a viewer's single buffer.

        Deforming the whole native Tigrex through this costs **0.44 ms against 2.38 ms**
        for 214 per-group calls: a big-monster group averages 19 vertices, so per-call
        array overhead dominates. :meth:`group_range` gives each group's slice back.
        """
        if self._merged is None:
            self._build_merged()
        return self._merged

    def group_range(self, index: int) -> Tuple[int, int]:
        """``(start, stop)`` of group ``index`` inside :attr:`merged`."""
        if self._ranges is None:
            self._build_merged()
        return self._ranges[index]

    def _build_merged(self) -> None:
        width = max((g.skin.max_influences for g in self.groups), default=1)
        total = self.n_vertices
        pos = np.zeros((total, 3), dtype=np.float64)
        bones = np.zeros((total, width), dtype=np.int64)
        weights = np.zeros((total, width), dtype=np.float64)
        ranges, at = [], 0
        for g in self.groups:
            n, k = g.n_vertices, g.skin.max_influences
            pos[at:at + n] = g.skin.positions
            bones[at:at + n, :k] = g.skin.bones
            weights[at:at + n, :k] = g.skin.weights
            ranges.append((at, at + n))
            at += n
        self._merged = SkinBinding(pos, bones, weights)
        self._ranges = ranges

    # ---- manifest cross-check ----------------------------------------- #
    def clip_mismatches(self) -> List[str]:
        """Where the manifest's clip fingerprint disagrees with the PAC. Empty = agree.

        `manifest.Clip` carries ``slot`` / ``frames`` / ``loop`` precisely so it can be
        checked against a build — the same fingerprint the clip-state block reads back
        live at ``ent+0x80 + slot*0x40``.
        """
        out: List[str] = []
        if self.manifest is None:
            return out
        for name, mc in sorted(self.manifest.clips.items()):
            got = self._by_slot.get(mc.slot)
            if got is None:
                out.append("clips.%s: slot %d is not in this PAC" % (name, mc.slot))
                continue
            if mc.frames is not None and mc.frames != got.frames:
                out.append("clips.%s: manifest says %d frames, the PAC has %d"
                           % (name, mc.frames, got.frames))
            if mc.loop is not None and bool(mc.loop) != got.loop:
                out.append("clips.%s: manifest says loop=%s, the PAC has %s"
                           % (name, mc.loop, got.loop))
        return out

    # ---- reporting ---------------------------------------------------- #
    def summary(self) -> str:
        looping = sum(1 for c in self.clips if c.loop)
        partial = [c.slot for c in self.clips if not c.whole_rig]
        tex = ", ".join("#%d %dx%d" % (t.index, t.width, t.height) for t in self.textures)
        lines = [
            "%s  [%s]%s" % (self.name, self.game, "  %s" % self.path if self.path else ""),
            "  skeleton  %d bones, %d roots, depth %d"
            % (self.rig.n_bones, int((self.rig.parents < 0).sum()), len(self.rig.levels) - 1),
            "  geometry  %d groups, %d vertices, %d triangles, max %d influences/vertex"
            % (len(self.groups), self.n_vertices,
               sum(g.n_faces for g in self.groups),
               max((g.skin.max_influences for g in self.groups), default=0)),
            "  textures  %d  (%s)" % (len(self.textures), tex or "none"),
            "  clips     %d  (%d looping%s), frames %d..%d"
            % (len(self.clips), looping,
               ", %d partial: %s" % (len(partial), partial) if partial else "",
               min((c.frames for c in self.clips), default=0),
               max((c.frames for c in self.clips), default=0)),
        ]
        if self.record_to_bone:
            js = sorted(self.record_to_bone.values())
            lines.append("  anim map  %d records -> joints %d..%d (p3rd_anim_map)"
                         % (len(js), js[0], js[-1]))
        for n in self.notes:
            lines.append("  ⚠️  %s" % n)
        for n in self.clip_mismatches():
            lines.append("  ⚠️  %s" % n)
        return "\n".join(lines)

    def __repr__(self) -> str:
        return ("<Scene %r %s: %d bones, %d groups, %d clips>"
                % (self.name, self.game, self.rig.n_bones, len(self.groups),
                   len(self.clips)))


# --------------------------------------------------------------------------- #
# sub-resource assembly
# --------------------------------------------------------------------------- #
def _textures(mm, notes) -> List[TextureImage]:
    raw = mm.texture.raw if getattr(mm, "texture", None) else b""
    if not raw:
        notes.append("no TMH sub-resource — the model has no textures")
        return []
    out = []
    for t in decode_tmh(raw):
        rgba = np.frombuffer(t["rgba"], dtype=np.uint8).reshape(t["height"], t["width"], 4)
        out.append(TextureImage(t["index"], t["width"], t["height"], rgba))
    if not out:
        notes.append("the TMH sub decoded to no images (DXT3/DXT5 are not supported)")
    return out


def _groups(model, n_bones: int, textures, notes) -> List[MeshGroup]:
    by_index = {t.index: i for i, t in enumerate(textures)}
    groups, unresolved, no_skin, dropped = [], set(), 0, 0
    for g in model.mesh_groups:
        tris, bad = _triangles(g)
        dropped += bad
        skin = SkinBinding.from_vertices(g.vertices, n_bones)
        if len(skin.rest) and skin.rest.max() > 0.5:
            no_skin += int((skin.rest > 0.5).sum())
        tex = by_index.get(g.material)
        if tex is None and textures:
            unresolved.add(g.material)
        groups.append(MeshGroup(
            index=g.index, material=g.material, texture=tex,
            mesh_record=g.mesh_record, vg_rec=g.vg_rec, boneref=g.boneref,
            positions=skin.positions, triangles=tris, skin=skin,
            uvs=_array(g.vertices, ("u", "v")),
            normals=_array(g.vertices, ("i", "j", "k")),
            dropped_faces=bad))
    if unresolved:
        notes.append("material index %s has no decoded texture — those groups draw "
                     "untextured" % sorted(unresolved))
    if no_skin:
        notes.append("%d vertices carry no usable bone influence and stay at bind pose"
                     % no_skin)
    if dropped:
        notes.append("%d triangles index past their group's vertex buffer and were "
                     "dropped" % dropped)
    return groups


def _mhfu_clips(pac, rig, notes) -> Tuple[List[Clip], None]:
    """MHFU's in-game 3-stream pack -> clips, with each stream's joint base applied."""
    sub = pac.find("anim")
    if sub is None:
        notes.append("no animation sub-resource — bind pose only")
        return [], None
    ig = AI.parse_ingame(sub.data)
    live = [si for si in BODY_STREAMS if si < len(ig.streams) and ig.streams[si].clips]
    stray = [si for si in range(len(ig.streams))
             if ig.streams[si].clips and si not in BODY_STREAMS]
    if stray:
        notes.append("streams %s carry clips but are not part of the joint partition "
                     "this loader knows about; they are ignored" % stray)
    if not live:
        notes.append("the animation sub has no populated body stream — bind pose only")
        return [], None
    bases = stream_joint_bases(ig)
    widths = {si: len(next(iter(ig.streams[si].clips.values())).bones) for si in live}
    pack = AI.to_flat_anim(ig, tuple(live))
    clips = []
    for anim in pack.animations:
        present = [si for si in live if anim.slot in ig.streams[si].clips]
        t2j, t = {}, 0
        for si in present:
            for k in range(widths[si]):
                t2j[t] = bases[si] + k
                t += 1
        frames = max((kf.frame for tr in anim.tracks for ch in tr.channels
                      for kf in ch.keyframes), default=0)
        driven = tuple(sorted({t2j[i] for i, tr in enumerate(anim.tracks)
                               if i in t2j and any(ch.keyframes for ch in tr.channels)}))
        clips.append(Clip(slot=anim.slot, frames=frames, loop=bool(anim.loop),
                          tracks=len(anim.tracks), track_to_joint=t2j, driven=driven,
                          whole_rig=len(present) == len(live), _anim=anim))
    partial = [c.slot for c in clips if not c.whole_rig]
    if partial:
        notes.append("slots %s hold a PARTIAL clip — present in only some of the %d "
                     "joint-partition streams. They pose the joints they own and leave "
                     "the rest at bind; concatenating them blindly onto joints 0..N is "
                     "what folds the monster in half." % (partial, len(live)))
    return sorted(clips, key=lambda c: c.slot), None


def _p3rd_clips(mm, rig, em_id, bone_offset, skip_bones, notes):
    """An MHP3rd donor's flat pack -> clips, mapped through `p3rd_anim_map`."""
    if mm.anim is None or not mm.anim.animations:
        notes.append("no MHP3rd moveset was loaded (pass the anim companion) — "
                     "bind pose only")
        return [], None
    anims = mm.anim.animations
    n_rec = max(len(a.tracks) for a in anims)
    if em_id is None or em_id < 0:
        notes.append("🔴 no MHP3rd em id for this donor, so the record->bone map falls "
                     "back to p3rd_anim_map.DEFAULT_BONE_OFFSET (%d) with no skip list "
                     "— very probably WRONG. Pin it with the fork rule and add a row to "
                     "p3rd_anim_map.EM_BY_MODEL_PAC." % P3AM.DEFAULT_BONE_OFFSET)
        b2r = P3AM.bone_to_record(n_rec, rig.n_bones, P3AM.DEFAULT_BONE_OFFSET, ())
    else:
        # 🔴 the porter's own map, read — never a locally re-derived offset.
        b2r = P3AM.for_monster(em_id, n_rec, rig.n_bones,
                               offset=bone_offset, skip=skip_bones)
    r2b = {r: b for b, r in b2r.items()}
    loc_bones = sorted({r2b[r] for a in anims for r, tr in enumerate(a.tracks)
                        if r in r2b and any(
                            _is_loc(ch.type) for ch in tr.channels if ch.keyframes)})
    bad = P3AM.loc_below_fork(list(rig.parents), loc_bones)
    if bad:
        notes.append("🔴 the record->bone map puts LOCATION channels on joints %s, "
                     "BELOW the body fork (joint %d) — half the animal will lift and "
                     "the waist will tear. That is a mapping fault, not a rig fault."
                     % (bad, P3AM.body_fork(list(rig.parents))))
    clips = []
    for a in anims:
        frames = max((kf.frame for tr in a.tracks for ch in tr.channels
                      for kf in ch.keyframes), default=0)
        driven = tuple(sorted({r2b[i] for i, tr in enumerate(a.tracks)
                               if i in r2b and any(ch.keyframes for ch in tr.channels)}))
        clips.append(Clip(slot=a.slot, frames=frames, loop=bool(a.loop),
                          tracks=len(a.tracks), track_to_joint=r2b, driven=driven,
                          whole_rig=True, _anim=a))
    return sorted(clips, key=lambda c: c.slot), r2b


def _is_loc(channel_type: int) -> bool:
    from mhfu_model import convert as C
    return C.channel_kind(channel_type)[0] == "loc"


def _probe_companions(path: Path, geo, anim, notes):
    """An MHP3rd model PAC N keeps its geometry at N+1 and its moveset at N+2."""
    if geo is not None and anim is not None:
        return Path(geo), Path(anim)
    stem = path.stem
    if not stem.startswith("file_") or not stem[5:].isdigit():
        if anim is None:
            notes.append("%s is not a `file_NNNNN` name, so the moveset companion "
                         "could not be probed — pass anim= to animate it" % path.name)
        return (Path(geo) if geo else None), (Path(anim) if anim else None)
    n = int(stem[5:])
    for off, cur in ((1, geo), (2, anim)):
        if cur is not None:
            continue
        cand = path.with_name("file_%05d.bin" % (n + off))
        if cand.exists():
            if off == 1:
                geo = cand
            else:
                anim = cand
        else:
            notes.append("%s (the %s companion) is missing"
                         % (cand.name, "geometry" if off == 1 else "moveset"))
    return (Path(geo) if geo else None), (Path(anim) if anim else None)


def _em_from_path(path: Path) -> int:
    """`p3rd_anim_map.EM_BY_MODEL_PAC` for a `file_NNNNN` donor, or -1 if unmapped.

    🔴 The id is a LOOKUP, not a formula — monsters occupy different numbers of files.
    Read it out of the data (the two files before a model PAC are its `emNNNm*.ovl` AI
    overlays and an `MWo3` header names the species) and add a row; do not guess.
    """
    stem = path.stem
    if stem.startswith("file_") and stem[5:].isdigit():
        return P3AM.em_for_model_pac(int(stem[5:]))
    return -1


# --------------------------------------------------------------------------- #
# the one door
# --------------------------------------------------------------------------- #
def open_scene(target, **kw) -> Scene:
    """Open a PAC path, a `port.toml` path, or a loaded :class:`PortManifest`.

    Keyword arguments go to :meth:`Scene.from_manifest` for a manifest (``pac``,
    ``root``, ``side``) and to :meth:`Scene.from_pac` for a PAC (``geo``, ``anim``,
    ``em_id``, ``bone_offset``, ``skip_bones``, ``name``).
    """
    if isinstance(target, PortManifest):
        return Scene.from_manifest(target, **kw)
    p = Path(target)
    if p.suffix == ".toml":
        return Scene.from_manifest(p, **kw)
    return Scene.from_pac(p, **kw)
