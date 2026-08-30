"""`port.toml` — the per-port manifest, and the only description of a ported monster.

One file per port. Hand-editable, read and written by this package, consumed by the
porter (`tools/build_p3rd_port.py --manifest`) and, later, by the runtime. It replaces
three things that used to live in three different places and drift apart:

  * the **flag soup** a port was built with (only ever recorded in a commit message or
    a Lua comment — `zinogre_v10.bin` is reproducible today because someone pasted the
    command line into `docs/agent_session_log.md`);
  * the Lua ``clips`` / ``moves`` tables in `framework/prx/mods/lua_host/scripts/*.lua`
    (`docs/MOD_PORTED_MONSTER.md` §3);
  * the hitzone volumes (`tools/mhfu_model/hitzone.py`) and the effect recipe
    (`docs/EFFECTS_AND_VFX.md` §5), neither of which has ever been written down per port.

**This module is pure data.** No rendering, no GL, no UI, stdlib only. Validation lives
next door in :mod:`mhfu_monster_editor.validate` because it needs the built PAC.

Shape
-----
::

    schema = 1

    [port]                       name / host_species / pac / fid / orig / replace
    [source]                     the donor game's file ids + its em id
    [build]                      what the porter needs (skin, source_skeleton, ...)
    [clips.<name>]               slot = executor a1, plus the frame fingerprint
    [moves.<name>]               main / sub / clip — the two-channel alignment
    [[hurtbox]]                  bone + radius (hitzone.Volume)
    [[effect]]                   move / frame / id / bone

Where this differs from the issue's sketch, and why
---------------------------------------------------
* **`source` file ids are integers, not `"file_05339"` strings.** The id is the identity
  — `p3rd_anim_map.em_for_model_pac` is keyed by it, and `build_p3rd_port.py` recovers it
  from the *filename* with a regex today. A manifest that carries the number skips the
  round trip through a string. `geo` and `anim` default to ``model + 1`` / ``model + 2``
  (`docs/MHP3RD_FILE_MONSTER_MAP.md`).
* **`bone_offset` moved out of `[port]` into `[build]`, and `em_id` into `[source]`.**
  The offset is an *override* of the measured value in `p3rd_anim_map.BONE_OFFSET`; the
  em id is what selects it. Naming the em id and leaving the offset unset is the correct
  authoring, and it is what both shipped ports do — pinning the number in the manifest
  forks it away from the map that the renderer and the porter both read.
* **`[moves]` has `latch` / `min_gap`, not `budget`.** There is no "budget" anywhere in
  the runtime. ``latch`` (how many executor dispatches the clip covers, default 1) and
  ``min_gap`` (ticks before the same move may be re-issued, default 2) are the two knobs
  `mhfu_port.lua` actually reads.
* **`[[hurtbox]]` does not carry `part` as a first-class field.** The sketch conflated
  two *different* tables in the host overlay: VOLUMES (`0x28` records: bone + radius, no
  part field at all) and WEAKNESS (`0x18` records, keyed by `part_id`). See
  `tools/mhfu_model/hitzone.py`. `part` survives here only as an optional back-reference
  for the editor's own labelling; nothing writes it into the volume record.
* **`frames` / `loop` on a clip are the fingerprint, and they are checkable.** Clips are
  stored unpadded at their authored length, so `(end, loop)` identifies a slot against
  the built PAC — which is what makes ``validate`` able to reject a clip slot that is not
  in the build (`docs/MOD_PORTED_MONSTER.md` §4b).

The data root is deliberately **not** in the manifest: `workspace/` is a machine-local
symlink to a game dump that is never committed. Pass it to :func:`PortManifest.build_args`.
"""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

SCHEMA = 1

#: extracted model-PAC id for a host species. The ENGINE asks for ``species + 6111``;
#: the extracted ids run one LOWER because `extract_iso.py` reads the TOC from offset 4
#: (root `CLAUDE.md` §4). Tigrex 75 -> file_06185, engine fid 6186.
SPECIES_TO_FRAME = 6110


class ManifestError(ValueError):
    """A manifest that cannot be loaded at all — a missing key, a bad type, a typo.

    Structural only. Anything that needs the built PAC or the action census is a
    :class:`mhfu_monster_editor.validate.Issue`, not an exception.
    """


# --------------------------------------------------------------------------- #
# records
# --------------------------------------------------------------------------- #
@dataclass
class Clip:
    """One entry of the port's animation vocabulary. ``slot`` IS the executor ``a1``.

    ``frames`` is the clip's own last keyframe and ``loop`` its loop flag — together the
    fingerprint the clip-state block (`ent+0x80 + slot*0x40`, `+0x1C`) reads back live,
    and what `validate` compares against the built PAC.
    """
    name: str
    slot: int
    frames: Optional[int] = None
    loop: Optional[bool] = None
    label: str = ""
    #: frame at which the move connects. Not derivable offline today — issue #4.
    impact_frame: Optional[int] = None


@dataclass
class Move:
    """The alignment: a host behaviour pair, plus the clip shown while it runs.

    ``main``/``sub`` go to `act_set` (`entity+0x298`/`+0x299`) and own the hitbox, the
    damage and the effects; ``clip`` names a :class:`Clip` and only paints the animation
    channel. Forcing a pair the engine never enters bounces out in one tick — which is
    what `validate` rejects when action intel is available.
    """
    name: str
    main: int
    sub: int
    clip: Optional[str] = None
    #: raw executor a1, for a move that bypasses the clip vocabulary (`mv.anim`)
    anim: Optional[int] = None
    latch: int = 1
    min_gap: int = 2
    label: str = ""


@dataclass
class Hurtbox:
    """A bone-attached collision sphere — one `hitzone.Volume` (`0x28` record)."""
    bone: int
    radius: float
    #: optional back-reference into the separate WEAKNESS table; not part of the volume
    part: Optional[int] = None
    label: str = ""


@dataclass
class Effect:
    """`at frame F, effect E at bone B`, anchored to a move.

    Fired with `mhfu.spawn_effect(entity, id, bone)` from inside an override callback.
    There is no effect data to port — the ids are MHFU's own (`docs/EFFECTS_AND_VFX.md`).
    """
    move: str
    frame: int
    id: int
    bone: int
    label: str = ""


@dataclass
class Source:
    """The donor monster: which game, which species, which three files."""
    model: int
    game: str = "mhp3rd"
    #: donor species id (Zinogre 40, Brute Tigrex 58). Selects the anim record->bone map
    #: in `p3rd_anim_map`; a NEGATIVE value opts out and builds unmapped.
    em_id: Optional[int] = None
    geo: Optional[int] = None
    anim: Optional[int] = None

    def __post_init__(self) -> None:
        if self.geo is None:
            self.geo = self.model + 1
        if self.anim is None:
            self.anim = self.model + 2


@dataclass
class Build:
    """Everything `build_p3rd_port.py` needs. Defaults match the CLI's own defaults."""
    source_skeleton: bool = False
    skin: str = "auto"
    ground_lift: float = 0.0
    #: animated-bone count override (= the anim 3-stream partition total)
    animated: Optional[int] = None
    #: override `p3rd_anim_map.BONE_OFFSET[em_id]`. Leave unset — measure, do not guess.
    bone_offset: Optional[int] = None
    #: override `p3rd_anim_map.SKIPPED_BONES[em_id]`
    skip_bones: Optional[List[int]] = None
    drop_joints: Optional[List[int]] = None
    nb: int = 3
    hops: int = 1
    #: 🔴 SUPERSEDED (tears the auxiliary plate). Kept so the experiment reproduces.
    reweight_undriven: bool = False

    SKINS = ("auto", "transfer", "source")


@dataclass
class PortManifest:
    name: str
    host_species: int
    pac: str
    source: Source
    build: Build = field(default_factory=Build)
    #: extracted file id of the host PAC. Defaults to ``host_species + 6110``.
    host_frame: Optional[int] = None
    #: engine file id the injector announces. Defaults to ``host_frame + 1``.
    fid: Optional[int] = None
    #: the untouched host PAC the relocate injector keeps alongside. Derived by default.
    orig: Optional[str] = None
    #: quest monster ids swapped for ``host_species`` at QUEST_TARGETS_BUILDING
    replace: List[int] = field(default_factory=list)
    clips: Dict[str, Clip] = field(default_factory=dict)
    moves: Dict[str, Move] = field(default_factory=dict)
    hurtboxes: List[Hurtbox] = field(default_factory=list)
    effects: List[Effect] = field(default_factory=list)
    schema: int = SCHEMA
    #: where it was loaded from, when it was. Not part of identity — two manifests
    #: with the same content are equal however they were read.
    path: Optional[Path] = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        if self.host_frame is None:
            self.host_frame = self.host_species + SPECIES_TO_FRAME
        if self.fid is None:
            self.fid = self.host_frame + 1
        if self.orig is None:
            self.orig = "file_%05d.bin.orig" % self.host_frame

    # ---- paths -------------------------------------------------------- #
    def host_pac_path(self, root: os.PathLike | str) -> Path:
        """Path of the MHFU host frame PAC under an extract root."""
        return Path(root) / "extracted" / "data_files" / ("file_%05d.bin" % self.host_frame)

    def source_paths(self, root: os.PathLike | str) -> Dict[str, Path]:
        """``{model, geo, anim}`` paths of the donor files under an extract root."""
        sub = {"mhp3rd": "extracted_mhp3"}.get(self.source.game, "extracted_" + self.source.game)
        base = Path(root) / sub / "data_files"
        return {k: base / ("file_%05d.bin" % getattr(self.source, k))
                for k in ("model", "geo", "anim")}

    # ---- the porter --------------------------------------------------- #
    def build_args(self, root: os.PathLike | str = "workspace") -> Dict[str, Any]:
        """Manifest -> `build_p3rd_port.py` argparse dests.

        Only keys the manifest actually determines; the CLI's own defaults cover the
        rest, and an explicit flag on the command line always wins over this.
        """
        src = self.source_paths(root)
        args: Dict[str, Any] = {
            "model": str(src["model"]),
            "geo": str(src["geo"]),
            "anim": str(src["anim"]),
            "frame": str(self.host_pac_path(root)),
            "skin": self.build.skin,
            "source_skeleton": self.build.source_skeleton,
            "ground_lift": self.build.ground_lift,
            "nb": self.build.nb,
            "hops": self.build.hops,
            "reweight_undriven": self.build.reweight_undriven,
        }
        if self.source.em_id is not None:
            args["em_id"] = self.source.em_id
        if self.build.animated is not None:
            args["animated"] = self.build.animated
        if self.build.bone_offset is not None:
            args["anim_bone_offset"] = self.build.bone_offset
        if self.build.drop_joints:
            args["drop_joints"] = ",".join(str(x) for x in self.build.drop_joints)
        return args

    def build_argv(self, root: os.PathLike | str = "workspace",
                   out: Optional[str] = None) -> List[str]:
        """The equivalent command line, for logs, docs and reproduction."""
        a = self.build_args(root)
        argv: List[str] = []
        for k in ("model", "geo", "anim", "frame"):
            argv += ["--" + k, a[k]]
        argv += ["--skin", a["skin"]]
        if a["source_skeleton"]:
            argv += ["--source-skeleton"]
        if a["ground_lift"]:
            argv += ["--ground-lift", repr(a["ground_lift"])]
        for dest, flag in (("em_id", "--em-id"), ("animated", "--animated"),
                           ("anim_bone_offset", "--anim-bone-offset"),
                           ("drop_joints", "--drop-joints")):
            if dest in a:
                argv += [flag, str(a[dest])]
        if a["reweight_undriven"]:
            argv += ["--reweight-undriven"]
        argv += ["--out", out or self.pac]
        return argv


# --------------------------------------------------------------------------- #
# load
# --------------------------------------------------------------------------- #
def _need(d: dict, key: str, typ, where: str):
    if key not in d:
        raise ManifestError("%s: missing required key %r" % (where, key))
    return _typed(d[key], typ, "%s.%s" % (where, key))


def _typed(v, typ, where: str):
    if typ is float and isinstance(v, int) and not isinstance(v, bool):
        return float(v)
    if typ is bool and not isinstance(v, bool):
        raise ManifestError("%s: expected a boolean, got %r" % (where, v))
    if typ is int and isinstance(v, bool):
        raise ManifestError("%s: expected an integer, got a boolean" % where)
    if not isinstance(v, typ):
        raise ManifestError("%s: expected %s, got %r" % (where, typ.__name__, v))
    return v


def _opt(d: dict, key: str, typ, where: str, default=None):
    if key not in d or d[key] is None:
        return default
    return _typed(d[key], typ, "%s.%s" % (where, key))


def _int_list(d: dict, key: str, where: str, default=None):
    if key not in d:
        return default
    v = d[key]
    if not isinstance(v, list):
        raise ManifestError("%s.%s: expected a list of integers" % (where, key))
    return [_typed(x, int, "%s.%s[%d]" % (where, key, i)) for i, x in enumerate(v)]


def _reject_unknown(d: dict, known, where: str):
    extra = sorted(set(d) - set(known))
    if extra:
        raise ManifestError(
            "%s: unknown key(s) %s — a typo here is silent otherwise. Known: %s"
            % (where, ", ".join(repr(e) for e in extra), ", ".join(sorted(known))))


_PORT_KEYS = ("name", "host_species", "host_frame", "pac", "fid", "orig", "replace")
_SOURCE_KEYS = ("game", "em_id", "model", "geo", "anim")
_BUILD_KEYS = ("source_skeleton", "skin", "ground_lift", "animated", "bone_offset",
               "skip_bones", "drop_joints", "nb", "hops", "reweight_undriven")
_CLIP_KEYS = ("slot", "frames", "loop", "label", "impact_frame")
_MOVE_KEYS = ("main", "sub", "clip", "anim", "latch", "min_gap", "label")
_HURTBOX_KEYS = ("bone", "radius", "part", "label")
_EFFECT_KEYS = ("move", "frame", "id", "bone", "label")
_TOP_KEYS = ("schema", "port", "source", "build", "clips", "moves", "hurtbox", "effect")


def loads(text: str, *, path: Optional[os.PathLike | str] = None) -> PortManifest:
    """Parse manifest TOML. Raises :class:`ManifestError` on anything structural."""
    try:
        raw = tomllib.loads(text)
    except tomllib.TOMLDecodeError as e:
        raise ManifestError("%s: not valid TOML — %s" % (path or "<string>", e)) from e
    return from_dict(raw, path=path)


def load(path: os.PathLike | str) -> PortManifest:
    """Load a `port.toml` from disk."""
    p = Path(path)
    return loads(p.read_text(encoding="utf-8"), path=p)


def from_dict(raw: dict, *, path: Optional[os.PathLike | str] = None) -> PortManifest:
    where = str(path or "<manifest>")
    _reject_unknown(raw, _TOP_KEYS, where)

    schema = _opt(raw, "schema", int, where, SCHEMA)
    if schema != SCHEMA:
        raise ManifestError("%s: schema = %d, this loader speaks %d"
                            % (where, schema, SCHEMA))

    port = _typed(raw.get("port", {}), dict, where + ".port")
    _reject_unknown(port, _PORT_KEYS, "port")
    src_raw = _typed(raw.get("source", {}), dict, where + ".source")
    _reject_unknown(src_raw, _SOURCE_KEYS, "source")
    build_raw = _typed(raw.get("build", {}), dict, where + ".build")
    _reject_unknown(build_raw, _BUILD_KEYS, "build")

    source = Source(
        model=_need(src_raw, "model", int, "source"),
        game=_opt(src_raw, "game", str, "source", "mhp3rd"),
        em_id=_opt(src_raw, "em_id", int, "source"),
        geo=_opt(src_raw, "geo", int, "source"),
        anim=_opt(src_raw, "anim", int, "source"),
    )

    skin = _opt(build_raw, "skin", str, "build", "auto")
    if skin not in Build.SKINS:
        raise ManifestError("build.skin: %r is not one of %s"
                            % (skin, ", ".join(Build.SKINS)))
    build = Build(
        source_skeleton=_opt(build_raw, "source_skeleton", bool, "build", False),
        skin=skin,
        ground_lift=_opt(build_raw, "ground_lift", float, "build", 0.0),
        animated=_opt(build_raw, "animated", int, "build"),
        bone_offset=_opt(build_raw, "bone_offset", int, "build"),
        skip_bones=_int_list(build_raw, "skip_bones", "build"),
        drop_joints=_int_list(build_raw, "drop_joints", "build"),
        nb=_opt(build_raw, "nb", int, "build", 3),
        hops=_opt(build_raw, "hops", int, "build", 1),
        reweight_undriven=_opt(build_raw, "reweight_undriven", bool, "build", False),
    )

    clips: Dict[str, Clip] = {}
    for cname, c in _typed(raw.get("clips", {}), dict, where + ".clips").items():
        w = "clips.%s" % cname
        c = _typed(c, dict, w)
        _reject_unknown(c, _CLIP_KEYS, w)
        clips[cname] = Clip(
            name=cname, slot=_need(c, "slot", int, w),
            frames=_opt(c, "frames", int, w), loop=_opt(c, "loop", bool, w),
            label=_opt(c, "label", str, w, ""),
            impact_frame=_opt(c, "impact_frame", int, w))

    moves: Dict[str, Move] = {}
    for mname, m in _typed(raw.get("moves", {}), dict, where + ".moves").items():
        w = "moves.%s" % mname
        m = _typed(m, dict, w)
        _reject_unknown(m, _MOVE_KEYS, w)
        moves[mname] = Move(
            name=mname, main=_need(m, "main", int, w), sub=_need(m, "sub", int, w),
            clip=_opt(m, "clip", str, w), anim=_opt(m, "anim", int, w),
            latch=_opt(m, "latch", int, w, 1), min_gap=_opt(m, "min_gap", int, w, 2),
            label=_opt(m, "label", str, w, ""))

    hurtboxes = []
    for i, h in enumerate(_typed(raw.get("hurtbox", []), list, where + ".hurtbox")):
        w = "hurtbox[%d]" % i
        h = _typed(h, dict, w)
        _reject_unknown(h, _HURTBOX_KEYS, w)
        hurtboxes.append(Hurtbox(
            bone=_need(h, "bone", int, w), radius=_need(h, "radius", float, w),
            part=_opt(h, "part", int, w), label=_opt(h, "label", str, w, "")))

    effects = []
    for i, e in enumerate(_typed(raw.get("effect", []), list, where + ".effect")):
        w = "effect[%d]" % i
        e = _typed(e, dict, w)
        _reject_unknown(e, _EFFECT_KEYS, w)
        effects.append(Effect(
            move=_need(e, "move", str, w), frame=_need(e, "frame", int, w),
            id=_need(e, "id", int, w), bone=_need(e, "bone", int, w),
            label=_opt(e, "label", str, w, "")))

    m = PortManifest(
        name=_need(port, "name", str, "port"),
        host_species=_need(port, "host_species", int, "port"),
        pac=_need(port, "pac", str, "port"),
        source=source, build=build,
        host_frame=_opt(port, "host_frame", int, "port"),
        fid=_opt(port, "fid", int, "port"),
        orig=_opt(port, "orig", str, "port"),
        replace=_int_list(port, "replace", "port", []) or [],
        clips=clips, moves=moves, hurtboxes=hurtboxes, effects=effects,
        schema=schema, path=Path(path) if path else None)

    # cross-references are structural: a move pointing at a clip that is not declared
    # is a typo, not a policy question, and it silently becomes `mv.clip == nil` in Lua.
    for mv in m.moves.values():
        if mv.clip is not None and mv.clip not in m.clips:
            raise ManifestError("moves.%s: clip %r is not declared in [clips]"
                                % (mv.name, mv.clip))
        if mv.clip is None and mv.anim is None:
            raise ManifestError("moves.%s: needs either `clip` (a name from [clips]) "
                                "or `anim` (a raw executor a1)" % mv.name)
    for i, ef in enumerate(m.effects):
        if ef.move not in m.moves:
            raise ManifestError("effect[%d]: move %r is not declared in [moves]"
                                % (i, ef.move))
    seen: Dict[int, str] = {}
    for c in m.clips.values():
        if c.slot in seen:
            raise ManifestError("clips.%s and clips.%s both claim slot %d — one name "
                                "per slot, or `play()` cannot tell them apart"
                                % (seen[c.slot], c.name, c.slot))
        seen[c.slot] = c.name
    return m


# --------------------------------------------------------------------------- #
# save — a small deterministic TOML emitter
# --------------------------------------------------------------------------- #
# There is no TOML *writer* in the stdlib (3.11 gives `tomllib`, read-only) and this
# package takes no third-party dependency, so this is hand-rolled. It only has to emit
# the value types the schema uses; anything else raises rather than guessing. The
# round trip `loads(dumps(m)) == m` is pinned by tests/test_manifest.py.
#
# ⚠️ Comments are NOT preserved. `tomllib` drops them on read, so a manifest written
# back by the editor loses the prose in the hand-authored ports/*.toml. Keep an
# explanation that must survive in the docs, not in the file.
# A basic TOML string may not carry a raw control character, and the editor writes
# free text here — a clip LABEL typed with a newline in it (issue #8) would otherwise
# emit a file `tomllib` refuses to read back.
_ESCAPES = {"\\": "\\\\", '"': '\\"', "\b": "\\b", "\t": "\\t",
            "\n": "\\n", "\f": "\\f", "\r": "\\r"}


def _escape(v: str) -> str:
    out = []
    for ch in v:
        esc = _ESCAPES.get(ch)
        if esc is not None:
            out.append(esc)
        elif ch < "\x20" or ch == "\x7f":
            out.append("\\u%04X" % ord(ch))
        else:
            out.append(ch)
    return "".join(out)


def _atom(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return repr(v)                       # repr round-trips a float exactly
    if isinstance(v, str):
        return '"%s"' % _escape(v)
    if isinstance(v, (list, tuple)):
        return "[%s]" % ", ".join(_atom(x) for x in v)
    raise TypeError("no TOML rendering for %r" % type(v).__name__)


def _kv(out: List[str], key: str, v: Any) -> None:
    if v is None or v == "" or v == []:
        return
    out.append("%s = %s" % (key, _atom(v)))


def dumps(m: PortManifest) -> str:
    """Canonical TOML for a manifest. Deterministic; round-trips through :func:`loads`."""
    out: List[str] = ["schema = %d" % m.schema, "", "[port]"]
    _kv(out, "name", m.name)
    _kv(out, "host_species", m.host_species)
    _kv(out, "host_frame", m.host_frame)
    _kv(out, "pac", m.pac)
    _kv(out, "fid", m.fid)
    _kv(out, "orig", m.orig)
    _kv(out, "replace", m.replace)

    out += ["", "[source]"]
    _kv(out, "game", m.source.game)
    _kv(out, "em_id", m.source.em_id)
    _kv(out, "model", m.source.model)
    _kv(out, "geo", m.source.geo)
    _kv(out, "anim", m.source.anim)

    out += ["", "[build]"]
    _kv(out, "source_skeleton", m.build.source_skeleton)
    _kv(out, "skin", m.build.skin)
    _kv(out, "ground_lift", m.build.ground_lift)
    _kv(out, "animated", m.build.animated)
    _kv(out, "bone_offset", m.build.bone_offset)
    _kv(out, "skip_bones", m.build.skip_bones)
    _kv(out, "drop_joints", m.build.drop_joints)
    _kv(out, "nb", m.build.nb)
    _kv(out, "hops", m.build.hops)
    _kv(out, "reweight_undriven", m.build.reweight_undriven)

    for name in sorted(m.clips):
        c = m.clips[name]
        out += ["", "[clips.%s]" % name]
        _kv(out, "slot", c.slot)
        _kv(out, "frames", c.frames)
        if c.loop is not None:
            _kv(out, "loop", c.loop)
        _kv(out, "impact_frame", c.impact_frame)
        _kv(out, "label", c.label)

    for name in sorted(m.moves):
        mv = m.moves[name]
        out += ["", "[moves.%s]" % name]
        _kv(out, "main", mv.main)
        _kv(out, "sub", mv.sub)
        _kv(out, "clip", mv.clip)
        _kv(out, "anim", mv.anim)
        _kv(out, "latch", mv.latch)
        _kv(out, "min_gap", mv.min_gap)
        _kv(out, "label", mv.label)

    for h in m.hurtboxes:
        out += ["", "[[hurtbox]]"]
        _kv(out, "bone", h.bone)
        _kv(out, "radius", h.radius)
        _kv(out, "part", h.part)
        _kv(out, "label", h.label)

    for e in m.effects:
        out += ["", "[[effect]]"]
        _kv(out, "move", e.move)
        _kv(out, "frame", e.frame)
        _kv(out, "id", e.id)
        _kv(out, "bone", e.bone)
        _kv(out, "label", e.label)

    return "\n".join(out).rstrip("\n") + "\n"


def save(m: PortManifest, path: os.PathLike | str) -> Path:
    """Write canonical TOML. ⚠️ Overwrites; comments in the old file are lost."""
    p = Path(path)
    p.write_text(dumps(m), encoding="utf-8")
    return p


def discover(root: os.PathLike | str = "ports") -> List[PortManifest]:
    """Every `*.toml` under ``root``, sorted by filename."""
    return [load(p) for p in sorted(Path(root).glob("*.toml"))]
