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
    [[hurtbox]]                  bone + radius + part + hitzone_row (where he is HIT)
    [[hitbox]]                   bone + radius + set — an ATTACK volume (where he HITS)
    [[attack]]                   id + power / element / volume — one attack record's levers
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
* **`[moves]` has `latch` / `min_gap`, not `budget`.** ``latch`` (how many executor
  dispatches the clip covers, default 1) and ``min_gap`` (ticks before the same move may
  be re-issued, default 2) are the two knobs `mhfu_port.lua` actually reads; there is no
  authorable budget beside them. (The engine's own `entity+0x414` frame budget is a
  different thing entirely — a per-action countdown that gates 27 of the Tigrex's 231
  actions, reported by `species/emNN.json` and *not* settable from a manifest.)
* **`[moves]` also has `allow_unentered`.** ``validate`` refuses a move bound to a
  `(main,sub)` the census MEASURED as never entered; this is the explicit override that
  downgrades it to a warning, so the decision is in the file rather than in a flag on
  someone's command line.
* **`[moves]` has `after` / `hold_max` — a move is one link of a chain.** The engine
  walks behaviour pairs in sequence (the charge `(1,4)` hands to the skid `(0,3)` when
  its run budget is spent — `species/emNN.json` `next`), and a pair written from Lua
  is NOT provisioned the way the translator provisions it, so it can park forever with
  its hitbox spent. ``after`` names the move the runtime hands to when this one is over
  or has stood `hold_max` ticks; declaring it is what turns "loop one pair" into "walk
  the sequence". A move whose pair the intel says never ends itself has to declare it.
  (`after`, not `then` — `then` is a Lua keyword and the script transcribes this table.)
* **`[moves]` has `claim`, and there are `[[rule]]` blocks — the native seams (em_vhook
  v3, issues #15/#16).** With the seam live the runtime enters a pair through the
  engine's OWN enter-action (provisioned: the charge gets its run budget and ends
  itself), so `after`/`hold_max` become the fallback. `claim = { main = 1 }` (or
  `{ main = [0, 1], sub = 7 }`) makes every enter-action the HOST brain issues for a
  pair in that set become this move — the host decides WHEN, the port decides WHAT.
  A `[[rule]]` is a trigger the 30 Hz stub evaluates every frame with no Lua in the
  loop: `from = "lunge"` (or `from_main = [0, 1]`), `min_frames`, `dist = [lo, hi]`,
  `receding` / `closing`, `play = "lunge_stop"`, `cooldown`, `count`. Four of each.
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
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    #: 🔴 WHICH BUILD this label was written against — `clips.build_id`, e.g.
    #: ``"zinogre_v10.bin@8f3c1a02"``. Clip ids are per PAC build: the porter files a
    #: source clip into the host slot of the SAME index, so rebuilding with a different
    #: host frame or a different source shifts them, and a label with no build recorded
    #: is a label that cannot be checked. `mhfu_monster_editor.clips.track_labels`
    #: compares it against the build in front of you and says CARRIED / MOVED / LOST
    #: instead of letting a stale name go on looking authoritative.
    labelled_build: Optional[str] = None


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
    #: bind this pair even though the census measured ZERO entries into it. An
    #: explicit, auditable override for the one thing `validate` refuses outright —
    #: 411 of 411 forced moves into such a pair survived exactly one tick, so the
    #: default is to refuse and the flag exists to be argued for in a comment.
    allow_unentered: bool = False
    #: the move the runtime hands to when this one is over (the engine left the pair)
    #: or has stood `hold_max` ticks — one link of a declared chain. None = none.
    #: (`after`, not `then`: `then` is a Lua keyword, and the script's `moves`
    #: table is this field transcribed by hand.)
    after: Optional[str] = None
    #: ticks (2 Hz) after which the runtime hands to `after` even if the pair still
    #: stands. None = only when the engine leaves the pair.
    hold_max: Optional[int] = None
    #: the host enter-actions this move is SUBSTITUTED for (the slot-32 seam):
    #: `{"main": [1]}` = every main-1 attack the host brain picks, `{"main": [1],
    #: "sub": 7}` = one of them. None = the host's own choices stand.
    claim: Optional["Claim"] = None


@dataclass
class Claim:
    """Which host enter-actions a move takes over: any `main` in the set, and either
    every sub of it or exactly `sub`."""
    mains: List[int]
    sub: Optional[int] = None

    @property
    def mask(self) -> int:
        m = 0
        for k in self.mains:
            m |= 1 << k
        return m


@dataclass
class Rule:
    """A native brain rule (the slot-29 seam, issue #16): evaluated every frame.

    Fires `play` when the live pair is in the `from` set (`from_move`, a move name,
    or `from_main`, a set of main states), has stood `min_frames`, the player is
    inside `dist`, and the gap is growing (`receding`) / shrinking (`closing`) if
    asked — then `cooldown` frames pass before it may fire again, `count` times.
    """
    play: str
    from_move: Optional[str] = None
    from_main: List[int] = field(default_factory=list)
    min_frames: int = 0
    dist: Tuple[float, float] = (0.0, 1.0e9)
    receding: bool = False
    closing: bool = False
    mode: int = 0
    cooldown: int = 0
    #: None = standing (the runtime passes EM_UNLIMITED)
    count: Optional[int] = None
    label: str = ""


#: the ten damage-type columns of a hitzone row, in file order.
#: `raw` and the five elements + `ko` are named by inference, not read out of the
#: game — `tools/mhfu_model/hitzone.COLUMN_PROVENANCE` says which is which.
HITZONE_COLUMNS = ("raw", "cut", "impact", "shot",
                   "fire", "water", "dragon", "thunder", "ice", "ko")
HITZONE_ROWS = 7
#: `entity+0x3B8[8]`, and the deposit masks the sphere's part field with 7.
PART_SLOTS = 8
SHAPES = ("sphere", "capsule")


@dataclass
class Hurtbox:
    """One collision volume on the port's OWN rig: a sphere, or a capsule.

    The host overlay's records are `0x28` bytes carrying two different indices, and
    conflating them is the mistake this schema used to make:

    * **`part`** (0..7) is the damage accumulator — `entity+0x3B8[part]` — and the
      thing that breaks or severs.
    * **`hitzone_row`** (0..6) chooses which row of `[[hitzone]]` percentages the hit
      is multiplied by.

    A Tigrex wing is part 6 and row 5. `offset` is BONE-RELATIVE, in engine units;
    a capsule sweeps the sphere from `offset` to `to`.

    ⚠️ Bone indices are indices into the rig THIS PORT SHIPS. Copying the host's
    across attaches the head sphere to whatever joint happens to sit at that index.
    """
    bone: int
    radius: float
    part: Optional[int] = None
    hitzone_row: Optional[int] = None
    shape: str = "sphere"
    offset: Optional[List[float]] = None
    #: capsule far end, bone-relative. Ignored for a sphere.
    to: Optional[List[float]] = None
    #: the record's `+0x08` word, shipped verbatim. The engine's walkers SKIP a
    #: record whose flags meet `hitzone.WALK_SKIP_MASK` (0x050A0A04); a hurtbox
    #: the hunter can hit is 0 or 0x101 on the Tigrex. Leave it 0 unless copying.
    flags: int = 0
    label: str = ""

    @property
    def is_capsule(self) -> bool:
        return self.shape == "capsule"

    @property
    def is_marker(self) -> bool:
        """Bones 0x7D..0x7F are walker markers (0x7D = the tail-sever skip), not
        joints. They draw nowhere and are kept in order when adopted."""
        return self.bone in (0x7D, 0x7E, 0x7F)


@dataclass
class Part:
    """One of the eight damage accumulators, named.

    The engine has no names for these — it has `entity+0x3B8[0..7]`. Naming them is
    the whole point of writing them down: "part 6" is unreadable and "left wing" is
    not, and the runtime move table and the break rules both key off the index.
    """
    name: str
    index: int
    #: the grid row this part's spheres read. Advisory: the row is per SPHERE and a
    #: part may legitimately use more than one (the Tigrex head, part 1, sits on
    #: rows 1 and 2).
    hitzone_row: Optional[int] = None
    #: severable in the MH sense — the tail comes off. Recorded, NOT yet implemented:
    #: the sever mechanic is deferred (`docs/BRUTE_TIGREX_PORT.md`).
    severable: bool = False
    label: str = ""


@dataclass
class HitzoneState:
    """One `0x48` grid block: seven rows of ten percentages.

    A species has one to three. The engine picks by `entity.s8[+0x481]`, which is
    what makes "break it first and THEN it gets weak" expressible.

    🔴 The grid is SPECIES data, shared map-wide. A port riding host 75 inherits the
    native Tigrex's grid, so authoring one here is a statement about what the port
    WANTS, not something the build can apply on its own — see `docs/ASSETS.md` and
    issue #19. `validate` says so.
    """
    name: str
    #: 7 rows x 10 columns, in `HITZONE_COLUMNS` order.
    rows: List[List[int]] = field(default_factory=list)
    label: str = ""

    def value(self, row: int, column: str) -> int:
        return self.rows[row][HITZONE_COLUMNS.index(column)]


#: bones that are a COORDINATE SPACE in an attack volume, not a joint (`hitbox.py`):
#: 125 a joiner with no geometry of its own, 126 a capsule between the NODE's own two
#: points, 127 a sphere at the node's own position — i.e. at the attacker.
HITBOX_BONE_JOINER = 0x7D
HITBOX_BONE_NODE_CAPSULE = 0x7E
HITBOX_BONE_NODE_SPHERE = 0x7F
HITBOX_MARKER_BONES = (HITBOX_BONE_JOINER, HITBOX_BONE_NODE_CAPSULE,
                       HITBOX_BONE_NODE_SPHERE)


@dataclass
class Hitbox:
    """One ATTACK volume on the port's own rig — where he hits YOU (issue #33).

    The same `0x28` record as a :class:`Hurtbox` with `part` and `hitzone_row`
    unused: an attack volume does not need to say where it can be hit. What it
    carries instead is **`set`** — which of the host overlay's volume sets it belongs
    to. A handler spawns an attack by id, the attack record's `+0x0A` names a set,
    and the engine walks that set's records; so the sets are the unit the runtime
    replaces IN PLACE (each at its own address, each with its own capacity) and the
    unit the editor lists moves by: `[moves.lunge]` is `(1,4)`, `(1,4)` spawns
    attacks 6 and 31, both on set 2 — edit set 2 and the lunge hits where you put it.

    ⚠️ Bone indices are indices into the rig THIS PORT SHIPS, exactly as for a
    hurtbox: the host's set 2 sits on Tigrex bones 10/18/34/4/2/41/42/43, and on the
    Zinogre's rig those numbers are other joints. That misalignment is what this
    block exists to fix.

    🔴 The sets are SPECIES data in the overlay: a port REPLACING its host owns
    them; beside a native monster of the host species it re-arms the native too.
    """
    bone: int
    radius: float
    set: int
    shape: str = "sphere"
    offset: Optional[List[float]] = None
    to: Optional[List[float]] = None
    #: the record's `+0x08` word, shipped verbatim. Attack volumes in em75 carry 0.
    flags: int = 0
    label: str = ""

    @property
    def is_capsule(self) -> bool:
        return self.shape == "capsule"

    @property
    def is_marker(self) -> bool:
        """125/126/127: not a joint. 125 draws nowhere; 126/127 hang on the NODE's
        own position, which for a body attack is the attacker's origin."""
        return self.bone in HITBOX_MARKER_BONES

    @property
    def is_node_space(self) -> bool:
        """126/127 — geometry, but at the node's position rather than on a joint."""
        return self.bone in (HITBOX_BONE_NODE_CAPSULE, HITBOX_BONE_NODE_SPHERE)


@dataclass
class Attack:
    """The measured levers on one `0x18` attack record, by record id.

    Three fields were edited live and the game followed each (#33): `power`
    (`+0x02`; 64 -> 10 took the Tigrex charge from -72 HP to -11), `element`
    (`+0x09`, the byte the player-damage resolver masks to pick a resistance) and
    `volume` (`+0x0A`, which set the node walks). Nothing else on the record is
    decoded, so nothing else is authorable here — `None` leaves the host's byte.

    `id` is the RECORD index — for the host's own species the handler literal; a
    species sharing the overlay adds its offset (`intel.AttackIntel.id_offset`).
    """
    id: int
    power: Optional[int] = None
    element: Optional[int] = None
    volume: Optional[int] = None
    label: str = ""

    @property
    def is_empty(self) -> bool:
        return self.power is None and self.element is None and self.volume is None


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
    parts: Dict[str, Part] = field(default_factory=dict)
    hitzones: List[HitzoneState] = field(default_factory=list)
    hitboxes: List[Hitbox] = field(default_factory=list)
    attacks: List[Attack] = field(default_factory=list)
    effects: List[Effect] = field(default_factory=list)
    rules: List[Rule] = field(default_factory=list)
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


def _bounded(d: dict, key: str, where: str, limit: int) -> Optional[int]:
    """An optional index that must land inside `0..limit-1`. Out of range is a typo,
    not a policy question: a `hitzone_row` of 9 indexes past the end of a 0x48 block
    and a `part` of 9 is silently folded to 1 by the engine's `& 7`."""
    v = _opt(d, key, int, where)
    if v is not None and not 0 <= v < limit:
        raise ManifestError("%s.%s: %d is outside 0..%d" % (where, key, v, limit - 1))
    return v


def _vec3(d: dict, key: str, where: str) -> Optional[List[float]]:
    v = d.get(key)
    if v is None:
        return None
    v = _typed(v, list, "%s.%s" % (where, key))
    if len(v) != 3:
        raise ManifestError("%s.%s: %d value(s), not 3 (x, y, z in engine units)"
                            % (where, key, len(v)))
    return [float(x) for x in v]


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


def _claim(v, where: str) -> Optional["Claim"]:
    if v is None:
        return None
    w = where + ".claim"
    if isinstance(v, int) and not isinstance(v, bool):
        v = {"main": v}
    v = _typed(v, dict, w)
    _reject_unknown(v, _CLAIM_KEYS, w)
    mains = v.get("main")
    if isinstance(mains, int) and not isinstance(mains, bool):
        mains = [mains]
    if not isinstance(mains, list) or not mains:
        raise ManifestError("%s: needs main = <state> or [states]" % w)
    mains = [_typed(x, int, "%s.main[%d]" % (w, i)) for i, x in enumerate(mains)]
    for k in mains:
        if not 0 <= k <= 7:
            raise ManifestError("%s: main %d is not a main state (0..7)" % (w, k))
    return Claim(mains=sorted(set(mains)), sub=_opt(v, "sub", int, w))


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
_CLIP_KEYS = ("slot", "frames", "loop", "label", "impact_frame",
              "labelled_build")
_MOVE_KEYS = ("main", "sub", "clip", "anim", "latch", "min_gap", "label",
              "allow_unentered", "after", "hold_max", "claim")
_CLAIM_KEYS = ("main", "sub")
_RULE_KEYS = ("play", "from", "from_main", "min_frames", "dist", "receding", "closing",
              "mode", "cooldown", "count", "label")
_HURTBOX_KEYS = ("bone", "radius", "part", "hitzone_row", "shape", "offset",
                 "to", "flags", "label")
_PART_KEYS = ("index", "hitzone_row", "severable", "label")
_HITZONE_KEYS = ("state", "rows", "label")
_HITBOX_KEYS = ("bone", "radius", "set", "shape", "offset", "to", "flags", "label")
_ATTACK_KEYS = ("id", "power", "element", "volume", "label")
_EFFECT_KEYS = ("move", "frame", "id", "bone", "label")
_TOP_KEYS = ("schema", "port", "source", "build", "clips", "moves", "hurtbox",
             "parts", "hitzone", "hitbox", "attack", "effect", "rule")


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
            impact_frame=_opt(c, "impact_frame", int, w),
            labelled_build=_opt(c, "labelled_build", str, w))

    moves: Dict[str, Move] = {}
    for mname, m in _typed(raw.get("moves", {}), dict, where + ".moves").items():
        w = "moves.%s" % mname
        m = _typed(m, dict, w)
        _reject_unknown(m, _MOVE_KEYS, w)
        moves[mname] = Move(
            name=mname, main=_need(m, "main", int, w), sub=_need(m, "sub", int, w),
            clip=_opt(m, "clip", str, w), anim=_opt(m, "anim", int, w),
            latch=_opt(m, "latch", int, w, 1), min_gap=_opt(m, "min_gap", int, w, 2),
            label=_opt(m, "label", str, w, ""),
            allow_unentered=_opt(m, "allow_unentered", bool, w, False),
            after=_opt(m, "after", str, w), hold_max=_opt(m, "hold_max", int, w),
            claim=_claim(m.get("claim"), w))
    for mname, mv in moves.items():
        if mv.after is not None and mv.after not in moves:
            raise ManifestError("moves.%s: after = %r names no [moves.%s]"
                                % (mname, mv.after, mv.after))
        if mv.after == mname:
            raise ManifestError("moves.%s: after = itself — that is the loop on one pair "
                                "this field exists to replace" % mname)
        if mv.hold_max is not None and mv.hold_max < 1:
            raise ManifestError("moves.%s: hold_max must be >= 1 tick" % mname)

    hurtboxes = []
    for i, h in enumerate(_typed(raw.get("hurtbox", []), list, where + ".hurtbox")):
        w = "hurtbox[%d]" % i
        h = _typed(h, dict, w)
        _reject_unknown(h, _HURTBOX_KEYS, w)
        shape = _opt(h, "shape", str, w, "sphere")
        if shape not in SHAPES:
            raise ManifestError("%s: shape %r is not one of %s"
                                % (w, shape, ", ".join(SHAPES)))
        hurtboxes.append(Hurtbox(
            bone=_need(h, "bone", int, w), radius=_need(h, "radius", float, w),
            part=_bounded(h, "part", w, PART_SLOTS),
            hitzone_row=_bounded(h, "hitzone_row", w, HITZONE_ROWS),
            shape=shape, offset=_vec3(h, "offset", w), to=_vec3(h, "to", w),
            flags=_opt(h, "flags", int, w, 0),
            label=_opt(h, "label", str, w, "")))

    parts: Dict[str, Part] = {}
    for pname, pd in sorted(_typed(raw.get("parts", {}), dict, where + ".parts").items()):
        w = "parts.%s" % pname
        pd = _typed(pd, dict, w)
        _reject_unknown(pd, _PART_KEYS, w)
        idx = _need(pd, "index", int, w)
        if not 0 <= idx < PART_SLOTS:
            raise ManifestError("%s: index %d is outside 0..%d — the engine masks "
                                "the part field with 7" % (w, idx, PART_SLOTS - 1))
        parts[pname] = Part(
            name=pname, index=idx,
            hitzone_row=_bounded(pd, "hitzone_row", w, HITZONE_ROWS),
            severable=_opt(pd, "severable", bool, w, False),
            label=_opt(pd, "label", str, w, ""))

    hitzones = []
    for i, hz in enumerate(_typed(raw.get("hitzone", []), list, where + ".hitzone")):
        w = "hitzone[%d]" % i
        hz = _typed(hz, dict, w)
        _reject_unknown(hz, _HITZONE_KEYS, w)
        rows = _typed(hz.get("rows", []), list, w + ".rows")
        if len(rows) != HITZONE_ROWS:
            raise ManifestError("%s: %d row(s), not %d — a grid block is seven rows "
                                "of ten" % (w, len(rows), HITZONE_ROWS))
        clean = []
        for ri, row in enumerate(rows):
            row = _typed(row, list, "%s.rows[%d]" % (w, ri))
            if len(row) != len(HITZONE_COLUMNS):
                raise ManifestError("%s.rows[%d]: %d value(s), not %d (%s)"
                                    % (w, ri, len(row), len(HITZONE_COLUMNS),
                                       ", ".join(HITZONE_COLUMNS)))
            for v in row:
                if not isinstance(v, int) or isinstance(v, bool) \
                        or not 0 <= v <= 255:
                    raise ManifestError("%s.rows[%d]: %r is not a percentage 0..255"
                                        % (w, ri, v))
            clean.append(list(row))
        hitzones.append(HitzoneState(name=_need(hz, "state", str, w), rows=clean,
                                     label=_opt(hz, "label", str, w, "")))

    hitboxes = []
    for i, h in enumerate(_typed(raw.get("hitbox", []), list, where + ".hitbox")):
        w = "hitbox[%d]" % i
        h = _typed(h, dict, w)
        _reject_unknown(h, _HITBOX_KEYS, w)
        shape = _opt(h, "shape", str, w, "sphere")
        if shape not in SHAPES:
            raise ManifestError("%s: shape %r is not one of %s"
                                % (w, shape, ", ".join(SHAPES)))
        st = _need(h, "set", int, w)
        if st < 0:
            raise ManifestError("%s: set %d — a volume-set index is 0 or more" % (w, st))
        hitboxes.append(Hitbox(
            bone=_need(h, "bone", int, w), radius=_need(h, "radius", float, w),
            set=st, shape=shape, offset=_vec3(h, "offset", w), to=_vec3(h, "to", w),
            flags=_opt(h, "flags", int, w, 0), label=_opt(h, "label", str, w, "")))

    attacks = []
    for i, a in enumerate(_typed(raw.get("attack", []), list, where + ".attack")):
        w = "attack[%d]" % i
        a = _typed(a, dict, w)
        _reject_unknown(a, _ATTACK_KEYS, w)
        aid = _need(a, "id", int, w)
        if aid < 0:
            raise ManifestError("%s: id %d — a record index is 0 or more" % (w, aid))
        attacks.append(Attack(
            id=aid, power=_bounded(a, "power", w, 256),
            element=_bounded(a, "element", w, 256),
            volume=_bounded(a, "volume", w, 256),
            label=_opt(a, "label", str, w, "")))
    seen_ids = [a.id for a in attacks]
    for aid in sorted(set(seen_ids)):
        if seen_ids.count(aid) > 1:
            raise ManifestError("attack: record %d is declared %d times — one block "
                                "per record, or the last write wins silently"
                                % (aid, seen_ids.count(aid)))

    effects = []
    for i, e in enumerate(_typed(raw.get("effect", []), list, where + ".effect")):
        w = "effect[%d]" % i
        e = _typed(e, dict, w)
        _reject_unknown(e, _EFFECT_KEYS, w)
        effects.append(Effect(
            move=_need(e, "move", str, w), frame=_need(e, "frame", int, w),
            id=_need(e, "id", int, w), bone=_need(e, "bone", int, w),
            label=_opt(e, "label", str, w, "")))

    rules = []
    for i, r in enumerate(_typed(raw.get("rule", []), list, where + ".rule")):
        w = "rule[%d]" % i
        r = _typed(r, dict, w)
        _reject_unknown(r, _RULE_KEYS, w)
        dist = r.get("dist", [0.0, 1.0e9])
        if not (isinstance(dist, list) and len(dist) == 2
                and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in dist)):
            raise ManifestError("%s: dist must be [lo, hi] in units" % w)
        if dist[0] < 0 or dist[1] <= dist[0]:
            raise ManifestError("%s: dist = [%r, %r] — need 0 <= lo < hi" % (w, dist[0], dist[1]))
        fm = _int_list(r, "from_main", w, []) or []
        for k in fm:
            if not 0 <= k <= 7:
                raise ManifestError("%s: from_main %d is not a main state (0..7)" % (w, k))
        count = _opt(r, "count", int, w)
        if count is not None and count < 1:
            raise ManifestError("%s: count must be >= 1 (omit it for a standing rule)" % w)
        rules.append(Rule(
            play=_need(r, "play", str, w), from_move=_opt(r, "from", str, w),
            from_main=fm, min_frames=_opt(r, "min_frames", int, w, 0),
            dist=(float(dist[0]), float(dist[1])),
            receding=_opt(r, "receding", bool, w, False),
            closing=_opt(r, "closing", bool, w, False),
            mode=_opt(r, "mode", int, w, 0), cooldown=_opt(r, "cooldown", int, w, 0),
            count=count, label=_opt(r, "label", str, w, "")))
        if rules[-1].from_move is None and not fm:
            raise ManifestError("%s: needs `from` (a move) or `from_main` (main states)" % w)
        if rules[-1].receding and rules[-1].closing:
            raise ManifestError("%s: receding and closing cannot both be required" % w)
    if len(rules) > 4:
        raise ManifestError("rule: %d declared, the seam holds 4" % len(rules))

    m = PortManifest(
        name=_need(port, "name", str, "port"),
        host_species=_need(port, "host_species", int, "port"),
        pac=_need(port, "pac", str, "port"),
        source=source, build=build,
        host_frame=_opt(port, "host_frame", int, "port"),
        fid=_opt(port, "fid", int, "port"),
        orig=_opt(port, "orig", str, "port"),
        replace=_int_list(port, "replace", "port", []) or [],
        clips=clips, moves=moves, hurtboxes=hurtboxes, parts=parts,
        hitzones=hitzones, hitboxes=hitboxes, attacks=attacks, effects=effects,
        rules=rules, schema=schema, path=Path(path) if path else None)

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
    for i, r in enumerate(m.rules):
        if r.play not in m.moves:
            raise ManifestError("rule[%d]: play = %r is not declared in [moves]" % (i, r.play))
        if r.from_move is not None and r.from_move not in m.moves:
            raise ManifestError("rule[%d]: from = %r is not declared in [moves]"
                                % (i, r.from_move))
        if r.from_move == r.play:
            raise ManifestError("rule[%d]: from and play are both %r — the rule would "
                                "restart the pair it waits in" % (i, r.play))
    claimed: Dict[Tuple[int, Optional[int]], str] = {}
    for mv in m.moves.values():
        if mv.claim is None:
            continue
        for k in mv.claim.mains:
            key = (k, mv.claim.sub)
            if key in claimed:
                raise ManifestError("moves.%s and moves.%s both claim main %d%s — the "
                                    "first slot wins silently at runtime"
                                    % (claimed[key], mv.name, k,
                                       "" if mv.claim.sub is None else " sub %d" % mv.claim.sub))
            claimed[key] = mv.name
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
        _kv(out, "labelled_build", c.labelled_build)

    for name in sorted(m.moves):
        mv = m.moves[name]
        out += ["", "[moves.%s]" % name]
        _kv(out, "main", mv.main)
        _kv(out, "sub", mv.sub)
        _kv(out, "clip", mv.clip)
        _kv(out, "anim", mv.anim)
        _kv(out, "latch", mv.latch)
        _kv(out, "min_gap", mv.min_gap)
        if mv.allow_unentered:
            _kv(out, "allow_unentered", mv.allow_unentered)
        _kv(out, "after", mv.after)
        _kv(out, "hold_max", mv.hold_max)
        if mv.claim is not None:
            inner = "main = %s" % _atom(mv.claim.mains)
            if mv.claim.sub is not None:
                inner += ", sub = %d" % mv.claim.sub
            out.append("claim = { %s }" % inner)
        _kv(out, "label", mv.label)

    for name in sorted(m.parts):
        pt = m.parts[name]
        out += ["", "[parts.%s]" % name]
        _kv(out, "index", pt.index)
        _kv(out, "hitzone_row", pt.hitzone_row)
        if pt.severable:
            _kv(out, "severable", pt.severable)
        _kv(out, "label", pt.label)

    for h in m.hurtboxes:
        out += ["", hurtbox_block(h)]

    for hz in m.hitzones:
        out += ["", "[[hitzone]]"]
        _kv(out, "state", hz.name)
        out.append("# " + "  ".join("%7s" % c for c in HITZONE_COLUMNS))
        out.append("rows = [")
        for row in hz.rows:
            out.append("  [%s]," % ", ".join("%3d" % v for v in row))
        out.append("]")
        _kv(out, "label", hz.label)

    for h in m.hitboxes:
        out += ["", hitbox_block(h)]

    for a in m.attacks:
        out += ["", attack_block(a)]

    for e in m.effects:
        out += ["", "[[effect]]"]
        _kv(out, "move", e.move)
        _kv(out, "frame", e.frame)
        _kv(out, "id", e.id)
        _kv(out, "bone", e.bone)
        _kv(out, "label", e.label)

    for r in m.rules:
        out += ["", "[[rule]]"]
        _kv(out, "from", r.from_move)
        _kv(out, "from_main", r.from_main)
        if r.min_frames:
            _kv(out, "min_frames", r.min_frames)
        if r.dist != (0.0, 1.0e9):
            _kv(out, "dist", [r.dist[0], r.dist[1]])
        if r.receding:
            _kv(out, "receding", True)
        if r.closing:
            _kv(out, "closing", True)
        _kv(out, "play", r.play)
        if r.mode:
            _kv(out, "mode", r.mode)
        if r.cooldown:
            _kv(out, "cooldown", r.cooldown)
        _kv(out, "count", r.count)
        _kv(out, "label", r.label)

    return "\n".join(out).rstrip("\n") + "\n"


def hurtbox_block(h: Hurtbox) -> str:
    """One `[[hurtbox]]` block, as :func:`dumps` would write it."""
    out: List[str] = ["[[hurtbox]]"]
    _kv(out, "bone", h.bone)
    _kv(out, "radius", h.radius)
    _kv(out, "part", h.part)
    _kv(out, "hitzone_row", h.hitzone_row)
    if h.shape != "sphere":
        _kv(out, "shape", h.shape)
    _kv(out, "offset", h.offset)
    _kv(out, "to", h.to)
    if h.flags:
        out.append("flags = 0x%X" % h.flags)     # TOML hex; readable as a mask
    _kv(out, "label", h.label)
    return "\n".join(out)


def hitzone_block(hz: HitzoneState) -> str:
    """One `[[hitzone]]` block, with the column header as a comment."""
    out: List[str] = ["[[hitzone]]"]
    _kv(out, "state", hz.name)
    _kv(out, "label", hz.label)
    out.append("#        " + " ".join("%7s" % c for c in HITZONE_COLUMNS))
    out.append("rows = [")
    for i, row in enumerate(hz.rows):
        out.append("  [%s],   # row %d" % (", ".join("%3d" % v for v in row), i))
    out.append("]")
    return "\n".join(out)


def hitbox_block(h: Hitbox) -> str:
    """One `[[hitbox]]` block, as :func:`dumps` would write it."""
    out: List[str] = ["[[hitbox]]"]
    _kv(out, "set", h.set)
    _kv(out, "bone", h.bone)
    _kv(out, "radius", h.radius)
    if h.shape != "sphere":
        _kv(out, "shape", h.shape)
    _kv(out, "offset", h.offset)
    _kv(out, "to", h.to)
    if h.flags:
        out.append("flags = 0x%X" % h.flags)
    _kv(out, "label", h.label)
    return "\n".join(out)


def attack_block(a: Attack) -> str:
    """One `[[attack]]` block: the record id and only the levers that are set."""
    out: List[str] = ["[[attack]]"]
    _kv(out, "id", a.id)
    if a.power is not None:
        _kv(out, "power", a.power)
    if a.element is not None:
        out.append("element = 0x%02X" % a.element)      # a gate byte reads as a mask
    if a.volume is not None:
        _kv(out, "volume", a.volume)
    _kv(out, "label", a.label)
    return "\n".join(out)


def save(m: PortManifest, path: os.PathLike | str) -> Path:
    """Write canonical TOML. ⚠️ Overwrites; comments in the old file are lost."""
    p = Path(path)
    p.write_text(dumps(m), encoding="utf-8")
    return p


def discover(root: os.PathLike | str = "ports") -> List[PortManifest]:
    """Every `*.toml` under ``root``, sorted by filename."""
    return [load(p) for p in sorted(Path(root).glob("*.toml"))]


# --------------------------------------------------------------------------- #
# patch — edit a manifest's TEXT in place, keeping everything else byte for byte
# --------------------------------------------------------------------------- #
# :func:`dumps` re-emits a manifest from the parsed object, which is correct and
# lossy: `tomllib` drops comments, so writing a hand-authored `ports/*.toml` back
# through it destroys the prose. Both shipped manifests are about half comments, and
# the comments are the part that carries knowledge —
#
#     # 🔴 CLIP IDS ARE PER BUILD. docs/brute_tigrex_anim_ids.txt was labelled by
#     # filming an EARLIER Brute, and on `v67_hostslots` a1 = label - 1 ...
#
# — so the editor (issue #8) does not rewrite the file. It patches the LINES it is
# changing and leaves every other byte alone, which also makes `git diff` after a
# labelling session show the labels and nothing else.
#
# This is a text edit, not a TOML re-serialisation: it knows how to find a table, a
# key inside it, and where a value ends. Anything it cannot locate unambiguously
# raises rather than guessing, and :func:`patch` re-parses its own output before
# returning, so a patch that would produce a file this loader cannot read fails at
# the call instead of on disk.

_TABLE_RE = re.compile(r"^\s*\[\[?\s*(?P<name>[^\[\]]+?)\s*\]\]?\s*(?:#.*)?$")
_KEY_RE = re.compile(r"^(?P<indent>\s*)(?P<key>[A-Za-z0-9_\-]+|\"[^\"]*\")"
                     r"(?P<eq>\s*=\s*)(?P<rest>.*)$")


@dataclass(frozen=True)
class SetKey:
    """Set ``table.key`` to ``value``. ``value=None`` removes the key.

    The table and the key are both created when missing — a new `[clips.<name>]` is
    inserted after the last `[clips.*]` table so the file stays grouped.
    """
    table: str
    key: str
    value: Any


@dataclass(frozen=True)
class RenameClip:
    """Rename `[clips.old]` to `[clips.new]`, and every `moves.*.clip` that named it.

    Schema-aware on purpose: a clip's name is a *reference target*, so renaming the
    table alone silently turns `clip = "charge"` into a dangling pointer — which
    :func:`from_dict` would then refuse to load, on the next session, with no clue
    that a rename caused it.
    """
    old: str
    new: str


@dataclass(frozen=True)
class AppendBlock:
    """Append a whole `[[array]]` block to the end of the file.

    `SetKey` addresses `[table.key]` and cannot reach an array of tables: a file has
    many `[[hurtbox]]` blocks and the name does not say which. The editor writes
    those wholesale instead — a hurtbox record and a hitzone grid are generated
    numbers, not prose, so replacing one loses nothing a comment was carrying.
    """
    text: str


@dataclass(frozen=True)
class ReplaceBlock:
    """Replace the ``index``-th `[[name]]` block with ``text`` (``None`` deletes it).

    ⚠️ Comments INSIDE the replaced block are lost — it is rewritten, not edited.
    Everything outside it, including the prose above the header, is untouched, which
    is what keeps a hand-authored manifest readable after an editing session.
    """
    name: str
    index: int
    text: Optional[str] = None


Op = Any            # SetKey | RenameClip | AppendBlock | ReplaceBlock


def _value_end(rest: str, where: str) -> int:
    """Index in ``rest`` just past the value, so a trailing comment can be kept.

    Handles what this schema emits: basic and literal strings, single-line arrays,
    numbers and booleans. A value that does not finish on its line raises — a
    multi-line array is legal TOML that nothing here writes, and truncating one would
    corrupt the file.
    """
    i, n, depth = 0, len(rest), 0
    while i < n:
        ch = rest[i]
        if ch == '"' or ch == "'":
            quote, i = ch, i + 1
            while i < n and rest[i] != quote:
                i += 2 if (quote == '"' and rest[i] == "\\") else 1
            if i >= n:
                raise ManifestError("%s: unterminated string in %r" % (where, rest))
            i += 1
            continue
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
        elif ch == "#" and depth == 0:
            break
        i += 1
    if depth:
        raise ManifestError("%s: the value spans more than one line, which this "
                            "patcher does not edit: %r" % (where, rest))
    return len(rest[:i].rstrip())


def _tables(lines: List[str]) -> Dict[str, tuple]:
    """``{table name: (header index, first body line, stop)}`` for every `[table]`.

    ``stop`` is one past the table's last *content* line, so blank lines and the
    comment block that introduces the NEXT table stay outside it — inserting a key at
    ``stop`` puts it under the keys it belongs with rather than under someone else's
    heading. `[[array]]` tables are recognised (they end a table) but not addressed:
    nothing the editor writes lives in one.
    """
    heads = [(i, m.group("name")) for i, line in enumerate(lines)
             for m in (_TABLE_RE.match(line),) if m]
    out: Dict[str, tuple] = {}
    for k, (i, name) in enumerate(heads):
        end = heads[k + 1][0] if k + 1 < len(heads) else len(lines)
        stop = end
        while stop > i + 1 and (not lines[stop - 1].strip()
                                or lines[stop - 1].lstrip().startswith("#")):
            stop -= 1
        out.setdefault(name, (i, i + 1, stop))
    return out


def _find_key(lines: List[str], span: tuple, key: str) -> Optional[int]:
    _, start, stop = span
    for i in range(start, stop):
        m = _KEY_RE.match(lines[i])
        if m and m.group("key").strip('"') == key:
            return i
    return None


def _set_key(text: str, table: str, key: str, value: Any) -> str:
    lines = text.split("\n")
    tables = _tables(lines)
    where = "%s.%s" % (table, key)
    span = tables.get(table)

    if span is None:
        if value is None:
            return text
        head = table.split(".")[0]
        block = ["", "[%s]" % table, "%s = %s" % (key, _atom(value))]
        at = max((sp[2] for name, sp in tables.items()
                  if name == head or name.startswith(head + ".")), default=None)
        if at is None:
            at = len(lines)
            while at and not lines[at - 1].strip():
                at -= 1
        lines[at:at] = block
        return "\n".join(lines)

    at = _find_key(lines, span, key)
    if at is None:
        if value is None:
            return text
        indent = ""
        for i in range(span[1], span[2]):
            m = _KEY_RE.match(lines[i])
            if m:
                indent = m.group("indent")
        lines.insert(span[2], "%s%s = %s" % (indent, key, _atom(value)))
        return "\n".join(lines)

    if value is None:
        del lines[at]
        return "\n".join(lines)
    m = _KEY_RE.match(lines[at])
    rest = m.group("rest")
    tail = rest[_value_end(rest, where):]
    lines[at] = "%s%s%s%s%s" % (m.group("indent"), m.group("key"), m.group("eq"),
                                _atom(value), tail)
    return "\n".join(lines)


def _rename_clip(text: str, old: str, new: str) -> str:
    lines = text.split("\n")
    tables = _tables(lines)
    if ("clips." + old) not in tables:
        raise ManifestError("cannot rename clips.%s: it is not in this file" % old)
    if ("clips." + new) in tables:
        raise ManifestError("cannot rename clips.%s to %r: that table already exists"
                            % (old, new))
    i = tables["clips." + old][0]
    lines[i] = lines[i].replace("[clips.%s]" % old, "[clips.%s]" % new, 1)
    text = "\n".join(lines)
    for name, span in _tables(text.split("\n")).items():
        if not name.startswith("moves."):
            continue
        at = _find_key(text.split("\n"), span, "clip")
        if at is None:
            continue
        rows = text.split("\n")
        m = _KEY_RE.match(rows[at])
        rest = m.group("rest")
        cut = _value_end(rest, name + ".clip")
        if rest[:cut].strip() == _atom(old):
            rows[at] = "%s%s%s%s%s" % (m.group("indent"), m.group("key"),
                                       m.group("eq"), _atom(new), rest[cut:])
            text = "\n".join(rows)
    return text


def _array_blocks(lines: List[str], name: str) -> List[tuple]:
    """``[(header index, stop)]`` for each `[[name]]` block, in file order.

    ``stop`` is one past the block's last CONTENT line, so the blank line and any
    comment block introducing the next table stay outside — the same rule
    :func:`_tables` uses, for the same reason.
    """
    heads = [(i, m.group("name"), line.lstrip().startswith("[["))
             for i, line in enumerate(lines)
             for m in (_TABLE_RE.match(line),) if m]
    out = []
    for k, (i, nm, is_arr) in enumerate(heads):
        if not (is_arr and nm == name):
            continue
        end = heads[k + 1][0] if k + 1 < len(heads) else len(lines)
        stop = end
        while stop > i + 1 and (not lines[stop - 1].strip()
                                or lines[stop - 1].lstrip().startswith("#")):
            stop -= 1
        out.append((i, stop))
    return out


def _replace_block(text: str, name: str, index: int, body: Optional[str]) -> str:
    lines = text.split("\n")
    blocks = _array_blocks(lines, name)
    if not 0 <= index < len(blocks):
        raise ManifestError("no [[%s]] block at index %d (the file has %d)"
                            % (name, index, len(blocks)))
    i, stop = blocks[index]
    new = [] if body is None else body.rstrip("\n").split("\n")
    return "\n".join(lines[:i] + new + lines[stop:])


def _append_block(text: str, body: str) -> str:
    out = text.rstrip("\n")
    return out + "\n\n" + body.strip("\n") + "\n"


def patch(text: str, ops: List[Op], *, path: Optional[os.PathLike | str] = None) -> str:
    """Apply ``ops`` to manifest TEXT, preserving comments, order and formatting.

    Ops run in the order given, so a :class:`RenameClip` followed by a
    :class:`SetKey` on the new name does what it reads like. The result is parsed
    before it is returned: a patch that would write a file this loader cannot read
    raises :class:`ManifestError` here rather than landing on disk.
    """
    for op in ops:
        if isinstance(op, RenameClip):
            text = _rename_clip(text, op.old, op.new)
        elif isinstance(op, SetKey):
            text = _set_key(text, op.table, op.key, op.value)
        elif isinstance(op, AppendBlock):
            text = _append_block(text, op.text)
        elif isinstance(op, ReplaceBlock):
            text = _replace_block(text, op.name, op.index, op.text)
        else:                                                    # pragma: no cover
            raise TypeError("not a manifest op: %r" % (op,))
    loads(text, path=path)
    return text


def patch_file(path: os.PathLike | str, ops: List[Op]) -> str:
    """:func:`patch` a manifest on disk. Returns the new text; writes only on success."""
    p = Path(path)
    text = patch(p.read_text(encoding="utf-8"), ops, path=p)
    p.write_text(text, encoding="utf-8")
    return text
