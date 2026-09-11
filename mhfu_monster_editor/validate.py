"""Check a `port.toml` against the things that have actually cost this project builds.

:mod:`mhfu_monster_editor.manifest` already refuses anything structural — a missing key,
a bad type, a move pointing at an undeclared clip. What is left needs *evidence*: the
built PAC, and a census of what the host species' AI really does. Both are optional
inputs, and a check that has no evidence says so rather than passing quietly.

The three traps, and what each one looks like when it is not caught
------------------------------------------------------------------
1. **A clip slot that is not in the built PAC.** The porter files a source clip into the
   host slot of the SAME INDEX, so a source clip whose index the host pack does not have
   is dropped — silently, with every structural check still passing. Worse, a third of
   the a1 space is *filler*: a copy of the idle clip. Forcing one of those is a
   successful override onto nothing, and on screen it is indistinguishable from "the
   latch didn't work" (`docs/MOD_PORTED_MONSTER.md` §4b). Both are decidable offline
   from the built PAC's clip table.
2. **A move bound to a `(main,sub)` pair the engine never enters.** Its handler checks
   for a condition nobody created and returns immediately: **411 of 411 forced moves
   survived exactly one tick**, the clip restarted from frame 0 twice a second and no
   animation ever finished. `species/emNN.json` (`tools/em_intel.py`) carries it, and
   the file separates three states that look alike and are not: no census at all, a
   census that says nothing about this pair, and a census that measured ZERO entries.
   Only the last is an error. A port that means it may say `allow_unentered = true`.
   The same file also carries STATIC intel, so even with no census this checks that
   the host overlay dispatches the pair at all.
3. **A hurtbox bone out of range for the shipped skeleton.** The volume table is
   bone-indexed and a ported monster ships its own rig, so a host-derived index is not
   just wrong, it reads off the end of the joint array.

Usage::

    python -m mhfu_monster_editor.validate ports/zinogre.toml --pac tmp/zinogre_v10.bin

Exit status is 1 if anything came back at level ``error``.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .clips import clip_table as _clip_table
from .manifest import PortManifest, load as load_manifest
from .intel import (ActionIntel, MIN_DWELL_TICKS, PairIntel,  # noqa: F401
                    SpeciesIntel, find_intel)

ERROR = "error"
WARNING = "warning"

#: the reader of `species/emNN.json`. Issue #2 shipped a provisional one under this
#: name and called it "the single class #4 replaces"; #4 replaced it with
#: :class:`mhfu_monster_editor.intel.SpeciesIntel`, which still accepts the flat
#: shape the provisional reader documented. The alias keeps callers working.
JsonActionIntel = SpeciesIntel


@dataclass
class Issue:
    level: str          # ERROR | WARNING
    code: str           # stable, machine-readable
    where: str          # "clips.charge", "hurtbox[0]", "moves.charge"
    message: str

    def __str__(self) -> str:
        return "%-7s %-24s %-18s %s" % (self.level.upper(), self.code, self.where,
                                        self.message)


# --------------------------------------------------------------------------- #
# evidence from the built PAC
# --------------------------------------------------------------------------- #
def _tools_on_path() -> None:
    tools = str(Path(__file__).resolve().parent.parent / "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)


#: ``{slot: (last_keyframe, loop)}`` for every populated slot of a built PAC — the
#: fingerprint this validator checks a manifest's clips against. Lives in
#: :mod:`mhfu_monster_editor.clips` with the coverage classification that reads it, and
#: is re-exported here because that is the name this module has always had it under.
clip_table = _clip_table


def bone_count(pac: bytes) -> int:
    """Joint count of the skeleton the built PAC actually ships."""
    _tools_on_path()
    from mhfu_model import skeleton_p3rd as skp
    from mhfu_model.pac import MonsterPac

    sub = MonsterPac.from_bytes(pac).find("skeleton")
    if sub is None:
        raise ValueError("no skeleton sub-resource in this PAC")
    return len(skp.parse(sub.data).bones)


# --------------------------------------------------------------------------- #
# the validator
# --------------------------------------------------------------------------- #
def validate(m: PortManifest, *, pac: Optional[bytes | os.PathLike | str] = None,
             intel: Optional[ActionIntel] = None) -> List[Issue]:
    """Check a manifest. Both evidence sources are optional; missing ones warn.

    ``pac``   the BUILT port PAC — bytes, or a path. Without it the clip and bone
              checks cannot run and say so.
    ``intel`` anything satisfying :class:`ActionIntel` (issue #4). Without it the
              `(main,sub)` check cannot run and says so.
    """
    out: List[Issue] = []
    out += _check_derived(m)
    out += _check_pac(m, pac)
    out += _check_intel(m, intel)
    out += _check_parts(m, intel)
    out += _check_attacks(m, intel)
    return out


def _check_parts(m: PortManifest, intel: Optional[ActionIntel]) -> List[Issue]:
    """The part system: names, rows, and the two things that are easy to believe
    and wrong — that a part is a hitzone row, and that authoring a grid ships one."""
    out: List[Issue] = []
    by_index: Dict[int, List[str]] = {}
    for name, pt in sorted(m.parts.items()):
        by_index.setdefault(pt.index, []).append(name)
    for idx, names in sorted(by_index.items()):
        if len(names) > 1:
            out.append(Issue(ERROR, "PART_INDEX_DUPLICATE", "parts",
                             "%s all claim part %d. The index is the engine's "
                             "`entity+0x3B8` slot; two names for one slot means two "
                             "break bars that are secretly the same bar."
                             % (", ".join(repr(n) for n in names), idx)))

    named = {pt.index for pt in m.parts.values()}
    for i, h in enumerate(m.hurtboxes):
        if h.part is None:
            out.append(Issue(WARNING, "HURTBOX_NO_PART", "hurtbox[%d]" % i,
                             "no `part`, so a hit here deposits into slot 0 and "
                             "cannot be told apart from any other unassigned "
                             "volume. `hitzone_row` is a DIFFERENT field."))
        elif m.parts and h.part not in named:
            out.append(Issue(WARNING, "HURTBOX_PART_UNNAMED", "hurtbox[%d]" % i,
                             "part %d has no entry in [parts], so nothing in this "
                             "file says what it is." % h.part))
        if h.is_capsule and h.to is None:
            out.append(Issue(ERROR, "HURTBOX_CAPSULE_NO_END", "hurtbox[%d]" % i,
                             "shape = \"capsule\" but no `to` — a capsule with no "
                             "far end is a sphere at `offset`, so say which you "
                             "mean."))

    for hz in m.hitzones:
        if all(v == 0 for row in hz.rows for v in row):
            out.append(Issue(WARNING, "HITZONE_ALL_ZERO", "hitzone[%s]" % hz.name,
                             "every percentage is 0, which is a monster nothing can "
                             "hurt. Probably a stub that was never filled in."))
    names = [hz.name for hz in m.hitzones]
    for n in sorted(set(names)):
        if names.count(n) > 1:
            out.append(Issue(ERROR, "HITZONE_STATE_DUPLICATE", "hitzone",
                             "two states are both called %r; the engine picks a "
                             "state by INDEX (`entity+0x481`), so the name is the "
                             "only way a human tells them apart." % n))

    # what the host actually has, when we can see it
    pt = getattr(intel, "parts", None)
    if m.hitzones and pt is not None and getattr(pt, "has_grid", False):
        if len(m.hitzones) != pt.n_states:
            out.append(Issue(WARNING, "HITZONE_STATE_COUNT", "hitzone",
                             "%d state(s) authored but host species %d ships %d. The "
                             "state table's length is fixed by the host — an extra "
                             "block has nothing to point at it."
                             % (len(m.hitzones), getattr(intel, "host_species", -1),
                                pt.n_states)))
    if m.hitzones:
        out.append(Issue(WARNING, "HITZONE_SHARED", "hitzone",
                         "the damage grid is SPECIES data, shared map-wide: with the "
                         "port REPLACING the host it is his alone; beside a native "
                         "host monster it changes that one too. Shipped by "
                         "mhfu_monster_editor.runtime -> P.hit() over the species' "
                         "0x48 blocks (grid writes proven live 2026-06-28)."))
    cap = getattr(pt, "capacity", None) if pt is not None else None
    if m.hurtboxes and cap is not None and len(m.hurtboxes) > cap:
        out.append(Issue(WARNING, "HURTBOX_OVER_CAPACITY", "hurtbox",
                         "%d volume(s) but the host set holds %d: the runtime writes "
                         "them IN PLACE over that set and truncates the rest (#19)."
                         % (len(m.hurtboxes), cap)))
    if m.hurtboxes and not m.parts:
        out.append(Issue(WARNING, "PARTS_UNNAMED", "parts",
                         "%d hurtbox volume(s) and no [parts] — nothing in this file "
                         "says which part is the head." % len(m.hurtboxes)))
    return out


def _check_attacks(m: PortManifest, intel: Optional[ActionIntel]) -> List[Issue]:
    """The attack side (#33): sets that exist, records that exist, capacities, and
    the two things worth saying out loud — that the tables are shared species data,
    and that the spawner->table join is only MEASURED on em75."""
    out: List[Issue] = []
    if not (m.hitboxes or m.attacks):
        return out
    for i, h in enumerate(m.hitboxes):
        if h.is_capsule and h.to is None:
            out.append(Issue(ERROR, "HITBOX_CAPSULE_NO_END", "hitbox[%d]" % i,
                             "shape = \"capsule\" but no `to` — a capsule with no far "
                             "end is a sphere at `offset`, so say which you mean."))
    for i, a in enumerate(m.attacks):
        if a.is_empty:
            out.append(Issue(WARNING, "ATTACK_EMPTY", "attack[%d]" % i,
                             "record %d names no lever (power / element / volume), so "
                             "the block changes nothing." % a.id))

    at = getattr(intel, "attacks", None)
    sets = sorted({h.set for h in m.hitboxes})
    if at is None or not getattr(at, "present", False):
        out.append(Issue(WARNING, "HITBOX_UNCHECKED", "hitbox",
                         "no attacks block in species/em%02d.json, so set indices, "
                         "record ids and capacities are unchecked — and the runtime "
                         "export REFUSES until it is built (tools/em_intel.py --all)."
                         % m.host_species))
        return out

    n_sets, n_recs = len(at.sets), len(getattr(at.primary, "attacks", []) or [])
    for st in sets:
        hs = at.set(st)
        if hs is None:
            out.append(Issue(ERROR, "HITBOX_SET_UNKNOWN", "hitbox",
                             "set %d — host em%02d has %d volume set(s) (0..%d). The "
                             "runtime writes through the overlay's pointer table and "
                             "there is no entry to follow." % (st, m.host_species,
                                                                n_sets, n_sets - 1)))
            continue
        mine = [h for h in m.hitboxes if h.set == st]
        if len(mine) > hs.capacity:
            out.append(Issue(WARNING, "HITBOX_OVER_CAPACITY", "hitbox",
                             "set %d: %d volume(s) but the host's set holds %d — the "
                             "runtime writes them IN PLACE over that set and truncates "
                             "the rest." % (st, len(mine), hs.capacity)))
        if not hs.rigged:
            out.append(Issue(WARNING, "HITBOX_SET_UNRIGGED", "hitbox",
                             "set %d is un-rigged on the host (every record on bone "
                             "126/127, the node's own position — projectile-shaped). "
                             "A joint index here re-aligns nothing the engine reads."
                             % st))
        if not at.attacks_using(st):
            out.append(Issue(WARNING, "HITBOX_SET_UNUSED", "hitbox",
                             "no attack record points at set %d, so nothing the host "
                             "spawns walks it — unless an [[attack]] re-points one "
                             "(`volume = %d`)." % (st, st)))
    for i, a in enumerate(m.attacks):
        if at.attack(a.id) is None:
            out.append(Issue(ERROR, "ATTACK_RECORD_UNKNOWN", "attack[%d]" % i,
                             "record %d — host em%02d has %d record(s)."
                             % (a.id, m.host_species, n_recs)))
        if a.volume is not None and at.set(a.volume) is None:
            out.append(Issue(ERROR, "ATTACK_VOLUME_UNKNOWN", "attack[%d]" % i,
                             "volume %d — host em%02d has %d volume set(s)."
                             % (a.volume, m.host_species, n_sets)))
    if getattr(at, "join", "") != "measured":
        out.append(Issue(WARNING, "ATTACK_JOIN_INFERRED", "hitbox",
                         "host em%02d's spawner->table join is %s, not measured: only "
                         "em75's was walked live to the HP write. Which set a move "
                         "hits with is an inference here." % (m.host_species,
                                                               getattr(at, "join", "?"))))
    out.append(Issue(WARNING, "HITBOX_SHARED", "hitbox",
                     "attack sets and records are SPECIES data in the overlay: with "
                     "the port REPLACING the host they are his alone; beside a native "
                     "em%02d they re-arm the native too. Shipped by "
                     "mhfu_monster_editor.runtime -> P.hit() in place through the "
                     "overlay's set-pointer table (set replacement proven by RAM poke and the "
                     "generated path validated live in a running quest, both 2026-09-11)."
                     % m.host_species))
    return out


def _check_derived(m: PortManifest) -> List[Issue]:
    out: List[Issue] = []
    want_frame = m.host_species + 6110
    if m.host_frame != want_frame:
        out.append(Issue(WARNING, "HOST_FRAME_UNEXPECTED", "port",
                         "host_frame %d is not host_species %d + 6110 = %d. Legal if "
                         "you meant it; usually a typo." % (m.host_frame,
                                                            m.host_species, want_frame)))
    if m.fid != m.host_frame + 1:
        out.append(Issue(WARNING, "FID_UNEXPECTED", "port",
                         "fid %d is not host_frame %d + 1. The engine asks for "
                         "species + 6111." % (m.fid, m.host_frame)))
    if m.source.geo != m.source.model + 1 or m.source.anim != m.source.model + 2:
        out.append(Issue(WARNING, "SOURCE_FILES_UNEXPECTED", "source",
                         "geo/anim are normally model+1 / model+2 (%d/%d), got %d/%d"
                         % (m.source.model + 1, m.source.model + 2,
                            m.source.geo, m.source.anim)))
    if m.build.skin == "source" and not m.build.source_skeleton:
        out.append(Issue(WARNING, "SKIN_SOURCE_WITHOUT_RIG", "build",
                         "skin = \"source\" pairs the source bone palette with the "
                         "OUTPUT rig; without source_skeleton the indices are the "
                         "host's and every vertex lands on the wrong joint."))
    if m.build.reweight_undriven:
        out.append(Issue(WARNING, "REWEIGHT_UNDRIVEN", "build",
                         "reweight_undriven is SUPERSEDED — it tears the auxiliary "
                         "plate across the whole body. Use drop_joints."))
    if m.build.bone_offset is not None and m.source.em_id is not None:
        _tools_on_path()
        try:
            from mhfu_model import p3rd_anim_map as p3am
        except Exception:                                    # pragma: no cover
            p3am = None
        if p3am is not None:
            measured = p3am.BONE_OFFSET.get(m.source.em_id)
            if measured is not None and measured != m.build.bone_offset:
                out.append(Issue(WARNING, "BONE_OFFSET_OVERRIDE", "build",
                                 "bone_offset %d overrides the MEASURED offset %d for "
                                 "em%03d in p3rd_anim_map. The renderer reads the map, "
                                 "so the port would build at one bone map and render "
                                 "at another." % (m.build.bone_offset, measured,
                                                  m.source.em_id)))
    return out


def _check_pac(m: PortManifest, pac) -> List[Issue]:
    out: List[Issue] = []
    if pac is None:
        if m.clips or m.hurtboxes or m.effects:
            out.append(Issue(WARNING, "PAC_ABSENT", "-",
                             "no built PAC given: %d clip slot(s), %d hurtbox bone(s) "
                             "and %d effect bone(s) went unchecked. Build it and pass "
                             "--pac." % (len(m.clips), len(m.hurtboxes),
                                         len(m.effects))))
        return out

    blob = pac if isinstance(pac, (bytes, bytearray)) else Path(pac).read_bytes()
    blob = bytes(blob)

    try:
        table = clip_table(blob)
    except Exception as e:
        out.append(Issue(ERROR, "PAC_UNREADABLE", "-",
                         "cannot read the clip table out of the PAC: %s" % e))
        table = None

    if table is not None:
        # where each named clip's fingerprint actually IS in this build, so a moved
        # slot can be reported as moved instead of merely wrong. → #8
        from .clips import AMBIGUOUS, MOVED, track_labels
        moved = {t.name: t for t in track_labels(m, table)
                 if t.status in (MOVED, AMBIGUOUS)}
        idle = table.get(1)
        for name in sorted(m.clips):
            c = m.clips[name]
            w = "clips.%s" % name
            if c.slot not in table:
                out.append(Issue(ERROR, "CLIP_SLOT_MISSING", w,
                                 "slot %d is not populated in this build. The porter "
                                 "files a source clip into the host slot of the SAME "
                                 "index, so a source clip the host pack has no slot "
                                 "for is DROPPED — scripting it reaches nothing.%s"
                                 % (c.slot, _relocated(moved, name))))
                continue
            end, loop = table[c.slot]
            if c.frames is not None and c.frames != end:
                out.append(Issue(ERROR, "CLIP_FRAMES_MISMATCH", w,
                                 "declared frames = %d, the build's slot %d ends at "
                                 "%d. Clips are stored unpadded at their authored "
                                 "length, so this is not rounding — the slot holds a "
                                 "different clip.%s"
                                 % (c.frames, c.slot, end,
                                    _relocated(moved, name))))
            if c.loop is not None and bool(c.loop) != bool(loop):
                out.append(Issue(ERROR, "CLIP_LOOP_MISMATCH", w,
                                 "declared loop = %s, the build's slot %d has loop = %s"
                                 % (bool(c.loop), c.slot, bool(loop))))
            if idle is not None and c.slot != 1 and table[c.slot] == idle:
                out.append(Issue(WARNING, "CLIP_IS_FILLER", w,
                                 "slot %d holds a copy of the idle clip (%df, loop=%s) "
                                 "— the porter's filler. Forcing it plays IDLE, which "
                                 "on screen is identical to the override failing."
                                 % (c.slot, idle[0], bool(idle[1]))))
            if c.label and not c.labelled_build:
                out.append(Issue(WARNING, "LABEL_UNKEYED", w,
                                 "the label records no build, and clip ids are per "
                                 "build — so nothing can say whether it describes "
                                 "slot %d's clip or an earlier build's. Re-label it "
                                 "in the editor: it stamps the build." % c.slot))

    if m.hurtboxes or m.effects or m.hitboxes:
        try:
            nb = bone_count(blob)
        except Exception as e:
            out.append(Issue(ERROR, "PAC_UNREADABLE", "-",
                             "cannot read the skeleton out of the PAC: %s" % e))
            nb = None
        if nb is not None:
            for i, h in enumerate(m.hurtboxes):
                if not 0 <= h.bone < nb:
                    out.append(Issue(ERROR, "HURTBOX_BONE_RANGE", "hurtbox[%d]" % i,
                                     "bone %d is outside the shipped skeleton's %d "
                                     "joints. The volume table is bone-indexed and a "
                                     "port ships its OWN rig, so a host bone number "
                                     "does not transfer." % (h.bone, nb)))
                if h.radius <= 0:
                    out.append(Issue(ERROR, "HURTBOX_RADIUS", "hurtbox[%d]" % i,
                                     "radius %g is not a collision sphere" % h.radius))
            for i, h in enumerate(m.hitboxes):
                if h.is_marker:
                    continue           # 125/126/127 are a coordinate space, not a joint
                if not 0 <= h.bone < nb:
                    out.append(Issue(ERROR, "HITBOX_BONE_RANGE", "hitbox[%d]" % i,
                                     "bone %d is outside the shipped skeleton's %d "
                                     "joints. An attack volume is bone-indexed like a "
                                     "hurtbox, and the host's set sits on the HOST's "
                                     "joints." % (h.bone, nb)))
                if h.radius <= 0:
                    out.append(Issue(ERROR, "HITBOX_RADIUS", "hitbox[%d]" % i,
                                     "radius %g is not a volume that can touch anyone"
                                     % h.radius))
            for i, e in enumerate(m.effects):
                if not 0 <= e.bone < nb:
                    out.append(Issue(ERROR, "EFFECT_BONE_RANGE", "effect[%d]" % i,
                                     "bone %d is outside the shipped skeleton's %d "
                                     "joints; spawn_effect reads that bone's live "
                                     "world position." % (e.bone, nb)))
    return out


def _relocated(moved: Dict[str, object], name: str) -> str:
    """" — and where that clip went", when the fingerprint is elsewhere in the build."""
    t = moved.get(name)
    return "" if t is None else " " + t.message


def _check_intel(m: PortManifest, intel: Optional[ActionIntel]) -> List[Issue]:
    """The `(main,sub)` checks — static ones always, measured ones when a census exists.

    🔴 Three states that read alike and must never be collapsed:

    ``INTEL_ABSENT``            no measurements were supplied at all. Since #4 the
                                file usually EXISTS and is full of static intel while
                                still carrying no census, so this fires on the census,
                                not on the file.
    ``MOVE_PAIR_UNOBSERVED``    a census exists and has nothing on this pair.
    ``MOVE_PAIR_NEVER_ENTERED`` a census exists, looked, and measured zero entries.
                                The only one that is an ERROR.
    """
    out: List[Issue] = []
    if not m.moves:
        return out
    if intel is None:
        out.append(Issue(WARNING, "INTEL_ABSENT", "moves",
                         "no action intel for host species %d, so %d (main,sub) pair(s)"
                         " went unchecked. A pair the engine never enters bounces out "
                         "in ONE tick — 411 of 411 forced moves did. Build it with "
                         "tools/em_intel.py; measure with tools/em_state_census.py."
                         % (m.host_species, len(m.moves))))
        return out

    got = getattr(intel, "host_species", None)
    if got is not None and int(got) != m.host_species:
        out.append(Issue(ERROR, "INTEL_WRONG_SPECIES", "moves",
                         "the intel is for host species %d, this port rides %d. "
                         "(main,sub) is dispatched by the HOST's overlay, so intel "
                         "from another species says nothing." % (got, m.host_species)))
        return out

    # A file with static intel but no census is the NORMAL case: collecting a census
    # needs a cold boot with the observe-only probe deployed. Say so once, loudly,
    # rather than once per move.
    has_census = bool(getattr(intel, "has_census", True))
    has_static = bool(getattr(intel, "has_static", False))
    if not has_census:
        why = getattr(intel, "census_reason", "") or "no census was attached"
        out.append(Issue(WARNING, "INTEL_ABSENT", "moves",
                         "the intel for host species %d carries NO measurements (%s), "
                         "so whether the engine ever enters %d (main,sub) pair(s) is "
                         "UNKNOWN — not zero. Deploy the observe-only probe and re-run "
                         "tools/em_state_census.py / tools/em_intel.py --log."
                         % (m.host_species, why, len(m.moves))))

    for name in sorted(m.moves):
        mv = m.moves[name]
        w = "moves.%s" % name
        p = intel.pair(mv.main, mv.sub)
        if p is None:
            mains = getattr(intel, "enumerated_mains", set())
            if has_static and mv.main in mains:
                n = next((s.get("sub_states")
                          for s in getattr(intel, "main_states", [])
                          if s.get("main") == mv.main), None)
                out.append(Issue(ERROR, "MOVE_PAIR_NO_HANDLER", w,
                                 "the host overlay's action tick dispatches %s "
                                 "sub_state(s) under main %d and %d is not one of "
                                 "them, so act_set would land on nothing."
                                 % (n, mv.main, mv.sub)))
            else:
                out.append(Issue(WARNING, "MOVE_PAIR_UNOBSERVED", w,
                                 "nothing is known about (%d,%d): it is in neither "
                                 "the overlay's jump tables nor the census. Absent is "
                                 "not the same as never entered."
                                 % (mv.main, mv.sub)))
            continue

        if has_static and p.handler is None:
            out.append(Issue(WARNING, "MOVE_PAIR_NO_HANDLER", w,
                             "the dispatcher's case for (%d,%d) runs inline and calls "
                             "no handler, so nothing offline can say what it does."
                             % (mv.main, mv.sub)))

        if p.entered is None:
            if has_census:
                out.append(Issue(WARNING, "MOVE_PAIR_UNOBSERVED", w,
                                 "the census covers this species but says nothing "
                                 "about (%d,%d). Absent is not the same as never "
                                 "entered — the sample may simply not cover it."
                                 % (mv.main, mv.sub)))
            # with no census at all the file-level INTEL_ABSENT already said it
        elif p.entered <= 0:
            note = (" " + p.note) if getattr(p, "note", "") else ""
            lvl = WARNING if getattr(mv, "allow_unentered", False) else ERROR
            out.append(Issue(lvl, "MOVE_PAIR_NEVER_ENTERED", w,
                             "the census says the engine enters (%d,%d) ZERO times. "
                             "Its handler asks for a condition nothing has created and "
                             "returns at once: forced, it survives exactly one tick and "
                             "the clip restarts from frame 0 forever.%s%s"
                             % (mv.main, mv.sub, note,
                                " Allowed by allow_unentered."
                                if lvl == WARNING else "")))
        elif p.dwell_ticks and p.dwell_ticks < MIN_DWELL_TICKS:
            out.append(Issue(WARNING, "MOVE_PAIR_SHORT_DWELL", w,
                             "(%d,%d) holds for only %.1f ticks (%.1f s at 2 Hz) even "
                             "when the ENGINE picks it. Forced from the wrong range it "
                             "will bounce out." % (mv.main, mv.sub, p.dwell_ticks,
                                                   p.dwell_ticks / 2.0)))

        # static findings that are worth saying whatever the census knows
        if has_static and p.handler is not None:
            if mv.clip is not None and p.ends_on == "budget" and p.budget.gated:
                seeds = p.budget.phase0_seeds
                out.append(Issue(WARNING, "MOVE_PAIR_BUDGET_GATED", w,
                                 "(%d,%d) does not end when the clip ends — it runs on "
                                 "the +0x414 frame budget, so a longer ported clip is "
                                 "TRUNCATED. %s"
                                 % (mv.main, mv.sub,
                                    "Phase 0 re-seeds it with %s, so a slot-32 "
                                    "post-hook cannot raise it; use the slot-29 seam."
                                    % seeds if seeds else
                                    "A slot-32 post-hook owns the budget, so it can "
                                    "be raised (EM_OVERLAY_ABI §13).")))
    return out


# --------------------------------------------------------------------------- #
def format_report(issues: List[Issue]) -> str:
    if not issues:
        return "OK — no issues."
    lines = [str(i) for i in issues]
    n_err = sum(1 for i in issues if i.level == ERROR)
    lines.append("")
    lines.append("%d issue(s): %d error, %d warning"
                 % (len(issues), n_err, len(issues) - n_err))
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(
        prog="python -m mhfu_monster_editor.validate",
        description="Validate a port.toml against the built PAC and the action census.")
    ap.add_argument("manifest", nargs="+")
    ap.add_argument("--pac", help="the BUILT port PAC (enables the clip + bone checks)")
    ap.add_argument("--intel", help="species/emNN.json (issue #4); default: look it up")
    ap.add_argument("--intel-root", default="species")
    ap.add_argument("--strict", action="store_true",
                    help="treat warnings as errors too")
    a = ap.parse_args(argv)
    if a.pac and len(a.manifest) > 1:
        ap.error("--pac names ONE built PAC; give one manifest at a time, or the clip "
                 "fingerprints get checked against the wrong monster")

    worst = 0
    for path in a.manifest:
        m = load_manifest(path)
        intel = (JsonActionIntel.from_path(a.intel) if a.intel
                 else find_intel(m.host_species, a.intel_root))
        issues = validate(m, pac=a.pac, intel=intel)
        print("== %s (%s -> host species %d)" % (path, m.name, m.host_species))
        print(format_report(issues))
        print()
        if any(i.level == ERROR for i in issues) or (a.strict and issues):
            worst = 1
    return worst


if __name__ == "__main__":                                   # pragma: no cover
    sys.exit(main())
