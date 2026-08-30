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
   animation ever finished. `tools/em_state_census.py` measures it; issue #4 turns that
   into machine-readable `species/emNN.json`. Until then this check WARNS — see
   :class:`ActionIntel`.
3. **A hurtbox bone out of range for the shipped skeleton.** The volume table is
   bone-indexed and a ported monster ships its own rig, so a host-derived index is not
   just wrong, it reads off the end of the joint array.

Usage::

    python -m mhfu_monster_editor.validate ports/zinogre.toml --pac tmp/zinogre_v10.bin

Exit status is 1 if anything came back at level ``error``.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Protocol, runtime_checkable

from .manifest import PortManifest, load as load_manifest

ERROR = "error"
WARNING = "warning"

#: how long, in ticks, a pair must hold before it is worth scripting. The census
#: prints dwell in ticks at 2 Hz; anything that bounces out in one or two is the
#: never-entered failure wearing a different hat.
MIN_DWELL_TICKS = 3.0


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
# the seam for issue #4 — action intel
# --------------------------------------------------------------------------- #
@dataclass
class PairIntel:
    """What the census knows about one `(main, sub)` behaviour pair."""
    main: int
    sub: int
    #: how many times the ENGINE entered the pair on its own. 0 = NEVER ENTERED.
    entered: int = 0
    #: mean dwell in 2 Hz ticks while the engine held it
    dwell_ticks: float = 0.0
    #: executor a1 values observed under it
    a1: List[int] = field(default_factory=list)
    note: str = ""


@runtime_checkable
class ActionIntel(Protocol):
    """The whole interface issue #4 has to satisfy. Deliberately two members.

    ``host_species`` is the MHFU species whose overlay the census was taken from — the
    intel is a property of the HOST, not of the port, because `(main,sub)` is dispatched
    by the host's own AI overlay. Two different ports riding the same host share it.

    ``pair(main, sub)`` returns :class:`PairIntel`, or ``None`` when the census has
    nothing to say about that pair (which is NOT the same as "never entered" — an
    absent pair means an unobserved one).
    """
    host_species: int

    def pair(self, main: int, sub: int) -> Optional[PairIntel]: ...


class JsonActionIntel:
    """Provisional reader for `species/emNN.json`, the file issue #4 will produce.

    ⚠️ The on-disk shape is #4's to define; this accepts the obvious one and is the
    single class to replace when it lands::

        {"host_species": 75,
         "pairs": [{"main": 2, "sub": 8, "entered": 46, "dwell_ticks": 23.7,
                    "a1": [15]}, ...]}

    A pair absent from ``pairs`` is unobserved. A pair present with ``entered`` 0 is the
    census saying it looked and the engine never went there — the rejectable case.
    """

    def __init__(self, host_species: int, pairs: Iterable[PairIntel],
                 source: str = "") -> None:
        self.host_species = int(host_species)
        self._by_pair = {(p.main, p.sub): p for p in pairs}
        self.source = source

    def pair(self, main: int, sub: int) -> Optional[PairIntel]:
        return self._by_pair.get((int(main), int(sub)))

    def __len__(self) -> int:
        return len(self._by_pair)

    @classmethod
    def from_dict(cls, d: dict, source: str = "") -> "JsonActionIntel":
        pairs = [PairIntel(main=int(p["main"]), sub=int(p["sub"]),
                           entered=int(p.get("entered", 0)),
                           dwell_ticks=float(p.get("dwell_ticks", 0.0)),
                           a1=[int(x) for x in p.get("a1", [])],
                           note=str(p.get("note", "")))
                 for p in d.get("pairs", [])]
        return cls(int(d.get("host_species", -1)), pairs, source)

    @classmethod
    def from_path(cls, path: os.PathLike | str) -> "JsonActionIntel":
        p = Path(path)
        return cls.from_dict(json.loads(p.read_text(encoding="utf-8")), str(p))


def find_intel(host_species: int,
               root: os.PathLike | str = "species") -> Optional[JsonActionIntel]:
    """`species/emNN.json` for a host species, or ``None`` when #4 has not run yet."""
    p = Path(root) / ("em%02d.json" % int(host_species))
    return JsonActionIntel.from_path(p) if p.exists() else None


# --------------------------------------------------------------------------- #
# evidence from the built PAC
# --------------------------------------------------------------------------- #
def _tools_on_path() -> None:
    tools = str(Path(__file__).resolve().parent.parent / "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)


def clip_table(pac: bytes) -> Dict[int, tuple]:
    """``{slot: (last_keyframe, loop)}`` for every populated slot of a built PAC.

    Same derivation as `tools/port_clip_probe.py::pac_clip_table`; kept here so the
    validator does not drag in that module's websockets/debugger imports.
    """
    _tools_on_path()
    from mhfu_model import anim_ingame as ig
    from mhfu_model.pac import MonsterPac

    sub = MonsterPac.from_bytes(pac).find("anim")
    if sub is None:
        raise ValueError("no animation sub-resource in this PAC")
    a = ig.parse_ingame(sub.data)
    out: Dict[int, tuple] = {}
    for slot in sorted({s for st in a.streams for s in st.clips}):
        blk = next(st.clips[slot] for st in a.streams if slot in st.clips)
        end = max((kf.frame for bn in blk.bones for ch in bn.channels
                   for kf in ch.keyframes), default=0)
        out[slot] = (end, blk.loop)
    return out


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
        idle = table.get(1)
        for name in sorted(m.clips):
            c = m.clips[name]
            w = "clips.%s" % name
            if c.slot not in table:
                out.append(Issue(ERROR, "CLIP_SLOT_MISSING", w,
                                 "slot %d is not populated in this build. The porter "
                                 "files a source clip into the host slot of the SAME "
                                 "index, so a source clip the host pack has no slot "
                                 "for is DROPPED — scripting it reaches nothing."
                                 % c.slot))
                continue
            end, loop = table[c.slot]
            if c.frames is not None and c.frames != end:
                out.append(Issue(ERROR, "CLIP_FRAMES_MISMATCH", w,
                                 "declared frames = %d, the build's slot %d ends at "
                                 "%d. Clips are stored unpadded at their authored "
                                 "length, so this is not rounding — the slot holds a "
                                 "different clip." % (c.frames, c.slot, end)))
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

    if m.hurtboxes or m.effects:
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
            for i, e in enumerate(m.effects):
                if not 0 <= e.bone < nb:
                    out.append(Issue(ERROR, "EFFECT_BONE_RANGE", "effect[%d]" % i,
                                     "bone %d is outside the shipped skeleton's %d "
                                     "joints; spawn_effect reads that bone's live "
                                     "world position." % (e.bone, nb)))
    return out


def _check_intel(m: PortManifest, intel: Optional[ActionIntel]) -> List[Issue]:
    out: List[Issue] = []
    if not m.moves:
        return out
    if intel is None:
        out.append(Issue(WARNING, "INTEL_ABSENT", "moves",
                         "no action intel for host species %d, so %d (main,sub) pair(s)"
                         " went unchecked. A pair the engine never enters bounces out "
                         "in ONE tick — 411 of 411 forced moves did. Measure with "
                         "tools/em_state_census.py; issue #4 makes it a file."
                         % (m.host_species, len(m.moves))))
        return out

    got = getattr(intel, "host_species", None)
    if got is not None and int(got) != m.host_species:
        out.append(Issue(ERROR, "INTEL_WRONG_SPECIES", "moves",
                         "the intel is for host species %d, this port rides %d. "
                         "(main,sub) is dispatched by the HOST's overlay, so intel "
                         "from another species says nothing." % (got, m.host_species)))
        return out

    for name in sorted(m.moves):
        mv = m.moves[name]
        w = "moves.%s" % name
        p = intel.pair(mv.main, mv.sub)
        if p is None:
            out.append(Issue(WARNING, "MOVE_PAIR_UNOBSERVED", w,
                             "(%d,%d) is not in the census. Absent is not the same as "
                             "never entered — the sample may simply not cover it."
                             % (mv.main, mv.sub)))
        elif p.entered <= 0:
            out.append(Issue(ERROR, "MOVE_PAIR_NEVER_ENTERED", w,
                             "the census says the engine enters (%d,%d) ZERO times. "
                             "Its handler asks for a condition nothing has created and "
                             "returns at once: forced, it survives exactly one tick and "
                             "the clip restarts from frame 0 forever."
                             % (mv.main, mv.sub)))
        elif p.dwell_ticks and p.dwell_ticks < MIN_DWELL_TICKS:
            out.append(Issue(WARNING, "MOVE_PAIR_SHORT_DWELL", w,
                             "(%d,%d) holds for only %.1f ticks (%.1f s at 2 Hz) even "
                             "when the ENGINE picks it. Forced from the wrong range it "
                             "will bounce out." % (mv.main, mv.sub, p.dwell_ticks,
                                                   p.dwell_ticks / 2.0)))
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
