"""Read `species/emNN.json` — what the HOST action expects, per `(main, sub)`.

`tools/em_intel.py` writes the file; this reads it. It is the editor's answer to the
only question that decides whether a ported move works at all: *if I bind this move to
host pair (main, sub), what does the engine already do there?*

Three kinds of fact live in one record and they must never be confused
----------------------------------------------------------------------
**static** — read out of the species overlay's MIPS. Which `(main,sub)` exist, which
handler runs, which executor `a1` clips it plays, what ENDS it (the clip, a cursor
frame, or the `+0x414` budget), and which effect literals it spawns. A property of the
ISO: identical for every run, every player, every session.

**measured** — `tools/em_state_census.py` watching a real game. How often the engine
entered the pair *on its own*, how long it held, whether the monster moved. A sample,
not a law.

**absent** — no measurement was supplied. 🔴 This is the normal case, not an edge
case: collecting a census needs a cold boot with the observe-only probe deployed. An
absent measurement is **UNKNOWN, never zero** — :attr:`PairIntel.entered` is ``None``,
and every consumer must keep that apart from a measured ``0``, which is the only
finding that justifies refusing a bind:

    entered is None   the census said nothing (or there was no census)
    entered == 0      the census LOOKED and the engine never went there — 411 of 411
                      forced moves into such a pair survived exactly one tick
    entered > 0       the engine uses it

Usage::

    from mhfu_monster_editor.intel import find_intel
    intel = find_intel(75)                       # species/em75.json, or None
    p = intel.pair(2, 8)
    intel.bindable(4, 15).ok                     # False once a census exists

    python -m mhfu_monster_editor.intel species/em75.json --pair 2,8

→ `tools/em_intel.py`, `docs/EM_OVERLAY_ABI.md`, the `monster-ai` skill
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Protocol, Tuple, \
    runtime_checkable

SCHEMA = "mhfu.species_intel/1"

#: provenance markers, as written by the generator.
STATIC = "static"
MEASURED = "measured"
ABSENT = "absent"

#: how long, in 2 Hz ticks, a pair must hold before it is worth scripting.
MIN_DWELL_TICKS = 3.0

# bind verdicts
BIND_OK = "OK"
BIND_NO_HANDLER = "NO_HANDLER"
BIND_NEVER_ENTERED = "NEVER_ENTERED"
BIND_SHORT_DWELL = "SHORT_DWELL"
BIND_UNMEASURED = "UNMEASURED"
BIND_UNKNOWN_PAIR = "UNKNOWN_PAIR"


def _addr(v: Any) -> Optional[int]:
    if v is None or v == "":
        return None
    return int(v, 0) if isinstance(v, str) else int(v)


# --------------------------------------------------------------------------- #
@dataclass
class EffectRecipe:
    """One `spawn_effect` literal, recovered from the handler's own code.

    ⚠️ `bone` is the HOST SPECIES' bone. Effect 60 "at bone 33" is only where the
    lightning leaves the mouth if bone 33 is *that* species' mouth; a port ships its
    own rig and the same number lands somewhere else. `frame` is ``None`` unless the
    site went through the frame-gated primitive.
    """
    id: int
    bone: Optional[int] = None
    frame: Optional[int] = None
    site: Optional[int] = None
    via: str = ""
    fn: Optional[int] = None

    def __str__(self) -> str:                     # em_effects' own notation
        s = "%d@b%s" % (self.id, "?" if self.bone is None else self.bone)
        return s if self.frame is None else s + "@f%d" % self.frame

    @classmethod
    def from_dict(cls, d: dict) -> "EffectRecipe":
        return cls(id=int(d["id"]),
                   bone=None if d.get("bone") is None else int(d["bone"]),
                   frame=None if d.get("frame") is None else int(d["frame"]),
                   site=_addr(d.get("site")), via=str(d.get("via", "")),
                   fn=_addr(d.get("fn")))


@dataclass
class BudgetIntel:
    """The `entity+0x414` frame budget — a MINORITY gate, and a settable one."""
    gated: bool = False
    #: literals the handler's phase-0 block re-seeds it with, if any
    phase0_seeds: List[int] = field(default_factory=list)
    #: True when a slot-32 POST-hook can own the budget (phase 0 only consumes it)
    post_hook_owns: Optional[bool] = None

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "BudgetIntel":
        d = d or {}
        return cls(gated=bool(d.get("gated")),
                   phase0_seeds=[int(x) for x in d.get("phase0_seeds", [])],
                   post_hook_owns=d.get("post_hook_owns"))


@dataclass
class PairIntel:
    """What is known about one `(main, sub)` behaviour pair, from all four analysers.

    The first six fields are the interface `validate.py` was written against; the rest
    are what issue #4's join adds. :attr:`provenance` maps a field name to ``static`` /
    ``measured`` / ``absent`` so a UI can render an inferred number differently from a
    measured one without hard-coding which is which.
    """
    main: int
    sub: int
    #: how many times the ENGINE entered the pair on its own.
    #: **None = not measured. 0 = measured ZERO — never entered.**
    entered: Optional[int] = None
    #: mean dwell in 2 Hz ticks while the engine held it (measured; 0.0 if absent)
    dwell_ticks: float = 0.0
    #: executor a1 values — measured if a census saw any, else the static ones
    a1: List[int] = field(default_factory=list)
    note: str = ""

    # --- static -------------------------------------------------------------
    #: the overlay function this pair dispatches to. None = the case runs inline
    handler: Optional[int] = None
    a1_static: List[int] = field(default_factory=list)
    #: a handler that also reaches the executor with a1 from a register/table
    a1_computed: bool = False
    #: ``clip`` | ``clip+cursor`` | ``budget`` | ``cursor`` | ``unknown``
    ends_on: str = ""
    #: cursor thresholds the handler tests, in clip frames. None = loaded from data
    event_frames: List[Optional[float]] = field(default_factory=list)
    windows: int = 0
    budget: BudgetIntel = field(default_factory=BudgetIntel)
    effects: List[EffectRecipe] = field(default_factory=list)

    #: why this pair's static record is odd — e.g. its dispatcher case runs inline.
    #: Kept apart from :attr:`note`, which prefers what the CENSUS said, so the two
    #: never get spliced into one sentence that is half static and half measured.
    static_note: str = ""

    # --- measured -----------------------------------------------------------
    a1_measured: List[int] = field(default_factory=list)
    #: mean units the monster travelled per tick while in this pair. None = the pair
    #: never occurred on two consecutive ticks co-located, so movement is UNMEASURED
    move_per_tick: Optional[float] = None
    move_samples: int = 0

    provenance: Dict[str, str] = field(default_factory=dict)

    # --- derived ------------------------------------------------------------
    @property
    def measured(self) -> bool:
        """True when a census had something to say about this pair."""
        return self.entered is not None

    @property
    def never_entered(self) -> bool:
        """Measured zero entries — the rejectable case. Absent evidence is False."""
        return self.entered == 0

    @property
    def a1_provenance(self) -> str:
        """Where :attr:`a1` came from.

        ⚠️ Not ``provenance["a1"]`` — that describes the JSON key of the same name,
        which is always the static list. :attr:`a1` is the *effective* one, and the
        census wins over the dispatcher when it has anything to say.
        """
        if self.a1_measured:
            return MEASURED
        return STATIC if self.a1_static else ABSENT

    @property
    def ends_on_clip(self) -> bool:
        """The action lasts as long as the clip does, so a port may choose it."""
        return self.ends_on in ("clip", "clip+cursor")

    @property
    def fixed_event_frames(self) -> List[float]:
        """Frames the handler fires things at — a ported clip must hit these."""
        return [f for f in self.event_frames if f is not None]

    def __str__(self) -> str:
        bits = ["(%d,%d)" % (self.main, self.sub)]
        bits.append("0x%08X" % self.handler if self.handler else "inline")
        if self.a1:
            bits.append("a1=" + ",".join(str(x) for x in self.a1))
        if self.ends_on:
            bits.append("ends:" + self.ends_on)
        if self.effects:
            bits.append("fx " + " ".join(str(e) for e in self.effects))
        bits.append("entered=%s" % ("?" if self.entered is None else self.entered))
        return "  ".join(bits)

    @classmethod
    def from_dict(cls, d: dict) -> "PairIntel":
        """Read one pair from either the joined shape or the flat provisional one."""
        meas = d.get("measured") or {}
        a1_static = [int(x) for x in d.get("a1", [])] if "measured" in d else []
        a1_meas = [int(x) for x in meas.get("a1", [])]
        if "measured" in d:
            entered = None if not meas else int(meas.get("entered", 0))
            dwell = float(meas.get("dwell_ticks", 0.0))
            note = str(meas.get("note", "") or d.get("note", ""))
        else:
            # the provisional shape validate.py documented before #4 landed:
            # {"main":2,"sub":8,"entered":46,"dwell_ticks":23.7,"a1":[15]}
            entered = None if d.get("entered") is None else int(d["entered"])
            dwell = float(d.get("dwell_ticks", 0.0))
            a1_meas = [int(x) for x in d.get("a1", [])]
            note = str(d.get("note", ""))
        return cls(
            main=int(d["main"]), sub=int(d["sub"]),
            entered=entered, dwell_ticks=dwell,
            a1=a1_meas or a1_static, note=note,
            handler=_addr(d.get("handler")),
            a1_static=a1_static, a1_computed=bool(d.get("a1_computed")),
            ends_on=str(d.get("ends_on", "")),
            event_frames=[None if f is None else float(f)
                          for f in d.get("event_frames", [])],
            windows=int(d.get("windows", 0)),
            budget=BudgetIntel.from_dict(d.get("budget")),
            effects=[EffectRecipe.from_dict(e) for e in d.get("effects", [])],
            a1_measured=a1_meas, static_note=str(d.get("note", "")),
            move_per_tick=meas.get("move_per_tick"),
            move_samples=int(meas.get("move_samples", 0)),
            provenance=dict(d.get("provenance", {})),
        )


@dataclass
class Bind:
    """May a move be bound to this pair, and if not, what would make it legal?"""
    ok: bool
    code: str
    reason: str
    #: True when the caller passed an explicit override to get `ok`
    overridden: bool = False
    #: True when `ok` rests on absent evidence rather than on a measurement
    unverified: bool = False

    def __bool__(self) -> bool:
        return self.ok


@runtime_checkable
class ActionIntel(Protocol):
    """The narrow seam `validate.py` consumes. Deliberately two members.

    ``host_species`` is the MHFU species whose overlay this describes — the intel is a
    property of the HOST, not of the port, because `(main,sub)` is dispatched by the
    host's own AI overlay. Two different ports riding the same host share it.

    ``pair(main, sub)`` returns :class:`PairIntel`, or ``None`` when the file says
    nothing at all about that pair.
    """
    host_species: int

    def pair(self, main: int, sub: int) -> Optional[PairIntel]: ...


# --------------------------------------------------------------------------- #
class SpeciesIntel:
    """`species/emNN.json`, joined from all four offline analysers.

    Also accepts the flat provisional shape `validate.py` documented before #4, so a
    hand-written census file keeps working; :attr:`has_static` is then False and the
    static checks quietly do not run.
    """

    def __init__(self, host_species: int, pairs: Iterable[PairIntel],
                 source: str = "", *,
                 has_census: Optional[bool] = None,
                 has_static: bool = False,
                 census_reason: str = "",
                 census_transitions: int = 0,
                 enumerated_mains: Optional[Iterable[int]] = None,
                 main_states: Optional[List[dict]] = None,
                 unattributed_effects: Optional[List[dict]] = None,
                 overlay: Optional[dict] = None) -> None:
        self.host_species = int(host_species)
        self._by_pair: Dict[Tuple[int, int], PairIntel] = {
            (p.main, p.sub): p for p in pairs}
        self.source = source
        self.has_static = bool(has_static)
        self.main_states = list(main_states or [])
        self.unattributed_effects = list(unattributed_effects or [])
        self.overlay = dict(overlay or {})
        self.census_reason = census_reason
        self.census_transitions = int(census_transitions)
        if has_census is None:
            has_census = any(p.measured for p in self._by_pair.values())
        self.has_census = bool(has_census)
        if enumerated_mains is None:
            enumerated_mains = {m["main"] for m in self.main_states
                                if m.get("enumerated")}
        self.enumerated_mains = set(enumerated_mains)

    # -- the ActionIntel seam ------------------------------------------------
    def pair(self, main: int, sub: int) -> Optional[PairIntel]:
        return self._by_pair.get((int(main), int(sub)))

    # -- convenience ---------------------------------------------------------
    def __len__(self) -> int:
        return len(self._by_pair)

    def __iter__(self) -> Iterator[PairIntel]:
        for k in sorted(self._by_pair):
            yield self._by_pair[k]

    def pairs_with_effects(self) -> List[PairIntel]:
        return [p for p in self if p.effects]

    def budget_gated(self) -> List[PairIntel]:
        return [p for p in self if p.budget.gated]

    def bindable(self, main: int, sub: int, *, override: bool = False) -> Bind:
        """May a move be bound to `(main, sub)`?

        Refuses exactly one thing outright: a pair the census **measured** as never
        entered. That is not a style preference — its handler asks for a condition
        nothing created and returns at once, so a forced move survives one tick, the
        clip restarts from frame 0 twice a second and nothing ever plays through.
        `override=True` lets a caller do it anyway, and the verdict says so.

        Everything else is allowed, but a bind resting on absent evidence comes back
        with ``unverified=True`` so a UI can badge it rather than pretend it checked.
        """
        p = self.pair(int(main), int(sub))
        if p is None:
            if self.has_static and int(main) in self.enumerated_mains:
                n = next((m.get("sub_states") for m in self.main_states
                          if m["main"] == int(main)), None)
                return Bind(False, BIND_NO_HANDLER,
                            "main %d dispatches %s sub_state(s) and %d is not one of "
                            "them — act_set would land on nothing."
                            % (int(main), n, int(sub)))
            return Bind(True, BIND_UNKNOWN_PAIR,
                        "nothing is known about (%d,%d): it is neither in the "
                        "overlay's jump tables nor in the census."
                        % (int(main), int(sub)), unverified=True)
        if p.handler is None:
            return Bind(True, BIND_NO_HANDLER,
                        "the dispatcher's case for (%d,%d) runs inline and calls no "
                        "handler, so nothing offline can say what it does."
                        % (p.main, p.sub), unverified=True)
        if p.never_entered:
            reason = ("the census says the engine enters (%d,%d) ZERO times. Forced, "
                      "it survives exactly one tick — 411 of 411 did."
                      % (p.main, p.sub))
            if override:
                return Bind(True, BIND_NEVER_ENTERED, reason, overridden=True)
            return Bind(False, BIND_NEVER_ENTERED, reason)
        if not p.measured:
            return Bind(True, BIND_UNMEASURED,
                        "no census covers (%d,%d), so whether the engine ever enters "
                        "it is UNKNOWN — not zero.%s"
                        % (p.main, p.sub,
                           " " + self.census_reason if self.census_reason else ""),
                        unverified=True)
        if p.dwell_ticks and p.dwell_ticks < MIN_DWELL_TICKS:
            return Bind(True, BIND_SHORT_DWELL,
                        "(%d,%d) holds for only %.1f ticks (%.1f s at 2 Hz) even when "
                        "the ENGINE picks it." % (p.main, p.sub, p.dwell_ticks,
                                                  p.dwell_ticks / 2.0),
                        unverified=True)
        return Bind(True, BIND_OK,
                    "the engine entered (%d,%d) %d time(s) and held it %.1f tick(s)."
                    % (p.main, p.sub, p.entered, p.dwell_ticks))

    # -- loading -------------------------------------------------------------
    @classmethod
    def from_dict(cls, d: dict, source: str = "") -> "SpeciesIntel":
        pairs = [PairIntel.from_dict(p) for p in d.get("pairs", [])]
        census = d.get("census")
        has_census = None if census is None else bool(census.get("present"))
        static = d.get("static") or {}
        has_static = bool(static.get("present")) or "main_states" in d
        return cls(int(d.get("host_species", -1)), pairs, source,
                   has_census=has_census, has_static=has_static,
                   census_reason=("" if not census else
                                  str(census.get("reason", ""))),
                   census_transitions=0 if not census
                   else int(census.get("transitions", 0)),
                   main_states=d.get("main_states"),
                   unattributed_effects=d.get("unattributed_effects"),
                   overlay=d.get("overlay"))

    @classmethod
    def from_path(cls, path: os.PathLike | str) -> "SpeciesIntel":
        p = Path(path)
        return cls.from_dict(json.loads(p.read_text(encoding="utf-8")), str(p))

    def describe(self) -> str:
        """One paragraph a UI (or a session) can print instead of the whole file."""
        handled = [p for p in self if p.handler is not None]
        lines = ["%s: %d pair(s), %d with a handler%s"
                 % (self.overlay.get("name", "em%02d" % self.host_species),
                    len(self), len(handled),
                    "" if self.has_static else "  (no static intel in this file)")]
        if self.has_census:
            ok = [p for p in self if p.entered]
            lines.append("  census PRESENT: %d pair(s) the engine actually enters"
                         % len(ok))
        else:
            lines.append("  census ABSENT — every `entered` is UNKNOWN, not zero."
                         + (" " + self.census_reason if self.census_reason else ""))
        lines.append("  %d pair(s) carry effect recipes; %d budget-gated"
                     % (len(self.pairs_with_effects()), len(self.budget_gated())))
        return "\n".join(lines)


def find_intel(host_species: int,
               root: os.PathLike | str = "species") -> Optional[SpeciesIntel]:
    """`species/emNN.json` for a host species, or ``None`` if it has not been built.

    Regenerate with ``python tools/em_intel.py --all`` — the files are derived from
    the game's own overlays and are never committed (`docs/ASSETS.md` §E).
    """
    p = Path(root) / ("em%02d.json" % int(host_species))
    return SpeciesIntel.from_path(p) if p.exists() else None


# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:                # pragma: no cover
    import argparse
    ap = argparse.ArgumentParser(
        prog="python -m mhfu_monster_editor.intel",
        description="Read species/emNN.json — the host action intel.")
    ap.add_argument("path", help="species/emNN.json, or a species number")
    ap.add_argument("--pair", help="M,S — everything known about one pair")
    ap.add_argument("--effects", action="store_true",
                    help="the pairs that carry effect recipes")
    ap.add_argument("--bindable", help="M,S — may a move be bound here?")
    ap.add_argument("--override", action="store_true")
    a = ap.parse_args(argv)

    if a.path.isdigit():
        si = find_intel(int(a.path))
        if si is None:
            print("no species/em%02d.json — run tools/em_intel.py" % int(a.path))
            return 1
    else:
        si = SpeciesIntel.from_path(a.path)

    if a.pair:
        m, s = (int(x) for x in a.pair.split(","))
        p = si.pair(m, s)
        if p is None:
            print("(%d,%d) is not in %s" % (m, s, si.source))
            return 1
        print(p)
        print("  ends on        : %s" % (p.ends_on or "?"))
        print("  event frames   : %s" % (p.event_frames or "-"))
        print("  budget         : %s" % p.budget)
        print("  effects        : %s" % (" ".join(str(e) for e in p.effects) or "-"))
        print("  measured       : %s" % ("entered %s, dwell %.1f ticks, %s u/tick"
                                         % (p.entered, p.dwell_ticks, p.move_per_tick)
                                         if p.measured else "ABSENT"))
        print("  provenance     : %s" % p.provenance)
        return 0
    if a.bindable:
        m, s = (int(x) for x in a.bindable.split(","))
        b = si.bindable(m, s, override=a.override)
        print("%-6s %-14s %s" % ("BIND" if b.ok else "REFUSE", b.code, b.reason))
        return 0 if b.ok else 1
    if a.effects:
        for p in si.pairs_with_effects():
            print("(%2d,%3d)  %s" % (p.main, p.sub,
                                     " ".join(str(e) for e in p.effects)))
        n = sum(len(u.get("sites", [])) for u in si.unattributed_effects)
        print("\n%d further spawn site(s) in %d function(s) no pair handler reaches "
              "— the species' vocabulary, but which action fires them is not "
              "decidable offline." % (n, len(si.unattributed_effects)))
        return 0
    print(si.describe())
    return 0


if __name__ == "__main__":                                        # pragma: no cover
    import sys
    sys.exit(main())
