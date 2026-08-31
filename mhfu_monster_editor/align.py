"""Put the HOST action's expectations and YOUR clip on the same frame axis.

The join that makes this a porting tool rather than a model viewer, and it needs no new
RE — only the JSON `tools/em_intel.py` already writes. A move is an alignment between a
host behaviour pair and a clip: `(main, sub)` goes to `act_set` and owns the hitbox, the
damage and the effects, while the clip only paints the animation channel. So the
question that decides whether a ported move *works* is not "does it play" but **does it
put its impact where the handler looks**.

    align(manifest, "charge", intel, clip_frames=382)

Four facts land on the timeline
-------------------------------
**Cursor gates.** The handler is a small phase machine gated on the clip's own cursor:
`0x08864408(block, slot, F)` is "the cursor has reached F" and `0x08864348` its windowed
form — the shape of a hitbox-active test, 280 call sites in em75. 🔴 **113 of em75's 231
actions test fixed frame numbers.** A ported clip may be any length, but a gate past its
last keyframe is a branch that never runs, and a gate the animation's contact does not
sit on fires the hit at the wrong moment.

**What ends it.** 182 of 231 end when the clip does, so their length is free. 27 end on
the `entity+0x414` frame budget instead — a countdown the clip has no say in, and one
that IS settable from a mod (15 via a slot-32 post-hook, 12 via the slot-29 one-shot).
Which of the two a move sits on changes what "make the clip longer" means.

**Effect spawns.** `id @ bone @ frame`, recovered as literal immediates from the
handler. ⚠️ **The bone index is the EMITTING species' own.** A port ships its own rig,
so the same number lands somewhere else entirely — which is exactly why the Zinogre
threw snowballs out of Tigrex mouth bones. This reports where that number lands on
*your* skeleton, and refuses to pretend it knows where it ought to.

**Measured dwell**, and the one loud finding: a pair the census measured as never
entered. 411 of 411 forced moves into such a pair survived exactly one tick.

Stdlib only, like `manifest` and `intel` — it joins two pure-data modules and adds no
dependency of its own. The rig facts it needs come in as :class:`PortRig`, so the caller
(the UI, a test) supplies them from wherever it has them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set

from .intel import PairIntel, SpeciesIntel
from .manifest import PortManifest

# marker kinds, in the order a timeline should stack them
GATE = "gate"          # `cursor >= F` — the handler tests this frame
WINDOW = "window"      # the windowed form: a hitbox-active-shaped test
EFFECT = "effect"      # a spawn_effect literal, host-side
OURS = "ours"          # an effect this manifest scripts
IMPACT = "impact"      # where the CLIP's own contact is, per the manifest

ERROR = "error"
WARN = "warn"
INFO = "info"

#: how far the clip's impact may sit from the frame the handler tests before it is
#: worth saying so. The engine compares `threshold <= cursor` once per game frame, and
#: the cursor advances by `speed` (2.0 and 2.4 both observed), so anything under a
#: couple of frames is inside one dispatch and not worth a finding.
IMPACT_TOLERANCE = 2.0


@dataclass
class PortRig:
    """What the alignment needs to know about the skeleton the PORT actually ships.

    ``driven`` is the set of joints some clip animates; a joint outside it stays at its
    bind position for the whole quest, so an effect anchored there does not follow the
    animal. ``vertices`` is only used to say how much geometry hangs off the joint,
    which is what tells a mouth bone from an unused one at a glance.
    """
    n_bones: int
    driven: Optional[Set[int]] = None
    vertices: Dict[int, int] = field(default_factory=dict)

    def describe(self, bone: int) -> str:
        if not 0 <= bone < self.n_bones:
            return "joint %d does not exist on this rig (%d joints)" % (bone,
                                                                        self.n_bones)
        bits = ["joint %d" % bone]
        n = self.vertices.get(bone, 0)
        bits.append("%d vertices" % n if n else "no geometry")
        if self.driven is not None:
            bits.append("driven" if bone in self.driven else "NOT ANIMATED")
        return ", ".join(bits)


@dataclass
class Marker:
    """One frame the timeline should draw, in the CLIP's own frame space."""
    frame: float
    kind: str
    label: str
    detail: str = ""
    #: the clip is shorter than this frame, so the cursor never reaches it
    unreachable: bool = False


@dataclass
class Finding:
    level: str
    code: str
    message: str

    def __str__(self) -> str:
        return "%-5s %-22s %s" % (self.level.upper(), self.code, self.message)


@dataclass
class Alignment:
    """One move, checked against what its host pair expects."""
    move: str
    main: int
    sub: int
    clip: Optional[str] = None
    slot: Optional[int] = None
    #: the clip's last keyframe, from the BUILD when one was given
    frames: Optional[int] = None
    #: `clips.<name>.impact_frame` — where the animation's contact is
    impact: Optional[float] = None
    markers: List[Marker] = field(default_factory=list)
    findings: List[Finding] = field(default_factory=list)
    headline: str = ""
    pair: Optional[PairIntel] = None

    # -- convenience ------------------------------------------------------ #
    @property
    def errors(self) -> List[Finding]:
        return [f for f in self.findings if f.level == ERROR]

    @property
    def warnings(self) -> List[Finding]:
        return [f for f in self.findings if f.level == WARN]

    @property
    def gates(self) -> List[float]:
        """Every frame the handler names, of either kind."""
        return sorted({m.frame for m in self.markers if m.kind in (GATE, WINDOW)})

    def marker_map(self) -> Dict[float, str]:
        """``{frame: label}`` — the shape the timeline strip draws."""
        out: Dict[float, str] = {}
        for m in self.markers:
            out[m.frame] = m.label if m.frame not in out else out[m.frame] + " " + m.label
        return out

    def report(self) -> str:
        lines = ["%s -> (%d,%d)%s" % (self.move, self.main, self.sub,
                                      "  clip %s" % self.clip if self.clip else ""),
                 "  " + self.headline]
        for f in self.findings:
            lines.append("  " + str(f))
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
def align(m: PortManifest, move: str, intel: Optional[SpeciesIntel] = None, *,
          clip_frames: Optional[int] = None,
          rig: Optional[PortRig] = None) -> Alignment:
    """What does the host action expect, and does this MOVE's clip deliver it?

    ``clip_frames`` is the clip's last keyframe as the BUILD has it — preferred over
    the manifest's own `frames`, because a rebuild moves clips and the manifest may be
    describing a slot that has since changed. ``rig`` lets the effect check say where a
    host bone number lands; without it that check reports the number and stops.
    """
    mv = m.moves.get(move)
    if mv is None:
        raise KeyError("no move named %r (have: %s)"
                       % (move, ", ".join(sorted(m.moves)) or "none"))
    clip = m.clips.get(mv.clip) if mv.clip else None
    return align_pair(m, mv.main, mv.sub, intel, move=move, clip=mv.clip,
                      slot=clip.slot if clip else mv.anim,
                      clip_frames=(clip_frames if clip_frames is not None
                                   else (clip.frames if clip else None)),
                      impact=clip.impact_frame if clip else None,
                      allow_unentered=mv.allow_unentered, rig=rig)


def align_pair(m: PortManifest, main: int, sub: int,
               intel: Optional[SpeciesIntel] = None, *,
               move: Optional[str] = None, clip: Optional[str] = None,
               slot: Optional[int] = None, clip_frames: Optional[int] = None,
               impact: Optional[float] = None, allow_unentered: bool = False,
               rig: Optional[PortRig] = None) -> Alignment:
    """The same answer for a pair that is NOT bound yet — the deciding view.

    Choosing a `(main,sub)` for a clip is the actual authoring act, and it happens
    before there is a move to inspect: `ports/zinogre.toml` ships with `[moves]`
    deliberately empty because aligning an unlabelled vocabulary is guessing. So the
    inspector answers for any pair against whatever clip is on screen, and a move is
    what you write down once the answer is good.
    """
    a = Alignment(move=move or "(%d,%d)" % (main, sub), main=main, sub=sub,
                  clip=clip, slot=slot, frames=clip_frames, impact=impact)

    if intel is None:
        a.headline = ("no species/em%02d.json, so nothing is known about (%d,%d). "
                      "Build it: python tools/em_intel.py --all"
                      % (m.host_species, main, sub))
        a.findings.append(Finding(WARN, "INTEL_ABSENT", a.headline))
        _impact_marker(a)
        return a

    a.pair = intel.pair(main, sub)
    _census(a, intel, allow_unentered)
    if a.pair is None:
        a.headline = (
            "(%d,%d) is in neither the overlay's jump tables nor the census, so there "
            "is nothing to align to." % (main, sub))
        _impact_marker(a)
        _our_effects(a, m, move, rig)
        return a

    _gates(a)
    _ends_on(a)
    _effects(a, rig)
    _our_effects(a, m, move, rig)
    _impact_marker(a)
    a.headline = _headline(a)
    return a


def align_all(m: PortManifest, intel: Optional[SpeciesIntel] = None, *,
              clip_frames: Optional[Dict[int, int]] = None,
              rig: Optional[PortRig] = None) -> List[Alignment]:
    """Every move in the manifest. ``clip_frames`` maps SLOT -> last keyframe."""
    out = []
    for name in sorted(m.moves):
        mv = m.moves[name]
        c = m.clips.get(mv.clip) if mv.clip else None
        end = None if (clip_frames is None or c is None) else clip_frames.get(c.slot)
        out.append(align(m, name, intel, clip_frames=end, rig=rig))
    return out


# --------------------------------------------------------------------------- #
# the four facts
# --------------------------------------------------------------------------- #
def _census(a: Alignment, intel: SpeciesIntel, allow_unentered: bool) -> None:
    """The one loud finding, and the measured numbers when there are any."""
    b = intel.bindable(a.main, a.sub, override=allow_unentered)
    p = a.pair
    if p is not None and p.never_entered:
        a.findings.append(Finding(
            WARN if allow_unentered else ERROR, "NEVER_ENTERED",
            "🔴 the census measured ZERO entries into (%d,%d). Its handler asks for a "
            "condition nothing created and returns at once: forced, the move survives "
            "exactly ONE tick and the clip restarts from frame 0 twice a second. 411 "
            "of 411 did.%s" % (a.main, a.sub,
                               "  (allow_unentered is set, so this is your call.)"
                               if allow_unentered else "")))
        return
    if p is not None and p.measured:
        a.findings.append(Finding(
            INFO, "DWELL",
            "the engine entered (%d,%d) %d time(s) on its own and held it %.1f tick(s)"
            " (%.1f s at 2 Hz)%s"
            % (a.main, a.sub, p.entered, p.dwell_ticks, p.dwell_ticks / 2.0,
               "" if p.move_per_tick is None
               else ", travelling %.0f u/tick" % p.move_per_tick)))
        if b.code == "SHORT_DWELL":
            a.findings.append(Finding(WARN, "SHORT_DWELL", b.reason))
        return
    if b.code == "NO_HANDLER" and not b.ok:
        a.findings.append(Finding(ERROR, "NO_HANDLER", b.reason))
    a.findings.append(Finding(
        WARN, "UNMEASURED",
        "no census covers (%d,%d), so whether the engine ever enters it is UNKNOWN — "
        "not zero. Everything below is STATIC: read out of the overlay, true of every "
        "run.%s" % (a.main, a.sub,
                    " " + intel.census_reason if intel.census_reason else "")))


def _gates(a: Alignment) -> None:
    """The frames the handler tests, and whether the clip is long enough to reach them."""
    p = a.pair
    for frames, kind, what in ((p.fixed_event_frames, GATE, "cursor >= %g"),
                               (p.fixed_window_frames, WINDOW, "window edge %g")):
        for f in sorted(set(frames)):
            past = a.frames is not None and f > a.frames
            a.markers.append(Marker(
                float(f), kind, "%g" % f, what % f, unreachable=past))
    if p.handler is None:
        a.findings.append(Finding(
            WARN, "INLINE_CASE",
            "the dispatcher's case for (%d,%d) runs inline and calls no handler, so "
            "nothing offline can say what frames it tests." % (p.main, p.sub)))
        return
    if not a.gates:
        a.findings.append(Finding(
            INFO, "NO_FIXED_FRAMES",
            "this handler tests no fixed frame numbers, so the clip's internal timing "
            "is yours. 113 of em75's 231 actions do test them; this is not one."))
        return
    unreachable = [mk for mk in a.markers if mk.unreachable]
    if unreachable:
        a.findings.append(Finding(
            ERROR, "GATE_BEYOND_CLIP",
            "the handler tests frame%s %s but the clip ends at %d — the cursor never "
            "gets there, so %s. A ported clip may be any length, but not shorter than "
            "the frames the handler names."
            % ("s" if len(unreachable) > 1 else "",
               ", ".join("%g" % mk.frame for mk in sorted(unreachable,
                                                          key=lambda k: k.frame)),
               a.frames,
               "those branches never run" if len(unreachable) > 1
               else "that branch never runs")))


def _ends_on(a: Alignment) -> None:
    """Whether the clip's LENGTH is free, which is a different question from timing."""
    p = a.pair
    if p.ends_on_clip:
        a.findings.append(Finding(
            INFO, "ENDS_ON_CLIP",
            "the action lasts as long as the clip does (the last phase waits on the "
            "clip-playing flag), so its total length is free — only the frames above "
            "are fixed."))
        return
    if p.budget.gated:
        who = ("a slot-32 POST-hook can own the budget (phase 0 only consumes it)"
               if p.budget.post_hook_owns else
               "the handler RE-SEEDS the budget in its phase-0 block%s, so a post-hook "
               "is overwritten one frame later" % (
                   " with %s" % ", ".join(str(s) for s in p.budget.phase0_seeds)
                   if p.budget.phase0_seeds else ""))
        a.findings.append(Finding(
            WARN, "ENDS_ON_BUDGET",
            "this action ends on the entity+0x414 frame BUDGET, not on the clip — one "
            "of only 27 of em75's 231 that do. A longer clip is cut off mid-play and a "
            "shorter one leaves the monster in the action after the animation stops. "
            "The budget is settable: %s." % who))
        return
    a.findings.append(Finding(
        WARN, "ENDS_ON_%s" % (p.ends_on or "UNKNOWN").upper().replace("+", "_"),
        "what ends this action reads as %r rather than clip-done, so the clip's length "
        "is not simply free here." % (p.ends_on or "unknown")))


def _effects(a: Alignment, rig: Optional[PortRig]) -> None:
    """The host's own spawn recipes — ids and bones that are NOT in the port's terms."""
    p = a.pair
    if not p.effects:
        return
    a.findings.append(Finding(
        INFO, "HOST_EFFECTS",
        "the host handler spawns %s. %s"
        % (", ".join(str(e) for e in p.effects),
           "%d of them name no frame — that site did not go through the frame-gated "
           "primitive, so WHEN it fires is not decidable offline."
           % sum(1 for e in p.effects if e.frame is None)
           if any(e.frame is None for e in p.effects)
           else "Every one names its frame.")))

    for e in p.effects:
        if e.frame is not None:
            past = a.frames is not None and e.frame > a.frames
            a.markers.append(Marker(
                float(e.frame), EFFECT, "fx%d" % e.id,
                "the host spawns effect %d at ITS bone %s on frame %d"
                % (e.id, e.bone, e.frame), unreachable=past))
            if past:
                a.findings.append(Finding(
                    WARN, "EFFECT_BEYOND_CLIP",
                    "effect %d fires at frame %d, past this clip's last frame (%d) — "
                    "it never spawns." % (e.id, e.frame, a.frames)))

    bones = sorted({e.bone for e in p.effects if e.bone is not None})
    if rig is not None:
        for bone in bones:
            if not 0 <= bone < rig.n_bones:
                a.findings.append(Finding(
                    ERROR, "EFFECT_BONE_RANGE",
                    "the host anchors effect(s) %s to its bone %d, and this port's rig "
                    "has only %d joints. spawn_effect reads that joint's live world "
                    "position, so the number reads off the end of the array."
                    % (", ".join(str(e.id) for e in p.effects if e.bone == bone),
                       bone, rig.n_bones)))
                continue
            if rig.driven is not None and bone not in rig.driven:
                a.findings.append(Finding(
                    WARN, "EFFECT_BONE_UNDRIVEN",
                    "bone %d — where the host anchors effect(s) %s — is driven by NO "
                    "clip on this rig. It stays at its bind position, so the effect "
                    "would not follow the animal."
                    % (bone, ", ".join(str(e.id) for e in p.effects
                                       if e.bone == bone))))
    if bones:
        where = ("" if rig is None else
                 "  On YOUR rig: %s." % "; ".join(rig.describe(b) for b in bones))
        a.findings.append(Finding(
            WARN, "EFFECT_BONES_ARE_THE_HOSTS",
            "⚠️ bone %s %s the HOST species' own, recovered as literal immediates from "
            "its handler — nothing here maps them onto the port. The same number on a "
            "port's own rig lands somewhere else entirely, which is how the Zinogre "
            "threw snowballs out of Tigrex mouth bones. Watch the joint in the "
            "viewport and write the port's own recipe into [[effect]].%s"
            % (", ".join(str(b) for b in bones),
               "is" if len(bones) == 1 else "are", where)))


def _our_effects(a: Alignment, m: PortManifest, move: Optional[str],
                 rig: Optional[PortRig]) -> None:
    """The `[[effect]]` rows this manifest scripts for the move — in the PORT's terms."""
    for i, e in enumerate(m.effects):
        if move is None or e.move != move:
            continue
        past = a.frames is not None and e.frame > a.frames
        a.markers.append(Marker(
            float(e.frame), OURS, "fx%d*" % e.id,
            "this port spawns effect %d at ITS bone %d on frame %d%s"
            % (e.id, e.bone, e.frame,
               "" if rig is None else " (%s)" % rig.describe(e.bone)),
            unreachable=past))
        if past:
            a.findings.append(Finding(
                WARN, "OUR_EFFECT_BEYOND_CLIP",
                "effect[%d] fires at frame %d, past the clip's last frame (%d)."
                % (i, e.frame, a.frames)))
        if rig is not None and not 0 <= e.bone < rig.n_bones:
            a.findings.append(Finding(
                ERROR, "OUR_EFFECT_BONE_RANGE",
                "effect[%d] names bone %d; this rig has %d joints."
                % (i, e.bone, rig.n_bones)))


def _impact_marker(a: Alignment) -> None:
    if a.impact is None:
        return
    a.markers.append(Marker(float(a.impact), IMPACT, "impact",
                            "where this clip's own contact is (clips.%s.impact_frame)"
                            % a.clip))


# --------------------------------------------------------------------------- #
def _nearest(frames: Sequence[float], to: float) -> Optional[float]:
    return min(frames, key=lambda f: abs(f - to)) if frames else None


def _gate_list(gates: Sequence[float], cap: int = 6) -> str:
    shown = ", ".join("%g" % f for f in gates[:cap])
    return shown + (" …" if len(gates) > cap else "")


def _headline(a: Alignment) -> str:
    """The one sentence #9 exists for: the handler's frame against yours."""
    gates = a.gates
    plural = "s" if len(gates) > 1 else ""
    if a.impact is None:
        if not gates:
            return ("(%d,%d) names no fixed frames, so only the clip's LENGTH matters "
                    "here." % (a.main, a.sub))
        return ("this handler tests frame%s %s; this clip records no impact frame yet "
                "— scrub to its contact and set it, and the gap is the answer."
                % (plural, _gate_list(gates)))
    if not gates:
        return ("this clip's impact is at frame %g, and (%d,%d) tests no fixed frames "
                "— nothing has to line up." % (a.impact, a.main, a.sub))
    near = _nearest(gates, a.impact)
    gap = a.impact - near
    head = ("this handler tests frame%s %s; this clip's impact is at frame %g"
            % (plural, _gate_list(gates), a.impact))
    if abs(gap) <= IMPACT_TOLERANCE:
        return "%s — on the %g test (%+.1f)." % (head, near, gap)
    a.findings.append(Finding(
        WARN, "IMPACT_OFF_GATE",
        "the impact lands %.1f frame(s) %s the nearest tested frame (%g). The clip is "
        "free to be any length, but the handler fires at ITS number, so the hit and "
        "the animation part company by that much."
        % (abs(gap), "after" if gap > 0 else "before", near)))
    return "%s — %.1f frame(s) %s (the nearest test is %g)." % (
        head, abs(gap), "late" if gap > 0 else "early", near)


# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:                # pragma: no cover
    import argparse

    from .intel import find_intel
    from .manifest import load

    ap = argparse.ArgumentParser(
        prog="python -m mhfu_monster_editor.align",
        description="What the host action expects, against your clip. Offline.")
    ap.add_argument("manifest", help="ports/<name>.toml")
    ap.add_argument("--move", help="one move (default: all of them)")
    ap.add_argument("--pair", metavar="M,S",
                    help="a pair that is NOT bound yet — the deciding view. Pair it "
                         "with --slot to name the clip you would use.")
    ap.add_argument("--slot", type=int,
                    help="with --pair: the clip slot to align against")
    ap.add_argument("--impact", type=float,
                    help="with --pair: where that clip's contact is, if the manifest "
                         "does not say")
    ap.add_argument("--pac", help="the built PAC, for the clips' real lengths")
    ap.add_argument("--species",
                    help="another overlay to read the pairs from — a species NUMBER or "
                         "a species/emNN.json path. Default: the manifest's host. "
                         "🔴 Comparing hosts, not re-hosting: the engine dispatches "
                         "the overlay port.host_species names.")
    a = ap.parse_args(argv)

    m = load(a.manifest)
    if not a.species:
        intel = find_intel(m.host_species)
    elif a.species.isdigit():
        intel = find_intel(int(a.species))
    else:
        intel = SpeciesIntel.from_path(a.species)
    ends = None
    if a.pac:
        from .clips import clip_table
        with open(a.pac, "rb") as fh:
            ends = {slot: fp[0] for slot, fp in clip_table(fh.read()).items()}
    if a.pair:
        main, sub = (int(x) for x in a.pair.split(","))
        c = next((x for x in m.clips.values() if x.slot == a.slot), None)
        print(align_pair(m, main, sub, intel,
                         clip=c.name if c else None, slot=a.slot,
                         clip_frames=None if ends is None else ends.get(a.slot),
                         impact=a.impact if a.impact is not None
                         else (c.impact_frame if c else None)).report())
        return 0
    if not m.moves:
        print("%s declares no [moves] yet. Try a pair against a clip first:\n"
              "  python -m mhfu_monster_editor.align %s --pair 1,13 --slot 10 "
              "--pac tmp/%s" % (a.manifest, a.manifest, m.pac))
        return 0
    names = [a.move] if a.move else sorted(m.moves)
    for name in names:
        mv = m.moves[name]
        c = m.clips.get(mv.clip) if mv.clip else None
        end = None if (ends is None or c is None) else ends.get(c.slot)
        print(align(m, name, intel, clip_frames=end).report())
        print()
    return 0


if __name__ == "__main__":                                        # pragma: no cover
    raise SystemExit(main())
