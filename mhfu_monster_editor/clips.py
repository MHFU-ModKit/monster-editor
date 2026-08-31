"""The clip vocabulary: what is in each slot, what we call it, and which build said so.

Two questions about a port's animation slots, both answerable offline, neither answered
anywhere the manifest can see:

**What is actually in slot N?** The porter files a source clip into the host slot of the
SAME INDEX. A source clip whose index the host pack has no slot for is DROPPED — with
every structural check still passing — and a host slot with no source clip of that index
is filled with a COPY of the source's idle. 🔴 **A filler slot playing idle is
indistinguishable on screen from an override that did not fire**, and one third of the
Zinogre's a1 space is filler, so every "the latch didn't work" report has to rule it out
first. `tools/port_clip_probe.py --coverage` computes this; that module imports the
debugger stack to do it, so the derivation lives here and is importable with nothing but
`mhfu_model`.

**And what did we decide slot N MEANS?** Clip meanings used to live in
`docs/brute_tigrex_anim_ids.txt`, ninety lines of `61 -> roar precharge?` written by
filming a build nobody recorded. 🔴 **Clip ids are per PAC build** — on `v67_hostslots`
every id had shifted by one — so that file is a trap by construction: it goes on looking
authoritative after the numbers underneath it have moved. Labels belong in the manifest,
next to the `(frames, loop)` fingerprint that can be checked, and keyed to the build they
were written against. :func:`track_labels` then reports which ones MOVED instead of
letting them mislabel silently.

    python -m mhfu_monster_editor.clips ports/zinogre.toml
    python -m mhfu_monster_editor.clips ports/zinogre.toml --pac tmp/zinogre_v10.bin
    python -m mhfu_monster_editor.clips ports/brute_tigrex.toml \\
        --import-labels docs/brute_tigrex_anim_ids.txt --write

Stdlib at import time. The three readers lazily import `mhfu_model` the way
:mod:`mhfu_monster_editor.validate` does, so everything that is pure classification —
which is all of the logic, and all of the tests — runs on a machine with no game data.
"""
from __future__ import annotations

import hashlib
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .manifest import PortManifest, RenameClip, SetKey, load as load_manifest

#: ``(last_keyframe, loop)`` — the fingerprint. Clips are stored UNPADDED at their
#: authored length (`docs/ANIMATION_FORMAT.md`), so this identifies a clip across
#: builds far better than its slot does, and it is what the live clip-state block
#: reads back at ``ent+0x80 + slot*0x40 + 0x1C``.
Fingerprint = Tuple[int, bool]

# ---- slot coverage --------------------------------------------------------- #
CARRIED = "CARRIED"      # the source clip of this index landed here, intact
FILLER = "FILLER"        # a copy of the idle clip — forcing this a1 plays IDLE
HOST = "HOST"            # the host species' own clip is still in the slot
ALTERED = "ALTERED"      # populated, but matches neither the source nor the host
DROPPED = "DROPPED"      # a SOURCE clip with no host slot of that index to land in
UNKNOWN = "UNKNOWN"      # no donor moveset to compare against — nothing is decidable

# ---- label tracking -------------------------------------------------------- #
CURRENT = "CURRENT"          # labelled against THIS build, fingerprint agrees
STILL_VALID = "STILL_VALID"  # labelled against another build, fingerprint agrees
MOVED = "MOVED"              # that clip is in this build, at a different slot
AMBIGUOUS = "AMBIGUOUS"      # the fingerprint matches several slots
LOST = "LOST"                # nothing in this build has that fingerprint
UNCHECKABLE = "UNCHECKABLE"  # the manifest records no fingerprint to look for

#: statuses that mean "do not trust this label until someone looks again"
SUSPECT = (MOVED, AMBIGUOUS, LOST, UNCHECKABLE)


def _tools_on_path() -> None:
    tools = str(Path(__file__).resolve().parent.parent / "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)


# --------------------------------------------------------------------------- #
# reading tables out of files
# --------------------------------------------------------------------------- #
def clip_table(pac: bytes) -> Dict[int, Fingerprint]:
    """``{slot: (last_keyframe, loop)}`` for every populated slot of an MHFU pack.

    Works on a native monster PAC and on a BUILT port alike — a port is MHFU-format
    by construction. Same derivation as `tools/port_clip_probe.py::pac_clip_table`.
    """
    _tools_on_path()
    from mhfu_model import anim_ingame as ig
    from mhfu_model.pac import MonsterPac

    sub = MonsterPac.from_bytes(pac).find("anim")
    if sub is None:
        raise ValueError("no animation sub-resource in this PAC")
    a = ig.parse_ingame(sub.data)
    out: Dict[int, Fingerprint] = {}
    for slot in sorted({s for st in a.streams for s in st.clips}):
        blk = next(st.clips[slot] for st in a.streams if slot in st.clips)
        end = max((kf.frame for bn in blk.bones for ch in bn.channels
                   for kf in ch.keyframes), default=0)
        out[slot] = (int(end), bool(blk.loop))
    return out


def source_clip_table(moveset: bytes) -> Dict[int, Fingerprint]:
    """``{slot: (last_keyframe, loop)}`` for an MHP3rd donor moveset (`file_0NNNN`).

    The DONOR side of the coverage question: these are the clips the porter had to
    place, before the host pack's slot layout got a vote.
    """
    _tools_on_path()
    from mhfu_model import anim as A

    pk = A.parse_p3rd(moveset)
    return {an.slot: (int(max((kf.frame for tr in an.tracks for ch in tr.channels
                               for kf in ch.keyframes), default=0)), bool(an.loop))
            for an in pk.animations}


def build_id(path: os.PathLike | str, data: Optional[bytes] = None) -> str:
    """``"zinogre_v10.bin@8f3c1a02"`` — the identity a label is keyed to.

    The filename alone is not enough (``tmp/zinogre_v10.bin`` gets rebuilt) and the
    digest alone is not readable, so it is both. Short digest on purpose: this goes in
    a hand-edited file and it only has to distinguish the builds on one disk.
    """
    p = Path(path)
    if data is None:
        data = p.read_bytes()
    return "%s@%s" % (p.name, hashlib.sha1(data).hexdigest()[:8])


# --------------------------------------------------------------------------- #
# coverage
# --------------------------------------------------------------------------- #
@dataclass
class SlotCoverage:
    """What one executor ``a1`` slot of the BUILT pack actually holds."""
    slot: int
    kind: str
    frames: int
    loop: bool
    #: the source clip of the same index, when the donor has one
    source: Optional[Fingerprint] = None

    @property
    def scriptable(self) -> bool:
        """Is forcing this ``a1`` a meaningful thing to do?

        Only :data:`CARRIED` is. :data:`FILLER` plays the idle, :data:`HOST` plays the
        host animal's motion on the port's rig, and :data:`ALTERED` is unidentified.
        """
        return self.kind == CARRIED

    def why(self) -> str:
        return {
            CARRIED: "the donor's own clip %d, intact (%df, loop=%s)"
                     % (self.slot, self.frames, self.loop),
            FILLER: "a COPY OF THE IDLE clip (%df, loop=%s). Forcing this a1 plays "
                    "idle, which on screen is identical to the override never firing."
                    % (self.frames, self.loop),
            HOST: "the HOST species' own clip (%df, loop=%s) — it survived the port, "
                  "so this a1 plays the host animal's motion on your rig."
                  % (self.frames, self.loop),
            ALTERED: "%df, loop=%s — matches neither the donor's clip %d nor the "
                     "host's. Converted, replaced, or a length collision."
                     % (self.frames, self.loop, self.slot),
            UNKNOWN: "%df, loop=%s. No donor moveset to compare against, so whether "
                     "this is a carried clip or the porter's filler is undecided."
                     % (self.frames, self.loop),
        }[self.kind]


@dataclass
class Coverage:
    """Per-slot verdicts for a built pack, plus the source clips that never landed."""
    slots: Dict[int, SlotCoverage] = field(default_factory=dict)
    #: source slot -> its fingerprint, for clips the host pack had no slot for
    dropped: Dict[int, Fingerprint] = field(default_factory=dict)
    idle: Optional[Fingerprint] = None
    #: True when a donor moveset was supplied. Without one NOTHING is decidable — see
    #: :func:`coverage` — so every slot comes back :data:`UNKNOWN`.
    has_source: bool = False
    has_host: bool = False

    def counts(self) -> Dict[str, int]:
        out = {k: 0 for k in (CARRIED, FILLER, HOST, ALTERED, UNKNOWN)}
        for c in self.slots.values():
            out[c.kind] += 1
        out[DROPPED] = len(self.dropped)
        return out

    def kind(self, slot: int) -> Optional[str]:
        c = self.slots.get(slot)
        return c.kind if c else None

    def summary(self) -> str:
        n = self.counts()
        if not self.has_source:
            return ("%d populated slot(s), all UNKNOWN: without the donor moveset "
                    "there is nothing to compare a slot against, and a native pack's "
                    "slots are not 'carried' from anywhere." % len(self.slots))
        lines = ["%d populated slot(s): %d carried, %d filler, %d host, %d altered"
                 % (len(self.slots), n[CARRIED], n[FILLER], n[HOST], n[ALTERED])]
        if not self.has_host:
            lines.append("  no host pack given — a slot still holding the HOST's own "
                         "clip reads as ALTERED")
        if self.dropped:
            lines.append("  %d donor clip(s) DROPPED: the host pack has no slot of "
                         "that index — %s" % (len(self.dropped),
                                              ", ".join(str(s) for s
                                                        in sorted(self.dropped))))
        lines.append("  => the scriptable vocabulary is %d slot(s), addressed by "
                     "executor a1 == the slot index" % n[CARRIED])
        return "\n".join(lines)


def coverage(port: Dict[int, Fingerprint],
             host: Optional[Dict[int, Fingerprint]] = None,
             source: Optional[Dict[int, Fingerprint]] = None) -> Coverage:
    """Classify every slot of a built pack against the host pack and the donor.

    🔴 The two findings that are invisible offline unless the packs are compared slot
    by slot, and that get brains written against them before anyone notices:

      * a donor clip whose index is not a populated host slot is :data:`DROPPED` —
        silently, because every structural check still passes;
      * a host slot with no donor clip of that index holds a copy of the donor's idle
        (:data:`FILLER`), so forcing that ``a1`` plays idle and looks exactly like a
        failed override.

    ``host`` and ``source`` are both optional and the report says which verdicts it
    could not reach without them, rather than guessing one.
    """
    host = host or {}
    source = source or {}
    idle = port.get(1)
    cov = Coverage(idle=idle, has_source=bool(source), has_host=bool(host))
    for slot in sorted(port):
        fp = port[slot]
        src = source.get(slot)
        if not source:
            # ⚠️ Every verdict here is a statement about the PORT — "the donor's clip
            # landed", "the porter's idle copy". A pack with no donor (a native
            # monster, or a port whose source file is not on this disk) has no such
            # statement to make, and guessing FILLER from "same length as slot 1"
            # would libel two clips that merely share an authored length.
            kind = UNKNOWN
        elif src is not None and fp == src:
            kind = CARRIED
        elif slot != 1 and idle is not None and fp == idle and src != idle:
            kind = FILLER
        elif host.get(slot) == fp:
            kind = HOST
        else:
            kind = ALTERED
        cov.slots[slot] = SlotCoverage(slot, kind, fp[0], fp[1], src)
    if host:
        for slot, fp in sorted(source.items()):
            if slot not in host:
                cov.dropped[slot] = fp
    return cov


# --------------------------------------------------------------------------- #
# label tracking
# --------------------------------------------------------------------------- #
@dataclass
class LabelTrack:
    """One manifest clip, checked against the build actually in front of you."""
    name: str
    slot: int
    status: str
    message: str
    label: str = ""
    #: where this clip's fingerprint IS in the current build, when it moved
    now_at: Optional[int] = None
    #: every slot that matches, when several do
    candidates: Tuple[int, ...] = ()
    labelled_build: Optional[str] = None

    @property
    def trusted(self) -> bool:
        return self.status in (CURRENT, STILL_VALID)


def track_labels(m: PortManifest, table: Dict[int, Fingerprint],
                 build: Optional[str] = None) -> List[LabelTrack]:
    """Do the manifest's clip names still point at the clips they were written for?

    The check the old label file could not do. A manifest clip carries a slot, a
    ``(frames, loop)`` fingerprint and the build it was labelled against; a rebuild
    keeps the fingerprint and moves the slot, so looking the fingerprint up in the new
    table answers *"your `charge` is now at slot 57"* instead of quietly labelling
    whatever landed at 61.

    ⚠️ A fingerprint is not unique — 7 of one pack's clips share a length with another
    of its own — so a match at several slots is reported as :data:`AMBIGUOUS` rather
    than resolved by picking the first.
    """
    out: List[LabelTrack] = []
    for name in sorted(m.clips):
        c = m.clips[name]
        got = table.get(c.slot)
        if c.frames is None:
            out.append(LabelTrack(
                name, c.slot, UNCHECKABLE,
                "no `frames` recorded, so there is no fingerprint to look for. Whatever "
                "is in slot %d now wears this name." % c.slot,
                c.label, labelled_build=c.labelled_build))
            continue
        want: Fingerprint = (int(c.frames), bool(c.loop) if c.loop is not None else None)
        matches = tuple(s for s, fp in sorted(table.items())
                        if fp[0] == want[0] and (want[1] is None or fp[1] == want[1]))
        if got is not None and c.slot in matches:
            same = build is not None and c.labelled_build == build
            out.append(LabelTrack(
                name, c.slot, CURRENT if same else STILL_VALID,
                "slot %d still holds a %df%s clip%s"
                % (c.slot, want[0], ", loop" if want[1] else "",
                   " — labelled against this very build" if same else
                   (" (labelled against %s)" % c.labelled_build
                    if c.labelled_build else
                    " (no build recorded for the label, so this rests on the "
                    "fingerprint alone)")),
                c.label, labelled_build=c.labelled_build))
        elif len(matches) == 1:
            out.append(LabelTrack(
                name, c.slot, MOVED,
                "the %df%s clip this name was written for is now at slot %d, not %d. "
                "Re-point it — a1 IS the slot index, so the old number now forces a "
                "different animation."
                % (want[0], ", loop" if want[1] else "", matches[0], c.slot),
                c.label, now_at=matches[0], candidates=matches,
                labelled_build=c.labelled_build))
        elif matches:
            out.append(LabelTrack(
                name, c.slot, AMBIGUOUS,
                "slot %d does not hold it any more and %d slots share its fingerprint "
                "(%s) — a length collision, so which one it is cannot be decided from "
                "the file." % (c.slot, len(matches),
                               ", ".join(str(s) for s in matches)),
                c.label, candidates=matches, labelled_build=c.labelled_build))
        else:
            out.append(LabelTrack(
                name, c.slot, LOST,
                "nothing in this build is %df%s. The clip this name describes is not "
                "here at all — a different donor, a different host frame, or a build "
                "that dropped it." % (want[0], ", loop" if want[1] else ""),
                c.label, labelled_build=c.labelled_build))
    return out


def unlabelled_slots(m: PortManifest, table: Dict[int, Fingerprint]) -> List[int]:
    """Populated slots the manifest has no name for, in slot order."""
    named = {c.slot for c in m.clips.values()}
    return [s for s in sorted(table) if s not in named]


# --------------------------------------------------------------------------- #
# writing a label back
# --------------------------------------------------------------------------- #
def clip_key(slot: int) -> str:
    """The default manifest table name for a slot — ``clip_07``, the ports' own style."""
    return "clip_%02d" % int(slot)


def label_ops(name: str, *, slot: Optional[int] = None, label: Optional[str] = None,
              frames: Optional[int] = None, loop: Optional[bool] = None,
              impact_frame: Optional[int] = None,
              build: Optional[str] = None,
              rename_from: Optional[str] = None) -> List[object]:
    """The manifest edits that write one label. Feed to :func:`manifest.patch`.

    ``build`` is stamped into ``labelled_build`` — which is the whole point of the
    exercise, so pass it whenever the PAC in front of you is known. Everything else is
    written only when given; ``label=""`` clears the key rather than writing an empty
    string.
    """
    ops: List[object] = []
    if rename_from and rename_from != name:
        ops.append(RenameClip(rename_from, name))
    table = "clips.%s" % name
    if slot is not None:
        ops.append(SetKey(table, "slot", int(slot)))
    for key, value in (("frames", frames), ("loop", loop),
                       ("impact_frame", impact_frame)):
        if value is not None:
            ops.append(SetKey(table, key, value))
    if label is not None:
        ops.append(SetKey(table, "label", label or None))
    if build is not None:
        ops.append(SetKey(table, "labelled_build", build or None))
    return ops


class LabelSession:
    """Name clips against ONE build, then write them into the manifest file.

    The editor's write path, kept out of the UI so it can be tested without a display:
    `ui/app.py` only supplies the text boxes. Edits are staged rather than written
    key by key, so a labelling session is one reviewable change to `ports/*.toml` —
    and so the file is untouched until :meth:`save`.

    🔴 The fingerprint and the build id are stamped from the PACK THAT IS OPEN, never
    copied from what the manifest already said. A name typed while looking at
    `zinogre_v10.bin` is a fact about `zinogre_v10.bin`; recording which build it was
    is the whole of #8, and it is what lets the next rebuild report the label as
    :data:`MOVED` instead of quietly re-labelling whatever lands in that slot.
    """

    def __init__(self, m: PortManifest, table: Dict[int, Fingerprint],
                 build: Optional[str] = None) -> None:
        self.manifest = m
        self.table = table
        self.build = build
        self.edits: List[object] = []

    # -- what is there now ---------------------------------------------- #
    def entry(self, slot: int):
        """The manifest's clip for a slot, or None."""
        return next((c for c in self.manifest.clips.values() if c.slot == slot), None)

    def default_name(self, slot: int) -> str:
        c = self.entry(slot)
        return c.name if c else clip_key(slot)

    @property
    def pending(self) -> int:
        return len(self.edits)

    def discard(self) -> None:
        self.edits = []

    # -- staging --------------------------------------------------------- #
    def stage(self, slot: int, name: str, label: str = "",
              impact_frame: Optional[int] = None) -> str:
        """Queue a name/label for ``slot``. Raises :class:`ManifestError` if it cannot.

        Also updates the in-memory manifest, so a list drawn from it redraws under the
        new name before anything is written.
        """
        from .manifest import Clip as _Clip, ManifestError

        name = name.strip()
        if not name:
            raise ManifestError("a clip needs a name")
        if not all(ch.isalnum() or ch in "-_" for ch in name):
            raise ManifestError("%r cannot be a TOML table name — letters, digits, "
                                "- and _" % name)
        clash = next((c for c in self.manifest.clips.values()
                      if c.name == name and c.slot != slot), None)
        if clash is not None:
            raise ManifestError("clips.%s already exists, on slot %d"
                                % (name, clash.slot))
        fp = self.table.get(slot)
        if fp is None:
            raise ManifestError("slot %d is not populated in this build — a name for "
                                "it would describe nothing" % slot)
        entry = self.entry(slot)
        self.edits += label_ops(name, slot=slot, label=label, frames=fp[0], loop=fp[1],
                                impact_frame=impact_frame, build=self.build,
                                rename_from=entry.name if entry else None)
        if entry is not None and entry.name != name:
            del self.manifest.clips[entry.name]
            entry.name = name
            self.manifest.clips[name] = entry
        elif entry is None:
            entry = _Clip(name=name, slot=slot)
            self.manifest.clips[name] = entry
        entry.label = label
        entry.frames, entry.loop = fp
        entry.labelled_build = self.build
        if impact_frame is not None:
            entry.impact_frame = impact_frame
        return "staged clips.%s — %d edit(s) pending" % (name, self.pending)

    def stage_move(self, name: str, main: int, sub: int,
                   clip: Optional[str] = None) -> str:
        """Queue a `[moves.<name>]` binding — a host pair plus the clip it paints.

        The conclusion of the action inspector (#9): you browse pairs against the clip
        on screen, and when the alignment is right you write it down. Only the three
        fields that ARE the alignment; `latch`, `min_gap` and `allow_unentered` keep
        their defaults and are edited in the file, where an override can be argued for
        in a comment.
        """
        from .manifest import ManifestError, Move, SetKey

        name = name.strip()
        if not all(ch.isalnum() or ch in "-_" for ch in name) or not name:
            raise ManifestError("%r cannot be a TOML table name — letters, digits, "
                                "- and _" % name)
        if clip is not None and clip not in self.manifest.clips:
            raise ManifestError("clip %r is not named in this manifest yet — name it "
                                "first, or the move points at nothing" % clip)
        table = "moves.%s" % name
        self.edits += [SetKey(table, "main", int(main)), SetKey(table, "sub", int(sub))]
        if clip is not None:
            self.edits.append(SetKey(table, "clip", clip))
        self.manifest.moves[name] = Move(name=name, main=int(main), sub=int(sub),
                                         clip=clip)
        return "staged moves.%s = (%d,%d)%s — %d edit(s) pending" % (
            name, main, sub, " on %s" % clip if clip else "", self.pending)

    # -- writing --------------------------------------------------------- #
    def save(self, path: Optional[os.PathLike | str] = None) -> str:
        """Patch the manifest file. Comments, order and formatting are preserved."""
        from .manifest import ManifestError, patch_file

        target = Path(path) if path else self.manifest.path
        if target is None:
            raise ManifestError("this manifest was not loaded from a file, so there "
                                "is nowhere to save it")
        if not self.edits:
            return "nothing to save"
        patch_file(target, self.edits)
        n, self.edits = self.pending, []
        return "wrote %d edit(s) to %s" % (n, target)


_LABEL_LINE = re.compile(r"^\s*(\d+)\s*->\s*(.+?)\s*$")


def parse_label_file(text: str) -> Dict[int, str]:
    """``61 -> roar precharge?`` lines — `docs/brute_tigrex_anim_ids.txt`'s format.

    🔴 The ids in such a file are keyed to whichever build was on screen when someone
    filmed it, which is usually recorded nowhere. Import them with the build named as
    unknown and let :func:`track_labels` say so; do not silently promote them.
    """
    out: Dict[int, str] = {}
    for line in text.splitlines():
        m = _LABEL_LINE.match(line)
        if m:
            out[int(m.group(1))] = m.group(2)
    return out


#: what to record as the provenance of a label file that does not say which build it
#: was written against. 🔴 `docs/brute_tigrex_anim_ids.txt` is such a file: 90 labels
#: made by forcing ids on a Brute nobody wrote down, and on `v67_hostslots` every id
#: had shifted by one. Importing them is fine; importing them as if they had been
#: measured against the build in front of you is the mislabelling this module exists
#: to stop.
UNRECORDED = "unrecorded — %s, written against a build nobody wrote down"


def import_ops(labels: Dict[int, str], m: PortManifest,
               table: Dict[int, Fingerprint], labelled_build: str,
               *, only_carried: Optional[Coverage] = None,
               overwrite: bool = False) -> List[object]:
    """Turn a hand-written label file into manifest edits, fingerprinting each one.

    🔴 ``labelled_build`` is where the LABELS came from, never the build being read.
    The fingerprint written beside them is measured from the current build, because
    that is what makes the NEXT rebuild checkable — but it says nothing about whether
    the label was right, and stamping this build as the label's provenance would claim
    it did. Pass :data:`UNRECORDED` when the file does not say.

    Slots the build does not populate are skipped — a label for a slot that is not
    there names nothing. ``only_carried`` skips filler and host slots as well:
    labelling a FILLER slot files a name for the idle clip. A slot that already
    carries a label is left alone unless ``overwrite``.
    """
    ops: List[object] = []
    by_slot = {c.slot: c for c in m.clips.values()}
    for slot in sorted(labels):
        fp = table.get(slot)
        if fp is None:
            continue
        if only_carried is not None and only_carried.kind(slot) != CARRIED:
            continue
        existing = by_slot.get(slot)
        if existing is not None and existing.label and not overwrite:
            continue
        ops += label_ops(existing.name if existing else clip_key(slot), slot=slot,
                         label=labels[slot], frames=fp[0], loop=fp[1],
                         build=labelled_build)
    return ops


# --------------------------------------------------------------------------- #
# one survey, one report
# --------------------------------------------------------------------------- #
@dataclass
class Vocabulary:
    """Everything #8 shows about one pack's clip slots, in one object.

    Assembled from tables the caller has already read, so this stays importable with
    no game data and no `mhfu_model` — the UI reads the files, this classifies them.
    """
    coverage: Coverage
    tracks: List[LabelTrack] = field(default_factory=list)
    unlabelled: List[int] = field(default_factory=list)
    #: `build_id` of the pack in front of you, when its file is known
    build: Optional[str] = None
    #: why part of the report is missing — a file that is not on this machine
    notes: List[str] = field(default_factory=list)

    @property
    def suspect(self) -> List[LabelTrack]:
        """Labels that must not be trusted against this build."""
        return [t for t in self.tracks if not t.trusted]

    def track(self, slot: int) -> Optional[LabelTrack]:
        return next((t for t in self.tracks if t.slot == slot), None)

    def kind(self, slot: int) -> Optional[str]:
        return self.coverage.kind(slot)


def survey(m: Optional[PortManifest], port: Dict[int, Fingerprint],
           host: Optional[Dict[int, Fingerprint]] = None,
           source: Optional[Dict[int, Fingerprint]] = None,
           build: Optional[str] = None,
           notes: Optional[Iterable[str]] = None) -> Vocabulary:
    """Coverage + label health for one pack. ``m=None`` for a PAC with no manifest."""
    cov = coverage(port, host, source)
    return Vocabulary(coverage=cov,
                      tracks=[] if m is None else track_labels(m, port, build),
                      unlabelled=sorted(port) if m is None
                      else unlabelled_slots(m, port),
                      build=build, notes=list(notes or []))


def report(m: PortManifest, port: Dict[int, Fingerprint],
           host: Optional[Dict[int, Fingerprint]] = None,
           source: Optional[Dict[int, Fingerprint]] = None,
           build: Optional[str] = None) -> str:
    """Coverage and label health for one port, as text."""
    v = survey(m, port, host, source, build)
    lines = ["%s  %s" % (m.name, build or "(build not identified)"),
             v.coverage.summary(), ""]
    if not v.tracks:
        lines.append("no clips are named in this manifest yet")
    else:
        lines.append("%d named clip(s), %d trustworthy against this build"
                     % (len(v.tracks), len(v.tracks) - len(v.suspect)))
        for t in v.tracks:
            lines.append("  %s %-14s slot %-3d %-12s %s"
                         % (" " if t.trusted else "!", t.name, t.slot, t.status,
                            t.message))
    scriptable = [s for s in v.unlabelled if v.kind(s) == CARRIED]
    lines.append("")
    lines.append("%d populated slot(s) unnamed, %d of them CARRIED and worth naming: %s"
                 % (len(v.unlabelled), len(scriptable),
                    ", ".join(str(s) for s in scriptable[:24])
                    + (" …" if len(scriptable) > 24 else "")))
    return "\n".join(lines)


def _tables_for(m: PortManifest, pac: Optional[str], root: str):
    """``(port, host, source, build)`` tables, each None when its file is not here."""
    target = Path(pac) if pac else Path("tmp") / m.pac
    if not target.exists():
        raise SystemExit(
            "%s: the built PAC is not there. It is game data and is never committed "
            "— build it with\n  python tools/build_p3rd_port.py --manifest %s --out %s"
            % (target, m.path or "ports/%s.toml" % m.name, target))
    blob = target.read_bytes()
    host_path = m.host_pac_path(root)
    src_path = m.source_paths(root)["anim"]
    host = clip_table(host_path.read_bytes()) if host_path.exists() else None
    source = source_clip_table(src_path.read_bytes()) if src_path.exists() else None
    return clip_table(blob), host, source, build_id(target, blob)


def main(argv: Optional[List[str]] = None) -> int:              # pragma: no cover
    import argparse

    from .manifest import patch_file

    ap = argparse.ArgumentParser(
        prog="python -m mhfu_monster_editor.clips",
        description="Slot coverage and label health for a port, offline.")
    ap.add_argument("manifest", help="ports/<name>.toml")
    ap.add_argument("--pac", help="the built PAC (default tmp/<port.pac>)")
    ap.add_argument("--root", default="workspace", help="where the extracts live")
    ap.add_argument("--slots", action="store_true",
                    help="one line per slot instead of the summary")
    ap.add_argument("--import-labels", metavar="FILE",
                    help="fold a `N -> what I saw` file (docs/brute_tigrex_anim_ids"
                         ".txt) into the manifest, fingerprinting every line")
    ap.add_argument("--labels-from", metavar="BUILD",
                    help="🔴 which build those labels were WRITTEN against. Not this "
                         "one unless you filmed this one: ids shift per build. "
                         "Defaults to recording the file as unrecorded.")
    ap.add_argument("--overwrite", action="store_true",
                    help="with --import-labels, replace labels the manifest already "
                         "carries (default: leave them alone)")
    ap.add_argument("--all-slots", action="store_true",
                    help="with --import-labels, also label FILLER and HOST slots "
                         "(default: only CARRIED ones, which are the real vocabulary)")
    ap.add_argument("--write", action="store_true",
                    help="with --import-labels, actually edit the manifest (comments "
                         "and layout are preserved); otherwise print the diff")
    a = ap.parse_args(argv)

    m = load_manifest(a.manifest)
    port, host, source, build = _tables_for(m, a.pac, a.root)
    cov = coverage(port, host, source)

    if a.import_labels:
        labels = parse_label_file(Path(a.import_labels).read_text(encoding="utf-8"))
        provenance = a.labels_from or UNRECORDED % a.import_labels
        ops = import_ops(labels, m, port, provenance,
                         only_carried=None if a.all_slots else cov,
                         overwrite=a.overwrite)
        print("%s: %d label(s), %d land on a populated%s slot this manifest does not "
              "already name\n  recorded as labelled against: %s"
              % (a.import_labels, len(labels), len(set(
                  op.table for op in ops if isinstance(op, SetKey))),
                 "" if a.all_slots else " CARRIED", provenance))
        if a.write:
            patch_file(a.manifest, ops)
            print("wrote %s" % a.manifest)
        else:
            import difflib
            before = Path(a.manifest).read_text(encoding="utf-8")
            from .manifest import patch
            after = patch(before, ops, path=a.manifest)
            print("".join(difflib.unified_diff(
                before.splitlines(True), after.splitlines(True),
                a.manifest, a.manifest + " (proposed)")) or "(no change)")
            print("--- not written; pass --write ---")
        return 0

    if a.slots:
        for slot in sorted(port):
            c = cov.slots[slot]
            named = [n for n, cl in m.clips.items() if cl.slot == slot]
            print("a1 %-4d %-8s %-14s %s" % (slot, c.kind, ",".join(named) or "-",
                                             c.why()))
        return 0

    print(report(m, port, host, source, build))
    return 0


if __name__ == "__main__":                                      # pragma: no cover
    raise SystemExit(main())
