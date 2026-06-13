"""Constraint validator — the engine's rules as first-class, checked predicates.

This is the reason the library exists: every edit/export op runs through `validate`
so a modder cannot author something the engine can't represent. Each predicate
returns structured `Result`s (`{level, code, message, fix_hint}`) that drive both
the CLI and the Blender "MHFU Compatibility" panel; an export is blocked when any
ERROR is present.

Rules encoded (see specs/002-model-anim-pipeline/tasks.md "Game constraints"):
  * bone-count <-> animation-record-count lockstep
  * s16 range / quantization overflow on keyframes
  * channel mask vs the channels actually present
  * mesh-group <-> bone binding completeness + ordering
  * keyframe-count / anim-size sanity
  * species "template" match (bone count + parent tree vs a reference monster)
  * advisory engine limits (new clip needs AI; ~2-combatant damage cap)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

ERROR = "error"
WARN = "warn"

# s16 quantization domains (engine units)
ROT_UNIT = 4096.0     # = 90 deg
LOC_UNIT = 16.0       # = 1.0
SCL_UNIT = 256.0      # = 1.0
S16_MIN, S16_MAX = -32768, 32767


@dataclass
class Result:
    level: str          # ERROR | WARN
    code: str
    message: str
    fix_hint: str = ""

    def __str__(self):
        tag = "ERROR" if self.level == ERROR else "warn "
        s = "[%s] %s: %s" % (tag, self.code, self.message)
        return s + ("  (fix: %s)" % self.fix_hint if self.fix_hint else "")


@dataclass
class Report:
    results: List[Result] = field(default_factory=list)

    def add(self, level, code, message, fix_hint=""):
        self.results.append(Result(level, code, message, fix_hint))

    @property
    def errors(self):
        return [r for r in self.results if r.level == ERROR]

    @property
    def warnings(self):
        return [r for r in self.results if r.level == WARN]

    @property
    def ok(self) -> bool:
        return not self.errors

    def __bool__(self):
        return self.ok

    def __str__(self):
        if not self.results:
            return "OK — no constraint issues."
        return "\n".join(str(r) for r in self.results)


# --------------------------------------------------------------------------- #
# Species templates (bone budget + parent tree of a reference monster)
# --------------------------------------------------------------------------- #
@dataclass
class SpeciesTemplate:
    name: str
    bone_count: int
    parents: List[int]                 # parent index per bone (index order)


_TEMPLATES: Dict[str, SpeciesTemplate] = {}


def register_template(name: str, skeleton) -> SpeciesTemplate:
    """Record a reference monster's skeleton layout as the template `name`.

    Built from a known-good PAC (e.g. `register_template("tigrex",
    load_pac("file_06134.bin").skeleton)`) so an edited skeleton can be checked
    against the source species' bone budget + tree.
    """
    t = SpeciesTemplate(name=name, bone_count=len(skeleton.bones),
                        parents=[b.parent for b in skeleton.bones])
    _TEMPLATES[name] = t
    return t


def template(name: Optional[str]) -> Optional[SpeciesTemplate]:
    return _TEMPLATES.get(name) if name else None


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _as_anim_list(animations):
    if animations is None:
        return []
    if hasattr(animations, "animations"):      # an AnimationPack
        return animations.animations
    return list(animations)


# --------------------------------------------------------------------------- #
# Predicates
# --------------------------------------------------------------------------- #
def check_bone_lockstep(rep, skeleton, anims):
    """Keep skeleton + animation bone counts in lockstep.

    Real engine data (verified across all 49 PACs): every anim in a pack animates
    the SAME number of bones (the first N joints, in index order), and that N is
    <= the skeleton's joint count (an anim may animate a subset; e.g. 23 tracks
    over a 26-bone skeleton). So the hard rules are:
      * each anim's header bone_count == its track count;
      * all anims in the pack share one track count (a divergent anim = a desync);
      * track count must not exceed the skeleton's joint count (else the engine
        indexes past its Joint array).
    """
    if not anims:
        return
    counts = [len(a.tracks) for a in anims]
    consensus = max(set(counts), key=counts.count)
    for a in anims:
        if a.bone_count != len(a.tracks):
            rep.add(ERROR, "ANIM_RECORD_COUNT",
                    "anim slot %d: header bone_count=%d but %d tracks present"
                    % (a.slot, a.bone_count, len(a.tracks)),
                    "set Animation.bone_count == len(tracks)")
        if len(a.tracks) != consensus:
            rep.add(ERROR, "ANIM_TRACK_DESYNC",
                    "anim slot %d animates %d bones; the pack's other anims animate "
                    "%d — every clip must cover the same bone set, in index order"
                    % (a.slot, len(a.tracks), consensus),
                    "edit all animations' bone records together")
    if skeleton is not None:
        joints = max(skeleton.bone_count, len(skeleton.bones))
        if consensus > joints:
            rep.add(ERROR, "BONE_COUNT_MISMATCH",
                    "animations animate %d bones but the skeleton only has %d joints"
                    % (consensus, joints),
                    "the skeleton needs at least as many joints as the animations "
                    "address — adding/removing skeleton bones means updating the "
                    "animation records too")


def check_keyframe_s16(rep, anims):
    for a in anims:
        for bi, tr in enumerate(a.tracks):
            for ch in tr.channels:
                for ki, kf in enumerate(ch.keyframes):
                    for fld in ("value", "frame", "ease_in", "ease_out"):
                        v = getattr(kf, fld)
                        if not (S16_MIN <= v <= S16_MAX):
                            rep.add(ERROR, "S16_OVERFLOW",
                                    "anim %d bone %d kf %d: %s=%d out of s16 range "
                                    "[%d,%d]" % (a.slot, bi, ki, fld, v,
                                                 S16_MIN, S16_MAX),
                                    "rotation 4096=90deg, loc 16=1.0, scl 256=1.0; "
                                    "reduce the value so quantization fits s16")


def check_frame_order(rep, anims):
    for a in anims:
        for bi, tr in enumerate(a.tracks):
            for ch in tr.channels:
                frames = [kf.frame for kf in ch.keyframes]
                if any(f < 0 for f in frames):
                    rep.add(ERROR, "NEG_FRAME",
                            "anim %d bone %d: negative keyframe frame" % (a.slot, bi),
                            "frames are >= 0")
                if frames != sorted(frames):
                    rep.add(WARN, "FRAME_ORDER",
                            "anim %d bone %d: keyframes not in ascending frame order"
                            % (a.slot, bi), "sort keyframes by frame")


def check_channel_mask(rep, anims):
    """The bone-record mask has exactly one set bit per channel stored.

    Verified across all 49 PACs (60539 tracks): popcount(mask) == channel count
    without exception. (The mask's bit layout differs from the per-channel `type`
    tag; only the count relationship is the engine's hard invariant.)
    """
    for a in anims:
        for bi, tr in enumerate(a.tracks):
            nbits = bin(tr.mask).count("1")
            if nbits != len(tr.channels):
                rep.add(ERROR, "CHANNEL_MASK_COUNT",
                        "anim %d bone %d: mask 0x%x has %d bits set but %d channels "
                        "present" % (a.slot, bi, tr.mask, nbits, len(tr.channels)),
                        "set exactly one mask bit per channel record (add/remove a "
                        "channel and its mask bit together)")


def check_mesh_binding(rep, model, skeleton):
    if model is None or not model.mesh_groups:
        return
    idx = [g.index for g in model.mesh_groups]
    expect = list(range(len(idx)))
    if idx != expect:
        rep.add(ERROR, "MESH_ORDER",
                "mesh-group draw indices %s are not contiguous 0..%d in order"
                % (idx[:8] + (["..."] if len(idx) > 8 else []), len(idx) - 1),
                "rigid binding maps draw order -> bone index; keep groups "
                "contiguous and ordered")
    empties = [g.index for g in model.mesh_groups if g.vertex_count == 0]
    if empties:
        rep.add(WARN, "MESH_EMPTY_GROUP",
                "mesh groups with no geometry: %s" % empties,
                "an empty group still consumes a bind slot")
    if skeleton is not None and len(model.mesh_groups) != len(skeleton.bones):
        rep.add(WARN, "MESH_BONE_COUNT",
                "%d mesh groups vs %d skeleton bones — binding relies on the "
                "geometry-less skip list; verify the mapping"
                % (len(model.mesh_groups), len(skeleton.bones)))


def check_anim_sanity(rep, anims):
    for a in anims:
        if a.bone_count <= 0:
            rep.add(ERROR, "ANIM_NO_BONES",
                    "anim slot %d has no bone records" % a.slot)
        for bi, tr in enumerate(a.tracks):
            for ch in tr.channels:
                if not ch.keyframes:
                    rep.add(WARN, "EMPTY_CHANNEL",
                            "anim %d bone %d: a channel has zero keyframes"
                            % (a.slot, bi), "remove the empty channel or add a key")


def check_species_template(rep, skeleton, target_species):
    t = template(target_species)
    if t is None:
        if target_species:
            rep.add(WARN, "NO_TEMPLATE",
                    "no registered template for species %r — skeleton not checked "
                    "against a reference" % target_species,
                    "register_template(name, reference_skeleton)")
        return
    if skeleton is None:
        return
    if len(skeleton.bones) != t.bone_count:
        rep.add(ERROR, "TEMPLATE_BONE_COUNT",
                "skeleton has %d bones; species %r template expects %d"
                % (len(skeleton.bones), t.name, t.bone_count),
                "match the source monster's bone budget (its skeleton + AI assume it)")
        return
    parents = [b.parent for b in skeleton.bones]
    if parents != t.parents:
        diffs = [i for i, (p, q) in enumerate(zip(parents, t.parents)) if p != q]
        rep.add(ERROR, "TEMPLATE_TREE",
                "skeleton parent tree differs from species %r template at bones %s"
                % (t.name, diffs[:8]),
                "keep the bone hierarchy identical to the source species")


def check_engine_limits(rep, anims, advisories):
    """Advisory-only flags for limits outside the asset itself."""
    if not advisories:
        return
    if advisories.get("new_anim_slots"):
        rep.add(WARN, "NEW_CLIP_NEEDS_AI",
                "new animation slots %s added — a clip only plays if the species "
                "AI requests its index" % advisories["new_anim_slots"],
                "drive it via the action-force seam mhfu_on_bigmonster_action")
    if advisories.get("extra_combatants"):
        rep.add(WARN, "COMBAT_CAP",
                "adding fighters beyond ~2 manager-driven combatants: extras render "
                "+ animate but won't deal/receive player damage (unsolved cap)",
                "see memory player-damage-path / megatigrex-clone-swarm")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def validate(model=None, skeleton=None, animations=None,
             target_species: Optional[str] = None,
             advisories: Optional[dict] = None) -> Report:
    """Run every constraint predicate; return a Report (truthy iff no errors)."""
    rep = Report()
    anims = _as_anim_list(animations)
    check_bone_lockstep(rep, skeleton, anims)
    check_keyframe_s16(rep, anims)
    check_frame_order(rep, anims)
    check_channel_mask(rep, anims)
    check_mesh_binding(rep, model, skeleton)
    check_anim_sanity(rep, anims)
    check_species_template(rep, skeleton, target_species)
    check_engine_limits(rep, anims, advisories)
    return rep
