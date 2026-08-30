"""MHP3rd animation record -> skeleton bone. **NOT positional.**

🔴 This module exists because both this repo's porter and its Blender renderer
assumed `record i drives bone i`, and MHP3rd does not work that way. The mapping is
the one `Kurogami2134/blender_p3rd_anim` implements in `import_p3a.py`::

    skeleton_bone = record_index + bone_offset + skipped_so_far

i.e. walk the skeleton from ``bone_offset``, stepping OVER a per-monster list of
bones the moveset never drives. That repo ships those lists in ``skipped_bones.md``.

**Why it matters.** The Zinogre (em040) has a 51-bone rig and a 37-record moveset.
Read positionally, records 0..36 land on bones 0..36 — so the tail (39..45) and part
of the jaw get NO data and freeze at bind pose while the body animates, and the skin
spanning the two stretches. Read correctly, the same 37 records cover bones 1..46,
**including the tail**. Nothing about the data was missing; it was being read onto
the wrong bones.

**Pinning the offset — use the FORK RULE, not symmetry (2026-08-30).**
``skipped_bones.md`` gives the skip list but not the offset (the addon defaults to 2).
The test that actually decides it is structural:

> **Every record carrying LOCATION channels must land on a joint at or ABOVE the body
> fork** — the first joint with more than one child, where the rig splits into front
> and rear.

A location channel translates its joint's whole subtree. Land one BELOW the fork and it
lifts one half of the animal and not the other, and the geometry spanning the waist is
stretched between them. On the Zinogre that is a ~230-unit lift of the front half: the
model rears up, and the skin from the shoulders to the tail base becomes one long flat
sheet. It looks exactly like a rigging fault, and it is a mapping fault.

    em040: fork = bone 1 (children 2 = chest/neck/head/forelegs, 25 = hind legs/tail)
      offset 0 -> loc on bones 0, 1   ✅ at and above the fork
      offset 1 -> loc on bones 1, 2   ❌ bone 2 is the FRONT branch only
      offset 2 -> loc on bones 2, 3   ❌ worse

    em058 (Brute Tigrex): fork = bone 1 (children 2, 29, 34, 39) — SAME answer, offset 0.
      He is not in ``skipped_bones.md`` at all, so he took DEFAULT_BONE_OFFSET (2) for
      months: two loc records on bones 2 and 3, his middle stretched and a wing swept
      away. ⚠️ It was INVISIBLE while `render_anim_clips` re-skinned with `auto_skin` —
      a blended guess smears a wrong bone map into something plausible. Switching to the
      PMO's authentic palette is what made it show. A skinning guess does not just lose
      fidelity; it HIDES mapping bugs.

The native MHFU Tigrex obeys the same rule — its fork is bone 2 (children 3, 21, 26, 40)
and its loc channels sit on bones 1 and 2. ⚠️ Comparing INDEX positions instead of TREE
positions is what got this wrong for a day: "native carries loc on 1 and 2, so the port
should too" is only true when the fork is in the same place, and it is not.

Two corroborations for offset 0 on em040: the 37 records then consume the 37 real body
bones EXACTLY, ending on bone 45 (the tail tip); and the leftovers are the skip list plus
**46–50, the severed-tail carve object**, which the native Tigrex also leaves undriven
(its own is 45–47). Offset 1 instead leaves bone 0 — the actual root — undriven and
drives bone 46, the severed tail's root.

⚠️ **LEFT/RIGHT symmetry cannot pick the offset.** ``score_offsets()`` scored em040 at
0.7615 for offset 1 against 0.6424 for 2 and 0.4231 positional, and it is *wrong*: a
mirrored pair stays a mirrored pair under any whole-rig shift, so the score only rejects
mappings that break the pairing (like a positional read against a skip list), not one
that is uniformly off by one. Use it to sanity-check the SKIP LIST; use the fork rule to
pick the OFFSET.

"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

# From Kurogami2134/blender_p3rd_anim `skipped_bones.md` — bones the moveset does
# NOT drive, per MHP3rd monster id. Keyed by em id (the "40" column = em040).
SKIPPED_BONES: Dict[int, List[int]] = {
    1:  [8, 15, 25],
    2:  [8, 15, 25],
    7:  [21, 22],
    15: [8, 15, 25],
    16: [22, 23, 24],
    18: [8, 15, 25],
    22: [8],
    40: [9, 10, 16, 17, 24, 31, 37, 38, 42, 47],      # Zinogre
    41: [8, 9, 15, 16, 25, 27, 30, 31, 32, 34, 43, 48],
    47: [17],
    56: [22, 23, 24],
}

# MHP3rd model-PAC base id -> em id, for callers that only know the file number
# (`file_05339.bin` = the Zinogre's model PAC = em040). The stride between PACs is
# NOT constant — monsters occupy different numbers of files — so this is a lookup,
# not a formula. Add a row when you port a new monster; an unknown id falls back to
# DEFAULT_BONE_OFFSET with no skips, which is the addon's own default and is very
# probably WRONG — pin it with the FORK RULE (module docstring). ⚠️ NOT with
# score_offsets(): that cannot pick an offset at all, it only validates a skip list.
EM_BY_MODEL_PAC: Dict[int, int] = {
    5248: 58,      # Brute Tigrex — file_05246/47 are `em058m0.ovl`/`em058m1.ovl`
    5339: 40,      # Zinogre      — file_05337/38 are `em040m0.ovl`/`em040m1.ovl`
}
# 🔴 READ THE ID OUT OF THE DATA, don't guess it. The two files immediately BEFORE a
# monster's model pac are its AI overlays, and an `MWo3` header carries the name at
# offset 32 — `em058m0.ovl` names the species outright. The Brute went unmapped for
# months and silently took DEFAULT_BONE_OFFSET, which is wrong for him.


def em_for_model_pac(pac_id: int) -> int:
    """em id for an MHP3rd model-PAC file number, or -1 if we have not mapped it."""
    return EM_BY_MODEL_PAC.get(int(pac_id), -1)


# Measured per monster (see score_offsets); the addon's own default is 2.
BONE_OFFSET: Dict[int, int] = {
    40: 0,      # Zinogre — the FORK RULE (see module docstring). NOT 1: that lands a
    #             location record on bone 2, the front branch, and hoists the front half
    #             of the animal ~230 units above the rear.
    58: 0,      # Brute Tigrex — the SAME rule, and it was reached the same way. His fork
    #             is bone 1 (children 2, 29, 34, 39), so the addon's default of 2 put his
    #             two location records on bones 2 and 3 — the FRONT branch — and tore his
    #             middle. Corroborated twice, exactly as em040 was: 43 records over
    #             offset 0 consume bones 0..42, i.e. every real body bone EXACTLY, and
    #             what is left undriven is 43/44/45 — a second orphan ROOT chain that is
    #             the severed-tail carve object, the same thing em040 leaves at 46..50.
}
DEFAULT_BONE_OFFSET = 2


def record_to_bone(n_records: int, n_bones: int, offset: int,
                   skip: Iterable[int]) -> Dict[int, int]:
    """``{record -> skeleton bone}``, walking from ``offset`` and stepping over ``skip``."""
    skip = set(skip)
    out: Dict[int, int] = {}
    b = offset
    for rec in range(n_records):
        while b in skip:
            b += 1
        if b >= n_bones:
            break
        out[rec] = b
        b += 1
    return out


def bone_to_record(n_records: int, n_bones: int, offset: int,
                   skip: Iterable[int]) -> Dict[int, int]:
    """Inverse of :func:`record_to_bone` — ``{skeleton bone -> record}``.

    Bones absent from the result are genuinely undriven (bone < offset, a skipped
    bone, or a trailing bone past the last record) and must be posed at REST, not
    left without a section.
    """
    return {b: r for r, b in record_to_bone(n_records, n_bones, offset, skip).items()}


def for_monster(em_id: int, n_records: int, n_bones: int,
                offset: Optional[int] = None,
                skip: Optional[Sequence[int]] = None) -> Dict[int, int]:
    """``{skeleton bone -> record}`` for a known MHP3rd monster id."""
    if offset is None:
        offset = BONE_OFFSET.get(em_id, DEFAULT_BONE_OFFSET)
    if skip is None:
        skip = SKIPPED_BONES.get(em_id, [])
    return bone_to_record(n_records, n_bones, offset, skip)


def score_offsets(pack, parents, bind_world, n_bones: int, n_records: int,
                  skip: Iterable[int], offsets=(0, 1, 2, 3, 4)) -> List[tuple]:
    """Rank candidate offsets by LEFT/RIGHT symmetry. Returns [(score, offset), ...].

    Assumes only that the rig is bilaterally symmetric — never that the pose "looks
    right" — so it works on a monster nobody has ever posed before.
    """
    pairs = []
    for i in range(n_bones):
        for j in range(i + 1, n_bones):
            xi, yi, zi = bind_world[i]
            xj, yj, zj = bind_world[j]
            if abs(xi + xj) < 12 and abs(yi - yj) < 12 and abs(zi - zj) < 12 and abs(xi) > 25:
                pairs.append((i, j))

    def energy(anim):
        e = {}
        for r, tr in enumerate(anim.tracks):
            e[r] = sum(abs(k.value) for ch in tr.channels
                       if (getattr(ch, "type", getattr(ch, "ctype", 0)) & 0xFFF) in (0x8, 0x10, 0x20)
                       for k in ch.keyframes)
        return e

    out = []
    for off in offsets:
        inv = bone_to_record(n_records, n_bones, off, skip)
        score = n = 0.0, 0
        score, n = 0.0, 0
        for anim in pack.animations:
            e = energy(anim)
            for i, j in pairs:
                a = e.get(inv.get(i, -1), 0)
                b = e.get(inv.get(j, -1), 0)
                if a == 0 and b == 0:
                    continue
                score += min(a, b) / max(a, b)
                n += 1
        out.append((score / max(1, n), off))
    return sorted(out, reverse=True)


# --------------------------------------------------------------------------- #
# the fork rule — the check that picks the offset
# --------------------------------------------------------------------------- #
def body_fork(parents: Sequence[int]) -> int:
    """Lowest joint with more than one child: where the rig splits front from rear.

    Everything above it moves the whole animal; everything below it moves one half.
    """
    kids: Dict[int, int] = {}
    for i, p in enumerate(parents):
        if p is not None and p >= 0:
            kids[p] = kids.get(p, 0) + 1
    multi = [j for j, n in kids.items() if n > 1]
    return min(multi) if multi else 0


def _ancestors(parents: Sequence[int], j: int) -> List[int]:
    out, seen = [], set()
    while j is not None and j >= 0 and j not in seen:
        out.append(j)
        seen.add(j)
        j = parents[j] if j < len(parents) else -1
    return out


def loc_below_fork(parents: Sequence[int], loc_bones: Iterable[int]) -> List[int]:
    """Which of `loc_bones` sit BELOW the body fork — i.e. translate only one half.

    Empty list = the mapping passes the fork rule. Anything in it is a bug that will
    show up as the animal tearing at the waist, not as anything the renderer can blame
    on skinning. See the module docstring.
    """
    fork = body_fork(parents)
    bad = []
    for b in loc_bones:
        if b is None or b < 0:
            continue
        anc = _ancestors(parents, b)
        # "at or above the fork" == the fork is this joint or one of its descendants,
        # equivalently this joint is the fork or one of the fork's ancestors.
        if b != fork and b not in _ancestors(parents, fork):
            bad.append(b)
    return sorted(bad)
