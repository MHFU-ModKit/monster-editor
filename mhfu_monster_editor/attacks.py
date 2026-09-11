"""The attack system's WRITE path: the sets a move hits with, on the port's own rig.

The mirror of `parts.py`, for issue #33's finding that the attack hitbox is DATA: a
handler spawns an attack by id, the attack record names a volume SET, and the set is
the same `0x28` sphere records the hurtbox editor already edits — minus `part` and
`hitzone_row`, plus the set index that says which attack it belongs to.

Two things can be authored, and neither is free of a caveat:

* **the volumes of a set** — `[[hitbox]]` records keyed by `set`. ⚠️ Bone-indexed
  against the rig the port SHIPS. The host's set 2 (the Tigrex charge) sits on
  Tigrex bones 10/18/34/4/2/41/42/43; on the Zinogre's rig those numbers are other
  joints, which is exactly the misalignment this session exists to fix.
  :meth:`AttackSession.adopt_set` copies the host's records anyway, because a wrong
  starting point you can SEE beats a blank one, and returns the count off the rig.
  🔴 The runtime writes each set IN PLACE over the host's own, sentinel-terminated,
  so a set holds at most as many records as the host's had (em75 set 2: 10).
  :meth:`over_capacity` says by how many; the runtime truncates and logs.
* **an attack record's levers** — `[[attack]]`: `power`, `element`, `volume`, the
  three fields that were edited live and followed (#33). Nothing else on the `0x18`
  record is decoded, so nothing else is offered.

🔴 Both are SPECIES data in the overlay, shared by every entity of that species on
the map — with the port REPLACING its host it owns them; beside a native monster of
the host species it re-arms the native too. Same caveat as the damage grid.

Everything lands through `manifest.patch`, so a session shows up in `git diff` as the
volumes and the records and nothing else.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Dict, Iterable, List, Optional, Tuple

from .manifest import (AppendBlock, Attack, Hitbox, ManifestError, PortManifest,
                       ReplaceBlock, SHAPES, attack_block, hitbox_block, patch,
                       patch_file)
from .parts import VolumeAdoption


class AttackSession:
    """Staged edits to a manifest's `[[hitbox]]` and `[[attack]]` blocks.

    Nothing touches disk until :meth:`save`. Indices into :meth:`volumes` mean the
    same thing in the panel, in the viewport and in :meth:`ops`, exactly as in
    `PartSession` — the list as it would be saved, staged edits applied.
    """

    def __init__(self, manifest: PortManifest, n_bones: Optional[int] = None) -> None:
        self.m = manifest
        #: the rig the port SHIPS, for the off-rig check. None = unknown, skipped.
        self.n_bones = n_bones
        #: `(file_index | None, Hitbox)` — None = appended this session
        self._work: List[Tuple[Optional[int], Hitbox]] = [
            (i, h) for i, h in enumerate(manifest.hitboxes)]
        #: record id -> staged Attack (a whole block: replaces the file's or appends)
        self._attacks: Dict[int, Attack] = {}
        #: record ids whose block is to be removed
        self._drop_attacks: set = set()
        #: set index -> how many records fit IN PLACE at runtime. Missing = unknown,
        #: and the check is skipped rather than guessed at.
        self.capacities: Dict[int, int] = {}

    # ---- reading what is there ---------------------------------------- #
    def volumes(self) -> List[Hitbox]:
        """The `[[hitbox]]` list as it would be saved, in order."""
        return [h for _, h in self._work]

    def sets(self) -> List[int]:
        """The set indices the port authors, sorted."""
        return sorted({h.set for _, h in self._work})

    def volumes_of(self, set_index: int) -> List[Tuple[int, Hitbox]]:
        """`(index into volumes(), Hitbox)` for one set, in order."""
        return [(i, h) for i, (_, h) in enumerate(self._work) if h.set == int(set_index)]

    def volume_changed(self, index: int) -> bool:
        src, h = self._work[index]
        return src is None or self.m.hitboxes[src] != h

    def attack(self, id: int) -> Optional[Attack]:
        """The staged record for ``id``, else the manifest's, else None."""
        if int(id) in self._drop_attacks:
            return None
        if int(id) in self._attacks:
            return self._attacks[int(id)]
        for a in self.m.attacks:
            if a.id == int(id):
                return a
        return None

    def attacks(self) -> List[Attack]:
        """The `[[attack]]` list as it would be saved, by record id."""
        ids = {a.id for a in self.m.attacks} | set(self._attacks)
        out = [self.attack(i) for i in sorted(ids)]
        return [a for a in out if a is not None]

    def over_capacity(self, set_index: int) -> int:
        """How many of a set's volumes would NOT fit in place (0 = all fit)."""
        cap = self.capacities.get(int(set_index))
        if cap is None:
            return 0
        return max(0, len(self.volumes_of(set_index)) - cap)

    def over_capacity_all(self) -> Dict[int, int]:
        """`{set: excess}` for every authored set that does not fit."""
        out = {}
        for st in self.sets():
            n = self.over_capacity(st)
            if n:
                out[st] = n
        return out

    @property
    def pending_volumes(self) -> int:
        return self._volume_ops()[1]

    @property
    def pending_attacks(self) -> int:
        return self._attack_ops()[1]

    @property
    def pending(self) -> int:
        return self.pending_volumes + self.pending_attacks

    def discard(self) -> None:
        self._work = [(i, h) for i, h in enumerate(self.m.hitboxes)]
        self._attacks.clear()
        self._drop_attacks.clear()

    # ---- the volumes ---------------------------------------------------- #
    _VOL_FIELDS = ("bone", "radius", "set", "shape", "offset", "to", "flags", "label")

    def _check_index(self, index: int) -> None:
        if not 0 <= index < len(self._work):
            raise ManifestError("no volume %d — the list has %d"
                                % (index, len(self._work)))

    def edit_volume(self, index: int, **fields) -> Hitbox:
        """Change one or more fields of volume ``index``. Same rules as the loader,
        so a bad value is refused at the keystroke rather than in the game."""
        self._check_index(index)
        bad = [k for k in fields if k not in self._VOL_FIELDS]
        if bad:
            raise ManifestError("a hitbox has no field %r" % bad[0])
        src, cur = self._work[index]
        new = replace(cur, **fields)
        if int(new.set) < 0:
            raise ManifestError("set %d — a volume-set index is 0 or more" % new.set)
        if new.shape not in SHAPES:
            raise ManifestError("shape %r is not one of %s" % (new.shape, ", ".join(SHAPES)))
        if float(new.radius) < 0.0:
            raise ManifestError("a negative radius is not a volume")
        if new.bone < 0 or new.bone > 0xFFFF:
            raise ManifestError("bone %d does not fit the record's u16" % new.bone)
        if new.offset is not None and len(new.offset) != 3:
            raise ManifestError("offset must be three numbers")
        if new.to is not None and len(new.to) != 3:
            raise ManifestError("to must be three numbers")
        self._work[index] = (src, new)
        return new

    def scale_volume(self, index: int, factor: float) -> Hitbox:
        self._check_index(index)
        return self.edit_volume(index, radius=float(self._work[index][1].radius)
                                * float(factor))

    def remove_volume(self, index: int) -> None:
        self._check_index(index)
        del self._work[index]

    def keep_only(self, index: int) -> int:
        """Drop every OTHER volume of the same set. Returns how many went.

        Within the set, not the whole list: the other sets are other attacks, and
        the experiment — one sphere, a hit lands there or nowhere — is per attack.
        """
        self._check_index(index)
        keep = self._work[index]
        st = keep[1].set
        before = len(self._work)
        self._work = [w for i, w in enumerate(self._work) if w[1].set != st or i == index]
        return before - len(self._work)

    def add_volume(self, h: Hitbox) -> int:
        """Append a volume. Returns its index into :meth:`volumes`."""
        if int(h.set) < 0:
            raise ManifestError("set %d — a volume-set index is 0 or more" % h.set)
        self._work.append((None, h))
        return len(self._work) - 1

    def drop_set(self, set_index: int) -> int:
        """Remove every authored volume of one set — the host's stays as it is at
        runtime. Returns how many went."""
        before = len(self._work)
        self._work = [w for w in self._work if w[1].set != int(set_index)]
        return before - len(self._work)

    def adopt_set(self, set_index: int, spheres: Iterable, *,
                  source: str = "the host overlay", label: str = "") -> VolumeAdoption:
        """Seed one set from the host's own records, REPLACING what the port had
        authored for that set.

        ⚠️ The bone indices are the HOST's. This is a starting point you can see in
        the viewport, not a correct answer, and the return value counts what already
        does not fit the port's rig. 125/126/127 are a coordinate space, expected
        off-rig, and are kept in order — the walker needs them where they were.
        """
        self.drop_set(set_index)
        out = VolumeAdoption(source=source)
        for s in spheres:
            b = getattr(s, "b", None)
            capsule = bool(getattr(s, "is_capsule", False)) and b is not None
            h = Hitbox(bone=int(s.bone), radius=float(s.radius), set=int(set_index),
                       shape="capsule" if capsule else "sphere",
                       offset=[float(v) for v in getattr(s, "a", (0, 0, 0))],
                       to=[float(v) for v in b] if capsule else None,
                       flags=int(getattr(s, "flags", 0) or 0), label=label)
            self._work.append((None, h))
            out.adopted += 1
            if self.n_bones is not None and not 0 <= h.bone < self.n_bones \
                    and not h.is_marker:
                out.off_rig.append(h.bone)
        return out

    # ---- the records ---------------------------------------------------- #
    _LEVERS = ("power", "element", "volume")

    def set_attack(self, id: int, **levers) -> Attack:
        """Stage `power` / `element` / `volume` on record ``id``; a lever passed as
        None is cleared (the host's byte stays). Refuses anything outside a byte."""
        bad = [k for k in levers if k not in self._LEVERS and k != "label"]
        if bad:
            raise ManifestError("an attack record has no lever %r — only %s are "
                                "decoded" % (bad[0], ", ".join(self._LEVERS)))
        if int(id) < 0:
            raise ManifestError("id %d — a record index is 0 or more" % id)
        for k, v in levers.items():
            if k != "label" and v is not None and not 0 <= int(v) <= 255:
                raise ManifestError("%s %r is outside a byte (0..255)" % (k, v))
        cur = self.attack(id) or Attack(id=int(id))
        new = replace(cur, **levers)
        self._drop_attacks.discard(int(id))
        self._attacks[int(id)] = new
        return new

    def clear_attack(self, id: int) -> None:
        """Drop the port's block for record ``id`` — the host's record stands."""
        self._attacks.pop(int(id), None)
        if any(a.id == int(id) for a in self.m.attacks):
            self._drop_attacks.add(int(id))

    # ---- writing ------------------------------------------------------- #
    def _volume_ops(self) -> Tuple[List[object], int]:
        ops: List[object] = []
        n = 0
        kept = {src for src, _ in self._work if src is not None}
        for src, h in self._work:
            if src is not None and self.m.hitboxes[src] != h:
                ops.append(ReplaceBlock("hitbox", src, hitbox_block(h)))
                n += 1
        for src, h in self._work:
            if src is None:
                ops.append(AppendBlock(hitbox_block(h)))
                n += 1
        # deletions LAST and from the back, so every earlier index stays valid
        for src in sorted((i for i in range(len(self.m.hitboxes)) if i not in kept),
                          reverse=True):
            ops.append(ReplaceBlock("hitbox", src, None))
            n += 1
        return ops, n

    def _attack_ops(self) -> Tuple[List[object], int]:
        ops: List[object] = []
        n = 0
        file_index = {a.id: i for i, a in enumerate(self.m.attacks)}
        for aid in sorted(self._attacks):
            a = self._attacks[aid]
            if aid in file_index:
                if self.m.attacks[file_index[aid]] != a:
                    ops.append(ReplaceBlock("attack", file_index[aid], attack_block(a)))
                    n += 1
            else:
                ops.append(AppendBlock(attack_block(a)))
                n += 1
        for aid in sorted((i for i in self._drop_attacks if i in file_index),
                          key=lambda i: file_index[i], reverse=True):
            ops.append(ReplaceBlock("attack", file_index[aid], None))
            n += 1
        return ops, n

    def ops(self) -> List[object]:
        """The `manifest.patch` ops this session would apply, in order."""
        vol_ops, _ = self._volume_ops()
        atk_ops, _ = self._attack_ops()
        return vol_ops + atk_ops

    def preview(self, text: str) -> str:
        """The patched TEXT, without writing it. Raises if it would not load."""
        return patch(text, self.ops())

    def save(self, path=None) -> Optional[str]:
        """Apply the staged edits to the manifest file. Returns a one-line summary."""
        if not self.pending:
            return None
        target = path or (self.m.path if self.m.path else None)
        if target is None:
            raise ManifestError("this manifest was not loaded from a file, so there "
                                "is nowhere to save it")
        patch_file(target, self.ops())
        bits = []
        nv = self.pending_volumes
        if nv:
            bits.append("%d hitbox change(s)" % nv)
        na = self.pending_attacks
        if na:
            bits.append("%d attack record(s)" % na)
        self.discard()
        return "saved " + ", ".join(bits)


# --------------------------------------------------------------------------- #
def summarise(m: PortManifest, host=None) -> List[str]:
    """A few lines a panel — or a session — can print instead of the whole thing."""
    out = []
    sets = sorted({h.set for h in m.hitboxes})
    if m.hitboxes:
        out.append("hitboxes: %d volume(s) over set(s) %s"
                   % (len(m.hitboxes), ", ".join(str(s) for s in sets)))
    else:
        out.append("hitboxes: none authored — the port hits with the host's own sets")
    if m.attacks:
        out.append("attacks: %d record(s) tuned — %s"
                   % (len(m.attacks), ", ".join(
                       "%d(%s)" % (a.id, ",".join(k for k in ("power", "element", "volume")
                                                  if getattr(a, k) is not None))
                       for a in m.attacks)))
    elif host is not None and getattr(host, "present", False):
        out.append("attacks: none tuned; the host's %d record(s) stand" % len(host.attacks))
    return out
