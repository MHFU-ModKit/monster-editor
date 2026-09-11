"""The part system's WRITE path: naming parts, editing the grid, adopting the host's.

Kept out of `ui/app.py` for the reason `clips.LabelSession` is: the editing rules are
the part of issue #10 worth testing, and a test for them should not need a window.

Three things can be authored, and they are not equally safe:

* **names for the eight parts.** Free. The engine has no names — it has
  `entity+0x3B8[0..7]` — so naming them costs nothing and is what makes every other
  panel readable.
* **the damage grid.** 🔴 Species data, shared map-wide. Riding host 75 means
  *inheriting the native Tigrex's grid*, and changing it changes his too. Authoring
  one here records what the port WANTS; nothing in this repo has ever written one into
  a running game and confirmed the effect (issue #19).
* **the collision volumes.** ⚠️ Bone-indexed against the rig the port SHIPS. Adopting
  the host's copies numbers that mean different joints — the Brute's manifest says it
  outright: "the native Tigrex's four volumes are indices into the Tigrex rig; this
  port ships its own 47-joint rig, so copying them across would attach the head sphere
  to whatever joint happens to sit at index 10." :meth:`PartSession.adopt_volumes`
  does it anyway, because a wrong starting point you can see in the viewport beats a
  blank one — but it returns the count that lands off the end of the rig, and the
  caller is expected to say so.

* **editing a volume** (2026-09-11). :meth:`PartSession.edit_volume`,
  :meth:`scale_volume`, :meth:`remove_volume`, :meth:`keep_only`, :meth:`add_volume`
  work on the list as it would be saved — the manifest's blocks with the staged
  edits applied — so an index means the same thing in the panel, in the viewport
  and in `ops()`. ⚠️ The runtime writes the list IN PLACE over the host's own set,
  sentinel-terminated, so it holds at most as many records as that set had
  (em75: 42). :meth:`PartSession.over_capacity` says by how many; the runtime
  truncates and logs rather than walking off the end of the table.

Everything lands through `manifest.patch`, so a session shows up in `git diff` as the
parts and nothing else.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .manifest import (AppendBlock, HITZONE_COLUMNS, HITZONE_ROWS, Hurtbox,
                       HitzoneState, ManifestError, PART_SLOTS, Part, PortManifest,
                       ReplaceBlock, SHAPES, SetKey, hitzone_block, hurtbox_block,
                       patch, patch_file)

#: the eight accumulator slots, with the names MH itself uses where they are obvious.
#: Suggestions only — a monster's parts are its own, and slot 0 really is "nobody".
SUGGESTED = {0: "", 1: "head", 2: "body", 3: "tail", 4: "left_wing",
             5: "left_leg", 6: "right_wing", 7: "right_leg"}

_VALID = set("abcdefghijklmnopqrstuvwxyz0123456789_")


def valid_name(name: str) -> bool:
    """A bare TOML key. `[parts.left wing]` is a different thing entirely."""
    return bool(name) and not name[0].isdigit() and set(name) <= _VALID


def blank_grid() -> List[List[int]]:
    return [[0] * len(HITZONE_COLUMNS) for _ in range(HITZONE_ROWS)]


@dataclass
class VolumeAdoption:
    """What :meth:`PartSession.adopt_volumes` actually did — never just a count."""
    adopted: int = 0
    #: volumes whose bone is off the end of the port's rig. They are STILL written,
    #: because a hurtbox the validator can complain about beats one nobody wrote.
    off_rig: List[int] = field(default_factory=list)
    source: str = ""

    @property
    def clean(self) -> bool:
        return not self.off_rig

    def describe(self) -> str:
        if not self.adopted:
            return "nothing to adopt"
        if self.clean:
            return ("%d volume(s) from %s. Every bone index exists on this rig — "
                    "which is not the same as pointing at the right joint."
                    % (self.adopted, self.source))
        return ("%d volume(s) from %s, and %d name a bone this rig does not have "
                "(%s). They are bone indices into the HOST's skeleton; this port "
                "ships its own." % (self.adopted, self.source, len(self.off_rig),
                                    ", ".join(str(b) for b in self.off_rig[:8])))


class PartSession:
    """Staged edits to a manifest's `[parts]`, `[[hurtbox]]` and `[[hitzone]]`.

    Nothing touches disk until :meth:`save`. :attr:`pending` counts what would be
    written, so a panel can show it and a test can assert it.
    """

    def __init__(self, manifest: PortManifest, n_bones: Optional[int] = None) -> None:
        self.m = manifest
        #: the rig the port SHIPS, for the off-rig check. None = unknown, and the
        #: check is then skipped rather than guessed at.
        self.n_bones = n_bones
        self._names: Dict[int, Part] = {}
        self._grids: Dict[int, HitzoneState] = {}
        self._drop_grids: set = set()
        #: the volumes as they would be saved: `(file_index | None, Hurtbox)`.
        #: None = appended this session. Rebuilt from the manifest on discard.
        self._work: List[Tuple[Optional[int], Hurtbox]] = [
            (i, h) for i, h in enumerate(manifest.hurtboxes)]
        #: how many records fit IN PLACE at runtime; None = unknown, no check
        self.capacity: Optional[int] = None

    # ---- reading what is there ---------------------------------------- #
    def part(self, index: int) -> Optional[Part]:
        """The staged part for a slot, else the manifest's, else None."""
        if index in self._names:
            return self._names[index]
        for p in self.m.parts.values():
            if p.index == index:
                return p
        return None

    def name_of(self, index: int) -> str:
        p = self.part(index)
        return p.name if p else ""

    def states(self) -> List[HitzoneState]:
        """The grid as it would be saved: the manifest's, with staged edits applied."""
        out = []
        for i, hz in enumerate(self.m.hitzones):
            if i in self._drop_grids:
                continue
            out.append(self._grids.get(i, hz))
        for i in sorted(k for k in self._grids if k >= len(self.m.hitzones)):
            out.append(self._grids[i])
        return out

    def state(self, index: int) -> Optional[HitzoneState]:
        s = self.states()
        return s[index] if 0 <= index < len(s) else None

    def volumes(self) -> List[Hurtbox]:
        """The `[[hurtbox]]` list as it would be saved, in order."""
        return [h for _, h in self._work]

    def volume_changed(self, index: int) -> bool:
        """Is volume ``index`` (of :meth:`volumes`) staged, i.e. not what the
        file has? Appended ones always are."""
        src, h = self._work[index]
        return src is None or self.m.hurtboxes[src] != h

    def _volume_ops(self) -> Tuple[List[object], int]:
        """Replace / append / delete ops for the volumes, and how many changes."""
        ops: List[object] = []
        n = 0
        kept = {src for src, _ in self._work if src is not None}
        for src, h in self._work:
            if src is not None and self.m.hurtboxes[src] != h:
                ops.append(ReplaceBlock("hurtbox", src, hurtbox_block(h)))
                n += 1
        for src, h in self._work:
            if src is None:
                ops.append(AppendBlock(hurtbox_block(h)))
                n += 1
        # deletions LAST and from the back, so every earlier index stays valid
        for src in sorted((i for i in range(len(self.m.hurtboxes)) if i not in kept),
                          reverse=True):
            ops.append(ReplaceBlock("hurtbox", src, None))
            n += 1
        return ops, n

    @property
    def pending_volumes(self) -> int:
        return self._volume_ops()[1]

    @property
    def over_capacity(self) -> int:
        """How many volumes would NOT fit in place at runtime (0 = all fit)."""
        if self.capacity is None:
            return 0
        return max(0, len(self._work) - self.capacity)

    @property
    def pending(self) -> int:
        return len(self._names) + len(self._grids) + self.pending_volumes \
            + len(self._drop_grids)

    def discard(self) -> None:
        self._names.clear()
        self._grids.clear()
        self._drop_grids.clear()
        self._work = [(i, h) for i, h in enumerate(self.m.hurtboxes)]

    # ---- staging ------------------------------------------------------- #
    def name_part(self, index: int, name: str, *, label: str = "",
                  severable: bool = False,
                  hitzone_row: Optional[int] = None) -> Part:
        """Give slot ``index`` a name. Raises on anything the schema would refuse."""
        if not 0 <= index < PART_SLOTS:
            raise ManifestError("part %d is outside 0..%d — the engine masks the "
                                "part field with 7" % (index, PART_SLOTS - 1))
        name = name.strip()
        if not valid_name(name):
            raise ManifestError("%r is not a bare TOML key: lower-case letters, "
                                "digits and underscore, not starting with a digit"
                                % name)
        clash = [p for p in self.m.parts.values() if p.name == name
                 and p.index != index]
        clash += [p for i, p in self._names.items() if p.name == name and i != index]
        if clash:
            raise ManifestError("part %r already names slot %d. Two names for one "
                                "slot means two break bars that are secretly one."
                                % (name, clash[0].index))
        if hitzone_row is not None and not 0 <= hitzone_row < HITZONE_ROWS:
            raise ManifestError("hitzone_row %d is outside 0..%d"
                                % (hitzone_row, HITZONE_ROWS - 1))
        p = Part(name=name, index=index, hitzone_row=hitzone_row,
                 severable=bool(severable), label=label)
        self._names[index] = p
        return p

    def set_hitzone(self, state: int, row: int, column, value: int) -> None:
        """Edit one percentage. ``column`` is a name or an index."""
        cur = self.state(state)
        if cur is None:
            raise ManifestError("no grid state %d — adopt or add one first" % state)
        col = HITZONE_COLUMNS.index(column) if isinstance(column, str) else int(column)
        if not 0 <= row < HITZONE_ROWS:
            raise ManifestError("row %d is outside 0..%d" % (row, HITZONE_ROWS - 1))
        if not 0 <= col < len(HITZONE_COLUMNS):
            raise ManifestError("column %r is not one of %s"
                                % (column, ", ".join(HITZONE_COLUMNS)))
        if not 0 <= int(value) <= 255:
            raise ManifestError("%r is not a percentage 0..255" % (value,))
        rows = [list(r) for r in cur.rows]
        rows[row][col] = int(value)
        self._grids[state] = HitzoneState(name=cur.name, rows=rows, label=cur.label)

    def fill_row(self, state: int, row: int, value: int,
                 columns: Sequence = ("cut", "impact", "shot")) -> None:
        """Set several columns of one row at once — `255` on cut/impact/shot is
        "every weapon class does the most the grid can express" (a byte)."""
        for c in columns:
            self.set_hitzone(state, row, c, value)

    # ---- the volumes ---------------------------------------------------- #
    _VOL_FIELDS = ("bone", "radius", "part", "hitzone_row", "shape", "offset",
                   "to", "flags", "label")

    def _check_index(self, index: int) -> None:
        if not 0 <= index < len(self._work):
            raise ManifestError("no volume %d — the list has %d"
                                % (index, len(self._work)))

    def edit_volume(self, index: int, **fields) -> Hurtbox:
        """Change one or more fields of volume ``index``. Same rules as the loader:
        a part outside 0..7 or a row outside 0..6 is refused here, not in the game."""
        self._check_index(index)
        bad = [k for k in fields if k not in self._VOL_FIELDS]
        if bad:
            raise ManifestError("a hurtbox has no field %r" % bad[0])
        src, cur = self._work[index]
        new = replace(cur, **fields)
        if new.part is not None and not 0 <= int(new.part) < PART_SLOTS:
            raise ManifestError("part %d is outside 0..%d" % (new.part, PART_SLOTS - 1))
        if new.hitzone_row is not None and not 0 <= int(new.hitzone_row) < HITZONE_ROWS:
            raise ManifestError("hitzone_row %d is outside 0..%d"
                                % (new.hitzone_row, HITZONE_ROWS - 1))
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

    def scale_volume(self, index: int, factor: float) -> Hurtbox:
        """Multiply the radius — the one-line way to make a hurtbox obviously
        different, which is what issue #19's test needs."""
        self._check_index(index)
        return self.edit_volume(index, radius=float(self._work[index][1].radius)
                                * float(factor))

    def remove_volume(self, index: int) -> None:
        self._check_index(index)
        del self._work[index]

    def keep_only(self, index: int) -> int:
        """Drop every volume but ``index``. Returns how many went. This is the
        #19 experiment in one call: one sphere, and a hit lands there or nowhere."""
        self._check_index(index)
        keep = self._work[index]
        n = len(self._work) - 1
        self._work = [keep]
        return n

    def add_volume(self, h: Hurtbox) -> int:
        """Append a volume. Returns its index."""
        self._work.append((None, h))
        return len(self._work) - 1

    def add_state(self, name: str, rows: Optional[Sequence[Sequence[int]]] = None,
                  label: str = "") -> int:
        """Append a grid state. Returns its index."""
        existing = [s.name for s in self.states()]
        if name in existing:
            raise ManifestError("a state called %r is already there; the engine "
                                "picks by INDEX, so the name is all a human has"
                                % name)
        rows = blank_grid() if rows is None else [list(r) for r in rows]
        if len(rows) != HITZONE_ROWS:
            raise ManifestError("%d row(s), not %d" % (len(rows), HITZONE_ROWS))
        idx = len(self.states())
        self._grids[max(idx, len(self.m.hitzones))] = HitzoneState(
            name=name, rows=rows, label=label)
        return idx

    def adopt_grid(self, states, names: Optional[Sequence[str]] = None) -> int:
        """Seed the grid from the host's own — the only real numbers available.

        ``states`` is anything with ``.rows``: `intel.HitzoneState`s straight out of
        `species/emNN.json`. Starting from the host's values is honest — the port
        INHERITS them at runtime whether or not they are written down, so copying
        them in makes visible what is already true.
        """
        default = ("normal", "enraged", "third")
        n = 0
        for i, st in enumerate(states):
            nm = (names[i] if names and i < len(names)
                  else (default[i] if i < len(default) else "state%d" % i))
            rows = [list(r) for r in st.rows]
            if len(rows) != HITZONE_ROWS:
                continue
            self._grids[i] = HitzoneState(
                name=nm, rows=rows,
                label="adopted from the host species; the port inherits these at "
                      "runtime either way")
            n += 1
        for i in range(n, len(self.m.hitzones)):
            self._drop_grids.add(i)
        return n

    def adopt_volumes(self, spheres: Iterable, *, source: str = "the host overlay",
                      parts: Optional[Iterable[int]] = None) -> VolumeAdoption:
        """Seed `[[hurtbox]]` from the host's collision spheres.

        ⚠️ The bone indices are the HOST's. See the module docstring — this is a
        starting point you can see in the viewport, not a correct answer, and the
        return value counts what already does not fit.
        """
        keep = None if parts is None else {int(p) & 7 for p in parts}
        out = VolumeAdoption(source=source)
        for s in spheres:
            part = int(getattr(s, "part", 0)) & 7
            if keep is not None and part not in keep:
                continue
            b = getattr(s, "b", None)
            capsule = bool(getattr(s, "is_capsule", False)) and b is not None
            h = Hurtbox(bone=int(s.bone), radius=float(s.radius), part=part,
                        hitzone_row=int(getattr(s, "hitzone_row", 0)),
                        shape="capsule" if capsule else "sphere",
                        offset=[float(v) for v in getattr(s, "a", (0, 0, 0))],
                        to=[float(v) for v in b] if capsule else None,
                        flags=int(getattr(s, "flags", 0) or 0),
                        label=self.name_of(part))
            self._work.append((None, h))
            out.adopted += 1
            # markers (bone 0x7D..0x7F) are not joints and are expected off-rig
            if self.n_bones is not None and not 0 <= h.bone < self.n_bones \
                    and not h.is_marker:
                out.off_rig.append(h.bone)
        return out

    # ---- writing ------------------------------------------------------- #
    def ops(self) -> List[object]:
        """The `manifest.patch` ops this session would apply, in order."""
        ops: List[object] = []
        for idx in sorted(self._names):
            p = self._names[idx]
            table = "parts.%s" % p.name
            ops.append(SetKey(table, "index", p.index))
            if p.hitzone_row is not None:
                ops.append(SetKey(table, "hitzone_row", p.hitzone_row))
            if p.severable:
                ops.append(SetKey(table, "severable", True))
            if p.label:
                ops.append(SetKey(table, "label", p.label))
        vol_ops, _ = self._volume_ops()
        ops += vol_ops
        # grids: replace in place where the block exists, append where it does not.
        # Deletions run LAST and from the back, so an earlier index stays valid.
        for i in sorted(self._grids):
            block = hitzone_block(self._grids[i])
            if i < len(self.m.hitzones):
                ops.append(ReplaceBlock("hitzone", i, block))
            else:
                ops.append(AppendBlock(block))
        for i in sorted(self._drop_grids, reverse=True):
            ops.append(ReplaceBlock("hitzone", i, None))
        return ops

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
        if self._names:
            bits.append("%d part name(s)" % len(self._names))
        nv = self.pending_volumes
        if nv:
            bits.append("%d volume change(s)" % nv)
        if self._grids:
            bits.append("%d grid state(s)" % len(self._grids))
        if self._drop_grids:
            bits.append("removed %d" % len(self._drop_grids))
        self.discard()
        return "saved " + ", ".join(bits)


# --------------------------------------------------------------------------- #
def summarise(m: PortManifest, host=None) -> List[str]:
    """A few lines a panel — or a session — can print instead of the whole thing."""
    out = []
    if m.parts:
        named = ", ".join("%d=%s" % (p.index, n) for n, p in sorted(
            m.parts.items(), key=lambda kv: kv[1].index))
        out.append("parts: " + named)
    else:
        out.append("parts: none named — the engine has eight slots and no names")
    out.append("volumes: %d authored" % len(m.hurtboxes))
    if m.hitzones:
        out.append("grid: %d state(s) — %s" % (len(m.hitzones),
                                               ", ".join(s.name for s in m.hitzones)))
    elif host is not None and getattr(host, "has_grid", False):
        out.append("grid: none authored; the port INHERITS the host's %d state(s)"
                   % host.n_states)
    else:
        out.append("grid: none authored")
    return out
