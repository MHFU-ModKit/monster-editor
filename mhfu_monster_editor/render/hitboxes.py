"""The hurtbox gizmos: where the monster can be hit, drawn on the pose that is playing.

Issue #10 asks for the collision volumes "attached to the live posed skeleton, so you
can see a sphere sitting on the wrong bone the moment a clip plays". That is the whole
argument for drawing them at all — a bone index is a number, and on a ported rig the
host's numbers point at the wrong joints, which is invisible in a table and obvious the
instant the animal moves and a sphere stays behind.

A record is a **sphere on a bone** or a **capsule between two bone-relative points**,
and both endpoints ride the bone's world matrix, so this needs `Pose.world` — the
`(n,4,4)` matrices — not just the joint positions the bone overlay uses.

🔴 **Colour is by PART, not by hitzone row.** They are different fields on the same
record (`monster-porting`, `tools/mhfu_model/hitzone.py`): `part` is the damage
accumulator that breaks or severs, `hitzone_row` picks the percentages. A Tigrex wing
is part 6 and row 5. Colouring by part is the right default because the parts are what
a player names — head, tail, wings — and because a part whose spheres are scattered
across the animal is a real fault the picture shows immediately.

⚠️ A sphere on a bone the rig does not have is drawn **nowhere**, and
:attr:`HitboxOverlay.orphans` counts them. Silently skipping would make a port whose
volumes were copied from a 48-joint host onto a 46-joint rig look correct.

**The ATTACK volumes draw through the same overlay** (issue #33) — the same `0x28`
record, so the same gizmo — with two differences the overlay carries rather than the
caller: they are coloured and filtered by their volume SET (`Volume.group`; for a
hurtbox the group IS the part, so nothing above changes), and bones 126/127 are a
coordinate space, not a joint: 127 is a sphere at the NODE's own position, 126 a
capsule between the node's own two points, and for a body attack the node sits at the
attacker's origin. They draw there, at the actor's origin, and are not orphans. 125 is
a joiner with no geometry and draws nowhere, like the hurtboxes' `0x7D` tail marker.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .overlay import Lines

#: one colour per `entity+0x3B8` slot. Eight, because the engine masks the part
#: field with 7 and there is no ninth accumulator to colour.
PART_COLORS = (
    (0.55, 0.58, 0.64),      # 0 — unassigned; deliberately grey, it means "nobody"
    (0.95, 0.35, 0.35),      # 1
    (0.98, 0.68, 0.25),      # 2
    (0.92, 0.88, 0.30),      # 3
    (0.42, 0.82, 0.45),      # 4
    (0.30, 0.75, 0.80),      # 5
    (0.45, 0.60, 0.95),      # 6
    (0.80, 0.50, 0.92),      # 7
)
#: a volume whose bone is off the end of the rig. Drawn nowhere; see `orphans`.
C_ORPHAN = (1.00, 0.15, 0.55, 1.0)

#: bones that are a coordinate space rather than a joint (`hitbox.py`)
BONE_JOINER = 0x7D
BONE_NODE_CAPSULE = 0x7E
BONE_NODE_SPHERE = 0x7F
NODE_SPACE_BONES = (BONE_NODE_CAPSULE, BONE_NODE_SPHERE)
MARKER_BONES = (BONE_JOINER, BONE_NODE_CAPSULE, BONE_NODE_SPHERE)

#: the two palettes. "part": one colour per accumulator slot, eight. "set": a hue
#: walk for attack volume sets — em75 has 56 and a move shows one or two at a time,
#: so what matters is that two sets on screen together are never the same colour,
#: not that set 37 has a memorable one. Warm-leaning on purpose: these are where he
#: hits YOU, and the hurtboxes' cool blues/greens should read as the other thing.
PALETTE_PART = "part"
PALETTE_SET = "set"


def set_color(group: int) -> Tuple[float, float, float]:
    """A hue for volume set ``group``, by golden-angle walk, warm-biased."""
    import colorsys
    hue = (0.02 + (int(group) * 0.381966)) % 1.0
    # fold the cool half towards warm: hurtboxes own blue/green
    hue = hue * 0.55 if hue < 0.5 else 0.72 + (hue - 0.5) * 0.56
    r, g, b = colorsys.hsv_to_rgb(hue % 1.0, 0.72, 0.96)
    return (r, g, b)


def group_color(group: int, palette: str) -> Tuple[float, float, float]:
    if palette == PALETTE_SET:
        return set_color(group)
    return PART_COLORS[int(group) % len(PART_COLORS)]

#: an unselected part while something else IS selected. Low enough to read as
#: context rather than as clutter — with 153 volumes on screen, anything brighter
#: buries the part you asked to look at.
DIM_ALPHA = 0.05
LIVE_ALPHA = 0.85

#: rings per sphere, and segments per ring. Three great circles read as a sphere from
#: any angle and cost 3*SEGMENTS line segments; more is prettier and not more useful.
RINGS = 3
SEGMENTS = 24

#: the SELECTED part also gets its shell filled, faintly. An outline alone reads as a
#: flat scribble on top of the animal — a translucent surface is what makes a sphere
#: look like it encloses something, and which side of a limb it is on.
#: Only the selection is filled: 153 filled volumes would be a fog, and the fill is
#: the selection indicator.
FILL_ALPHA = 0.16
#: latitude bands per hemisphere and longitude segments. Modest on purpose — the
#: surface is rebuilt every frame while a clip plays, and at 0.16 alpha nobody is
#: counting facets.
FILL_LAT = 8
FILL_LON = 14


def sphere_geometry(centre, radius: float, segments: int = SEGMENTS) -> np.ndarray:
    """Three great circles as a line list: ``(3*segments*2, 3)``."""
    c = np.asarray(centre, dtype=np.float64).reshape(3)
    t = np.linspace(0.0, 2.0 * np.pi, segments + 1)
    cos, sin = np.cos(t) * radius, np.sin(t) * radius
    z = np.zeros_like(cos)
    rings = [np.stack([cos, sin, z], axis=1),      # XY
             np.stack([cos, z, sin], axis=1),      # XZ
             np.stack([z, cos, sin], axis=1)]      # YZ
    out = []
    for ring in rings:
        p = ring + c
        out.append(np.stack([p[:-1], p[1:]], axis=1).reshape(-1, 3))
    return np.concatenate(out, axis=0)


def capsule_geometry(a, b, radius: float, segments: int = SEGMENTS) -> np.ndarray:
    """Two end spheres plus four rails along the axis — a swept sphere, readably."""
    a = np.asarray(a, dtype=np.float64).reshape(3)
    b = np.asarray(b, dtype=np.float64).reshape(3)
    parts = [sphere_geometry(a, radius, segments), sphere_geometry(b, radius, segments)]
    axis = b - a
    n = float(np.linalg.norm(axis))
    if n > 1e-9:
        axis = axis / n
        # any vector not parallel to the axis gives a usable first perpendicular
        seed = np.array([0.0, 1.0, 0.0])
        if abs(float(axis @ seed)) > 0.9:
            seed = np.array([1.0, 0.0, 0.0])
        u = np.cross(axis, seed)
        u /= np.linalg.norm(u)
        v = np.cross(axis, u)
        for d in (u, -u, v, -v):
            off = d * radius
            parts.append(np.stack([a + off, b + off], axis=0))
    return np.concatenate(parts, axis=0)


def _grid_tris(pts: np.ndarray) -> np.ndarray:
    """Triangulate an ``(R, C, 3)`` quad lattice into a ``(N, 3)`` triangle soup."""
    a = pts[:-1, :-1]
    b = pts[1:, :-1]
    c = pts[1:, 1:]
    d = pts[:-1, 1:]
    tris = np.stack([a, b, c, a, c, d], axis=2)          # (R-1, C-1, 6, 3)
    return tris.reshape(-1, 3)


def sphere_surface(centre, radius: float, lat: int = FILL_LAT,
                   lon: int = FILL_LON) -> np.ndarray:
    """A filled UV sphere as a triangle soup."""
    theta = np.linspace(0.0, np.pi, 2 * lat + 1)[:, None]
    phi = np.linspace(0.0, 2.0 * np.pi, lon + 1)[None, :]
    pts = np.stack([np.sin(theta) * np.cos(phi),
                    np.tile(np.cos(theta), (1, phi.shape[1])),
                    np.sin(theta) * np.sin(phi)], axis=2) * radius
    return _grid_tris(pts + np.asarray(centre, dtype=np.float64).reshape(1, 1, 3))


def capsule_surface(a, b, radius: float, lat: int = FILL_LAT,
                    lon: int = FILL_LON) -> np.ndarray:
    """A filled capsule: two hemispheres and the tube between them, in one lattice.

    The trick is that the CENTRE moves with latitude — ``b`` for the bands above the
    equator and ``a`` for the ones below — so a single sphere lattice becomes a
    capsule. The equator ring is duplicated (once per centre) and the quads between
    those two copies are the tube, which is why there is no separate cylinder here.
    """
    a = np.asarray(a, dtype=np.float64).reshape(3)
    b = np.asarray(b, dtype=np.float64).reshape(3)
    axis = b - a
    n = float(np.linalg.norm(axis))
    if n <= 1e-9:
        return sphere_surface(a, radius, lat, lon)
    axis = axis / n
    seed = np.array([0.0, 1.0, 0.0])
    if abs(float(axis @ seed)) > 0.9:
        seed = np.array([1.0, 0.0, 0.0])
    u = np.cross(axis, seed)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)

    half = np.linspace(0.0, np.pi / 2.0, lat + 1)
    theta = np.concatenate([half, np.pi - half[::-1]])[:, None]   # equator twice
    phi = np.linspace(0.0, 2.0 * np.pi, lon + 1)[None, :]
    radial = (u[None, None, :] * np.cos(phi)[..., None]
              + v[None, None, :] * np.sin(phi)[..., None])        # (1, lon+1, 3)
    dirs = (axis[None, None, :] * np.cos(theta)[..., None]
            + radial * np.sin(theta)[..., None])
    centre = np.where((np.cos(theta) >= 0.0)[..., None], b[None, None, :],
                      a[None, None, :])
    return _grid_tris(centre + dirs * radius)


class Volume:
    """One drawable collision volume, in the rig's own indices.

    Deliberately duck-typed on :class:`mhfu_monster_editor.intel.HitSphere` and on
    :class:`mhfu_monster_editor.manifest.Hurtbox` alike — the host's volumes and the
    port's own authored ones draw through the same path, which is what makes a
    side-by-side comparison possible at all.
    """

    __slots__ = ("bone", "part", "hitzone_row", "radius", "a", "b", "label", "group")

    def __init__(self, bone: int, radius: float, part: int = 0,
                 hitzone_row: int = 0, a=(0.0, 0.0, 0.0), b=None,
                 label: str = "", group: Optional[int] = None) -> None:
        self.bone = int(bone)
        self.radius = float(radius)
        self.part = int(part) & 7
        self.hitzone_row = int(hitzone_row)
        self.a = tuple(float(v) for v in a)
        self.b = None if b is None else tuple(float(v) for v in b)
        self.label = label
        #: the colour/filter key: the PART for a hurtbox, the SET for an attack
        #: volume. Defaults to the part so the hurtbox path is unchanged.
        self.group = self.part if group is None else int(group)

    @property
    def is_capsule(self) -> bool:
        return self.b is not None

    @property
    def is_node_space(self) -> bool:
        """126/127: geometry at the NODE's own position — the actor's origin."""
        return self.bone in NODE_SPACE_BONES

    @property
    def is_marker(self) -> bool:
        return self.bone in MARKER_BONES

    @classmethod
    def from_intel(cls, s, group: Optional[int] = None) -> "Volume":
        return cls(bone=s.bone, radius=s.radius, part=getattr(s, "part", 0),
                   hitzone_row=getattr(s, "hitzone_row", 0), a=s.a,
                   b=s.b if getattr(s, "is_capsule", False) else None, group=group)

    @classmethod
    def from_manifest(cls, h) -> "Volume":
        if hasattr(h, "set"):            # manifest.Hitbox — an attack volume
            return cls(bone=h.bone, radius=h.radius, part=0, hitzone_row=0,
                       a=h.offset or (0.0, 0.0, 0.0),
                       b=h.to if getattr(h, "is_capsule", False) else None,
                       label=getattr(h, "label", ""), group=int(h.set))
        return cls(bone=h.bone, radius=h.radius, part=h.part or 0,
                   hitzone_row=h.hitzone_row or 0,
                   a=h.offset or (0.0, 0.0, 0.0),
                   b=h.to if getattr(h, "is_capsule", False) else None,
                   label=getattr(h, "label", ""))

    def __repr__(self) -> str:                                # pragma: no cover
        return "<Volume bone=%d part=%d row=%d group=%d r=%g%s>" % (
            self.bone, self.part, self.hitzone_row, self.group, self.radius,
            " capsule" if self.is_capsule else "")


def volumes_from(source: Iterable, group: Optional[int] = None) -> List[Volume]:
    """Adapt intel spheres or manifest hurtboxes/hitboxes — whichever you hand it.
    ``group`` stamps a set index onto intel spheres (an `AttackSet`'s), so the host's
    attack volumes colour and filter by set like the port's authored ones."""
    out = []
    for s in source:
        if isinstance(s, Volume):
            out.append(s)
        elif hasattr(s, "offset"):
            out.append(Volume.from_manifest(s))
        else:
            out.append(Volume.from_intel(s, group=group))
    return out


def attack_volumes_from(sets: Iterable) -> List[Volume]:
    """The host's attack sets (`intel.AttackSet`s) as one list, grouped by index."""
    out: List[Volume] = []
    for st in sets:
        out += volumes_from(st.spheres, group=st.index)
    return out


def _upload(batch, pos, col) -> None:
    if pos:
        batch.set(np.concatenate(pos, axis=0), np.concatenate(col, axis=0))
    else:
        batch.set(np.zeros((0, 3), dtype="f4"), np.zeros((0, 4), dtype="f4"))


class HitboxOverlay:
    """Collision volumes, transformed by the pose each frame.

    The geometry is rebuilt whenever the pose moves, which is once per frame while a
    clip plays. That is ~150 volumes x 3 rings x 24 segments on a Tigrex — small
    beside the skinned mesh, and it keeps the code honest: there is no cached
    world-space copy to go stale against the animation.
    """

    def __init__(self, ctx, volumes: Sequence[Volume], n_bones: int,
                 palette: str = PALETTE_PART) -> None:
        self.ctx = ctx
        self.volumes = list(volumes)
        self.n_bones = int(n_bones)
        #: "part" for hurtboxes, "set" for attack volumes — see `group_color`
        self.palette = palette
        #: the GROUP filter and selection: parts for hurtboxes, sets for attacks.
        #: The `*_part` names are kept because the hurtbox panel and its tests use
        #: them; on an attack overlay they mean the set.
        self.visible_parts: Optional[frozenset] = None
        self.selected_part: Optional[int] = None
        #: ONE volume, by index into :attr:`volumes`, singled out for editing: it
        #: alone gets the shell and everything else dims, whatever its part.
        self.selected_volume: Optional[int] = None
        #: volumes whose bone is off the end of this rig — drawn NOWHERE, counted
        #: here. Marker bones are NOT orphans: 125 is a joiner (no geometry), 126/127
        #: draw at the actor's origin.
        self.orphans: Tuple[Volume, ...] = tuple(
            v for v in self.volumes
            if not 0 <= v.bone < self.n_bones and not v.is_marker)
        self._lines = Lines(ctx)
        #: the selected part's translucent shell. Same flat program, TRIANGLES.
        self._fill = Lines(ctx, mode=ctx.TRIANGLES)
        self._world: Optional[np.ndarray] = None
        self._dirty = True

    # ---- state -------------------------------------------------------- #
    def set_pose(self, world: np.ndarray) -> None:
        """``world`` is ``(n,4,4)`` bone matrices — :attr:`core.pose.Pose.world`."""
        self._world = np.asarray(world, dtype=np.float64)
        self._dirty = True

    def _key(self, g: int) -> int:
        return int(g) & 7 if self.palette == PALETTE_PART else int(g)

    def set_visible_parts(self, parts: Optional[Iterable[int]]) -> None:
        """Show only these GROUPS (parts, or sets on an attack overlay)."""
        v = None if parts is None else frozenset(self._key(p) for p in parts)
        if v != self.visible_parts:
            self.visible_parts = v
            self._dirty = True

    def set_selected_part(self, part: Optional[int]) -> None:
        """Focus one GROUP: it gets the shell, the rest dim."""
        p = None if part is None else self._key(part)
        if p != self.selected_part:
            self.selected_part = p
            self._dirty = True

    # the same two, under the name that is honest on an attack overlay
    set_visible_groups = set_visible_parts
    set_selected_group = set_selected_part

    def set_selected_volume(self, index: Optional[int]) -> None:
        """Single out one volume. Out-of-range clears, rather than raising in the
        middle of a frame after a list the panel just shortened."""
        i = None if index is None or not 0 <= int(index) < len(self.volumes) \
            else int(index)
        if i != self.selected_volume:
            self.selected_volume = i
            self._dirty = True

    def index_of(self, v: Volume) -> Optional[int]:
        for i, x in enumerate(self.volumes):
            if x is v:
                return i
        return None

    def _drawable(self, v: Volume) -> bool:
        return (0 <= v.bone < self.n_bones) or v.is_node_space

    def shown(self) -> List[Volume]:
        """The volumes that will actually be drawn: orphans and joiners excluded,
        node-space ones (126/127) included at the origin."""
        return [v for v in self.volumes
                if self._drawable(v)
                and (self.visible_parts is None or v.group in self.visible_parts)]

    def parts(self) -> List[int]:
        """The groups present — parts, or sets on an attack overlay."""
        return sorted({v.group for v in self.volumes})

    groups = parts

    def by_part(self) -> Dict[int, List[Volume]]:
        out: Dict[int, List[Volume]] = {}
        for v in self.volumes:
            out.setdefault(v.group, []).append(v)
        return out

    by_group = by_part

    # ---- geometry ----------------------------------------------------- #
    def _place(self, v: Volume) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """The volume's endpoints in world space under the current pose.

        A node-space volume (bone 126/127) is placed at the ACTOR's origin — the
        node's position is the owner's, and the owner's origin is world zero in the
        viewport — so it neither follows a joint nor counts as off the rig.
        """
        if v.is_node_space or self._world is None \
                or not 0 <= v.bone < len(self._world):
            return np.asarray(v.a), (None if v.b is None else np.asarray(v.b))
        m = self._world[v.bone]
        rot, trans = m[:3, :3], m[:3, 3]
        a = rot @ np.asarray(v.a) + trans
        b = None if v.b is None else rot @ np.asarray(v.b) + trans
        return a, b

    def world_centres(self) -> np.ndarray:
        """``(k,3)`` — one point per shown volume, for picking and for framing."""
        shown = self.shown()
        if not shown:
            return np.zeros((0, 3))
        out = np.empty((len(shown), 3))
        for i, v in enumerate(shown):
            a, b = self._place(v)
            out[i] = a if b is None else (a + b) * 0.5
        return out

    def _rebuild(self) -> None:
        pos, col, fpos, fcol = [], [], [], []
        one = (None if self.selected_volume is None
               else self.volumes[self.selected_volume])
        for v in self.shown():
            a, b = self._place(v)
            g = (sphere_geometry(a, v.radius) if b is None
                 else capsule_geometry(a, b, v.radius))
            rgb = group_color(v.group, self.palette)
            alpha = LIVE_ALPHA
            if one is not None:
                # a single volume singled out overrides the part focus: it is the
                # one being edited, and the question is "where is THIS one"
                focus = v is one
            else:
                focus = self.selected_part is not None and v.group == self.selected_part
            if (one is not None or self.selected_part is not None) and not focus:
                alpha = DIM_ALPHA
            pos.append(g)
            col.append(np.tile(np.array((*rgb, alpha), dtype="f4"), (len(g), 1)))
            if focus:
                f = (sphere_surface(a, v.radius) if b is None
                     else capsule_surface(a, b, v.radius))
                fpos.append(f)
                fcol.append(np.tile(np.array((*rgb, FILL_ALPHA), dtype="f4"),
                                    (len(f), 1)))
        _upload(self._lines, pos, col)
        _upload(self._fill, fpos, fcol)
        self._dirty = False

    # ---- drawing ------------------------------------------------------ #
    def render(self, mvp: np.ndarray, *, alpha: float = 1.0) -> None:
        if self._dirty:
            self._rebuild()
        # the shell first, with depth WRITES off: it is transparent, so letting it
        # write depth would make it occlude its own outline and the volumes behind
        # it. (Under x-ray the depth test is already off and this changes nothing.)
        if self._fill.count:
            self.ctx.depth_mask = False
            self._fill.alpha = alpha
            self._fill.render(mvp)
            self.ctx.depth_mask = True
        self._lines.alpha = alpha
        self._lines.render(mvp)

    def pick(self, mvp: np.ndarray, size: Tuple[int, int], x: float, y: float,
             radius: float = 18.0) -> Optional[Volume]:
        """The shown volume whose centre is nearest ``(x, y)`` in PANEL pixels.

        ``y`` from the TOP, like `SkeletonOverlay.pick`. Volumes behind the camera
        are excluded rather than wrapped by the perspective divide.
        """
        shown = self.shown()
        c = self.world_centres()
        if not len(c):
            return None
        clip = np.concatenate([c, np.ones((len(c), 1))], axis=1) @ np.asarray(mvp).T
        w = clip[:, 3]
        ok = w > 1e-9
        if not ok.any():
            return None
        ndc = np.zeros((len(c), 3))
        ndc[ok] = clip[ok, :3] / w[ok, None]
        px = (ndc[:, 0] * 0.5 + 0.5) * size[0]
        py = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * size[1]
        d = np.hypot(px - x, py - y)
        d[~ok] = np.inf
        i = int(np.argmin(d))
        return shown[i] if d[i] <= radius else None

    def release(self) -> None:
        self._lines.release()
        self._fill.release()

    def __repr__(self) -> str:                                # pragma: no cover
        return "<HitboxOverlay %d volume(s), %d orphan(s), %s %s>" % (
            len(self.volumes), len(self.orphans),
            "sets" if self.palette == PALETTE_SET else "parts", self.parts())
