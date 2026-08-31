"""Engine FK and linear-blend skinning, vectorised — the maths the viewer runs per frame.

🔴 **`tools/mhfu_model/stretch.py` IS THE ORACLE, and it stays.** Its scalar
``world_matrices`` / ``deform_matrices`` / ``skin_group`` are the engine-correct
reference: posed joints computed with them equal the built PAC's to **0.00 units on
all 46 driven joints of every clip** (`blender_mhfu/CLAUDE.md`). Nothing here is a
better idea about the maths — it is the same maths with the Python loops replaced by
array ops, and `tests/test_core_pose.py` pins it term for term against the scalar. A
numpy rewrite whose only oracle is itself is worthless.

What is reproduced, exactly
---------------------------
* **Rotations are ABSOLUTE LOCAL.** A channel gives a joint's own orientation, not a
  delta from a rest orientation — the bug that cost a week in `blender_mhfu`
  (`fk_bake.py`). ``bind_rot`` is all zeros, so a joint's rest orientation is identity
  and the bind pose is pure translation.
* **Euler XYZ, composed `Rz @ Ry @ Rx`** — Blender's ``Euler('XYZ')``, and
  `stretch._euler_xyz` term for term.
* **`world = parent_world @ local`**, `local = [R(rot) | loc]`, `loc` defaulting to the
  joint's ``bind_pos`` on any axis the clip does not drive and `rot` to 0.
* **`deform = posed @ bind⁻¹`**, blended per vertex by its own normalised weights; a
  vertex with no usable influence stays at bind.
* **Channels interpolate linearly and CLAMP** outside their own first/last key — per
  channel, not per clip, so a joint whose rotation stops keying at frame 40 holds that
  value while its neighbours keep moving.

What is NOT reproduced: `stretch.sample`'s positional record→bone assumption. It is
right for MHFU's own flat clips and wrong for an MHP3rd source moveset, so :func:`sample`
takes the map (`mhfu_model.p3rd_anim_map`) as an argument. See :mod:`.scene`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from ._paths import tools_on_path

tools_on_path()

from mhfu_model import convert as C            # noqa: E402  (needs tools/ on the path)

#: an influence below this is dropped before normalising — `stretch.skin_group`'s own
#: threshold. A PMO palette pads unused slots with weight 0.0, and a 0-weight slot
#: still names a bone, so without the filter every vertex would look 8-way blended.
WEIGHT_EPS = 1e-4


# --------------------------------------------------------------------------- #
# skeleton -> arrays
# --------------------------------------------------------------------------- #
def bone_ids(skel) -> List[int]:
    return [b.index for b in skel.bones]


def effective_parents(skel) -> np.ndarray:
    """``(n,) int32`` parent array, ``-1`` for a root — what the FK actually walks.

    Replicates `stretch.world_matrices`'s guard exactly: a parent that is not a bone
    of this skeleton, or that is already on the path being resolved (a cycle), is
    dropped and the joint becomes a root. Doing it once here means the vectorised FK
    can assume a clean forest.
    """
    by = {b.index: b for b in skel.bones}
    used: Dict[int, int] = {}
    done: set = set()

    def resolve(i: int, seen: tuple = ()) -> None:
        if i in done:
            return
        b = by[i]
        par = b.parent if (b.parent in by and b.parent not in seen) else None
        used[i] = -1 if par is None else par
        if par is not None:
            resolve(par, seen + (i,))
        done.add(i)

    for b in skel.bones:
        resolve(b.index)
    return np.array([used[b.index] for b in skel.bones], dtype=np.int32)


def _levels(parents: np.ndarray) -> List[np.ndarray]:
    """Joint indices grouped by tree depth, roots first — the FK's batching order."""
    n = len(parents)
    depth = np.full(n, -1, dtype=np.int32)
    depth[parents < 0] = 0
    for _ in range(n):
        todo = depth < 0
        if not todo.any():
            break
        ok = todo & (depth[parents] >= 0)
        if not ok.any():
            raise ValueError("skeleton parent array is cyclic after effective_parents()")
        depth[ok] = depth[parents[ok]] + 1
    return [np.flatnonzero(depth == d) for d in range(int(depth.max()) + 1)]


def _euler_xyz(rot: np.ndarray) -> np.ndarray:
    """``(n,3)`` radians -> ``(n,3,3)``. `stretch._euler_xyz`, term for term."""
    cx, cy, cz = np.cos(rot[:, 0]), np.cos(rot[:, 1]), np.cos(rot[:, 2])
    sx, sy, sz = np.sin(rot[:, 0]), np.sin(rot[:, 1]), np.sin(rot[:, 2])
    m = np.empty(rot.shape[:1] + (3, 3), dtype=np.float64)
    m[:, 0, 0] = cy * cz
    m[:, 0, 1] = cz * sx * sy - cx * sz
    m[:, 0, 2] = cx * cz * sy + sx * sz
    m[:, 1, 0] = cy * sz
    m[:, 1, 1] = cx * cz + sx * sy * sz
    m[:, 1, 2] = -cz * sx + cx * sy * sz
    m[:, 2, 0] = -sy
    m[:, 2, 1] = cy * sx
    m[:, 2, 2] = cy * cx
    return m


class Rig:
    """A bind skeleton, arranged for the FK: parents, local bind offsets, depth levels.

    Bone *index* must equal position in ``skel.bones``. Every consumer in this repo
    already assumes it (`stretch.cross_fork_tear` builds its parent list by position),
    so this refuses a skeleton where it does not hold rather than mis-indexing quietly.
    """

    def __init__(self, parents: Sequence[int], bind_local: np.ndarray) -> None:
        self.parents = np.asarray(parents, dtype=np.int32)
        self.bind_local = np.asarray(bind_local, dtype=np.float64)
        if self.bind_local.shape != (len(self.parents), 3):
            raise ValueError("bind_local must be (n_bones, 3)")
        self.levels = _levels(self.parents)
        self.bind_world = self._world(np.zeros_like(self.bind_local), self.bind_local)
        # bind is pure translation, so its inverse is a translation by the negation.
        self.bind_inverse = np.tile(np.eye(4), (self.n_bones, 1, 1))
        self.bind_inverse[:, :3, 3] = -self.bind_world[:, :3, 3]

    # ---- construction ------------------------------------------------- #
    @classmethod
    def from_skeleton(cls, skel) -> "Rig":
        ids = bone_ids(skel)
        if ids != list(range(len(ids))):
            raise ValueError(
                "skeleton bone indices are not 0..n-1 (%r...) — every consumer in this "
                "repo keys joints by list position" % ids[:8])
        return cls(effective_parents(skel),
                   np.array([b.bind_pos for b in skel.bones], dtype=np.float64))

    # ---- geometry ----------------------------------------------------- #
    @property
    def n_bones(self) -> int:
        return len(self.parents)

    @property
    def bind_joints(self) -> np.ndarray:
        """``(n,3)`` world bind positions."""
        return self.bind_world[:, :3, 3]

    def _world(self, rot: np.ndarray, loc: np.ndarray) -> np.ndarray:
        n = len(self.parents)
        local = np.zeros((n, 4, 4), dtype=np.float64)
        local[:, :3, :3] = _euler_xyz(rot)
        local[:, :3, 3] = loc
        local[:, 3, 3] = 1.0
        world = np.empty_like(local)
        root = self.levels[0]
        world[root] = local[root]
        for lvl in self.levels[1:]:
            world[lvl] = world[self.parents[lvl]] @ local[lvl]
        return world

    def world(self, rot: Optional[np.ndarray] = None,
              loc: Optional[np.ndarray] = None) -> np.ndarray:
        """``(n,4,4)`` world matrices. Both arguments ``None`` gives the BIND pose."""
        return self._world(np.zeros_like(self.bind_local) if rot is None else rot,
                           self.bind_local if loc is None else loc)

    def deform(self, world: np.ndarray) -> np.ndarray:
        """``posed @ bind⁻¹`` — what a vertex's palette blends."""
        return world @ self.bind_inverse


# --------------------------------------------------------------------------- #
# clip sampling
# --------------------------------------------------------------------------- #
def sample(anim, record_to_bone: Optional[Dict[int, int]] = None) -> Dict[int, dict]:
    """``{joint: {"rot"/"loc": {axis: [(frame, value), ...]}}}`` — `stretch.sample`'s shape.

    With ``record_to_bone=None`` this is `stretch.sample` verbatim (positional: track
    *i* drives joint *i*), which is right for MHFU's own clips and for a BUILT port.

    🔴 It is wrong for an MHP3rd **source** moveset. There ``bone = record + offset +
    skipped``, and reading it positionally leaves the Zinogre's tail frozen at bind
    while his body animates. Pass the map from `mhfu_model.p3rd_anim_map.for_monster`
    — the same map the porter reads, never a locally re-derived offset.
    """
    out: Dict[int, dict] = {}
    for j, tr in enumerate(anim.tracks):
        if record_to_bone is not None:
            mapped = record_to_bone.get(j)
            if mapped is None:
                continue
            j = mapped
        per = out.setdefault(j, {"rot": {}, "loc": {}})
        for ch in tr.channels:
            kind, axis = C.channel_kind(ch.type)
            if kind in ("rot", "loc") and ch.keyframes:
                per[kind][axis] = sorted(
                    (kf.frame, C.dequantize(kind, kf.value)) for kf in ch.keyframes)
    return out


class Curves:
    """One clip's channels as padded arrays, evaluable at any (fractional) frame.

    The scalar reference walks a per-channel keyframe list with a Python loop; this
    keeps every channel's keys in one ``(C, Kmax)`` block and finds all the brackets
    with a single comparison. Keyframe *frames* are s16 integers, so the piecewise-
    linear result is identical, including the per-channel clamp outside its own range.
    """

    _KIND = ("rot", "loc")

    def __init__(self, samples: Dict[int, dict], n_bones: int,
                 bind_local: np.ndarray) -> None:
        rows: List[tuple] = []          # (kind_index, joint, axis)
        keys: List[list] = []
        for joint in sorted(samples):
            if not 0 <= joint < n_bones:
                continue
            per = samples[joint]
            for ki, kind in enumerate(self._KIND):
                for axis, pts in sorted(per.get(kind, {}).items()):
                    if not pts:
                        continue
                    rows.append((ki, joint, axis))
                    keys.append(pts)
        self.n_bones = int(n_bones)
        self._bind_local = np.asarray(bind_local, dtype=np.float64)
        self._rows = np.array(rows, dtype=np.int32).reshape(-1, 3)
        width = max((len(k) for k in keys), default=1)
        self._frames = np.full((len(keys), width), np.inf, dtype=np.float64)
        self._values = np.zeros((len(keys), width), dtype=np.float64)
        self._lens = np.array([len(k) for k in keys], dtype=np.int64).reshape(-1)
        for i, pts in enumerate(keys):
            self._frames[i, :len(pts)] = [p[0] for p in pts]
            self._values[i, :len(pts)] = [p[1] for p in pts]
        self.driven = tuple(sorted({int(j) for _k, j, _a in rows}))
        self.last_frame = int(self._frames[np.isfinite(self._frames)].max()) if len(keys) else 0

    def __len__(self) -> int:
        return len(self._lens)

    def eval(self, frame: float):
        """``(rot(n,3), loc(n,3))`` at ``frame`` — undriven channels at their default."""
        rot = np.zeros((self.n_bones, 3), dtype=np.float64)
        loc = self._bind_local.copy()
        if not len(self._lens):
            return rot, loc
        hi = (self._frames < frame).sum(axis=1)
        lo = np.maximum(hi - 1, 0)
        hi = np.minimum(hi, self._lens - 1)
        idx = np.arange(len(self._lens))
        f0, f1 = self._frames[idx, lo], self._frames[idx, hi]
        v0, v1 = self._values[idx, lo], self._values[idx, hi]
        den = f1 - f0
        t = np.where(den > 0, (frame - f0) / np.where(den > 0, den, 1.0), 0.0)
        val = v0 + (v1 - v0) * t
        is_loc = self._rows[:, 0] == 1
        rot[self._rows[~is_loc, 1], self._rows[~is_loc, 2]] = val[~is_loc]
        loc[self._rows[is_loc, 1], self._rows[is_loc, 2]] = val[is_loc]
        return rot, loc


# --------------------------------------------------------------------------- #
# skinning
# --------------------------------------------------------------------------- #
class SkinBinding:
    """One mesh group's vertices as ``(position, bones, weights)`` arrays.

    ``weights`` are NORMALISED (each skinned vertex sums to 1.0), because that is what
    the deform blends — `stretch.skin_group` divides by the surviving total too. A
    vertex whose influences were all dropped carries ``weights`` of 0 and ``rest`` 1,
    which leaves it at its bind position exactly as the scalar reference does.
    """

    def __init__(self, positions: np.ndarray, bones: np.ndarray,
                 weights: np.ndarray) -> None:
        self.positions = np.ascontiguousarray(positions, dtype=np.float64)
        self.bones = np.ascontiguousarray(bones, dtype=np.int64)
        self.weights = np.ascontiguousarray(weights, dtype=np.float64)
        self.rest = 1.0 - self.weights.sum(axis=1)
        self._homog = np.concatenate(
            [self.positions, np.ones((len(self.positions), 1))], axis=1)

    @classmethod
    def from_vertices(cls, vertices, n_bones: int) -> "SkinBinding":
        pos = np.array([(v["x"], v["y"], v["z"]) for v in vertices], dtype=np.float64) \
            if vertices else np.zeros((0, 3))
        pal = []
        width = 1
        for v in vertices:
            infl = [(int(b), float(w)) for b, w in (v.get("influences") or ())
                    if w > WEIGHT_EPS and 0 <= b < n_bones]
            pal.append(infl)
            width = max(width, len(infl))
        bones = np.zeros((len(vertices), width), dtype=np.int64)
        weights = np.zeros((len(vertices), width), dtype=np.float64)
        for i, infl in enumerate(pal):
            if not infl:
                continue
            tot = sum(w for _b, w in infl) or 1.0
            for k, (b, w) in enumerate(infl):
                bones[i, k] = b
                weights[i, k] = w / tot
        return cls(pos, bones, weights)

    @property
    def n_vertices(self) -> int:
        return len(self.positions)

    @property
    def max_influences(self) -> int:
        return self.bones.shape[1]

    def dominant(self) -> np.ndarray:
        """``(v,)`` the heaviest bone per vertex; ``-1`` for an unskinned vertex.

        This is `stretch.skin_group`'s second return value — what the tear analysis
        blames an edge on.
        """
        out = self.bones[np.arange(len(self.bones)), self.weights.argmax(axis=1)]
        return np.where(self.rest > 0.5, -1, out)

    def apply(self, deform: np.ndarray) -> np.ndarray:
        """``(v,3)`` deformed positions under ``(n,4,4)`` deform matrices.

        Blends the MATRICES and then transforms once, where `stretch.skin_group`
        transforms once per influence and blends the results. Linearity makes those
        the same quantity — ``Σ wₖ(Mₖp) == (Σ wₖMₖ)p`` — and it is measured: the two
        agree to **1.1e-13 units** over the whole native Tigrex. It is also 4.7x
        faster, because the blend collapses the (v,K,3,4) gather to one (v,3,4)
        transform, and it is what a GPU skinner does.
        """
        if not len(self.positions):
            return self.positions.copy()
        blended = np.einsum("vk,vkij->vij", self.weights, deform[self.bones][:, :, :3, :])
        out = np.einsum("vij,vj->vi", blended, self._homog)
        out += self.positions * self.rest[:, None]
        return out

    def apply_directions(self, deform: np.ndarray, dirs: np.ndarray) -> np.ndarray:
        """``(v,3)`` NORMALS carried through the same blend, re-normalised.

        The 3x3 part only: a direction has no origin, so the translation column must
        not touch it — blending the full matrix and then transforming a point at the
        normal's coordinates would drag every normal to wherever the joint moved.

        Strictly this is the inverse-transpose that a non-uniform scale would need, and
        the engine's own FK has **no scale channels at all** (a scale in a converted
        clip is one of the two things that crash MHFU outright — `monster-porting`), so
        every deform here is a rotation and a translation and the 3x3 IS its own
        inverse-transpose. Should that ever stop being true, this is the line to fix.

        Unskinned vertices keep their bind normal, exactly as :meth:`apply` keeps their
        bind position.
        """
        d = np.ascontiguousarray(dirs, dtype=np.float64).reshape(-1, 3)
        if len(d) != len(self.positions):
            raise ValueError("%d vertices but %d directions"
                             % (len(self.positions), len(d)))
        if not len(d):
            return d.copy()
        blended = np.einsum("vk,vkij->vij", self.weights,
                            deform[self.bones][:, :, :3, :3])
        out = np.einsum("vij,vj->vi", blended, d)
        out += d * self.rest[:, None]
        n = np.linalg.norm(out, axis=1, keepdims=True)
        return np.divide(out, n, out=np.zeros_like(out), where=n > 1e-12)


# --------------------------------------------------------------------------- #
# a posed instant
# --------------------------------------------------------------------------- #
@dataclass
class Pose:
    """The rig at one instant: world matrices, and the deform a vertex blends."""
    rig: Rig
    frame: float
    world: np.ndarray                       # (n,4,4)
    slot: Optional[int] = None
    _deform: Optional[np.ndarray] = None

    @property
    def deform(self) -> np.ndarray:
        if self._deform is None:
            self._deform = self.rig.deform(self.world)
        return self._deform

    @property
    def joints(self) -> np.ndarray:
        """``(n,3)`` world joint positions."""
        return self.world[:, :3, 3]

    def skin(self, binding: SkinBinding) -> np.ndarray:
        """``(v,3)`` deformed vertices of one mesh group."""
        return binding.apply(self.deform)
