"""How far does a clip pull the mesh apart? — the measurement a render only hints at.

A port can be structurally perfect and still look torn: geometry whose vertices ride
joints that separate under animation stretches between them. Judging that by eye costs
a render per guess and is subjective. This skins the mesh at bind and at a frame with
the ENGINE's deform (``anim_world[j] @ bind_world[j]⁻¹``, blended by the vertex's own
palette weights) and reports how much each edge GREW, in world units.

🔴 **Rank by absolute growth, never by ratio.** A ratio puts a 2-unit sliver that became
20 above a 300-unit edge that became 900; the sliver is invisible and the 600-unit tear
is the whole problem. Ranked by ratio the native Tigrex looks as bad as a broken port.

🔴 **And magnitude alone does not prove a defect.** Swept over every clip, the native
Tigrex peaks at **239 u** — at an elbow, which is what blend skinning does. A Zinogre
built with the anim bone-offset one too high peaked *lower*, at 208 u, and was badly
broken. What separated them was WHERE: native tears at limb joints, the broken build
tore at the **waist**, every time. Hence `cross_fork_tear` below, which is the number
that actually discriminates:

    native Tigrex        0 u      (no vertex even blends across its fork)
    Zinogre offset 1   208 u      <- the visible "L-shaped back"
    Zinogre offset 0    52 u
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from . import convert as C
from .p3rd_anim_map import body_fork


# --------------------------------------------------------------------------- #
# minimal 4x4 maths (no numpy dependency in this package)
# --------------------------------------------------------------------------- #
def _mul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)] for i in range(4)]


def _euler_xyz(rx, ry, rz):
    """Same convention `blender_mhfu/fk_bake.py` bakes: Euler('XYZ') == Rz @ Ry @ Rx."""
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    return [[cy * cz, cz * sx * sy - cx * sz, cx * cz * sy + sx * sz, 0.0],
            [cy * sz, cx * cz + sx * sy * sz, -cz * sx + cx * sy * sz, 0.0],
            [-sy,     cy * sx,                cy * cx,                0.0],
            [0.0, 0.0, 0.0, 1.0]]


def _xform(m, p):
    return (m[0][0] * p[0] + m[0][1] * p[1] + m[0][2] * p[2] + m[0][3],
            m[1][0] * p[0] + m[1][1] * p[1] + m[1][2] * p[2] + m[1][3],
            m[2][0] * p[0] + m[2][1] * p[1] + m[2][2] * p[2] + m[2][3])


# --------------------------------------------------------------------------- #
# sampling / FK
# --------------------------------------------------------------------------- #
def sample(anim) -> Dict[int, dict]:
    """joint -> {"rot"/"loc": {axis: [(frame, value), ...]}} — positional.

    ⚠️ MHFU's own flat clips are positional (track i == joint i). An MHP3rd SOURCE
    moveset is NOT — map it through `p3rd_anim_map` first.
    """
    out: Dict[int, dict] = {}
    for j, tr in enumerate(anim.tracks):
        per = out.setdefault(j, {"rot": {}, "loc": {}})
        for ch in tr.channels:
            kind, axis = C.channel_kind(ch.type)
            if kind in ("rot", "loc") and ch.keyframes:
                per[kind][axis] = sorted(
                    (kf.frame, C.dequantize(kind, kf.value)) for kf in ch.keyframes)
    return out


def at(pts, frame, default=0.0):
    if not pts:
        return default
    if frame <= pts[0][0]:
        return pts[0][1]
    if frame >= pts[-1][0]:
        return pts[-1][1]
    for (f0, v0), (f1, v1) in zip(pts, pts[1:]):
        if f0 <= frame <= f1:
            t = 0.0 if f1 == f0 else (frame - f0) / (f1 - f0)
            return v0 + (v1 - v0) * t
    return pts[-1][1]


def world_matrices(skel, samples=None, frame=0):
    """joint -> 4x4 world matrix. `samples=None` gives the BIND pose."""
    by = {b.index: b for b in skel.bones}
    out: Dict[int, list] = {}

    def resolve(i, seen=()):
        if i in out:
            return out[i]
        b = by[i]
        s = (samples or {}).get(i, {"rot": {}, "loc": {}})
        loc = [at(s["loc"].get(a, []), frame, b.bind_pos[a]) for a in (0, 1, 2)]
        rot = [at(s["rot"].get(a, []), frame) for a in (0, 1, 2)]
        local = _euler_xyz(*rot)
        local[0][3], local[1][3], local[2][3] = loc
        par = b.parent if (b.parent in by and b.parent not in seen) else None
        out[i] = local if par is None else _mul(resolve(par, seen + (i,)), local)
        return out[i]

    for b in skel.bones:
        resolve(b.index)
    return out


def deform_matrices(skel, samples, frame):
    """joint -> ``posed @ bind⁻¹``. bind_rot is all zeros, so bind is pure translation."""
    bind = world_matrices(skel, None)
    posed = world_matrices(skel, samples, frame)
    out = {}
    for j, bm in bind.items():
        inv = [[1.0 if r == c else 0.0 for c in range(4)] for r in range(4)]
        inv[0][3], inv[1][3], inv[2][3] = -bm[0][3], -bm[1][3], -bm[2][3]
        out[j] = _mul(posed[j], inv)
    return out


def skin_group(group, deform):
    """[(posed_xyz, dominant_joint), ...] for one mesh group, in vertex order."""
    out = []
    for v in group.vertices:
        p = (v["x"], v["y"], v["z"])
        infl = [(b, w) for b, w in (v.get("influences") or ())
                if w > 1e-4 and b in deform]
        if not infl:
            out.append((p, -1))
            continue
        tot = sum(w for _b, w in infl) or 1.0
        acc = [0.0, 0.0, 0.0]
        for b, w in infl:
            q = _xform(deform[b], p)
            for k in range(3):
                acc[k] += q[k] * w / tot
        out.append((tuple(acc), max(infl, key=lambda t: t[1])[0]))
    return out


def _edges(group):
    for f in group.faces:
        idx = (f["v1"], f["v2"], f["v3"])
        if max(idx) >= len(group.vertices):
            continue
        for x, y in ((0, 1), (1, 2), (2, 0)):
            yield idx[x], idx[y]


# --------------------------------------------------------------------------- #
# the two measurements
# --------------------------------------------------------------------------- #
def worst_growth(model, skel, clip, frame):
    """(growth_in_units, (group, joint_a, joint_b)) — the single most stretched edge."""
    deform = deform_matrices(skel, sample(clip), frame)
    best, where = 0.0, None
    for g in model.mesh_groups:
        sk = skin_group(g, deform)
        for i, j in _edges(g):
            pa, pb = g.vertices[i], g.vertices[j]
            d0 = math.dist((pa["x"], pa["y"], pa["z"]), (pb["x"], pb["y"], pb["z"]))
            d1 = math.dist(sk[i][0], sk[j][0])
            if d1 - d0 > best:
                best, where = d1 - d0, (g.index, sk[i][1], sk[j][1])
    return best, where


def branch_labels(parents: Sequence[int], fork: int) -> Dict[int, int]:
    """joint -> which child-subtree of `fork` it is in. Absent = at or above the fork."""
    kids: Dict[int, List[int]] = {}
    for i, p in enumerate(parents):
        if p is not None and p >= 0:
            kids.setdefault(p, []).append(i)
    lab: Dict[int, int] = {}
    for c in kids.get(fork, []):
        stack = [c]
        while stack:
            j = stack.pop()
            lab[j] = c
            stack += kids.get(j, [])
    return lab


def cross_fork_tear(model, skel, clips, fractions=(0.25, 0.5, 0.75)):
    """Worst stretch of an edge whose ends sit in DIFFERENT branches of the body fork.

    This is the number that catches a location channel landing below the fork — the
    failure that lifts one half of the animal and leaves the waist to span the gap.
    A healthy build is near zero even while its limb joints stretch 200+ units.

    Returns ``(units, (slot, frame, joint_a, joint_b))`` or ``(0.0, None)``.
    """
    parents = [b.parent for b in skel.bones]
    fork = body_fork(parents)
    lab = branch_labels(parents, fork)
    best, where = 0.0, None
    for clip in clips:
        last = max((kf.frame for tr in clip.tracks for ch in tr.channels
                    for kf in ch.keyframes), default=0)
        if last < 4:
            continue
        for frac in fractions:
            frame = int(last * frac)
            deform = deform_matrices(skel, sample(clip), frame)
            for g in model.mesh_groups:
                sk = skin_group(g, deform)
                for i, j in _edges(g):
                    ja, jb = sk[i][1], sk[j][1]
                    if ja < 0 or jb < 0:
                        continue
                    if lab.get(ja) is None or lab.get(jb) is None or lab[ja] == lab[jb]:
                        continue
                    pa, pb = g.vertices[i], g.vertices[j]
                    d0 = math.dist((pa["x"], pa["y"], pa["z"]), (pb["x"], pb["y"], pb["z"]))
                    d1 = math.dist(sk[i][0], sk[j][0])
                    if d1 - d0 > best:
                        best, where = d1 - d0, (clip.slot, frame, ja, jb)
    return best, where
