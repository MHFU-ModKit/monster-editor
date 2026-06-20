"""Skeleton-to-skeleton bone correspondence for cross-game animation porting.

When a monster is ported from another game (e.g. MHP3rd) its animation keyframes
are indexed by the SOURCE skeleton's bone order, but the engine drives the TARGET
(MHFU) skeleton's joints. If the two skeletons differ in bone count, order, or have
different leading "structural root" bones, copying tracks 1:1 by index applies each
bone's motion to the wrong joint -> the mesh tears.

This module builds the correspondence ``target_joint -> source_bone`` so the anim
encoder can place each track on the joint it actually drives (and leave unmatched
target joints static). It matches by bind-world position (the same creature posed
the same way in both games) with a tree-depth tie-breaker, greedy nearest-first.

Generalizable: works for any (source, target) skeleton pair, not just Brute/Tigrex.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

Vec3 = Tuple[float, float, float]


# --------------------------------------------------------------------------- #
# tree helpers
# --------------------------------------------------------------------------- #
def bind_world_positions(parents: Sequence[int],
                         local_pos: Sequence[Vec3]) -> List[Vec3]:
    """Accumulate each bone's bind-WORLD position down the parent chain.

    Assumes ``bind_rot == 0`` (true for MHFU 0xC0000000 skeletons — rotation is
    anim-only), so world = sum of local offsets from the root.
    """
    n = len(parents)
    out: List[Optional[Vec3]] = [None] * n

    def resolve(i: int, _seen=None) -> Vec3:
        cached = out[i]
        if cached is not None:
            return cached
        if _seen is None:
            _seen = set()
        if i in _seen:                      # parent cycle / self-parent: treat as root
            return tuple(local_pos[i])
        _seen.add(i)
        lx, ly, lz = local_pos[i]
        p = parents[i]
        if p < 0 or p >= n:
            w = (lx, ly, lz)
        else:
            px, py, pz = resolve(p, _seen)
            w = (px + lx, py + ly, pz + lz)
        out[i] = w
        return w

    for i in range(n):
        resolve(i)
    return [w for w in out]  # type: ignore[misc]


def depth_of(parents: Sequence[int]) -> List[int]:
    """Tree depth (root = 0) for each bone."""
    n = len(parents)
    out = [-1] * n

    def d(i: int, _seen=None) -> int:
        if out[i] >= 0:
            return out[i]
        if _seen is None:
            _seen = set()
        if i in _seen:
            return 0
        _seen.add(i)
        p = parents[i]
        out[i] = 0 if (p < 0 or p >= n) else d(p, _seen) + 1
        return out[i]

    for i in range(n):
        d(i)
    return out


def walk_order(parents: Sequence[int]) -> List[int]:
    """Engine joint-walk order. MHFU 0xC0000000 skeletons store bones so that a
    parent always precedes its children (topological), and the engine binds anim
    sections in this storage order — so storage order 0..N-1 IS the walk order.
    Exposed as a function so callers don't hard-code the assumption.
    """
    return list(range(len(parents)))


# --------------------------------------------------------------------------- #
# the matcher
# --------------------------------------------------------------------------- #
def match_skeletons(src_parents: Sequence[int], src_local: Sequence[Vec3],
                    dst_parents: Sequence[int], dst_local: Sequence[Vec3],
                    max_dist: Optional[float] = None,
                    depth_penalty: float = 50.0) -> Dict[int, Optional[int]]:
    """Return ``{dst_idx: src_idx or None}`` matching each target (dst) joint to
    the source (src) bone it corresponds to.

    Match metric = euclidean bind-world distance + ``depth_penalty`` per level of
    tree-depth mismatch. Greedy: assign closest (dst,src) pairs first; each src
    bone is used at most once. A dst joint with no src within ``max_dist`` (or all
    src bones already taken) maps to ``None`` (stays static / empty in the anim).

    ``max_dist`` defaults to 25% of the source skeleton's bounding diagonal — large
    enough to absorb minor per-game proportion differences, small enough to leave
    genuinely-absent joints unmatched.
    """
    sw = bind_world_positions(src_parents, src_local)
    dw = bind_world_positions(dst_parents, dst_local)
    sd = depth_of(src_parents)
    dd = depth_of(dst_parents)

    if max_dist is None:
        xs = [p[0] for p in sw]; ys = [p[1] for p in sw]; zs = [p[2] for p in sw]
        if xs:
            diag = ((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2
                    + (max(zs) - min(zs)) ** 2) ** 0.5
            max_dist = 0.25 * diag
        else:
            max_dist = 1e9

    # all candidate (cost, dst, src) pairs
    cands: List[Tuple[float, int, int]] = []
    for di in range(len(dw)):
        dx, dy, dz = dw[di]
        for si in range(len(sw)):
            sx, sy, sz = sw[si]
            dist = ((dx - sx) ** 2 + (dy - sy) ** 2 + (dz - sz) ** 2) ** 0.5
            cost = dist + depth_penalty * abs(dd[di] - sd[si])
            if dist <= max_dist:
                cands.append((cost, di, si))
    cands.sort()

    out: Dict[int, Optional[int]] = {di: None for di in range(len(dw))}
    used_src = set()
    for cost, di, si in cands:
        if out[di] is None and si not in used_src:
            out[di] = si
            used_src.add(si)
    return out


def remap_tracks(tracks: list, bone_map: Dict[int, Optional[int]],
                 n_dst: int, empty_factory=None) -> list:
    """Reorder a source ``tracks`` list into target-joint order via ``bone_map``.

    Returns a list of length ``n_dst`` where entry ``d`` = ``tracks[bone_map[d]]``
    or ``empty_factory()`` (default ``None``) when ``bone_map[d]`` is None / OOB.
    """
    out = []
    for d in range(n_dst):
        s = bone_map.get(d)
        if s is not None and 0 <= s < len(tracks):
            out.append(tracks[s])
        else:
            out.append(empty_factory() if empty_factory else None)
    return out


# --------------------------------------------------------------------------- #
# skeleton-object convenience (mhfu_model.model.Skeleton)
# --------------------------------------------------------------------------- #
def _skel_arrays(skel) -> Tuple[List[int], List[Vec3]]:
    """Extract (parents, local_pos) from a mhfu_model.model.Skeleton."""
    bones = list(skel.bones)
    return ([b.parent for b in bones], [tuple(b.bind_pos) for b in bones])


def match_skeleton_objects(src_skel, dst_skel, **kw) -> Dict[int, Optional[int]]:
    """match_skeletons() taking two mhfu_model.model.Skeleton objects."""
    sp, sl = _skel_arrays(src_skel)
    dp, dl = _skel_arrays(dst_skel)
    return match_skeletons(sp, sl, dp, dl, **kw)
