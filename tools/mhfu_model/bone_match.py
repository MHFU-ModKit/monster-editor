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

    _fix_leading_root_chain(out, sw, dw)
    return out


def _fix_leading_root_chain(out: Dict[int, Optional[int]],
                            sw: Sequence[Vec3], dw: Sequence[Vec3],
                            eps: float = 1.0) -> None:
    """Correct the matching of the leading *structural root chain* in-place.

    Both skeletons store bones topologically (parent before child), so the leading
    run of bones whose bind-WORLD position is the origin is the structural root
    chain (zero-length joints). Pure position matching can't tell these apart (all
    at the same point), so the greedy matcher pairs them in storage order — which is
    WRONG when the two chains differ in length (e.g. the MHFU Tigrex host has 3
    origin bones, the MHP3rd Brute source 2): it then pairs host0↔src0, host1↔src1
    and STARVES host2 (the hip — the bone that carries the vertical positioning
    ``locY``), so the ported monster never lifts and sinks into the floor.

    The chains correspond from the TAIL (nearest the first real, distinctly-positioned
    bone), not the head: align host[Lh-1]↔src[Ls-1], host[Lh-2]↔src[Ls-2], …; any
    surplus LEADING host bones (the extra structural roots) stay unmatched (None) —
    exactly the native layout (leading empty placeholder joints). No-op when the two
    chains are the same length (1:1, unchanged) — safe for same-rig / equal-root ports.
    """
    def lead_origin(world: Sequence[Vec3]) -> int:
        c = 0
        for w in world:
            if (w[0] * w[0] + w[1] * w[1] + w[2] * w[2]) ** 0.5 < eps:
                c += 1
            else:
                break
        return c

    Lh = lead_origin(dw)
    Ls = lead_origin(sw)
    if Lh <= 0 or Ls <= 0:
        return
    k = min(Lh, Ls)
    for d in range(Lh):                       # clear current (mis)assignments
        out[d] = None
    for j in range(k):                        # re-align from the tail
        d = Lh - 1 - j
        s = Ls - 1 - j
        for dd in list(out):                  # free this src from any other dst
            if out[dd] == s:
                out[dd] = None
        out[d] = s


def fill_unmatched(bone_map: Dict[int, Optional[int]],
                   dst_parents: Sequence[int], dst_local: Sequence[Vec3],
                   max_dist: Optional[float] = None) -> Dict[int, Optional[int]]:
    """Fill ``None`` (unmatched) target joints from their nearest MATCHED neighbour.

    When the target rig has a LONGER chain than the source (e.g. the MHFU Tigrex tail
    has 5 joints but the MHP3rd Brute tail has 4), greedy 1:1 matching leaves the extra
    target joint unmatched — it then stays at bind pose while its animated neighbours
    move, kinking the chain and scrambling the chain-skinned geometry (the Brute tail
    bug). This post-pass assigns each unmatched joint the source bone of the closest
    matched joint by bind-WORLD distance, so a too-long target chain is driven
    coherently (the source's last bone covers the extra tips). Returns a NEW map.
    """
    dw = bind_world_positions(dst_parents, dst_local)
    out = dict(bone_map)
    matched = [d for d, s in out.items() if s is not None]
    if not matched:
        return out
    xs = [p[0] for p in dw]; ys = [p[1] for p in dw]; zs = [p[2] for p in dw]
    diag = ((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2
            + (max(zs) - min(zs)) ** 2) ** 0.5 if xs else 1e9
    if max_dist is None:
        max_dist = 0.5 * diag
    for d, s in list(out.items()):
        if s is not None:
            continue
        dx, dy, dz = dw[d]
        best = None
        for m in matched:
            mx, my, mz = dw[m]
            dist = ((dx - mx) ** 2 + (dy - my) ** 2 + (dz - mz) ** 2) ** 0.5
            if dist <= max_dist and (best is None or dist < best[0]):
                best = (dist, m)
        if best is not None:
            out[d] = out[best[1]]
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
