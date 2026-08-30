"""Skeleton (PAC sub-0, 0xC0000000 blob) <-> data model.

Decode logic ported verbatim from tools/skeleton.py (validated 2026-06-13 against
engine builder EU 0x088dc40c and the iOS m2jean format). See docs/ANIMATION_FORMAT.md.

`encode()` is a lossless passthrough (returns the original bytes) until the Phase 3
skeleton encoder replaces it; this keeps the round-trip byte-identical today.
"""
from __future__ import annotations

import struct

from .model import Bone, Skeleton

MAGIC = 0xC0000000
SECTION_MAGIC = 0x40000001
HDR_SIZE = 0x1C


def parse(blob: bytes) -> Skeleton:
    magic, bone_count, total_size = struct.unpack_from("<3I", blob, 0)
    if magic != MAGIC:
        raise ValueError("not a 0xC0000000 skeleton (magic=0x%08x)" % magic)
    bones = []
    o = HDR_SIZE
    for i in range(bone_count):
        if o + 0x1C > len(blob):
            break
        smag, flag, ssize = struct.unpack_from("<3I", blob, o)
        idx, parent, child, sibling = struct.unpack_from("<4i", blob, o + 0x0C)
        scale = struct.unpack_from("<3f", blob, o + 0x1C)
        rotation = struct.unpack_from("<3f", blob, o + 0x2C)
        position = struct.unpack_from("<3f", blob, o + 0x3C)
        section = blob[o:o + ssize] if (0 < ssize <= 0x400) else blob[o:o + 0x10C]
        bones.append(Bone(
            index=idx, parent=parent, child=child, sibling=sibling,
            bind_scale=scale, bind_rot=rotation, bind_pos=position,
            flag=flag, section_size=ssize, raw=section,
        ))
        if smag != SECTION_MAGIC or ssize == 0 or ssize > 0x400:
            break
        o += ssize
    return Skeleton(
        bone_count=bone_count, total_size=total_size, bones=bones,
        header=blob[:HDR_SIZE], raw=blob,
    )


def parse_p3rd(blob: bytes) -> Skeleton:
    """Parse a 0x80000000 MHP3rd skeleton blob into the Skeleton data model.

    Delegates to skeleton_p3rd.parse() which handles:
    - Section magic 0x40000002 (in addition to 0x40000001)
    - Optional extra 4-byte header word before bone sections (lobby PACs)

    The returned Skeleton is compatible with the Blender armature builder
    and exporter — same dataclass, same field layout.
    """
    from .skeleton_p3rd import parse as _p3rd
    return _p3rd(blob)


def _encode_bone(b: Bone) -> bytes:
    """Patch a bone's editable fields back into its raw section (same size).

    Keeps every non-field byte (the implementation-private matrix/aux region
    0x48..section_size) verbatim, so an unedited bone re-emits byte-identically
    and a bind-pose/tree edit changes only the intended words.
    """
    sec = bytearray(b.raw)
    if len(sec) < 0x48:
        return bytes(sec)                       # terminator / short final section
    struct.pack_into("<I", sec, 0x04, b.flag)
    struct.pack_into("<4i", sec, 0x0C, b.index, b.parent, b.child, b.sibling)
    struct.pack_into("<3f", sec, 0x1C, *b.bind_scale)
    struct.pack_into("<3f", sec, 0x2C, *b.bind_rot)
    struct.pack_into("<3f", sec, 0x3C, *b.bind_pos)
    return bytes(sec)


def assign_stream_ids(skel: Skeleton, split) -> Skeleton:
    """Set each bone's stream-id (`bone+0x50`, u16 → `joint+0x114`) so the anim FK
    partitions all bones into streams without overrunning.

    RE (2026-06-28, disasm `0x0886010c lw a2,0x50(a1)` → `0x08860110 sh a2,0x114(a0)`
    + native Tigrex `file_06185`): the stream-id is `bone+0x50`; a valid partition is
    **contiguous runs by bone index**. Native Tigrex = `{0:31, 1:9, 2:5, 3:3}`. An
    MHP3rd source skeleton ships `+0x50`=0 on every bone (all stream 0 → FK overrun
    crash), so a ported skeleton MUST be assigned before injection.

    `split` = per-stream bone counts (e.g. [31, 9, 5, 3]); must match the anim's
    stream partition. Patches each bone's raw section at +0x50 in place. Returns skel.
    """
    runs = []
    for sid, n in enumerate(split):
        runs += [sid] * n
    if len(runs) < len(skel.bones):
        runs += [len(split) - 1] * (len(skel.bones) - len(runs))   # trailing -> last stream
    for b, sid in zip(skel.bones, runs):
        if len(b.raw) >= 0x52:
            sec = bytearray(b.raw)
            struct.pack_into("<H", sec, 0x50, sid & 0xFFFF)
            b.raw = bytes(sec)
    return skel


def _default_split(n: int):
    """3-stream split mirroring native Tigrex's tail-stream sizes (…/9/5).

    A LAST RESORT. Prefer :func:`derive_stream_partition`, which reads the real
    partition off the bone tree instead of assuming the Tigrex's bone ordering.
    """
    if n >= 9 + 5 + 1:
        return [n - 14, 9, 5]
    return [n]


# --------------------------------------------------------------------------- #
# stream partition — derived from the bone tree, not assumed
# --------------------------------------------------------------------------- #
def _child_map(parents):
    kids = {}
    for i, p in enumerate(parents):
        kids.setdefault(p, []).append(i)
    return kids


def _subtree(kids, root):
    out, stack = {root}, [root]
    while stack:
        for c in kids.get(stack.pop(), ()):
            out.add(c)
            stack.append(c)
    return out


def _appendage(parents, kids, leaf, limit, taken, ceiling):
    """The subtree an extremity belongs to: climb from ``leaf`` to the highest
    ancestor whose subtree still fits in ``limit`` bones, stays inside ``ceiling``
    and does not touch an already-claimed stream.

    Always returns a COMPLETE subtree — starting from ``{leaf}`` would strand the
    leaf's own children in the body stream while the leaf moved to the end, putting
    a child ahead of its parent (13 of MHP3rd's 212 in-quest rigs hit exactly that).
    """
    best = _subtree(kids, leaf)
    if len(best) > limit or max(best) >= ceiling:
        return set()
    a = leaf
    while True:
        p = parents[a]
        if p is None or p <= 0:
            break
        st = _subtree(kids, p)
        if len(st) > limit or (st & taken) or max(st) >= ceiling:
            break
        a, best = p, st
    return best


def _appendages(parents, kids, limit, ceiling):
    """Every maximal subtree that could serve as a stream, by root bone."""
    out = {}
    for a in range(1, ceiling):
        if parents[a] is None or parents[a] < 0:
            continue
        st = _subtree(kids, a)
        if len(st) <= limit and max(st) < ceiling:
            out[a] = st
    return out


def _pick_appendage(parents, kids, bind_world, limit, taken, ceiling, want_max_z):
    """The stream at one end of the body: the subtree holding the most extreme
    bone along Z, falling back to the most extreme MULTI-bone subtree when that
    lands on a bare leaf (a one-bone stream is a legal but useless partition)."""
    live = [i for i in range(ceiling) if i not in taken]
    if not live:
        return set()
    tip = (max if want_max_z else min)(live, key=lambda i: bind_world[i][2])
    best = _appendage(parents, kids, tip, limit, taken, ceiling)
    if len(best) >= 2:
        return best
    cands = [st for st in _appendages(parents, kids, limit, ceiling).values()
             if len(st) >= 2 and not (st & taken)]
    if not cands:
        return best
    return (max if want_max_z else min)(
        cands, key=lambda st: (max if want_max_z else min)(
            bind_world[i][2] for i in st))


def derive_stream_partition(parents, bind_world, animated, streams=3,
                            max_frac=0.4):
    """Read MHFU's 3-stream bone partition off the SOURCE bone tree.

    MHFU's animation FK walks a rig in independent streams (native Tigrex =
    31 body / 9 head+neck / 5 tail), and each stream must be a **contiguous run of
    bone indices** (`bone+0x50`). Native monster rigs satisfy that because they are
    authored body-first with the head and tail as the two trailing subtrees — which
    is exactly what makes the old ``[n-14, 9, 5]`` formula appear to work.

    An MHP3rd rig is under no such obligation. The Zinogre's head sits at bones
    18..23, in the MIDDLE of its index order, so slicing off the last 14 bones puts a
    hind leg in the head stream. This finds the real subtrees geometrically — the
    stream holding the most **-Z** bone is the tail, the most **+Z** the head — and
    returns the permutation that moves them to the end.

    Returns ``(split, order)``: ``split`` = the per-stream bone counts, ``order`` =
    the new->old index permutation (identity when the rig is already native-shaped).
    Verified to reproduce the native Tigrex's [31, 9, 5] with an identity order.
    """
    n = len(parents)
    animated = max(0, min(animated, n))
    if animated < 3 or streams < 2:
        return [animated] if animated else [n], list(range(n))
    kids = _child_map(parents)
    limit = max(1, int(animated * max_frac))
    tail = _pick_appendage(parents, kids, bind_world, limit, set(), animated, False)
    head = _pick_appendage(parents, kids, bind_world, limit, tail, animated, True)
    if head & tail or len(head) + len(tail) >= animated:
        head = set()
    body = [i for i in range(animated) if i not in head and i not in tail]
    if not body:
        return [animated], list(range(n))
    split = [len(body)] + [len(x) for x in (head, tail) if x]
    order = body + sorted(head) + sorted(tail) + list(range(animated, n))
    return split, order


def reorder_bones(parents, order):
    """Apply a new->old permutation, returning the reindexed parent array.

    The engine builds its joint tree by index, so a parent must keep a LOWER index
    than its children; :func:`derive_stream_partition` only ever moves whole
    subtrees behind their parent, which preserves that.
    """
    pos = {old: new for new, old in enumerate(order)}
    out = []
    for old in order:
        p = parents[old]
        out.append(pos[p] if p is not None and p >= 0 and p in pos else -1)
    return out


def p3rd_to_mhfu(p3rd_blob: bytes, split=None, lead_pad: int = 0,
                 order=None, src_animated: int = None, reparent=None) -> bytes:
    """Convert a MHP3rd (0x80000000) skeleton to a native MHFU 0xC0000000 skeleton.

    The MHP3rd skeleton uses **0x5C** bone sections; the MHFU engine's joint builder
    needs **0x10C** sections (`0x40000001` magic). We rebuild each section from the
    parsed transform fields (idx/parent/child/sibling/scale/rot/pos at the canonical
    offsets — identical across both games) into the full 0x10C layout. The matrix/aux
    region `+0x54..0x10C` is left ZERO (the engine recomputes it at load — verified on
    native `file_06185`). Stream-ids (`+0x50`) are assigned contiguously per ``split``
    so the anim FK partitions cleanly (see :func:`assign_stream_ids`).

    ``lead_pad`` prepends N zero-length **placeholder origin bones** after which the
    source bones follow (a single-child chain root->ph1->...->src_root). The native
    big-mon OVERLAY hardcodes which joint index is the hip/ground anchor (Tigrex = joint
    2, after a 3-bone leading-origin chain); a source skeleton with a SHORTER leading
    chain lands its hip at the wrong joint → the overlay's lift never reaches it and the
    body sinks. Pad the leading chain to match the host's leading-origin count so the
    hip aligns (the caller shifts the anim/skin by the same ``lead_pad``).

    This is the source-skeleton (no-down-rig) path: a ported monster ships its OWN
    skeleton, so its own anim drives it 1:1 (the joint count is data-driven — the engine
    `malloc`s `bone_count*0x250` with no clamp). Returns the 0xC0000000 skeleton bytes.
    """
    from . import skeleton_p3rd as _skp
    ssk = _skp.parse(p3rd_blob)
    src = ssk.bones
    nsrc = len(src)
    n = nsrc + lead_pad
    src_bone_count = struct.unpack_from("<I", p3rd_blob, 4)[0]
    # ⚠️ +0x1C is NOT an animated-bone count (it reads 0x40000001 on most MHP3rd
    # rigs, and a plausible-but-wrong 46 on the Zinogre). Default to EVERY bone so
    # the partition spans the whole FK walk; the caller overrides via src_animated.
    if src_animated is None:
        src_animated = nsrc
    animated = (src_animated if 0 < src_animated <= nsrc else nsrc) + lead_pad
    sp = split or _default_split(animated)
    if sum(sp) != animated:
        # The FK walks `animated` joints across the stream sections; a partition
        # that does not cover exactly that many runs off the end of one.
        raise ValueError("split %r sums to %d, skeleton declares %d animated bones"
                         % (sp, sum(sp), animated))

    runs = []
    for sid, c in enumerate(sp):
        runs += [sid] * c
    while len(runs) < n:
        runs.append(len(sp))                       # non-animated tail -> own stream id

    # unified bone list: lead_pad placeholders (origin chain) then the source bones,
    # optionally REORDERED so each anim stream is a contiguous run of bone indices
    # (see derive_stream_partition — MHP3rd rigs are not authored native-shaped).
    perm = list(order) if order else list(range(nsrc))
    if sorted(perm) != list(range(nsrc)):
        raise ValueError("order must be a permutation of the %d source bones" % nsrc)
    pos = {old_i: new_i + lead_pad for new_i, old_i in enumerate(perm)}

    def sh(v):                     # source bone index -> output joint index
        return pos.get(v, -1) if v is not None and v >= 0 else -1

    parents = [(j - 1 if j > 0 else -1) for j in range(lead_pad)]
    for old_i in perm:
        p = src[old_i].parent
        if reparent and old_i in reparent:
            p = reparent[old_i]           # orphan root adopted; see port_p3rd
        parents.append(sh(p) if p is not None and p >= 0
                       else (lead_pad - 1 if lead_pad else -1))
    # child/sibling are fully derivable from the parent array and MUST be rebuilt
    # after a permutation (verified byte-identical to the shipped links on the native
    # Tigrex, the Brute, the Zinogre and file_05354 — the only difference is that two
    # ROOTS are not each other's sibling, which is preserved here).
    kids = {}
    for i, p in enumerate(parents):
        if p >= 0:
            kids.setdefault(p, []).append(i)
    child = [-1] * n
    sibling = [-1] * n
    for p, cs in kids.items():
        cs = sorted(cs)
        child[p] = cs[0]
        for a, b in zip(cs, cs[1:]):
            sibling[a] = b

    bl = []   # (flag, idx, parent, child, sibling, scale, rot, pos)
    for j in range(lead_pad):
        bl.append((1, j, parents[j], child[j], sibling[j],
                   (1.0, 1.0, 1.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)))
    for new_i, old_i in enumerate(perm):
        b = src[old_i]
        j = new_i + lead_pad
        bl.append((b.flag if getattr(b, "flag", 0) else 1,
                   j, parents[j], child[j], sibling[j],
                   tuple(b.bind_scale), tuple(b.bind_rot), tuple(b.bind_pos)))

    secs = bytearray()
    for i, (flag, idx, par, chld, sib, sc, ro, po) in enumerate(bl):
        sec = bytearray(0x10C)
        struct.pack_into("<I", sec, 0x00, SECTION_MAGIC)
        struct.pack_into("<I", sec, 0x04, flag)
        struct.pack_into("<I", sec, 0x08, 0x10C)
        struct.pack_into("<4i", sec, 0x0C, idx, par, chld, sib)
        struct.pack_into("<3f", sec, 0x1C, *sc)
        struct.pack_into("<f",  sec, 0x28, 1.0)
        struct.pack_into("<3f", sec, 0x2C, *ro)
        struct.pack_into("<f",  sec, 0x38, 1.0)
        struct.pack_into("<3f", sec, 0x3C, *po)
        struct.pack_into("<f",  sec, 0x48, 1.0)
        struct.pack_into("<i",  sec, 0x4C, -1)
        struct.pack_into("<I",  sec, 0x50, runs[i] & 0xFFFF)
        secs += sec

    total = 0x20 + len(secs)
    hdr = struct.pack("<8I", MAGIC, src_bone_count + lead_pad, total, 0, 2, 0x14, 0, animated)
    return bytes(hdr + secs)


def encode(skel: Skeleton) -> bytes:
    """Serialize a skeleton blob from the data model.

    In-place field patching (the Phase 3 "safe case"): header (verbatim) + per-bone
    sections rebuilt from fields + the opaque bytes after the last parsed section.
    Verified byte-identical for unedited skeletons across all 49 big-monster PACs;
    bind-pose and tree-link edits (same bone count) serialize correctly without
    disturbing the implementation-private matrix/aux region of each section.

    Header words (`bone_count`/`total_size`) are taken verbatim from the source —
    a genuine bone-count change must update the animation pack AND those words in
    lockstep (the validator blocks an unmatched change), so this encoder targets
    the safe same-count case and leaves the header authoritative.
    """
    header = bytearray(skel.header) if skel.header else bytearray(
        struct.pack("<3I", MAGIC, len(skel.bones), 0))
    # If this skeleton came from a MHP3rd PAC (0x80000000 magic), flip the magic to
    # MHFU's 0xC0000000 so the output PAC is recognised by the MHFU engine and by
    # MonsterPac.role().  All other header bytes (bone_count, total_size, extra p3rd
    # word at +0x1C if present) are preserved verbatim.
    if len(header) >= 4 and struct.unpack_from("<I", header, 0)[0] == 0x80000000:
        struct.pack_into("<I", header, 0, MAGIC)
    sections = b"".join(_encode_bone(b) for b in skel.bones)
    body = bytearray(header) + sections
    if skel.raw and len(skel.raw) > len(body):
        body += skel.raw[len(body):]              # preserve the opaque tail
    return bytes(body)
