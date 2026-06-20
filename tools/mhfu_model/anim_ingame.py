"""MHFU **in-game** animation codec (PAC sub-3, the recursive 3-stream format).

This is the format the engine's per-frame interpolator actually walks
(`0x0885fa0c → 08863198 → 088630d0 → 08863668`). It is DISTINCT from the flat
lobby / MHP3rd pack handled by ``anim.py`` — see docs/ANIMATION_FORMAT.md
("MHFU in-game (0x38) anim").

Container layout (decoded byte-exact on Tigrex ``file_06185`` sub[3]):

    +0x00  u32 magic          0x64
    +0x04  u32 hsize          0x38
    +0x08  5 × (u32 0x64, u32 stream_table_offset)   # the 5 sub-stream tables
    +0x30  u32 0
    +0x34  u32[num_slots]     MAIN stream slot table        (stream index 0)
    <stream_table_offset[k]>  u32[num_slots]  sub-stream k  (k = 0..4)
    <data>                    per-stream packed block regions

So there are up to **6 parallel slot tables** (main + 5 sub). In practice a
monster uses a subset: Tigrex populates main + sub[1] + sub[3] (the empty ones
are all-0xFFFFFFFF). Each animation *clip* = one block per **populated** stream;
the streams **partition the skeleton's animated bones** (Tigrex 31 + 9 + 5 = 45).

Every level shares the same recursive header ``{0x80000000|tag, count, size}``
(``size`` = total bytes incl. header):

    Block (per stream/clip)   tag=0x80000002  count=bone_count  size
                              + u32 loop  + f32 loop_start         (header 0x14)
        BoneSection           tag=0x80000000|channel_mask  count=nchan  size  (0x0C)
            Channel           tag=0x80000000|channel_type  count=nkf    size  (0x0C)
                Keyframe      s16 value, s16 frame, s16 ease_in, s16 ease_out  (8 B)

``parse_ingame`` → :class:`InGameAnim`; ``encode_ingame`` reproduces the source
bytes exactly for an unmodified parse (validated on file_06185).
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

FLAG = 0x80000000
EMPTY = 0xFFFFFFFF
BLOCK_TAG = 0x80000002          # the per-stream/clip block tag

# transform-channel bits (low half of a bone-section mask / channel type)
CHANNEL_BITS = [
    (0x008, "rotX"), (0x010, "rotY"), (0x020, "rotZ"),
    (0x040, "locX"), (0x080, "locY"), (0x100, "locZ"),
    (0x200, "sclX"), (0x400, "sclY"), (0x800, "sclZ"),
]


# --------------------------------------------------------------------------- #
# data model
# --------------------------------------------------------------------------- #
@dataclass
class Keyframe:
    value: int
    frame: int
    ease_in: int = 0
    ease_out: int = 0


@dataclass
class Channel:
    ctype: int                  # full tag incl. FLAG + extra bits (e.g. 0x80120008)
    keyframes: List[Keyframe] = field(default_factory=list)


@dataclass
class BoneSection:
    mask: int                   # full tag incl. FLAG (e.g. 0x800001f8)
    channels: List[Channel] = field(default_factory=list)


@dataclass
class Block:
    """One stream's block for one clip (a bone-partition's keyframes)."""
    tag: int = BLOCK_TAG
    loop: int = 0
    loop_start: float = 0.0
    bones: List[BoneSection] = field(default_factory=list)


@dataclass
class Stream:
    """One slot table + its clips, keyed by slot index."""
    table_offset: int                       # absolute byte offset of the table
    clips: Dict[int, Block] = field(default_factory=dict)
    # original on-disk offset each slot pointed at (preserves aliasing/order)
    slot_offsets: Dict[int, int] = field(default_factory=dict)


@dataclass
class InGameAnim:
    magic: int = 0x64
    hsize: int = 0x38
    num_slots: int = 100
    pair_magic: int = 0x64
    word30: int = 0
    # stream 0 = main (@0x34); streams 1..5 = the 5 header sub-tables
    streams: List[Stream] = field(default_factory=list)
    raw: Optional[bytes] = None


# --------------------------------------------------------------------------- #
# recursive block parse / encode
# --------------------------------------------------------------------------- #
def _parse_block(b: bytes, o: int) -> Block:
    tag, nbones, size = struct.unpack_from("<3I", b, o)
    loop = struct.unpack_from("<I", b, o + 0x0C)[0]
    loop_start = struct.unpack_from("<f", b, o + 0x10)[0]
    bones: List[BoneSection] = []
    p = o + 0x14
    end = o + size
    for _ in range(nbones):
        bmask, nch, bsz = struct.unpack_from("<3I", b, p)
        chans: List[Channel] = []
        c = p + 0x0C
        for _c in range(nch):
            ctag, nkf, csz = struct.unpack_from("<3I", b, c)
            kfs = [Keyframe(*struct.unpack_from("<4h", b, c + 0x0C + k * 8))
                   for k in range(nkf)]
            chans.append(Channel(ctype=ctag, keyframes=kfs))
            c += csz
        bones.append(BoneSection(mask=bmask, channels=chans))
        p += bsz
    assert p == end, f"block @{o:#x} bone walk ended {p:#x} != {end:#x}"
    return Block(tag=tag, loop=loop, loop_start=loop_start, bones=bones)


def _encode_channel(ch: Channel) -> bytes:
    nkf = len(ch.keyframes)
    out = bytearray(struct.pack("<3I", ch.ctype, nkf, 0x0C + nkf * 8))
    for kf in ch.keyframes:
        out += struct.pack("<4h", _s16(kf.value), _s16(kf.frame),
                           _s16(kf.ease_in), _s16(kf.ease_out))
    return bytes(out)


def _encode_bone(bn: BoneSection) -> bytes:
    body = b"".join(_encode_channel(c) for c in bn.channels)
    return struct.pack("<3I", bn.mask, len(bn.channels), 0x0C + len(body)) + body


def _encode_block(bl: Block) -> bytes:
    body = b"".join(_encode_bone(bn) for bn in bl.bones)
    size = 0x14 + len(body)
    head = struct.pack("<3I", bl.tag, len(bl.bones), size) + \
        struct.pack("<I", bl.loop) + struct.pack("<f", bl.loop_start)
    return head + body


def _s16(v: int) -> int:
    return max(-32768, min(32767, int(v)))


# --------------------------------------------------------------------------- #
# container parse / encode
# --------------------------------------------------------------------------- #
def parse_ingame(blob: bytes) -> InGameAnim:
    magic, hsize, pmagic1 = struct.unpack_from("<3I", blob, 0)
    pairs = [struct.unpack_from("<2I", blob, 0x08 + i * 8) for i in range(5)]
    word30 = struct.unpack_from("<I", blob, 0x30)[0]
    # Layout: header(0x34) + main table(100) + one 0xFFFFFFFF pad word + the 5
    # sub-tables(100 each). So main is [0x34, 0x34+ns*4) and sub0 begins one pad
    # word later. num_slots derived from the first declared sub-table offset.
    sub_offs = [off for _m, off in pairs]
    main_off = 0x34
    num_slots = (sub_offs[0] - main_off - 4) // 4
    table_offs = [main_off] + sub_offs

    streams: List[Stream] = []
    for toff in table_offs:
        table = struct.unpack_from("<%dI" % num_slots, blob, toff)
        st = Stream(table_offset=toff)
        cache: Dict[int, Block] = {}
        for slot, off in enumerate(table):
            if off == EMPTY or not (0 < off < len(blob)):
                continue
            st.slot_offsets[slot] = off
            if off not in cache:
                cache[off] = _parse_block(blob, off)
            st.clips[slot] = cache[off]
        streams.append(st)

    return InGameAnim(magic=magic, hsize=hsize, num_slots=num_slots,
                      pair_magic=pmagic1, word30=word30, streams=streams, raw=blob)


def encode_ingame(a: InGameAnim) -> bytes:
    """Serialize. Byte-exact for an unmodified parse; engine-valid for edits.

    The header + all six slot tables occupy a fixed prefix
    ``0x34 + 6*num_slots*4``; blocks are packed per-stream after it, in the order
    main, sub0..sub4, preserving intra-stream aliasing (each unique source offset
    emitted once, slots sharing it repointed to the same new offset).
    """
    ns = a.num_slots
    # Layout: header(0x34) + main(ns) + one pad word + 5 sub-tables(ns each).
    main_off = 0x34
    sub0 = main_off + ns * 4 + 4                    # +4 = the pad word
    sub_offs = [sub0 + ns * 4 * i for i in range(5)]
    prefix_end = sub_offs[-1] + ns * 4

    # header: magic, hsize, 5×(pair_magic, sub_offset), word30  → exactly 0x34 B
    head = bytearray(struct.pack("<2I", a.magic, a.hsize))
    for off in sub_offs:
        head += struct.pack("<2I", a.pair_magic, off)
    head += struct.pack("<I", a.word30)
    assert len(head) == main_off, (len(head), main_off)

    tables = [[EMPTY] * ns for _ in range(6)]      # main + 5 sub
    body = bytearray()
    pos = prefix_end

    for si, st in enumerate(a.streams):
        if not st.clips:
            continue
        # group slots by shared source block (preserve aliasing); emit in the
        # order blocks first appear by ascending original offset when available.
        order: List[int] = []
        groups: Dict[int, List[int]] = {}
        for slot in sorted(st.clips):
            key = st.slot_offsets.get(slot, id(st.clips[slot]))
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append(slot)
        # stable order by original offset if we have it
        if all(isinstance(k, int) and k in st.slot_offsets.values() for k in order):
            order.sort()
        for key in order:
            slots = groups[key]
            blk = _encode_block(st.clips[slots[0]])
            blk_off = pos
            body += blk
            pos += len(blk)
            for slot in slots:
                tables[si][slot] = blk_off

    out = bytearray(head)
    out += struct.pack("<%dI" % ns, *tables[0])    # main table
    out += struct.pack("<I", EMPTY)                 # pad word @ main_off+ns*4
    for ti in range(1, 6):                          # 5 sub-tables
        out += struct.pack("<%dI" % ns, *tables[ti])
    assert len(out) == prefix_end, (len(out), prefix_end)
    out += body
    return bytes(out)


# --------------------------------------------------------------------------- #
# generators
# --------------------------------------------------------------------------- #
def empty_bone() -> BoneSection:
    """A bone section with no channels → engine leaves the joint matrix UNSET.

    NOTE (proven in-game 2026-06-19): the engine only computes a joint's matrix for
    bones that have keyframe channels; an empty section leaves the matrix zeroed, so
    a rigid-skinned mesh COLLAPSES to the origin. Use :func:`rest_bone` for a visible
    bind pose. ``empty_bone`` is kept only for round-trip fidelity with native data
    (native uses it for the static root bones 0/1).
    """
    return BoneSection(mask=FLAG, channels=[])


# channel types observed in native data: FLAG | 0x120000 | <transform bit>.
# the 0x120000 nibble is the interpolation/format flag the engine expects.
_CT = 0x80120000
ROT_BITS = (0x008, 0x010, 0x020)         # rotX, rotY, rotZ


def rest_bone(end_frame: int = 180) -> BoneSection:
    """A bone section that poses the joint at its skeleton REST orientation.

    Emits the three rotation channels with identity (value 0) keyframes so the
    engine computes the joint matrix (identity rotation + the skeleton's bind_pos)
    instead of leaving it zeroed. Two keyframes (frame 0 and ``end_frame``) give the
    interpolator a proper span. Because every value is identity, this poses *any*
    bone correctly regardless of the stream→bone mapping.
    """
    chans = [Channel(ctype=_CT | bit,
                     keyframes=[Keyframe(0, 0, 0, 0), Keyframe(0, end_frame, 0, 0)])
             for bit in ROT_BITS]
    return BoneSection(mask=FLAG | 0x38, channels=chans)


def static_pose_for_bonecount(animated_bones: int, num_slots: int = 100,
                              split: Optional[List[int]] = None) -> InGameAnim:
    """Bind-pose in-game anim for a skeleton with ``animated_bones`` bones.

    ``split`` optionally partitions the animated bones across the 3 streams
    (main, sub1, sub3 — mirroring native Tigrex); it must sum to
    ``animated_bones``. Default = a single ``main`` stream covering all bones
    (the most layout-agnostic form: every bone gets an empty section → bind pose).
    """
    slots = list(range(num_slots))
    if not split:
        return make_static_pose([(animated_bones, slots)], num_slots)
    if sum(split) != animated_bones:
        raise ValueError("split %r must sum to animated_bones %d"
                         % (split, animated_bones))
    # native populates main@0, sub1@2, sub3@4 (data-model stream indices 0,2,4)
    specs: List[Tuple[int, List[int]]] = [(0, [])] * 6
    stream_idx = [0, 2, 4]
    for k, bc in enumerate(split[:3]):
        specs[stream_idx[k]] = (bc, slots) if bc else (0, [])
    return make_static_pose(specs, num_slots)


def animated_count_from_skeleton(skel_bytes: bytes) -> int:
    """Animated-bone count for a 0xC0000000 skeleton.

    Big-monster skeletons (file_06185, the converted Brute) carry a 0x20-byte
    header whose word at +0x1C is the animated-bone count (Tigrex 45, Brute 42) —
    the value the engine's anim setup uses to size the per-stream bone partition.
    A short 0x1C-header skeleton (section magic at +0x1C) has no such word; fall
    back to the bone_count field.
    """
    bone_count = struct.unpack_from("<I", skel_bytes, 4)[0]
    w1c = struct.unpack_from("<I", skel_bytes, 0x1C)[0]
    if w1c == 0x40000001:                 # section starts at +0x1C → 0x1C header
        return bone_count
    if 0 < w1c <= bone_count:             # plausible animated-bone count
        return w1c
    return bone_count


def default_split(animated: int) -> List[int]:
    """A 3-stream split mirroring native Tigrex's tail-stream sizes (…/9/5).

    The engine requires all three streams populated (a single stream null-derefs).
    For a bind pose the exact partition is immaterial (every bone falls back to the
    skeleton bind pose), so we keep native's sub1=9 / sub3=5 and give the remainder
    to main. Falls back to a single stream only if too few bones for 9+5.
    """
    if animated >= 9 + 5 + 1:
        return [animated - 14, 9, 5]
    return [animated]


def swap_anim_to_bindpose(pac_bytes: bytes, anim_index: int = 3,
                          skel_index: int = 0, split: Optional[List[int]] = None,
                          keep_size: bool = True):
    """Replace a big-monster PAC's animation sub with an in-game BIND-POSE anim.

    The canonical "make the addon's .bin animation-valid" step: derives the
    animated-bone count from the skeleton sub, builds a recursive 3-stream
    bind-pose (``static_pose_for_bonecount``), and swaps it into the anim sub.
    ``keep_size`` pads the new anim to the original sub size so the whole PAC stays
    the same total size (the proven same-size in-place inject path); set False to
    let the PAC shrink (use the relocate inject path then).

    Returns ``(pac_bytes, info)`` where info = dict(animated, split).
    """
    from .pac import MonsterPac, SubResource
    pac = MonsterPac.from_bytes(pac_bytes)
    n = animated_count_from_skeleton(pac.subs[skel_index].data)
    sp = split or default_split(n)
    if sum(sp) != n:
        raise ValueError("split %r must sum to animated count %d" % (sp, n))
    blob = encode_ingame(static_pose_for_bonecount(n, split=sp))
    if keep_size:
        blob = fit_anim_sub(blob, len(pac.subs[anim_index].data))
    subs = list(pac.subs)
    subs[anim_index] = SubResource(anim_index, blob)
    out = MonsterPac(subs=subs, tail=pac.tail).to_bytes()
    return out, {"animated": n, "split": sp}


def from_flat_anim(flat_pack, split: List[int], num_slots: int = 100) -> InGameAnim:
    """Convert a FLAT (lobby / P3rd, ``anim.py``) animation pack to the in-game
    recursive 3-stream format — the REAL-MOTION encoder.

    The flat block model and the in-game block model are the same recursive
    ``{tag,count,size} → bone → channel → keyframe`` structure (verified). The
    conversion per clip:
      * each flat bone track → an in-game bone section (mask = FLAG | track.tag),
      * channels re-sorted ASCENDING by transform bit (native in-game order:
        rot 0x08→loc 0x100→scl 0x800) — the flat pack stores loc-first,
      * each channel ctype = 0x80120000 | bit; keyframes copied verbatim,
      * the clip's bone sections PARTITIONED across the populated streams
        (main/sub1/sub3) per ``split`` (must sum to the flat clip's bone count).

    ``split`` is the per-stream bone count (e.g. [29,9,5]); it determines which
    consecutive flat bones go to which stream. The exact partition that matches the
    skeleton must be validated in-game (the rest-pose milestone) / via the runtime
    bone-remap (capture_bone_remap.py) before this produces correct motion.
    """
    bit_of = {b: i for i, (b, _n) in enumerate(CHANNEL_BITS)}

    def _ctype(c):                            # flat model uses .type, ours .ctype
        return getattr(c, "type", None) if hasattr(c, "type") else c.ctype

    # MHFU's in-game engine only supports rotation (0x08/0x10/0x20) and location
    # (0x40/0x80/0x100) channels — the engine's bit→ordinal table 0x089A5C44 maps
    # ONLY those bits. SCALE channels (0x200/0x400/0x800) index it OOB → garbage
    # joint ptr → crash (proven live: ctype 0x80120200 → table[0x200] = 0x01000000).
    # Native Tigrex anims never carry scale; we DROP scale channels on conversion.
    SUPPORTED = 0x1FF                         # bits 0x08..0x100 (rot + loc)

    def conv_track(tr) -> BoneSection:
        chans = [c for c in tr.channels if (_ctype(c) & 0xFFF) & SUPPORTED]
        chans.sort(key=lambda c: bit_of.get(_ctype(c) & 0xFFF, 99))
        out = []
        for c in chans:
            bit = _ctype(c) & SUPPORTED
            kfs = [Keyframe(k.value, k.frame, getattr(k, "ease_in", 0),
                            getattr(k, "ease_out", 0)) for k in c.keyframes]
            out.append(Channel(ctype=_CT | bit, keyframes=kfs))
        return BoneSection(mask=FLAG | (tr.tag & SUPPORTED), channels=out)

    stream_idx = [0, 2, 4]                     # main, sub1, sub3 (native layout)
    streams = [Stream(table_offset=0) for _ in range(6)]
    total = sum(split)
    for anim in flat_pack.animations:
        # The in-game joint walk iterates the HOST bone count (= sum(split)); the
        # flat clip may have FEWER tracks (e.g. Brute 43 vs host 45). Pad the
        # shortfall with rest bones (identity rotation) so every host joint has a
        # bone section — a stream shorter than its allocation desyncs the walk and
        # crashes (proven live: v20 42-bone vs host 45 → 0x088630d0 OOB).
        tracks = list(anim.tracks)[:total]
        base = 0
        for k, bc in enumerate(split):
            if bc <= 0:
                continue
            seg = tracks[base:base + bc]
            base += bc
            bones = [conv_track(t) for t in seg]
            while len(bones) < bc:                 # pad to the stream's allocation
                bones.append(rest_bone())
            blk = Block(tag=BLOCK_TAG, loop=anim.loop,
                        loop_start=float(getattr(anim, "loop_start", 0.0) or 0.0),
                        bones=bones)
            streams[stream_idx[k]].clips[anim.slot] = blk
    return InGameAnim(num_slots=num_slots, streams=streams)


def swap_anim_to_realmotion(pac_bytes: bytes, flat_pack, anim_index: int = 3,
                            skel_index: int = 0, split: Optional[List[int]] = None,
                            fill_slots: bool = True, keep_size: bool = True,
                            host_count: Optional[int] = None):
    """Replace a big-monster PAC's anim sub with REAL motion from a flat anim pack.

    The real-motion sibling of :func:`swap_anim_to_bindpose`. THE in-game joint walk
    iterates the **host** bone count (the overlay slot the monster is constructed in —
    Tigrex = 45, split 31/9/5), NOT the injected skeleton's own animated count. The
    anim bone partition, the skeleton's declared animated count, and the host must all
    AGREE or the walk runs off a stream and crashes (proven live: v20 42-bone/28-9-5 vs
    host 45 → ``0x088630d0`` OOB; v22 = 45/31-9-5 + skeleton synced fixed it).

    With ``host_count`` set (the proven path for a ported monster):
      * the bone count = ``host_count`` (split = ``default_split(host_count)`` unless
        given) — each flat clip's tracks are trimmed to / padded to ``host_count``
        (shortfall filled with rest bones by :func:`from_flat_anim`),
      * the skeleton sub's declared animated count (word ``+0x1C``) is bumped to
        ``host_count`` so the engine builds that many joints (the skeleton must have
        at least that many bones — total ``bone_count`` ``@+4``).
    Without ``host_count`` it falls back to the skeleton's own animated count (the
    pre-v22 behaviour, kept for native/same-rig monsters).

    Returns ``(pac_bytes, info)`` with info = dict(animated, skel_animated, split,
    host_count, clips, skel_synced).
    """
    import copy
    from .pac import MonsterPac, SubResource
    pac = MonsterPac.from_bytes(pac_bytes)
    skel_animated = animated_count_from_skeleton(pac.subs[skel_index].data)
    n = host_count if host_count else skel_animated
    sp = split or default_split(n)
    if sum(sp) != n:
        raise ValueError("split %r must sum to bone count %d" % (sp, n))
    # the skeleton must physically have >= n bones to back n joints
    skel_total = struct.unpack_from("<I", pac.subs[skel_index].data, 4)[0]
    if n > skel_total:
        raise ValueError("host_count %d > skeleton bone_count %d — too few bones"
                         % (n, skel_total))
    fp = copy.deepcopy(flat_pack)
    for a in fp.animations:
        if len(a.tracks) > n:
            a.tracks = a.tracks[:n]
        a.bone_count = len(a.tracks)
    ig = from_flat_anim(fp, split=sp)          # pads short clips to n with rest bones
    if fill_slots:
        for st in ig.streams:
            if st.clips:
                base = st.clips[min(st.clips)]
                for s in range(ig.num_slots):
                    st.clips.setdefault(s, base)
    blob = encode_ingame(ig)
    if keep_size:
        blob = fit_anim_sub(blob, len(pac.subs[anim_index].data))
    subs = list(pac.subs)
    subs[anim_index] = SubResource(anim_index, blob)
    # sync the skeleton's declared animated count to the bone count the walk uses
    skel_synced = False
    if skel_animated != n:
        sk = bytearray(pac.subs[skel_index].data)
        # only a 0x20-header skeleton carries the count word @+0x1C (a 0x1C-header
        # skeleton has a section magic there — leave it alone)
        if struct.unpack_from("<I", sk, 0x1C)[0] != 0x40000001:
            struct.pack_into("<I", sk, 0x1C, n)
            subs[skel_index] = SubResource(skel_index, bytes(sk))
            skel_synced = True
    out = MonsterPac(subs=subs, tail=pac.tail).to_bytes()
    return out, {"animated": n, "skel_animated": skel_animated, "split": sp,
                 "host_count": host_count, "clips": len(fp.animations),
                 "skel_synced": skel_synced}


def fit_anim_sub(anim_bytes: bytes, target_size: int) -> bytes:
    """Pad (or validate) an in-game anim sub to ``target_size`` bytes.

    Padding with trailing zeros after the last block keeps the PAC the same size
    as the source so the proven same-size in-place inject path applies. The block
    offset tables are absolute, so trailing pad is inert. Raises if the anim is
    already larger than the target (caller must use the relocate inject path).
    """
    if len(anim_bytes) > target_size:
        raise ValueError("anim %d B > target %d B — use the relocate inject path"
                         % (len(anim_bytes), target_size))
    return anim_bytes + b"\x00" * (target_size - len(anim_bytes))


def make_static_pose(stream_specs: List[Tuple[int, List[int]]],
                     num_slots: int = 100, bone_factory=rest_bone) -> InGameAnim:
    """Build a static pose in-game anim (one bind-pose block aliased to all slots).

    ``stream_specs`` is a list of up to 6 ``(bone_count, [slots])`` — one per
    stream in order (main, sub0..sub4). For each spec with bone_count>0 a single
    block of ``bone_count`` bone sections (built by ``bone_factory``) is created and
    aliased to every listed slot, so any clip index the AI requests resolves to the
    same static pose. Streams with bone_count==0 are left empty (all-0xFFFFFFFF).

    ``bone_factory`` defaults to :func:`rest_bone` (identity-rotation keyframes → the
    skeleton rest pose renders visibly). Pass :func:`empty_bone` only for native
    round-trip fidelity (an all-empty anim collapses the mesh — see ``empty_bone``).
    """
    streams: List[Stream] = []
    main_off = 0x34
    sub0 = main_off + num_slots * 4 + 4
    table_offs = [main_off] + [sub0 + num_slots * 4 * i for i in range(5)]
    for si in range(6):
        toff = table_offs[si]
        st = Stream(table_offset=toff)
        if si < len(stream_specs):
            bc, slots = stream_specs[si]
            if bc > 0 and slots:
                blk = Block(tag=BLOCK_TAG, loop=0, loop_start=0.0,
                            bones=[bone_factory() for _ in range(bc)])
                for s in slots:
                    st.clips[s] = blk
        streams.append(st)
    return InGameAnim(num_slots=num_slots, streams=streams)


# --------------------------------------------------------------------------- #
# diagnostics
# --------------------------------------------------------------------------- #
def summary(a: InGameAnim) -> str:
    lines = [f"magic={a.magic:#x} hsize={a.hsize:#x} slots={a.num_slots}"]
    names = ["main", "sub0", "sub1", "sub2", "sub3", "sub4"]
    for nm, st in zip(names, a.streams):
        if not st.clips:
            lines.append(f"  {nm} @{st.table_offset:#x}: empty")
            continue
        any_blk = next(iter(st.clips.values()))
        lines.append(f"  {nm} @{st.table_offset:#x}: {len(st.clips)} clips, "
                     f"bones/blk={len(any_blk.bones)}")
    return "\n".join(lines)
