#!/usr/bin/env python3
"""MHFU / MHP2G big-monster SKELETON parser (`0xC0000000` blob).

Big monsters (those with an `em*.ovl` AI overlay — Tigrex, etc.) ship a real bone
skeleton + bind-pose as **sub-resource 0** of their model PAC (`file_id = em_id +
0x17AB`; e.g. Tigrex em75 = file_06134). Small monsters (Popo/em01) have NO skeleton
sub-resource (their PAC sub-0 is the PMO) — they ride a rigid/shared path.

Format (verified 2026-06-13 against the engine builder EU `0x088dc40c`, and identical
to the iOS `m2jean/mhfu-ios-pmo-plugin` skeleton):

  Header (0x1C):
    +0x00 u32  magic        0xC0000000
    +0x04 u32  bone_count   (engine reads this; top bit of magic flags a -1 adjust)
    +0x08 u32  total_size
    +0x0C u32  ? (0)
    +0x10 u32  ? (1)
    +0x14 u32  ? (0x10)
    +0x18 u32  ? (0)
  Then `bone_count` bone sections, each (size from its +0x08 field, typically 0x10C):
    +0x00 u32  section_magic 0x40000001
    +0x04 u32  flag (1)
    +0x08 u32  section_size  (0x10C)
    +0x0C s32  index
    +0x10 s32  parent index       (-1 = none)   <- engine: record+0x10
    +0x14 s32  left_child index   (-1 = none)   <- engine: record+0x14
    +0x18 s32  right_sibling index(-1 = none)   <- engine: record+0x18
    +0x1C ..   bind-pose transform (4x4 matrix, row-major f32; identity rows = 1.0)
               followed by decomposed pos/rot/scale + aux (the engine's bind_pose).

The engine builds one `Joint` (stride 0x250) per section, links parent/child/sibling
by index, and stores the bind transform into `Joint.bind` (see docs/ANIMATION_FORMAT.md).

Usage:
  python skeleton.py <em.pac>            # auto-extract sub-0 and parse
  python skeleton.py --raw <skel.bin>    # parse a raw 0xC0000000 blob
"""
import array
import struct
import sys

MAGIC = 0xC0000000
SECTION_MAGIC = 0x40000001
HDR_SIZE = 0x1C


def extract_sub0(pac_bytes):
    cnt = array.array('I', pac_bytes[:4])[0]
    if cnt == 0 or cnt > 64:
        return None
    info = array.array('I', pac_bytes[4:4 + cnt * 8])
    off, sz = info[0], info[1]
    return pac_bytes[off:off + sz]


def parse(blob):
    magic, bone_count, total_size = struct.unpack_from('<3I', blob, 0)
    if magic != MAGIC:
        raise ValueError('not a 0xC0000000 skeleton (magic=0x%08x)' % magic)
    bones = []
    o = HDR_SIZE
    for i in range(bone_count):
        if o + 0x1C > len(blob):
            break
        smag, flag, ssize = struct.unpack_from('<3I', blob, o)
        idx, parent, child, sibling = struct.unpack_from('<4i', blob, o + 0x0C)
        # bind-pose = three padded vec4 rows at section +0x1C:
        #   scale @+0x1C, rotation(euler) @+0x2C, position @+0x3C  (each Vec3 + 1.0 pad)
        scale = struct.unpack_from('<3f', blob, o + 0x1C)
        rotation = struct.unpack_from('<3f', blob, o + 0x2C)
        position = struct.unpack_from('<3f', blob, o + 0x3C)
        bones.append({
            'i': i, 'off': o, 'section_magic': smag, 'size': ssize,
            'index': idx, 'parent': parent, 'child': child, 'sibling': sibling,
            'scale': scale, 'rotation': rotation, 'pos': position,
        })
        if smag != SECTION_MAGIC or ssize == 0 or ssize > 0x400:
            break
        o += ssize
    return {'bone_count': bone_count, 'total_size': total_size, 'bones': bones}


def parse_p3rd(blob):
    """Parse a 0x80000000 MHP3rd skeleton blob; returns same dict shape as parse().

    Delegates to mhfu_model.skeleton_p3rd which handles the two MHP3rd quirks:
    - section magic 0x40000002 (in addition to 0x40000001)
    - optional extra 4-byte header word before bone sections (lobby PACs)
    """
    import sys, os
    # Import via the mhfu_model package so relative imports inside skeleton_p3rd work
    tools_dir = os.path.dirname(os.path.abspath(__file__))
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)
    from mhfu_model.skeleton_p3rd import parse as _parse_p3rd
    sk = _parse_p3rd(blob)
    # Convert dataclass back to the dict shape used by this module's CLI/callers
    bones = []
    for i, b in enumerate(sk.bones):
        bones.append({
            'i': i, 'off': 0, 'section_magic': 0x40000001, 'size': b.section_size,
            'index': b.index, 'parent': b.parent, 'child': b.child, 'sibling': b.sibling,
            'scale': b.bind_scale, 'rotation': b.bind_rot, 'pos': b.bind_pos,
        })
    return {'bone_count': sk.bone_count, 'total_size': sk.total_size, 'bones': bones}


def _find_skel_sub(data, magic):
    """Find the first sub-resource with given u32 magic in a PAC blob."""
    try:
        cnt = struct.unpack_from('<I', data)[0]
        if cnt == 0 or cnt > 64:
            return None
        for i in range(cnt):
            off, sz = struct.unpack_from('<II', data, 4 + i * 8)
            if off + 4 <= len(data):
                m = struct.unpack_from('<I', data, off)[0]
                if m == magic:
                    return data[off:off + sz]
    except struct.error:
        pass
    return None


def main(argv):
    raw = False
    game = 'mhfu'
    args = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == '--raw':
            raw = True
        elif a == '--game' and i + 1 < len(argv):
            game = argv[i + 1]
            i += 1
        else:
            args.append(a)
        i += 1
    if not args:
        print(__doc__)
        return 1
    data = open(args[0], 'rb').read()
    if raw:
        blob = data
    elif game == 'p3rd':
        # MHP3rd: skeleton is 0x80000000, may be any sub; try to find it
        blob = _find_skel_sub(data, 0x80000000)
        if blob is None:
            blob = data  # caller passed raw blob directly
    else:
        blob = extract_sub0(data) or data
    if game == 'p3rd':
        sk = parse_p3rd(blob)
    else:
        sk = parse(blob)
    print('skeleton: %d bones, blob size %d' % (sk['bone_count'], sk['total_size']))
    print('  idx  parent  child  sibling   bind position')
    for b in sk['bones']:
        print('  %3d  %5d  %5d  %7d   (%9.2f %9.2f %9.2f)  sz=0x%x'
              % (b['index'], b['parent'], b['child'], b['sibling'],
                 b['pos'][0], b['pos'][1], b['pos'][2], b['size']))
    # sanity: root(s) = parent==-1
    roots = [b['index'] for b in sk['bones'] if b['parent'] == -1]
    print('  roots (parent==-1):', roots, ' parsed %d/%d bones'
          % (len(sk['bones']), sk['bone_count']))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
