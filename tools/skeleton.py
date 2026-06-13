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


def main(argv):
    raw = False
    args = []
    for a in argv:
        if a == '--raw':
            raw = True
        else:
            args.append(a)
    if not args:
        print(__doc__)
        return 1
    data = open(args[0], 'rb').read()
    blob = data if raw else (extract_sub0(data) or data)
    sk = parse(blob)
    print('skeleton: %d bones, blob size %d' % (sk['bone_count'], sk['total_size']))
    print('  idx  parent  child  sibling   bind position')
    for b in sk['bones']:
        print('  %3d  %5d  %5d  %7d   (%9.2f %9.2f %9.2f)  rec@0x%x sz=0x%x'
              % (b['index'], b['parent'], b['child'], b['sibling'],
                 b['pos'][0], b['pos'][1], b['pos'][2], b['off'], b['size']))
    # sanity: root(s) = parent==-1
    roots = [b['index'] for b in sk['bones'] if b['parent'] == -1]
    print('  roots (parent==-1):', roots, ' parsed %d/%d bones'
          % (len(sk['bones']), sk['bone_count']))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
