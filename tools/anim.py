#!/usr/bin/env python3
"""MHFU / MHP2G big-monster ANIMATION parser (PAC sub-resource 3).

Big-monster model PACs (`file_0{em_id+0x17AB}.bin`, the species with `em*.ovl`
overlays) store keyframe animation as **sub-resource 3** — a P3rd-style animation
pack. Skeleton (sub-0) + this share bone ordering. Decoded 2026-06-13; the engine
interpolates with the cubic `spline()` (ease-in/out tangents) in `model_base.cpp`.

Container:
  Header 0x18:  u32 magic(0x64) ; u32 header_size(0x18) ; u32 slot_count ;
                u32 ? ; u32 ? ; u32 first_offset
  Offset table @ +0x14:  slot_count × u32, each an offset (from sub-3 start) to an
                animation block, or 0xFFFFFFFF = empty slot.

Every level is a section `{ u32 (0x80000000|tag) ; u32 count ; u32 size }` whose
`size` includes the 12-byte header and chains exactly to the next sibling:

  Animation block:  tag flags, count = bone_count, size ; +0x0C u32 loop ;
                    +0x10 f32 loop_start ; then `bone_count` bone records.
                    (header is 0x14, bone records start at +0x14)
  Bone record:      tag = channel bitmask (P3rd: rotX/Y/Z=0x08/10/20,
                    locX/Y/Z=0x40/80/100, scaleX/Y/Z=0x200/400/800),
                    count = #channels ; then `count` channel records.
                    Empty bone = 12-byte header only (count 0).
  Channel record:   tag = channel type, count = #keyframes, size ;
                    then `count` keyframes.
  Keyframe (8B):    s16 value, s16 frame, s16 ease_in, s16 ease_out.

Value quantization (per the P3rd sibling addon): rotation 4096 = 90 deg,
location 16 = 1.0, scale 256 = 1.0.

Usage:
  python anim.py <big-mon PAC | --raw sub3.bin> [anim_index]
"""
import array
import struct
import sys

FLAG = 0x80000000
# P3rd transform-channel bits (low half of a bone record's tag)
CHANNEL_BITS = [
    (0x008, 'rotX'), (0x010, 'rotY'), (0x020, 'rotZ'),
    (0x040, 'locX'), (0x080, 'locY'), (0x100, 'locZ'),
    (0x200, 'sclX'), (0x400, 'sclY'), (0x800, 'sclZ'),
]


def extract_sub(pac, idx):
    cnt = array.array('I', pac[:4])[0]
    info = array.array('I', pac[4:4 + cnt * 8])
    return pac[info[idx * 2]:info[idx * 2] + info[idx * 2 + 1]]


def channel_names(mask):
    return [n for bit, n in CHANNEL_BITS if mask & bit]


def parse_pack(a):
    magic, hsize, slot_count = struct.unpack_from('<3I', a, 0)
    table = struct.unpack_from('<%dI' % slot_count, a, 0x14)
    anims = {i: off for i, off in enumerate(table) if off != 0xFFFFFFFF}
    return {'magic': magic, 'slot_count': slot_count, 'anims': anims, 'blob': a}


def parse_anim(a, ao):
    tag, bone_count, size = struct.unpack_from('<3I', a, ao)
    loop = struct.unpack_from('<I', a, ao + 0x0C)[0]
    loop_start = struct.unpack_from('<f', a, ao + 0x10)[0]
    bones = []
    o = ao + 0x14
    for _ in range(bone_count):
        btag, nch, bsz = struct.unpack_from('<3I', a, o)
        mask = btag & 0xFFFF
        channels = []
        co = o + 0x0C
        for _c in range(nch):
            ctag, nkf, csz = struct.unpack_from('<3I', a, co)
            kfs = [struct.unpack_from('<4h', a, co + 0x0C + k * 8) for k in range(nkf)]
            channels.append({'type': ctag, 'keyframes': kfs, 'size': csz})
            co += csz
        bones.append({'mask': mask, 'channels': channel_names(mask),
                      'nchan': nch, 'size': bsz, 'chan': channels})
        o += bsz
    return {'off': ao, 'bone_count': bone_count, 'size': size,
            'loop': loop, 'loop_start': loop_start, 'bones': bones, 'end': o - ao}


def main(argv):
    raw = False
    args = []
    for x in argv:
        if x == '--raw':
            raw = True
        else:
            args.append(x)
    if not args:
        print(__doc__)
        return 1
    data = open(args[0], 'rb').read()
    a = data if raw else extract_sub(data, 3)
    pk = parse_pack(a)
    print('anim pack: magic=0x%x  %d slots, %d populated' %
          (pk['magic'], pk['slot_count'], len(pk['anims'])))
    print('populated anim indices:', list(pk['anims'].keys()))
    idx = int(args[1]) if len(args) > 1 else next(iter(pk['anims']))
    an = parse_anim(a, pk['anims'][idx])
    print('\nanim[%d] @0x%x: %d bones, size 0x%x (walked 0x%x), loop=%d loop_start=%g'
          % (idx, an['off'], an['bone_count'], an['size'], an['end'],
             an['loop'], an['loop_start']))
    print('  chain %s' % ('OK' if an['end'] == an['size'] else 'MISMATCH'))
    for bi, b in enumerate(an['bones']):
        ch = ','.join(b['channels']) or '(empty)'
        kfc = sum(len(c['keyframes']) for c in b['chan'])
        print('  bone %2d: mask=0x%03x [%s] %d chan, %d keyframes'
              % (bi, b['mask'], ch, b['nchan'], kfc))
    # show a sample channel's keyframes (first animated bone)
    for b in an['bones']:
        if b['chan']:
            c = b['chan'][0]
            print('\n  sample channel (bone mask 0x%x, type 0x%x), %d keyframes:'
                  % (b['mask'], c['type'], len(c['keyframes'])))
            for v, f, ei, eo in c['keyframes'][:8]:
                print('    frame=%-5d value=%-7d ease_in=%-5d ease_out=%-5d  (rot~%.2f deg)'
                      % (f, v, ei, eo, v * 90.0 / 4096))
            break
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
