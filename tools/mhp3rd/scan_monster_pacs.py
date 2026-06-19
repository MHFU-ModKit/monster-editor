#!/usr/bin/env python3
"""
Scan an extracted MHP3rd (or MHFU) data_files/ tree and classify every PAC by the
sub-resources it contains, so we can locate big-monster MODEL packages.

PAC container layout (same family across MHP1/2/2G/3rd):
    u32 file_count
    file_count * (u32 offset, u32 size)   # sub-resource table
    ... data ...

Sub-resource magics:
    skeleton  : u32@0 == 0xC0000000             (big-monster bind-pose blob)
    pmo       : bytes[0:4] == b'pmo\\x00'        (geometry; ver bytes[4:8] = '1.0\\x00' or '102\\x00')
    tmh       : bytes[0:8] == b'.TMH0.14'        (texture package)
    anim      : u32@0 == 0x64 and u32@4 == 0x18  (P3rd-style keyframe pack)

A big-monster model PAC = has a skeleton sub AND a pmo sub. We print those plus a
one-line classification of every other PAC, and (with --detail) a full sub dump.

Usage:
    python tools/mhp3rd/scan_monster_pacs.py <data_files_dir> [--detail file_0NNNNN.bin]
"""
import argparse
import os
import struct
import sys


def classify_sub(blob: bytes) -> dict:
    """Return {'type':..., 'ver':...} for a sub-resource blob."""
    if len(blob) < 8:
        return {'type': 'tiny', 'ver': None, 'size': len(blob)}
    u0, u1 = struct.unpack_from('<II', blob, 0)
    # MHFU skeleton = 0xC0000000; MHP3rd (gen-3) skeleton = 0x80000000 (same blob family,
    # different top-bit flag). bone_count at +4 in both.
    if u0 in (0xC0000000, 0x80000000):
        return {'type': 'skeleton', 'ver': '%08x' % u0, 'size': len(blob), 'bones': u1}
    if blob[0:4] == b'pmo\x00':
        return {'type': 'pmo', 'ver': blob[4:8].rstrip(b'\x00').decode('latin1'), 'size': len(blob)}
    if blob[0:8] == b'.TMH0.14':
        return {'type': 'tmh', 'ver': None, 'size': len(blob)}
    if u0 == 0x64 and u1 == 0x18:
        return {'type': 'anim', 'ver': None, 'size': len(blob), 'slots': struct.unpack_from('<I', blob, 8)[0]}
    return {'type': 'other', 'ver': '%08x' % u0, 'size': len(blob)}


def parse_pac(data: bytes):
    """Yield (idx, offset, size, classify_sub(...)) or return None if not a PAC."""
    if len(data) < 12:
        return None
    count = struct.unpack_from('<I', data, 0)[0]
    if count == 0 or count > 64:
        return None
    tbl_end = 4 + count * 8
    if tbl_end > len(data):
        return None
    subs = []
    for i in range(count):
        off, sz = struct.unpack_from('<II', data, 4 + i * 8)
        if off == 0 or sz == 0 or off + sz > len(data) or off < tbl_end:
            return None  # not a coherent PAC table
        subs.append((i, off, sz, classify_sub(data[off:off + sz])))
    return subs


def scan(data_dir: str):
    bigmon = []
    rows = []
    for name in sorted(os.listdir(data_dir)):
        path = os.path.join(data_dir, name)
        if not os.path.isfile(path):
            continue
        try:
            with open(path, 'rb') as f:
                head = f.read()
        except OSError:
            continue
        subs = parse_pac(head)
        if not subs:
            continue
        types = [s[3]['type'] for s in subs]
        has_skel = 'skeleton' in types
        has_pmo = 'pmo' in types
        pmo_vers = sorted({s[3].get('ver') for s in subs if s[3]['type'] == 'pmo'})
        summary = '%-16s n=%d  %s' % (name, len(subs), ','.join(types))
        if pmo_vers:
            summary += '  pmo_ver=%s' % '/'.join(v or '?' for v in pmo_vers)
        rows.append(summary)
        if has_skel and has_pmo:
            bigmon.append((name, subs, pmo_vers))
    return bigmon, rows


def detail(path: str):
    with open(path, 'rb') as f:
        data = f.read()
    subs = parse_pac(data)
    if not subs:
        print('NOT a coherent PAC: %s' % path)
        return
    print('PAC %s : %d sub-resources' % (os.path.basename(path), len(subs)))
    for i, off, sz, c in subs:
        extra = ''
        if c['type'] == 'skeleton':
            extra = ' bones=%d' % c.get('bones', -1)
        elif c['type'] == 'pmo':
            extra = ' ver=%s' % c.get('ver')
        elif c['type'] == 'anim':
            extra = ' slots=%d' % c.get('slots', -1)
        print('  [%d] off=0x%08x size=%9d  %-9s%s' % (i, off, sz, c['type'], extra))


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument('data_dir')
    ap.add_argument('--detail', help='dump one PAC file fully (path or name under data_dir)')
    args = ap.parse_args(argv)

    if args.detail:
        p = args.detail
        if not os.path.isfile(p):
            p = os.path.join(args.data_dir, args.detail)
        detail(p)
        return

    bigmon, rows = scan(args.data_dir)
    print('=== BIG-MONSTER MODEL PACs (skeleton + pmo) : %d ===' % len(bigmon))
    for name, subs, pmo_vers in bigmon:
        layout = ','.join('%s' % s[3]['type'] for s in subs)
        print('  %-16s subs=%d  layout=[%s]  pmo_ver=%s' %
              (name, len(subs), layout, '/'.join(v or '?' for v in pmo_vers)))
    print('\n=== ALL %d PACs (classification) ===' % len(rows))
    for r in rows:
        print('  ' + r)


if __name__ == '__main__':
    main(sys.argv[1:])
