"""In-memory TMH texture decoder (PIL-free) for the Blender importer.

Ported from tools/mhff/psp/tmh.py (Seth VanHeulen, GPL) but returns raw RGBA8
bytes + dimensions instead of writing PNGs, so Blender's bundled Python (no PIL)
can build `bpy.data.images` directly. Decodes the swizzled/blocked PSP texture
modes 0-8 (+ CLUT-indexed); DXT3/DXT5 (modes 9/10) are not yet supported and
return None for that image.

    from mhfu_model.tmh import decode_tmh
    for tex in decode_tmh(mm.texture.raw):
        tex["index"], tex["width"], tex["height"], tex["rgba"]  # rgba = bytes WxHx4
"""
from __future__ import annotations

import array
import struct
from typing import List, Optional


def _deblock(mode, width, data):
    bw = (8, 8, 8, 4, 32, 16, 8, 4, 4, 4, 4)[mode]
    bh = (8, 8, 8, 8, 8, 8, 8, 8, 4, 4, 4)[mode]
    data = array.array('I', data)
    new = array.array('I')
    for i in range(len(data)):
        x = i % width
        y = i // width
        xb = x // bw
        x %= bw
        yb = y // bh
        y %= bh
        offset = bw * bh * xb + width * bh * yb + bw * y + x
        new.append(data[offset])
    return new.tobytes()


def _decode(mode, data):
    if mode == 0:
        data = array.array('H', data); new = bytearray()
        for i in data:
            new.append(round((i & 31) * (255 / 31)))
            new.append(round((i >> 5 & 63) * (255 / 63)))
            new.append(round((i >> 11 & 31) * (255 / 31)))
            new.append(255)
        return bytes(new)
    if mode == 1:
        data = array.array('H', data); new = bytearray()
        for i in data:
            new.append(round((i & 31) * (255 / 31)))
            new.append(round((i >> 5 & 31) * (255 / 31)))
            new.append(round((i >> 10 & 31) * (255 / 31)))
            new.append((i >> 15) * 255)
        return bytes(new)
    if mode == 2:
        new = bytearray()
        for i in data:
            new.append((i & 15) * 17); new.append((i >> 4) * 17)
        return bytes(new)
    if mode == 3:
        return data
    if mode == 4:
        new = bytearray()
        for i in data:
            new.append(i & 15); new.append(i >> 4)
        return bytes(new)
    if mode == 5:
        return data
    if mode == 6:
        return array.array('H', data)
    if mode == 7:
        return array.array('I', data)
    if mode == 8:
        new = bytearray()
        for i in range(0, len(data), 8):
            c = [_decode(0, data[i + 4:i + 6]), _decode(0, data[i + 6:i + 8])]
            temp = array.array('H', data[i + 4:i + 8])
            if temp[0] > temp[1]:
                c.append(bytes([(2 * c[0][0] + c[1][0]) // 3, (2 * c[0][1] + c[1][1]) // 3,
                                (2 * c[0][2] + c[1][2]) // 3, 255]))
                c.append(bytes([(c[0][0] + 2 * c[1][0]) // 3, (c[0][1] + 2 * c[1][1]) // 3,
                                (c[0][2] + 2 * c[1][2]) // 3, 255]))
            else:
                c.append(bytes([(c[0][0] + c[1][0]) // 2, (c[0][1] + c[1][1]) // 2,
                                (c[0][2] + c[1][2]) // 2, 255]))
                c.append(b'\x00\x00\x00\xff')
            for d in data[i:i + 4]:
                new.extend(c[d & 3]); new.extend(c[d >> 2 & 3])
                new.extend(c[d >> 4 & 3]); new.extend(c[d >> 6 & 3])
        return bytes(new)
    return None   # modes 9/10 = DXT3/DXT5, not yet supported


def decode_tmh(raw: bytes) -> List[dict]:
    """Decode a .TMH blob -> list of {index,width,height,rgba} (rgba = W*H*4 bytes,
    row 0 = TOP). Images that can't be decoded (DXT/odd modes) are skipped."""
    out: List[dict] = []
    if not raw or raw[:8] != b'.TMH0.14':
        return out
    hdr = struct.unpack_from('8s2I', raw, 0)
    off = 16
    for i in range(hdr[1]):
        try:
            image_header = struct.unpack_from('4I', raw, off); off += 16
            pixel_header = struct.unpack_from('3I2H', raw, off); off += 16
            size, mode, width, height = pixel_header[0], pixel_header[2], pixel_header[3], pixel_header[4]
            pixel_data = _decode(mode, raw[off:off + size - 16]); off += size - 16
            if image_header[3] == 1:           # CLUT-indexed
                clut_header = struct.unpack_from('4I', raw, off); off += 16
                clut_data = _decode(clut_header[2], raw[off:off + clut_header[0] - 16])
                off += clut_header[0] - 16
                exp = bytearray()
                for p in pixel_data:
                    exp.extend(clut_data[p * 4:p * 4 + 4])
                pixel_data = bytes(exp)
            if pixel_data is None:
                continue
            pixel_data = _deblock(mode, width, pixel_data)
            if mode > 7:                       # BGRA -> RGBA channel swap
                b = bytearray(pixel_data)
                b[0::4], b[2::4] = b[2::4], b[0::4]
                pixel_data = bytes(b)
            if len(pixel_data) < width * height * 4:
                continue
            out.append({"index": i, "width": width, "height": height,
                        "rgba": pixel_data[:width * height * 4]})
        except (struct.error, ValueError, IndexError):
            break
    return out
