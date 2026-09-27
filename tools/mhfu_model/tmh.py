# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2013 Seth VanHeulen (mhff, https://github.com/svanheulen/mhff)
# Copyright 2026 sp00ktober
"""In-memory TMH texture codec (PIL-free) for the Blender importer and the stage editor.

Ported from tools/mhff/psp/tmh.py (Seth VanHeulen, GPL) but returns raw RGBA8
bytes + dimensions instead of writing PNGs, so Blender's bundled Python (no PIL)
can build `bpy.data.images` directly. Decodes the swizzled/blocked PSP texture
modes 0-8 (+ CLUT-indexed); DXT3/DXT5 (modes 9/10) are not yet supported and
return None for that image.

    from mhfu_model.tmh import decode_tmh
    for tex in decode_tmh(mm.texture.raw):
        tex["index"], tex["width"], tex["height"], tex["rgba"]  # rgba = bytes WxHx4

`parse_tmh` gives the same bank structurally (byte offsets, not pixels) and
`encode_into` writes one image back **at exactly its original size** -- which is
the only kind of edit a resident sub-resource can take, since sub[1] is wedged
between sub[0] and sub[2] in the PAC with no slack. Every stage texture seen so
far is CLUT-indexed (mode 4 = 4bpp/16 colours, mode 5 = 8bpp/256), so the byte
size is fixed by (mode, width, height) and a same-dimension swap is automatically
size-identical. See `docs/STAGE_MAP_FORMAT.md` s4e.
"""
from __future__ import annotations

import array
import struct
from typing import List, Optional


# PSP swizzle block, in PIXELS, per pixel mode. A block is always 16 bytes wide
# by 8 rows, so the pixel width falls out of the bit depth (mode 4 = 4bpp -> 32).
_BW = (8, 8, 8, 4, 32, 16, 8, 4, 4, 4, 4)
_BH = (8, 8, 8, 8, 8, 8, 8, 8, 4, 4, 4)


def _perm(mode, width, n):
    """For each LINEAR pixel index, its index in the file's block order."""
    bw, bh = _BW[mode], _BH[mode]
    out = array.array('i', bytes(4 * n))
    for i in range(n):
        x = i % width
        y = i // width
        xb = x // bw
        x %= bw
        yb = y // bh
        y %= bh
        out[i] = bw * bh * xb + width * bh * yb + bw * y + x
    return out


def _deblock(mode, width, data):
    data = array.array('I', data)
    perm = _perm(mode, width, len(data))
    new = array.array('I')
    for i in range(len(data)):
        new.append(data[perm[i]])
    return new.tobytes()


def _swizzle_idx(mode, width, idx):
    """Inverse of `_deblock`, on one-byte palette indices: linear -> block order."""
    perm = _perm(mode, width, len(idx))
    out = bytearray(len(idx))
    for i in range(len(idx)):
        out[perm[i]] = idx[i]
    return bytes(out)


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


# ---------------------------------------------------------------------------
# Structural view + encoder
#
# A resident sub-resource cannot change size (sub[1] is wedged between sub[0]
# and sub[2] with a zero-byte gap), so everything below writes back in place.
# ---------------------------------------------------------------------------

_BPP = {4: 4, 5: 8}          # the CLUT-indexed modes, bits per pixel


class TmhImage:
    """One image in a bank, as byte ranges into the blob it came from."""

    __slots__ = ("index", "width", "height", "mode", "indexed",
                 "ihdr_off", "phdr_off", "pix_off", "pix_len",
                 "clut_hdr_off", "clut_off", "clut_len", "clut_mode",
                 "clut_entries", "end")

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))

    @property
    def colours(self):
        """How many palette entries this image's mode addresses (None if direct)."""
        return (1 << _BPP[self.mode]) if self.mode in _BPP else None

    def __repr__(self):
        return ("<TmhImage %d %dx%d mode=%d %s pix@0x%X+%d>"
                % (self.index, self.width, self.height, self.mode,
                   ("clut=%d" % self.clut_entries) if self.indexed else "direct",
                   self.pix_off, self.pix_len))


def parse_tmh(raw: bytes) -> List[TmhImage]:
    """Walk a .TMH bank and return every image as byte offsets. Raises ValueError on
    anything it cannot account for -- a silent partial walk is how a same-size
    edit ends up landing on the wrong image."""
    if not raw or raw[:8] != b'.TMH0.14':
        raise ValueError("not a .TMH0.14 blob (magic %r)" % raw[:8])
    count = struct.unpack_from('8s2I', raw, 0)[1]
    out: List[TmhImage] = []
    off = 16
    for i in range(count):
        ihdr_off = off
        ih = struct.unpack_from('4I', raw, off); off += 16
        phdr_off = off
        ph = struct.unpack_from('3I2H', raw, off); off += 16
        size, mode, width, height = ph[0], ph[2], ph[3], ph[4]
        pix_off, pix_len = off, size - 16
        off += pix_len
        img = TmhImage(index=i, width=width, height=height, mode=mode,
                       indexed=(ih[3] == 1), ihdr_off=ihdr_off, phdr_off=phdr_off,
                       pix_off=pix_off, pix_len=pix_len)
        if img.indexed:
            img.clut_hdr_off = off
            ch = struct.unpack_from('4I', raw, off); off += 16
            img.clut_mode, img.clut_entries = ch[2], ch[3]
            img.clut_off, img.clut_len = off, ch[0] - 16
            off += img.clut_len
        img.end = off
        out.append(img)
    if off != len(raw):
        raise ValueError("TMH walk ended at 0x%X but the blob is 0x%X bytes"
                         % (off, len(raw)))
    return out


def _pack_indices(mode, idx):
    """Palette indices (one byte each, block order) -> the file's pixel bytes."""
    if mode == 5:
        return bytes(idx)
    if mode == 4:
        if len(idx) & 1:
            raise ValueError("4bpp needs an even pixel count")
        return bytes((idx[k] & 15) | ((idx[k + 1] & 15) << 4)
                     for k in range(0, len(idx), 2))
    raise ValueError("mode %d is not a CLUT-indexed mode this encoder writes" % mode)


def _q(v, bits):
    """8-bit channel -> `bits`-bit, the exact inverse of `_decode`'s expansion."""
    m = (1 << bits) - 1
    return int(round(max(0, min(255, v)) * m / 255.0))


def _pack_clut(mode, palette):
    """Palette of (r,g,b,a) -> the file's CLUT bytes, in the CLUT's own pixel
    mode. Modes 0/1/2 are lossy for colours that did not come out of such a
    palette to begin with; 3 is exact."""
    out = bytearray()
    for c in palette:
        r, g, b, a = (max(0, min(255, int(v))) for v in c)
        if mode == 0:     # RGB565, no alpha
            out.extend(struct.pack('<H', _q(r, 5) | (_q(g, 6) << 5) | (_q(b, 5) << 11)))
        elif mode == 1:   # RGBA5551
            out.extend(struct.pack('<H', _q(r, 5) | (_q(g, 5) << 5)
                                   | (_q(b, 5) << 10) | ((1 if a >= 128 else 0) << 15)))
        elif mode == 2:   # RGBA4444
            out.extend(bytes((_q(r, 4) | (_q(g, 4) << 4), _q(b, 4) | (_q(a, 4) << 4))))
        elif mode == 3:   # RGBA8888
            out.extend(bytes((r, g, b, a)))
        else:
            raise ValueError("CLUT mode %d is not a colour mode this encoder "
                             "writes" % mode)
    return bytes(out)


def quantize(rgba: bytes, n: int, palette=None):
    """RGBA8 pixels -> (indices bytes, palette list) with at most `n` entries.

    Median cut over the *distinct* colours, weighted by how often each occurs, so
    a texture that is mostly one flat colour does not spend half its palette on
    it. Pass `palette` to keep an existing one and only re-map onto it."""
    px = [rgba[i:i + 4] for i in range(0, len(rgba), 4)]
    hist = {}
    for p in px:
        hist[p] = hist.get(p, 0) + 1

    if palette is None:
        uniq = sorted(hist)
        if len(uniq) <= n:
            palette = [tuple(c) for c in uniq]
        else:
            boxes = [[(tuple(c), w) for c, w in hist.items()]]
            while len(boxes) < n:
                bi, best = -1, -1
                for i, b in enumerate(boxes):
                    if len(b) < 2:
                        continue
                    sp = max(max(c[ch] for c, _ in b) - min(c[ch] for c, _ in b)
                             for ch in range(4))
                    if sp > best:
                        bi, best = i, sp
                if bi < 0:
                    break
                b = boxes.pop(bi)
                ch = max(range(4), key=lambda ch:
                         max(c[ch] for c, _ in b) - min(c[ch] for c, _ in b))
                b.sort(key=lambda t: t[0][ch])
                tot, acc, cut = sum(w for _, w in b), 0, 1
                for j, (_, w) in enumerate(b):
                    acc += w
                    if acc * 2 >= tot:
                        cut = max(1, min(j + 1, len(b) - 1))
                        break
                boxes.append(b[:cut]); boxes.append(b[cut:])
            palette = []
            for b in boxes:
                tot = sum(w for _, w in b)
                palette.append(tuple(round(sum(c[ch] * w for c, w in b) / tot)
                                     for ch in range(4)))
    palette = [tuple(c) for c in palette]
    if len(palette) > n:
        raise ValueError("palette has %d entries, mode holds %d" % (len(palette), n))

    # nearest match, memoised per DISTINCT colour (a 256-entry CLUT over 16k
    # pixels is 4M distance tests otherwise)
    exact = {c: i for i, c in enumerate(palette)}
    cache = {}
    for c in hist:
        t = tuple(c)
        i = exact.get(t)
        if i is None:
            i = min(range(len(palette)),
                    key=lambda k: sum((palette[k][ch] - t[ch]) ** 2 for ch in range(4)))
        cache[c] = i
    idx = bytes(cache[p] for p in px)
    return idx, palette + [(0, 0, 0, 0)] * (n - len(palette))


def encode_into(raw: bytes, index: int, rgba: bytes, palette=None) -> bytes:
    """Write one image back into a bank, same bytes in and out.

    `rgba` must be exactly width*height*4, row 0 = TOP, matching the image it
    replaces -- the PSP holds the dimensions in the header and the header is not
    ours to grow. Returns a new blob the same length as `raw`."""
    imgs = parse_tmh(raw)
    img = imgs[index]
    if not img.indexed:
        raise ValueError("image %d is not CLUT-indexed (mode %d); only the "
                         "indexed modes are written" % (index, img.mode))
    if img.mode not in _BPP:
        raise ValueError("image %d uses mode %d, which this encoder cannot write"
                         % (index, img.mode))
    want = img.width * img.height * 4
    if len(rgba) != want:
        raise ValueError("image %d is %dx%d -> %d bytes of RGBA, got %d"
                         % (index, img.width, img.height, want, len(rgba)))

    idx, pal = quantize(rgba, img.clut_entries, palette)
    pix = _pack_indices(img.mode, _swizzle_idx(img.mode, img.width, idx))
    clut = _pack_clut(img.clut_mode, pal)
    if len(pix) != img.pix_len:
        raise ValueError("encoded %d pixel bytes, the slot holds %d"
                         % (len(pix), img.pix_len))
    if len(clut) != img.clut_len:
        raise ValueError("encoded %d CLUT bytes, the slot holds %d"
                         % (len(clut), img.clut_len))
    out = bytearray(raw)
    out[img.pix_off:img.pix_off + img.pix_len] = pix
    out[img.clut_off:img.clut_off + img.clut_len] = clut
    return bytes(out)
