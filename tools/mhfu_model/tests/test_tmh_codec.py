"""The texture half of an edit list: can we write a stage's TMH bank back?

sub[1] is read LIVE out of the resident PAC every frame and it cannot change size, so
every property that matters here is a *byte* property, not a pixel one: the walk has to
account for the whole blob, an encode has to land in exactly the bytes it replaced, and a
re-encode of what was already there has to reproduce the shipped file. The last one is the
real oracle — it is checkable offline over all 246 banks in the game, which is why
`stage_tex.py verify` exists, and it is what says the swizzle and the bit packing are
right rather than merely plausible (docs/STAGE_MAP_FORMAT.md §4e).
"""
from __future__ import annotations

import os
import sys

TOOLS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)

from mhfu_model import stage as ST                                   # noqa: E402
from mhfu_model.tmh import (parse_tmh, decode_tmh, encode_into,      # noqa: E402
                            quantize, _decode, _deblock, _swizzle_idx)

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                    "workspace", "extracted", "data_files")
STAGE = 98


def _bank(n=STAGE):
    return ST.load(DATA, n).sub(1)


def _palette(raw, img):
    cl = _decode(img.clut_mode, raw[img.clut_off:img.clut_off + img.clut_len])
    return [tuple(cl[k * 4:k * 4 + 4]) for k in range(img.clut_entries)]


def test_the_walk_accounts_for_every_byte():
    # A partial walk is how a same-size edit lands on the wrong image, so
    # parse_tmh refuses rather than returning what it managed to read.
    raw = _bank()
    imgs = parse_tmh(raw)
    assert len(imgs) == 16
    assert imgs[0].ihdr_off == 16                 # straight after the 16-byte header
    assert imgs[-1].end == len(raw)               # and it lands exactly on the end
    for a, b in zip(imgs, imgs[1:]):
        assert a.end == b.ihdr_off                # no gaps, no overlap


def test_a_truncated_bank_raises_instead_of_truncating_silently():
    raw = _bank()
    try:
        parse_tmh(raw[:-64])
    except ValueError:
        return
    raise AssertionError("a short bank walked without complaint")


def test_swizzle_is_the_exact_inverse_of_deblock():
    # _deblock runs on 4-byte pixels and _swizzle_idx on 1-byte palette indices,
    # so drive both off the same permutation and check they undo each other.
    import struct
    for mode, w, h in ((4, 128, 128), (5, 128, 128), (4, 64, 64)):
        n = w * h
        idx = bytes((i * 7 + i // w) & 0xFF for i in range(n))
        swz = _swizzle_idx(mode, w, idx)
        assert len(swz) == n
        # deblock the same bytes widened to u32, then narrow again
        wide = b"".join(struct.pack("<I", b) for b in swz)
        back = _deblock(mode, w, wide)
        assert bytes(back[0::4]) == idx


def test_every_image_re_encodes_to_the_shipped_bytes():
    raw = _bank()
    dec = {d["index"]: d for d in decode_tmh(raw)}
    for img in parse_tmh(raw):
        out = encode_into(raw, img.index, dec[img.index]["rgba"],
                          palette=_palette(raw, img))
        assert out == raw, "image %d did not re-encode to itself" % img.index


def test_an_encode_never_changes_the_size_or_touches_another_image():
    raw = _bank()
    imgs = parse_tmh(raw)
    dec = {d["index"]: d for d in decode_tmh(raw)}
    tgt = imgs[7]
    # a genuinely different picture, so the bytes really do move
    other = dec[3]["rgba"]
    out = encode_into(raw, 7, other[:tgt.width * tgt.height * 4])
    assert len(out) == len(raw)
    assert out[:tgt.pix_off] == raw[:tgt.pix_off]
    assert out[tgt.end:] == raw[tgt.end:]
    assert out[tgt.pix_off:tgt.pix_off + tgt.pix_len] != \
        raw[tgt.pix_off:tgt.pix_off + tgt.pix_len]


def test_wrong_dimensions_are_refused_not_padded():
    raw = _bank()
    try:
        encode_into(raw, 7, b"\x00" * 16)
    except ValueError:
        return
    raise AssertionError("a short RGBA buffer was accepted")


def test_quantise_respects_the_palette_budget():
    # a gradient with far more colours than any stage palette holds
    px = bytearray()
    for y in range(64):
        for x in range(64):
            px += bytes((x * 4 & 0xFF, y * 4 & 0xFF, (x + y) * 2 & 0xFF, 255))
    for n in (16, 256):
        idx, pal = quantize(bytes(px), n)
        assert len(pal) == n
        assert len(idx) == 64 * 64
        assert max(idx) < n


def test_a_texture_imported_from_another_map_decodes_at_the_host_size():
    # the asset-browser case: st139's art in st098's slot. The host header owns
    # the dimensions, so the import must come back out at the HOST's size.
    host = _bank(98)
    donor = [d for d in decode_tmh(_bank(139))
             if (d["width"], d["height"]) == (128, 128)][0]
    out = encode_into(host, 7, donor["rgba"])
    assert len(out) == len(host)
    got = decode_tmh(out)[7]
    assert (got["width"], got["height"]) == (128, 128)
    assert got["rgba"] != decode_tmh(host)[7]["rgba"]


if __name__ == "__main__":
    import traceback
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("  ok   %s" % name)
            except Exception:
                fails += 1; print("  FAIL %s" % name); traceback.print_exc()
    raise SystemExit(1 if fails else 0)
