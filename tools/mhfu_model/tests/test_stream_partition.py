"""The anim stream partition is READ OFF THE BONE TREE, not assumed.

MHFU's animation FK walks a rig in independent streams, and each stream must be a
contiguous run of bone indices (`bone+0x50`). Native monster rigs satisfy that only
because they are authored body-first with the head and tail as the two TRAILING
subtrees — which is what made the old `[n-14, 9, 5]` formula look correct. An MHP3rd
rig owes us nothing: the Zinogre's head sits at bones 18..23, mid-order, so the
formula puts a hind leg in the head stream.

These tests pin the derivation against the one rig whose answer is known — the native
Tigrex, whose shipped partition is 31/9/5.
"""
import os
import struct
import sys

_TOOLS = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, _TOOLS)

from mhfu_model import skeleton as SK
from mhfu_model import skeleton_p3rd as SKP
from mhfu_model.bone_match import bind_world_positions

_DATA = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                     "workspace", "extracted", "data_files")
_P3 = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                   "workspace", "extracted_mhp3", "data_files")
TIGREX = os.path.join(_DATA, "file_06185.bin")
ZINOGRE = os.path.join(_P3, "file_05339.bin")


def _subs(b):
    n = struct.unpack_from("<I", b, 0)[0]
    return [struct.unpack_from("<II", b, 4 + i * 8) for i in range(n)]


def _skel(path, magic):
    b = open(path, "rb").read()
    for o, s in _subs(b):
        if b[o:o + 4] == magic:
            return b[o:o + s]
    raise AssertionError("no skeleton sub in %s" % path)


def _rig(blob):
    sk = SKP.parse(blob)
    par = [x.parent for x in sk.bones]
    bw = bind_world_positions(par, [tuple(x.bind_pos) for x in sk.bones])
    return sk, par, bw


def _stream_ids(blob):
    sk = SKP.parse(blob)
    out, o = [], len(sk.header)
    for b in sk.bones:
        out.append(struct.unpack_from("<H", blob, o + 0x50)[0])
        o += b.section_size
    return out


def test_native_tigrex_partition_is_reproduced():
    """The ground truth: the shipped native partition is 31/9/5, in bone order."""
    blob = _skel(TIGREX, b"\x00\x00\x00\xc0")
    _sk, par, bw = _rig(blob)
    split, order = SK.derive_stream_partition(par, bw, 45)
    assert split == [31, 9, 5]
    assert order == list(range(len(par))), "a native rig needs no reordering"
    # and that is exactly what the file itself says
    sids = _stream_ids(blob)
    assert sids[:31] == [0] * 31
    assert sids[31:40] == [1] * 9
    assert sids[40:45] == [2] * 5


def test_streams_are_head_and_tail_not_a_fixed_slice():
    """Each derived stream is one complete subtree — head forward, tail back."""
    blob = _skel(TIGREX, b"\x00\x00\x00\xc0")
    _sk, par, bw = _rig(blob)
    split, _ = SK.derive_stream_partition(par, bw, 45)
    body, head, tail = (range(0, 31), range(31, 40), range(40, 45))
    assert max(bw[i][2] for i in head) > max(bw[i][2] for i in body)   # snout
    assert min(bw[i][2] for i in tail) < min(bw[i][2] for i in body)   # tail tip
    for rng in (head, tail):                     # exactly one root per stream
        outside = [i for i in rng if par[i] not in rng]
        assert len(outside) == 1


def test_zinogre_needs_reordering_and_gets_it():
    """A rig whose head is mid-order: the formula fails, the derivation doesn't."""
    blob = _skel(ZINOGRE, b"\x00\x00\x00\x80")
    _sk, par, bw = _rig(blob)
    animated = 46
    assert SK._default_split(animated) == [32, 9, 5]      # the old, wrong answer
    split, order = SK.derive_stream_partition(par, bw, animated)
    assert split == [33, 6, 7]
    assert order != list(range(len(par)))
    assert sorted(order) == list(range(len(par)))
    # the head subtree really is bones 18..23 in SOURCE numbering
    assert order[33:39] == [18, 19, 20, 21, 22, 23]
    assert order[39:46] == [39, 40, 41, 42, 43, 44, 45]


def test_reorder_keeps_every_parent_ahead_of_its_child():
    for path, magic, animated in ((TIGREX, b"\x00\x00\x00\xc0", 45),
                                  (ZINOGRE, b"\x00\x00\x00\x80", 46)):
        _sk, par, bw = _rig(_skel(path, magic))
        _split, order = SK.derive_stream_partition(par, bw, animated)
        npar = SK.reorder_bones(par, order)
        assert all(npar[i] < i for i in range(len(npar)) if npar[i] >= 0)


def test_converted_skeleton_carries_the_reordered_partition():
    """p3rd_to_mhfu must apply the permutation to the sections AND the stream ids."""
    blob = _skel(ZINOGRE, b"\x00\x00\x00\x80")
    _sk, par, bw = _rig(blob)
    split, order = SK.derive_stream_partition(par, bw, 46)
    out = SK.p3rd_to_mhfu(blob, split=split, order=order)
    sids = _stream_ids(out)
    assert sids[:33] == [0] * 33
    assert sids[33:39] == [1] * 6
    assert sids[39:46] == [2] * 7
    # bind positions follow the permutation, bone for bone
    _sk2, npar, nbw = _rig(out)
    for new_i, old_i in enumerate(order):
        assert nbw[new_i] == bw[old_i]
    assert npar == SK.reorder_bones(par, order)


def test_child_and_sibling_links_are_rebuilt_correctly():
    """After a permutation the shipped child/sibling links are stale; they are
    derivable from the parent array and must be regenerated."""
    blob = _skel(ZINOGRE, b"\x00\x00\x00\x80")
    _sk, par, bw = _rig(blob)
    split, order = SK.derive_stream_partition(par, bw, 46)
    out = SK.p3rd_to_mhfu(blob, split=split, order=order)
    sk2 = SKP.parse(out)
    npar = [b.parent for b in sk2.bones]
    kids = {}
    for i, p in enumerate(npar):
        if p >= 0:
            kids.setdefault(p, []).append(i)
    for p, cs in kids.items():
        cs = sorted(cs)
        assert sk2.bones[p].child == cs[0]
        for a, b in zip(cs, cs[1:]):
            assert sk2.bones[a].sibling == b
        assert sk2.bones[cs[-1]].sibling == -1


def test_the_formula_split_fragments_the_zinogre_head():
    """The check has teeth: with the old formula, stream 1 is THREE fragments."""
    blob = _skel(ZINOGRE, b"\x00\x00\x00\x80")
    _sk, par, _bw = _rig(blob)
    out = SK.p3rd_to_mhfu(blob, split=SK._default_split(46))     # [32, 9, 5]
    npar = [b.parent for b in SKP.parse(out).bones]
    rng = set(range(32, 41))
    roots = [i for i in rng if npar[i] >= 0 and npar[i] not in rng]
    assert len(roots) == 3, roots           # a hind leg, a stub and the tail base


def test_authentic_skin_survives_the_reorder():
    """The permutation must reach the SKIN too, or every vertex moves to the
    wrong joint. Checked against the source, not against the porter."""
    import verify_port as VP
    geo = os.path.join(_P3, "file_05340.bin")
    port = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                        "tmp", "zinogre_streamfix.bin")
    if not os.path.exists(port):
        return                              # built by tools/build_p3rd_port.py
    res = VP.check(open(port, "rb").read(), open(ZINOGRE, "rb").read(),
                   open(geo, "rb").read())
    failed = [(n, d) for ok, n, d in res if not ok]
    assert not failed, failed


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    bad = 0
    for f in fns:
        try:
            f()
            print("  PASS  %s" % f.__name__)
        except Exception as e:
            bad += 1
            print("  FAIL  %s: %s" % (f.__name__, e))
    print("\n%d tests, %d failed" % (len(fns), bad))
    sys.exit(1 if bad else 0)
