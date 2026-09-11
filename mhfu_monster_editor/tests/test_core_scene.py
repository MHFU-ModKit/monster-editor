"""Does the scene describe what is actually in the PAC — and does the PORT pose right?

Two questions, and only the second one is hard.

**Loading** is checked against the numbers the repo already pins elsewhere: the native
Tigrex's 48 bones / 214 groups / 4128 vertices / 5 textures / 64 clips, the Zinogre's
51 / 181 / 4180 / 5 / 42, and the two partial slots (24, 25) every Tigrex-framed pack
carries.

**Posing** is checked the only way that is worth anything: the MHP3rd donor and the
BUILT port are two different files, in two different formats, read by two different
parsers, under two different record→bone conventions — the donor through
`p3rd_anim_map`, the build positionally. Pose both and compare joint for joint under
the porter's bone PERMUTATION, with its constant ground lift removed. If the map this
loader draws with were not the map the porter built with, they could not agree.

    zinogre       96 poses x 46 joints   worst 2.3e-13 u   lift 165.31 (lead pad 0)
    brute_tigrex 126 poses x 43 joints   worst 0.0e+00 u   lift   0.00 (lead pad 1)

🔴 That comparison must be done under the permutation, never by index: `port_p3rd`
reorders bones (source 18–23 → built 33–38, source 24–38 → built 18–32) and an
index-wise diff reports 570–830-unit differences on exactly that band while everything
else matches. The permutation is recovered here by tree isomorphism, the same way
`tools/port_anim_verify.py` recovers it — which needs Blender's `mathutils` and so
cannot run on this machine at all.

⚠️ Needs the extracts under `workspace/`; returns early without them. The built PAC is
game data (`docs/ASSETS.md` §D) and is never committed, so it is built into a temp dir.
"""
import os
import sys
import tempfile

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "tools"))

import numpy as np

from mhfu_monster_editor import manifest as MF
from mhfu_monster_editor.core import SceneError, open_scene

from mhfu_model import convert as C
from mhfu_model import p3rd_anim_map as P3AM

MHFU = os.path.join(_ROOT, "workspace", "extracted", "data_files")
MHP3 = os.path.join(_ROOT, "workspace", "extracted_mhp3", "data_files")
TIGREX = os.path.join(MHFU, "file_06185.bin")
ZINOGRE_SRC = os.path.join(MHP3, "file_05339.bin")
BRUTE_SRC = os.path.join(MHP3, "file_05248.bin")
PORTS = os.path.join(_ROOT, "ports")

#: the same tolerance the golden test uses. The measured worst here is 1.4e-13.
TOL = 1e-9


def _have_data():
    return all(os.path.exists(p) for p in (TIGREX, ZINOGRE_SRC, BRUTE_SRC))


def _built(name, tmpdir):
    """The built port PAC — reused from `tmp/` if it is there, else built (~1 s).

    Never committed: a built port is Capcom data spliced from two games
    (`docs/ASSETS.md` §D), so a fresh checkout has to make its own.
    """
    m = MF.load(os.path.join(PORTS, name + ".toml"))
    shipped = os.path.join(_ROOT, "tmp", m.pac)
    if os.path.exists(shipped):
        return shipped
    import build_p3rd_port
    out = os.path.join(tmpdir, m.pac)
    build_p3rd_port.main(["--manifest", os.path.join(PORTS, name + ".toml"),
                          "--data-root", os.path.join(_ROOT, "workspace"), "--out", out])
    return out


# --------------------------------------------------------------------------- #
# loading — both games, through the same door
# --------------------------------------------------------------------------- #
def test_the_native_tigrex_loads_as_a_whole_scene():
    if not _have_data():
        return                      # extracts are gitignored; docs/ASSETS.md rebuilds
    sc = open_scene(TIGREX)
    assert sc.game == "mhfu"
    assert sc.rig.n_bones == 48
    assert len(sc.groups) == 214
    assert sc.n_vertices == 4128
    assert sum(g.n_faces for g in sc.groups) == 2919
    assert len(sc.textures) == 5
    assert [(t.width, t.height) for t in sc.textures] == \
        [(128, 128), (128, 128), (128, 128), (128, 128), (64, 64)]
    assert all(t.rgba.shape == (t.height, t.width, 4) for t in sc.textures)
    assert len(sc.clips) == 64
    # every group binds a decoded texture and carries UVs — otherwise it draws blank
    assert all(g.texture is not None for g in sc.groups)
    assert all(g.uvs is not None and g.uvs.shape == (g.n_vertices, 2) for g in sc.groups)
    assert all(g.normals is not None for g in sc.groups)
    assert sc.record_to_bone is None, "an MHFU pack is positional"


def test_the_mhp3rd_donor_loads_as_a_whole_scene():
    """PMO v102 + its companion geometry file + its separate moveset file."""
    if not _have_data():
        return
    sc = open_scene(ZINOGRE_SRC)
    assert sc.game == "mhp3rd"
    assert sc.rig.n_bones == 51
    assert len(sc.groups) == 181
    assert sc.n_vertices == 4180
    assert len(sc.textures) == 5
    assert len(sc.clips) == 42
    assert all(g.uvs is not None for g in sc.groups)
    assert max(g.skin.max_influences for g in sc.groups) > 1, "the v102 palette is lost"


def test_a_pac_with_no_skeleton_is_refused_not_half_loaded():
    if not _have_data():
        return
    small = os.path.join(MHFU, "file_06060.bin")     # em01, a small monster: no skeleton
    if not os.path.exists(small):
        return
    try:
        open_scene(small)
    except SceneError as e:
        assert "skeleton" in str(e) or "geometry" in str(e), e
    else:
        raise AssertionError("a PAC with no big-monster skeleton was accepted")


# --------------------------------------------------------------------------- #
# the two mappings that are not identity
# --------------------------------------------------------------------------- #
def test_a_partial_clip_poses_the_joints_it_owns_not_joints_zero_upwards():
    """🔴 Slots 24 and 25 exist in ONE of the three joint-partition streams: a
    head-and-neck clip meant to play over an idle body. `to_flat_anim` concatenates
    what it finds, so read positionally those 9 tracks land on joints 0..8 — the root,
    the spine, a foreleg — and the monster folds in half. The joint base comes from the
    partition widths (31 + 9 + 5 on the native Tigrex), so they land on 31..39."""
    if not _have_data():
        return
    sc = open_scene(TIGREX)
    partial = [c for c in sc.clips if not c.whole_rig]
    assert [c.slot for c in partial] == [24, 25], [c.slot for c in partial]
    for c in partial:
        assert c.tracks == 9, c.tracks
        assert min(c.track_to_joint.values()) == 31, c.track_to_joint
        assert max(c.track_to_joint.values()) == 39, c.track_to_joint
        driven = sc.pose(c, c.frames // 2).joints - sc.rig.bind_joints
        moved = np.flatnonzero(np.linalg.norm(driven, axis=1) > 1e-6)
        assert moved.min() >= 31, "a partial clip moved joint %d" % moved.min()
    # and the whole-rig clips are untouched by all this
    assert sum(1 for c in sc.clips if c.whole_rig) == 62


def test_the_donor_map_comes_from_p3rd_anim_map_and_passes_the_fork_rule():
    """🔴 Never a locally re-derived offset. The Zinogre is em040, offset 0, and his 37
    records must consume joints 0..45 — leaving 46..50, the severed-tail carve object
    the moveset never drives. And the LOCATION records must land at or above the body
    fork, or half the animal lifts and the waist tears."""
    if not _have_data():
        return
    sc = open_scene(ZINOGRE_SRC)
    want = {r: b for b, r in P3AM.for_monster(40, 37, 51).items()}
    assert sc.record_to_bone == want
    assert min(want.values()) == 0 and max(want.values()) == 45
    assert sorted(set(range(51)) - set(want.values())) == \
        sorted(set(P3AM.SKIPPED_BONES[40]) | {46, 47, 48, 49, 50})
    loc = sorted({sc.record_to_bone[r]
                  for c in sc.clips for r, tr in enumerate(c._anim.tracks)
                  if r in sc.record_to_bone
                  and any(C.channel_kind(ch.type)[0] == "loc"
                          for ch in tr.channels if ch.keyframes)})
    assert P3AM.loc_below_fork(list(sc.rig.parents), loc) == [], loc
    assert not any("fork" in n for n in sc.notes), sc.notes


def test_an_unmapped_donor_says_so_loudly():
    """An MHP3rd monster with no row in `p3rd_anim_map` falls back to the addon's
    default offset of 2, which the fork rule says is wrong for both monsters we have
    measured. That silence is what hid the Brute's mismapping for months."""
    if not _have_data():
        return
    sc = open_scene(ZINOGRE_SRC, em_id=-1)
    assert any("DEFAULT_BONE_OFFSET" in n for n in sc.notes), sc.notes


# --------------------------------------------------------------------------- #
# the manifest
# --------------------------------------------------------------------------- #
def test_a_manifest_opens_both_sides_and_its_clip_fingerprints_check_out():
    """`manifest.Clip` carries slot/frames/loop precisely so it can be checked against
    a build. If it could not be, the manifest would just be a second place to be wrong."""
    if not _have_data():
        return
    m = MF.load(os.path.join(PORTS, "zinogre.toml"))
    with tempfile.TemporaryDirectory() as td:
        port = open_scene(m, pac=_built("zinogre", td))
    assert port.game == "mhfu" and port.rig.n_bones == 51
    assert port.clip_mismatches() == [], port.clip_mismatches()
    # slot 10 was `clip_10` until the 2026-09-10 labelling session named it
    c10 = next(c for c in port.clips if c.slot == 10)
    assert c10.name == "standing_charge_up" and c10.frames == 408
    assert port.clip("standing_charge_up").loop is True
    src = open_scene(m, side="source")
    assert src.game == "mhp3rd" and src.record_to_bone is not None
    assert src.manifest is m


def test_a_missing_built_pac_names_the_command_that_builds_it():
    m = MF.load(os.path.join(PORTS, "zinogre.toml"))
    try:
        open_scene(m, pac=os.path.join(tempfile.gettempdir(), "no_such_port.bin"))
    except SceneError as e:
        assert "build_p3rd_port.py" in str(e), e
    else:
        raise AssertionError("a missing built PAC loaded anyway")


# --------------------------------------------------------------------------- #
# THE evidence: the port plays its source moveset
# --------------------------------------------------------------------------- #
def _children(rig):
    out = {}
    for i, p in enumerate(rig.parents.tolist()):
        out.setdefault(p, []).append(i)
    return out


def bone_permutation(src, port):
    """``{source joint: built joint}`` by tree isomorphism — same shape, same offsets.

    `port_p3rd` may also insert a LEAD PAD of origin joints at the root, so anchoring
    blindly at built joint 0 can match almost nothing; try every node down the built
    root's single-child chain and keep the best. (`tools/port_anim_verify.py`.)
    """
    sk, pk = _children(src.rig), _children(port.rig)
    sb, pb = src.rig.bind_local, port.rig.bind_local

    def isomorphism(start):
        perm = {}

        def walk(si, bi):
            perm[si] = bi
            used = set()
            for s in sk.get(si, []):
                best, bd = None, 1e9
                for b in pk.get(bi, []):
                    if b in used:
                        continue
                    d = float(np.abs(sb[s] - pb[b]).max())
                    if d < bd:
                        best, bd = b, d
                if best is not None and bd < 0.5:      # a loose match is NOT a match
                    used.add(best)
                    walk(s, best)
        walk(0, start)
        return perm

    best, pad, node = {}, 0, 0
    for depth in range(8):
        cand = isomorphism(node)
        if len(cand) > len(best):
            best, pad = cand, depth
        kids = pk.get(node, [])
        if len(kids) != 1:
            break
        node = kids[0]
    return best, pad


def _plays_its_source_moveset(name):
    """Pose the donor and the build at 3 frames of every shared clip; return the worst.

    Returns ``(perm, pad, compared, worst, lifts, partial)``.
    """
    m = MF.load(os.path.join(PORTS, name + ".toml"))
    src = open_scene(m, side="source")
    with tempfile.TemporaryDirectory() as td:
        port = open_scene(m, pac=_built(name, td))
    perm, pad = bone_permutation(src, port)
    n_src = src.rig.n_bones
    assert len(perm) >= 0.8 * n_src, (
        "🔴 only %d of %d joints lined up — that is NOT a verdict on the port, it is "
        "the comparison failing. Refuse to judge." % (len(perm), n_src))

    s_idx = np.array(sorted(perm))
    b_idx = np.array([perm[i] for i in s_idx])
    worst, compared, lifts, partial = 0.0, 0, set(), []
    for c in src.clips:
        try:
            pc = port.clip(c.slot)
        except KeyError:
            continue                    # the build does not carry this source slot
        # ⚠️ not every port slot holds a whole-rig clip: the host fills 24/25 in one
        # stream only. Diffing a whole-rig source clip against one is meaningless.
        if len(pc.driven) * 2 < len(c.driven):
            partial.append(c.slot)
            continue
        for frac in (0.25, 0.5, 0.75):
            frame = int(max(c.frames, 1) * frac)
            a = src.pose(c, frame).joints[s_idx].copy()
            b = port.pose(pc, frame).joints[b_idx]
            # the porter's ground lift is a constant Y offset on every joint; take it
            # off explicitly so it can never mask a real difference.
            lift = float(port.pose(pc, frame).joints[perm[0]][1]
                         - src.pose(c, frame).joints[0][1])
            a[:, 1] += lift
            lifts.add(round(lift, 2))
            worst = max(worst, float(np.linalg.norm(a - b, axis=1).max()))
            compared += 1
    print("      %-13s %3d poses x %2d joints, worst %.2e u, lift %s, pad %d"
          % (name, compared, len(s_idx), worst, sorted(lifts), pad))
    assert len(lifts) == 1, sorted(lifts)
    assert abs(lifts.pop() - m.build.ground_lift) < 0.05
    assert partial == [24, 25], partial
    assert worst < TOL, worst
    return perm, pad, compared


def test_the_built_zinogre_plays_its_source_moveset_joint_for_joint():
    if not _have_data():
        return
    perm, pad, compared = _plays_its_source_moveset("zinogre")
    assert pad == 0 and compared >= 90, (pad, compared)
    assert len(perm) == 46, len(perm)
    assert sorted(set(range(51)) - set(perm)) == [46, 47, 48, 49, 50], \
        "only the severed-tail carve object may go unmatched"
    # the documented reordering, recovered here from the bind offsets alone
    assert [perm[i] for i in range(18, 24)] == [33, 34, 35, 36, 37, 38], perm
    assert [perm[i] for i in range(24, 39)] == list(range(18, 33)), perm


def test_the_built_brute_tigrex_plays_its_source_moveset_too():
    """The second port, and the one that matters most for the MAP: he is absent from
    upstream `skipped_bones.md`, so he silently took `DEFAULT_BONE_OFFSET` for months.
    He also exercises the LEAD PAD — `port_p3rd` builds 47 joints from his 46, so every
    source joint sits one deeper and anchoring at built joint 0 matches almost nothing.
    """
    if not _have_data():
        return
    perm, pad, compared = _plays_its_source_moveset("brute_tigrex")
    assert pad == 1, pad
    assert len(perm) == 43 and compared >= 120, (len(perm), compared)
    assert sorted(set(range(46)) - set(perm)) == [43, 44, 45], \
        "only his severed-tail chain may go unmatched"


def test_the_geometry_survives_the_port_untouched():
    """Same vertex count, same triangle count, same texture set as the donor — the
    port splices HIS mesh onto an MHFU frame, it does not rebuild it."""
    if not _have_data():
        return
    m = MF.load(os.path.join(PORTS, "zinogre.toml"))
    src = open_scene(m, side="source")
    with tempfile.TemporaryDirectory() as td:
        port = open_scene(m, pac=_built("zinogre", td))
    assert port.n_vertices == src.n_vertices == 4180
    assert len(port.groups) == len(src.groups) == 181
    assert sum(g.n_faces for g in port.groups) == sum(g.n_faces for g in src.groups)
    assert [(t.width, t.height) for t in port.textures] == \
        [(t.width, t.height) for t in src.textures]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    if not _have_data():
        print("  workspace/ extracts absent — the scene tests are skipped "
              "(docs/ASSETS.md)")
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
