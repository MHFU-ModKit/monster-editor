"""Slot coverage and label tracking — the two things a clip name can lie about.

Almost all of this runs on synthetic tables and needs neither game data nor a GL
driver, which is deliberate: the classification IS the feature, and it is worth being
able to check on any machine. The last two tests use the real packs and return early
without them, the same contract `test_ports_build.py` uses.

    venv/bin/python mhfu_monster_editor/tests/test_clips.py
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

from mhfu_monster_editor import clips as C
from mhfu_monster_editor import manifest as MF

PORTS = os.path.join(_ROOT, "ports")
ZINOGRE_PAC = os.path.join(_ROOT, "tmp", "zinogre_v10.bin")
HOST_PAC = os.path.join(_ROOT, "workspace", "extracted", "data_files", "file_06185.bin")
ZINOGRE_ANIM = os.path.join(_ROOT, "workspace", "extracted_mhp3", "data_files",
                            "file_05341.bin")

#: a toy port: slot 1 is the idle, 2 and 3 carried, 4 is a copy of the idle (filler),
#: 5 still holds the host's own clip, 6 is neither. Source slot 9 has nowhere to land.
IDLE = (180, True)
SOURCE = {1: IDLE, 2: (60, False), 3: (120, True), 6: (44, False), 9: (300, False)}
HOST = {1: (26, True), 2: (44, False), 3: (50, True), 4: (76, False), 5: (90, True),
        6: (154, False)}
PORT = {1: IDLE, 2: (60, False), 3: (120, True), 4: IDLE, 5: (90, True), 6: (99, True)}


def test_coverage_names_the_four_things_a_slot_can_hold():
    cov = C.coverage(PORT, HOST, SOURCE)
    assert cov.kind(2) == C.CARRIED and cov.kind(3) == C.CARRIED
    assert cov.kind(4) == C.FILLER, cov.kind(4)
    assert cov.kind(5) == C.HOST, cov.kind(5)
    assert cov.kind(6) == C.ALTERED, cov.kind(6)
    assert list(cov.dropped) == [9], cov.dropped
    assert cov.counts()[C.CARRIED] == 3, cov.counts()   # 1, 2 and 3
    assert cov.slots[4].scriptable is False and cov.slots[2].scriptable is True
    print("coverage          carried/filler/host/altered/dropped all separated")


def test_the_idle_slot_is_never_its_own_filler():
    """Slot 1 IS the idle. Calling it filler would flag the one clip that is real."""
    assert C.coverage(PORT, HOST, SOURCE).kind(1) == C.CARRIED
    print("coverage          slot 1 is the idle, not a copy of it")


def test_a_source_clip_that_equals_the_idle_is_carried_not_filler():
    """A donor whose clip 4 happens to BE its idle: the slot holds its own clip."""
    src = dict(SOURCE)
    src[4] = IDLE
    assert C.coverage(PORT, HOST, src).kind(4) == C.CARRIED
    print("coverage          a donor clip that equals the idle is not filler")


def test_without_the_donor_nothing_is_decidable_and_it_says_so():
    """🔴 The important refusal. 'Same length as slot 1' is not evidence of filler on a
    NATIVE pack — two clips sharing an authored length is ordinary."""
    cov = C.coverage(PORT)
    assert set(c.kind for c in cov.slots.values()) == {C.UNKNOWN}, cov.counts()
    assert not cov.has_source and not cov.dropped
    assert "nothing to compare" in cov.summary(), cov.summary()
    print("coverage          no donor -> every slot UNKNOWN, and it explains why")


def test_dropped_needs_the_host_pack_to_be_a_claim():
    """Without the host table, 'no host slot of that index' is not a fact anyone has."""
    assert not C.coverage(PORT, None, SOURCE).dropped
    print("coverage          DROPPED is withheld when the host pack is absent")


# --------------------------------------------------------------------------- #
# labels
# --------------------------------------------------------------------------- #
def _manifest(**clips) -> MF.PortManifest:
    body = ['[port]\nname = "t"\nhost_species = 75\npac = "t.bin"\n',
            "[source]\nmodel = 5339\n"]
    for name, (slot, frames, loop, build) in clips.items():
        body.append('[clips.%s]\nslot = %d\nframes = %d\nloop = %s\nlabel = "x"%s\n'
                    % (name, slot, frames, "true" if loop else "false",
                       '\nlabelled_build = "%s"' % build if build else ""))
    return MF.loads("\n".join(body))


def test_a_label_written_against_this_build_is_current():
    m = _manifest(charge=(2, 60, False, "t.bin@abc123"))
    got = C.track_labels(m, PORT, "t.bin@abc123")[0]
    assert got.status == C.CURRENT and got.trusted, got
    print("labels            same build + same fingerprint -> CURRENT")


def test_a_label_from_another_build_still_holds_if_the_clip_did_not_move():
    m = _manifest(charge=(2, 60, False, "old.bin@000000"))
    got = C.track_labels(m, PORT, "t.bin@abc123")[0]
    assert got.status == C.STILL_VALID and got.trusted, got
    print("labels            other build, fingerprint agrees -> STILL_VALID")


def test_a_rebuilt_pack_reports_which_label_MOVED():
    """🔴 The whole reason #8 exists. `docs/brute_tigrex_anim_ids.txt` was labelled on
    a build where every id later shifted by one, and nothing said so."""
    m = _manifest(charge=(2, 60, False, "old.bin@000000"))
    shifted = {slot + 1: fp for slot, fp in PORT.items()}
    got = C.track_labels(m, shifted, "new.bin@111111")[0]
    assert got.status == C.MOVED and not got.trusted, got
    assert got.now_at == 3, got.now_at
    assert "slot 3, not 2" in got.message, got.message
    print("labels            a shifted rebuild -> MOVED, with the new slot named")


def test_a_length_collision_is_ambiguous_not_a_guess():
    """Two slots with the same fingerprint cannot be told apart; say so."""
    table = {2: (99, True), 7: (60, False), 8: (60, False)}
    m = _manifest(charge=(2, 60, False, None))
    got = C.track_labels(m, table)[0]
    assert got.status == C.AMBIGUOUS and got.candidates == (7, 8), got
    print("labels            two slots share a fingerprint -> AMBIGUOUS, not the first")


def test_a_clip_that_is_not_in_the_build_at_all_is_lost():
    m = _manifest(charge=(2, 1234, False, None))
    got = C.track_labels(m, PORT)[0]
    assert got.status == C.LOST and not got.trusted, got
    print("labels            fingerprint nowhere in the build -> LOST")


def test_a_label_with_no_fingerprint_cannot_be_checked_and_admits_it():
    m = MF.loads('[port]\nname = "t"\nhost_species = 75\npac = "t.bin"\n'
                 '[source]\nmodel = 5339\n[clips.charge]\nslot = 2\n')
    got = C.track_labels(m, PORT)[0]
    assert got.status == C.UNCHECKABLE and not got.trusted, got
    print("labels            no frames recorded -> UNCHECKABLE, never a silent pass")


def test_unlabelled_slots_are_the_ones_worth_naming():
    m = _manifest(charge=(2, 60, False, None))
    assert C.unlabelled_slots(m, PORT) == [1, 3, 4, 5, 6]
    print("labels            unlabelled_slots lists what still has no name")


# --------------------------------------------------------------------------- #
# writing one back
# --------------------------------------------------------------------------- #
def test_label_ops_stamp_the_build_and_survive_a_round_trip():
    text = '[port]\nname = "t"\nhost_species = 75\npac = "t.bin"\n[source]\nmodel = 5339\n'
    ops = C.label_ops("charge", slot=61, label="crazy forward charge", frames=382,
                      loop=False, build="brute.bin@deadbeef")
    m = MF.loads(MF.patch(text, ops))
    c = m.clips["charge"]
    assert (c.slot, c.frames, c.loop) == (61, 382, False), c
    assert c.labelled_build == "brute.bin@deadbeef", c
    assert c.label == "crazy forward charge"
    print("write             label_ops -> patch -> loads round-trips the fingerprint")


def test_renaming_a_clip_follows_the_moves_that_point_at_it():
    text = ('[port]\nname = "t"\nhost_species = 75\npac = "t.bin"\n[source]\n'
            'model = 5339\n[clips.old]\nslot = 2\n[moves.mv]\nmain = 2\nsub = 8\n'
            'clip = "old"\n')
    m = MF.loads(MF.patch(text, C.label_ops("charge", slot=2, rename_from="old")))
    assert "charge" in m.clips and "old" not in m.clips
    assert m.moves["mv"].clip == "charge", m.moves["mv"]
    print("write             a rename re-points every move that named the clip")


def test_import_records_where_the_labels_CAME_from_not_where_we_are():
    """🔴 The hand file says nothing about which build it was filmed on. Stamping the
    build in front of us would manufacture the provenance the file never had."""
    m = _manifest(kept=(2, 60, False, None))
    labels = C.parse_label_file("1 -> idle\n 3 -> roar \n2 -> already named\nnoise\n")
    assert labels == {1: "idle", 3: "roar", 2: "already named"}, labels
    ops = C.import_ops(labels, m, PORT, C.UNRECORDED % "hand.txt")
    tables = {op.table for op in ops if isinstance(op, MF.SetKey)}
    assert tables == {"clips.clip_01", "clips.clip_03"}, tables
    assert not any(op.value == "already named" for op in ops
                   if isinstance(op, MF.SetKey)), "an existing label was clobbered"
    stamped = [op.value for op in ops
               if isinstance(op, MF.SetKey) and op.key == "labelled_build"]
    assert stamped and all("unrecorded" in v for v in stamped), stamped
    print("write             import keeps the labels' own provenance, and does not "
          "overwrite")


def test_import_can_skip_the_slots_that_only_play_the_idle():
    """Labelling a FILLER slot files a name for the idle clip."""
    m = _manifest()
    cov = C.coverage(PORT, HOST, SOURCE)
    ops = C.import_ops({4: "spin to win"}, m, PORT, "x", only_carried=cov)
    assert not ops, "a FILLER slot was labelled"
    assert C.import_ops({2: "charge"}, m, PORT, "x", only_carried=cov)
    print("write             --import only lands on CARRIED slots by default")


# --------------------------------------------------------------------------- #
# the editor's write path, with no window in sight
# --------------------------------------------------------------------------- #
SESSION_TOML = """# a comment that must survive a labelling session
schema = 1

[port]
name = "t"
host_species = 75
pac = "t.bin"

[source]
model = 5339

[clips.old_name]
slot = 2
frames = 60
loop = false

[moves.go]
main = 2
sub = 8
clip = "old_name"
"""


def _session(tmpdir, build="t.bin@abc123"):
    path = os.path.join(tmpdir, "t.toml")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(SESSION_TOML)
    return C.LabelSession(MF.load(path), PORT, build), path


def test_a_labelling_session_stages_before_it_writes():
    """The file is untouched until save — a session is ONE reviewable change."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        s, path = _session(d)
        s.stage(3, "roar", "screams, then walks back")
        assert s.pending and open(path).read() == SESSION_TOML, "it wrote early"
        s.save()
        m = MF.load(path)
        assert m.clips["roar"].label == "screams, then walks back"
        assert (m.clips["roar"].frames, m.clips["roar"].loop) == PORT[3], m.clips["roar"]
        assert m.clips["roar"].labelled_build == "t.bin@abc123"
        assert "# a comment that must survive" in open(path).read()
        assert s.pending == 0 and s.save() == "nothing to save"
    print("session           stage -> save writes once, keeps the comments")


def test_a_session_rename_follows_the_move_that_points_at_the_clip():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        s, path = _session(d)
        s.stage(2, "charge", "the 60f one")
        s.save()
        m = MF.load(path)
        assert "charge" in m.clips and "old_name" not in m.clips
        assert m.moves["go"].clip == "charge", m.moves["go"]
    print("session           renaming a clip re-points the move that used it")


def test_a_session_refuses_a_name_that_would_break_the_file():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        s, _ = _session(d)
        for name, why in (("", "empty"), ("has space", "not a bare key"),
                          ("old_name", "already taken by slot 2")):
            try:
                s.stage(3, name)
            except MF.ManifestError:
                pass
            else:
                raise AssertionError("staged %r (%s)" % (name, why))
        try:
            s.stage(99, "nope")
        except MF.ManifestError as e:
            assert "not populated" in str(e), e
        else:
            raise AssertionError("named a slot this build does not have")
        assert s.pending == 0, "a refused stage still queued something"
    print("session           empty/spaced/duplicate names and empty slots refused")


def test_staging_updates_the_manifest_in_memory_so_a_list_redraws():
    """The clip list is drawn from the manifest; a rename has to show at once."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        s, _ = _session(d)
        s.stage(2, "charge", "x")
        assert s.entry(2).name == "charge" and s.default_name(2) == "charge"
        assert s.entry(4) is None and s.default_name(4) == "clip_04"
    print("session           the in-memory manifest tracks the staged name")


# --------------------------------------------------------------------------- #
# the real packs
# --------------------------------------------------------------------------- #
def test_the_built_zinogre_matches_the_numbers_the_issue_quotes():
    """8 dropped, 30 filler — `tools/port_clip_probe.py --coverage`'s own figures."""
    missing = [p for p in (ZINOGRE_PAC, HOST_PAC, ZINOGRE_ANIM) if not os.path.exists(p)]
    if missing:
        print("SKIP: %s is not here (game data, never committed — docs/ASSETS.md)"
              % os.path.basename(missing[0]))
        return
    port = C.clip_table(open(ZINOGRE_PAC, "rb").read())
    host = C.clip_table(open(HOST_PAC, "rb").read())
    src = C.source_clip_table(open(ZINOGRE_ANIM, "rb").read())
    cov = C.coverage(port, host, src)
    n = cov.counts()
    assert len(cov.dropped) == 8, (len(cov.dropped), sorted(cov.dropped))
    assert n[C.FILLER] == 30, n
    assert n[C.CARRIED] == 34, n
    assert n[C.CARRIED] + n[C.FILLER] == len(port), n
    # and one third of the a1 space plays idle, which is the thing worth knowing
    assert n[C.FILLER] > len(port) / 4
    print("zinogre           %d slots: %d carried, %d filler, %d donor clips dropped"
          % (len(port), n[C.CARRIED], n[C.FILLER], len(cov.dropped)))


def test_the_shipped_manifests_labels_still_point_at_their_clips():
    if not os.path.exists(ZINOGRE_PAC):
        print("SKIP: no built PAC")
        return
    m = MF.load(os.path.join(PORTS, "zinogre.toml"))
    port = C.clip_table(open(ZINOGRE_PAC, "rb").read())
    build = C.build_id(ZINOGRE_PAC)
    tracks = C.track_labels(m, port, build)
    assert tracks and all(t.trusted for t in tracks), \
        [str(t) for t in tracks if not t.trusted]
    assert all(t.status == C.STILL_VALID for t in tracks), \
        "none of them records a build yet, so none can be CURRENT"
    assert "@" in build and build.startswith("zinogre_v10.bin"), build
    print("zinogre           %d manifest labels still point at their clips (%s)"
          % (len(tracks), build))


def main() -> int:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
        except AssertionError as e:
            failed += 1
            print("FAIL %s: %s" % (fn.__name__, e))
    print("\n%d tests, %d failed" % (len(fns), failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
