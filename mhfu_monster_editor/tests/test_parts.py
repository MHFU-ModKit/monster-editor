"""The part system's write path: naming, editing the grid, adopting the host's tables.

`parts.PartSession` is the editor's whole authoring surface for issue #10, and none of
it needs a window — which is the point of it living outside `ui/app.py`. What is worth
pinning here is not that a value round-trips but that the session refuses the things
the engine would silently fold:

* two names for one part index (two break bars that are one bar);
* a grid state that is not seven rows of ten;
* a percentage outside a byte.

and that adopting the host's volumes REPORTS the bone indices that do not fit, instead
of writing 153 records and calling it done.
"""
import os
import shutil
import sys
import tempfile

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

from mhfu_monster_editor import intel as I
from mhfu_monster_editor import manifest as MF
from mhfu_monster_editor import parts as P

PORTS = os.path.join(_ROOT, "ports")
ZINOGRE = os.path.join(PORTS, "zinogre.toml")

ROW = [100, 75, 65, 40, 0, 15, 5, 30, 20, 110]
GRID = [list(ROW)] + [[0] * 10 for _ in range(6)]


class _State:
    """The shape `adopt_grid` consumes — anything with `.rows`."""

    def __init__(self, rows):
        self.rows = rows


def _session(n_bones=46):
    return P.PartSession(MF.load(ZINOGRE), n_bones=n_bones)


# --------------------------------------------------------------------------- #
# naming
# --------------------------------------------------------------------------- #
def test_naming_a_part_stages_a_table_key_and_nothing_else():
    s = _session()
    s.name_part(1, "head", label="KO lands here", hitzone_row=2)
    assert s.pending == 1
    assert s.name_of(1) == "head"
    ops = s.ops()
    assert all(isinstance(o, MF.SetKey) for o in ops)
    assert {o.key for o in ops} == {"index", "hitzone_row", "label"}
    assert all(o.table == "parts.head" for o in ops)


def test_two_names_for_one_slot_are_refused_at_the_session():
    """The validator catches it too, but by then it is in the file. The session is
    where an author finds out, and the message has to say why it matters."""
    s = _session()
    s.name_part(1, "head")
    try:
        s.name_part(2, "head")
    except MF.ManifestError as e:
        assert "secretly one" in str(e), e
        return
    raise AssertionError("one name was staged onto two slots")


def test_a_part_index_the_engine_cannot_address_is_refused():
    s = _session()
    for idx in (-1, 8, 9):
        try:
            s.name_part(idx, "x")
        except MF.ManifestError:
            continue
        raise AssertionError("part %d was accepted" % idx)


def test_a_name_that_is_not_a_bare_toml_key_is_refused():
    s = _session()
    for bad in ("left wing", "Head", "2nd", "tail!", ""):
        try:
            s.name_part(1, bad)
        except MF.ManifestError:
            continue
        raise AssertionError("%r was accepted as a part name" % bad)
    s.name_part(1, "left_wing")           # the way to spell it


# --------------------------------------------------------------------------- #
# the grid
# --------------------------------------------------------------------------- #
def test_editing_a_cell_without_a_grid_says_so_rather_than_creating_one():
    """A manifest with no `[[hitzone]]` INHERITS the host's. Silently materialising
    a blank grid on the first keystroke would make a monster nothing can hurt."""
    s = _session()
    try:
        s.set_hitzone(0, 0, "cut", 50)
    except MF.ManifestError as e:
        assert "adopt" in str(e), e
        return
    raise AssertionError("a grid appeared out of nowhere")


def test_a_cell_edit_leaves_every_other_cell_alone():
    s = _session()
    s.add_state("normal", GRID)
    s.set_hitzone(0, 0, "thunder", 55)
    rows = s.state(0).rows
    assert rows[0][7] == 55
    assert rows[0][1] == 75 and rows[0][9] == 110
    assert rows[1] == [0] * 10


def test_a_percentage_outside_a_byte_is_refused():
    s = _session()
    s.add_state("normal", GRID)
    for v in (-1, 256, 1000):
        try:
            s.set_hitzone(0, 0, "cut", v)
        except MF.ManifestError:
            continue
        raise AssertionError("%r was accepted as a percentage" % v)


def test_a_column_can_be_addressed_by_name_or_by_index():
    s = _session()
    s.add_state("normal", GRID)
    s.set_hitzone(0, 2, "fire", 33)
    s.set_hitzone(0, 3, 4, 44)
    assert s.state(0).rows[2][4] == 33 and s.state(0).rows[3][4] == 44


def test_a_state_that_is_not_seven_rows_is_refused():
    s = _session()
    try:
        s.add_state("short", GRID[:6])
    except MF.ManifestError:
        return
    raise AssertionError("a six-row state was staged")


def test_two_states_cannot_share_a_name():
    s = _session()
    s.add_state("normal", GRID)
    try:
        s.add_state("normal", GRID)
    except MF.ManifestError as e:
        assert "INDEX" in str(e), e
        return
    raise AssertionError("two states called 'normal' were staged")


# --------------------------------------------------------------------------- #
# adopting the host's
# --------------------------------------------------------------------------- #
def test_adopting_the_grid_names_the_states_and_says_where_they_came_from():
    s = _session()
    n = s.adopt_grid([_State(GRID), _State([list(r) for r in GRID])])
    assert n == 2
    assert [x.name for x in s.states()] == ["normal", "enraged"]
    assert "inherits" in s.states()[0].label


def test_adopting_volumes_reports_the_bones_that_do_not_fit():
    """🔴 The fault the whole feature exists to expose. The host's indices belong to
    the HOST's rig; a port ships its own, so a copied sphere lands on whatever joint
    happens to sit at that number. Writing them and saying nothing is the bug."""
    class S:
        def __init__(self, bone, part=1, row=2, r=100.0):
            self.bone, self.part, self.hitzone_row, self.radius = bone, part, row, r
            self.a, self.b, self.is_capsule = (0.0, 0.0, 0.0), None, False

    s = _session(n_bones=46)
    got = s.adopt_volumes([S(2), S(45), S(46), S(125)], source="em75")
    assert got.adopted == 4
    assert got.off_rig == [46, 125], got.off_rig
    assert not got.clean
    assert "does not have" in got.describe()
    assert "em75" in got.describe()


def test_a_clean_adoption_still_refuses_to_claim_the_bones_are_right():
    """Every index existing is not every index meaning the right joint, and the
    difference is exactly what a porter gets wrong."""
    class S:
        bone, part, hitzone_row, radius = 2, 1, 2, 100.0
        a, b, is_capsule = (0.0, 0.0, 0.0), None, False

    s = _session(n_bones=46)
    got = s.adopt_volumes([S()])
    assert got.clean
    assert "not the same as pointing at the right joint" in got.describe()


def test_a_capsule_survives_adoption_as_a_capsule():
    class S:
        bone, part, hitzone_row, radius = 6, 4, 5, 65.0
        a, b, is_capsule = (35.0, 0.0, 0.0), (330.0, 0.0, 0.0), True

    s = _session()
    s.adopt_volumes([S()])
    h = s.volumes()[-1]
    assert h.shape == "capsule" and h.to == [330.0, 0.0, 0.0]


def test_adoption_can_be_limited_to_the_parts_you_asked_for():
    class S:
        def __init__(self, part):
            self.bone, self.part, self.hitzone_row, self.radius = 2, part, 0, 10.0
            self.a, self.b, self.is_capsule = (0.0, 0.0, 0.0), None, False

    s = _session()
    got = s.adopt_volumes([S(1), S(3), S(6)], parts=[1, 6])
    assert got.adopted == 2
    assert sorted(h.part for h in s.volumes()) == [1, 6]


def test_an_adopted_volume_takes_the_name_already_staged_for_its_part():
    class S:
        bone, part, hitzone_row, radius = 2, 1, 2, 97.0
        a, b, is_capsule = (0.0, 0.0, 0.0), None, False

    s = _session()
    s.name_part(1, "head")
    s.adopt_volumes([S()])
    assert s.volumes()[-1].label == "head"


# --------------------------------------------------------------------------- #
# writing
# --------------------------------------------------------------------------- #
def test_a_session_writes_only_the_parts_and_keeps_the_prose():
    """The whole reason the editor patches instead of re-emitting: a manifest that is
    half comments must come back with them."""
    src = open(ZINOGRE, encoding="utf-8").read()
    with tempfile.TemporaryDirectory() as td:
        dst = os.path.join(td, "zinogre.toml")
        shutil.copy(ZINOGRE, dst)
        m = MF.load(dst)
        s = P.PartSession(m, n_bones=46)
        s.name_part(1, "head", hitzone_row=2)
        s.name_part(3, "tail", severable=True)
        s.add_state("normal", GRID)
        s.set_hitzone(0, 0, "thunder", 55)
        msg = s.save()
        assert msg and "2 part name(s)" in msg
        assert s.pending == 0, "save did not clear the session"

        back = MF.load(dst)
        assert {n: p.index for n, p in back.parts.items()} == {"head": 1, "tail": 3}
        assert back.parts["tail"].severable is True
        assert back.hitzones[0].value(0, "thunder") == 55
        new = open(dst, encoding="utf-8").read()
        assert new.count("#") >= src.count("#"), "comments were lost"
        # the untouched half of the file is byte for byte what it was
        assert new.startswith(src[:src.index("[clips.")])


def test_saving_nothing_writes_nothing():
    with tempfile.TemporaryDirectory() as td:
        dst = os.path.join(td, "zinogre.toml")
        shutil.copy(ZINOGRE, dst)
        before = open(dst, "rb").read()
        assert P.PartSession(MF.load(dst)).save() is None
        assert open(dst, "rb").read() == before


def test_discard_leaves_the_manifest_as_it_was():
    s = _session()
    s.name_part(1, "head")
    s.add_state("normal", GRID)
    assert s.pending == 2
    s.discard()
    assert s.pending == 0 and s.name_of(1) == "" and s.states() == []


def test_the_preview_refuses_a_patch_that_would_not_load_back():
    """`patch` re-parses its own output, so a session that would produce an unreadable
    manifest fails at the call rather than on the next session."""
    s = _session()
    s._grids[0] = MF.HitzoneState(name="broken", rows=[[0] * 10] * 6)   # six rows
    try:
        s.preview(open(ZINOGRE, encoding="utf-8").read())
    except MF.ManifestError:
        return
    raise AssertionError("a six-row grid was written")


def test_a_manifest_with_no_path_says_where_it_would_go_rather_than_guessing():
    s = P.PartSession(MF.loads(open(ZINOGRE, encoding="utf-8").read()))
    s.name_part(1, "head")
    try:
        s.save()
    except MF.ManifestError as e:
        assert "nowhere to save" in str(e), e
        return
    raise AssertionError("a pathless manifest was saved somewhere")


# --------------------------------------------------------------------------- #
# against the real host
# --------------------------------------------------------------------------- #
def test_the_tigrex_volumes_do_not_fit_the_zinogre_rig():
    """The measured version of the warning: the Zinogre ships 46 joints and 19 of
    em75's 153 volumes name a bone it does not have. Adopting them is still the right
    starting point — you can SEE where they land — but the count has to be told."""
    si = I.find_intel(75)
    if si is None or not si.parts.present:
        return
    s = _session(n_bones=46)
    got = s.adopt_volumes(si.parts.spheres(), source="em75")
    assert got.adopted == len(si.parts.spheres())
    assert got.off_rig, "every host bone index fitted, which would be a surprise"
    assert s.preview(open(ZINOGRE, encoding="utf-8").read())


def test_summarise_says_the_port_inherits_a_grid_it_has_not_authored():
    si = I.find_intel(75)
    if si is None or not si.parts.present:
        return
    lines = P.summarise(MF.load(ZINOGRE), si.parts)
    assert any("INHERITS" in x for x in lines), lines


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
