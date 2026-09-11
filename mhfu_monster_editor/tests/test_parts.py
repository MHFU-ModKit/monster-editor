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
SHIPPED = os.path.join(PORTS, "zinogre.toml")


def _bare(shipped=SHIPPED):
    """The shipped manifest with its `[[hurtbox]]` / `[[hitzone]]` blocks stripped.

    Every test here starts from "nothing authored yet"; since 2026-09-11 the shipped
    file carries the user's own tables (the #19 experiment), so the fixture is a
    temp copy with those blocks patched out — the prose and the clips untouched.
    """
    text = open(shipped, encoding="utf-8").read()
    m = MF.loads(text)
    assert not m.parts, "the fixture cannot strip named [parts.*] tables"
    ops = [MF.ReplaceBlock("hurtbox", i, None) for i in range(len(m.hurtboxes) - 1, -1, -1)]
    ops += [MF.ReplaceBlock("hitzone", i, None) for i in range(len(m.hitzones) - 1, -1, -1)]
    bare = MF.patch(text, ops) if ops else text
    d = tempfile.mkdtemp(prefix="mhfu_bare_")
    path = os.path.join(d, "zinogre.toml")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(bare)
    return path


#: the bare fixture; a temp file, so `save()` in a test never touches ports/
ZINOGRE = _bare()

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
    # 125 (0x7D) is the walker's tail-sever MARKER, not a joint: expected off-rig,
    # kept in order, and not a fault to report
    assert got.off_rig == [46], got.off_rig
    assert not got.clean
    assert "does not have" in got.describe()
    assert "em75" in got.describe()
    assert s.volumes()[3].is_marker and not s.volumes()[2].is_marker


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
def test_the_tigrex_volumes_adopt_as_the_set_he_walks():
    """The measured version of the warning. Since 2026-09-11 the intel names the
    ONE set species 75 walks (42 records, bones up to 44 plus three 0x7D markers), so
    on the Zinogre's 46-joint rig every index EXISTS — which is precisely the case
    the adoption text refuses to call correct. An older intel file without the
    active set still hands over all 153, 19 of them off the rig."""
    si = I.find_intel(75)
    if si is None or not si.parts.present:
        return
    s = _session(n_bones=46)
    got = s.adopt_volumes(si.parts.spheres(), source="em75")
    assert got.adopted == len(si.parts.spheres())
    if si.parts.active is not None:
        assert got.adopted == si.parts.capacity == 42
        assert got.clean, got.off_rig
        assert sum(1 for v in s.volumes() if v.is_marker) == 3
        assert "not the same as pointing at the right joint" in got.describe()
    else:
        assert got.off_rig, "every host bone index fitted, which would be a surprise"
    assert s.preview(open(ZINOGRE, encoding="utf-8").read())


# --------------------------------------------------------------------------- #
# editing the volumes (2026-09-11)
# --------------------------------------------------------------------------- #
class _Sphere:
    def __init__(self, bone, part=1, row=2, r=100.0, flags=0):
        self.bone, self.part, self.hitzone_row, self.radius = bone, part, row, r
        self.a, self.b, self.is_capsule, self.flags = (0.0, 5.0, 0.0), None, False, flags


def _with_volumes(n=3):
    s = _session(n_bones=46)
    s.adopt_volumes([_Sphere(2, r=100.0), _Sphere(3, r=80.0, flags=0x101),
                     _Sphere(4, part=2, row=3, r=60.0)][:n])
    return s


def test_scaling_a_volume_changes_its_radius_and_nothing_else():
    s = _with_volumes()
    before = s.volumes()[1]
    got = s.scale_volume(1, 3.0)
    assert got.radius == 240.0
    assert (got.bone, got.part, got.hitzone_row, got.flags) == (3, 1, 2, 0x101)
    assert s.volumes()[0] == _with_volumes().volumes()[0]
    assert before.radius == 80.0, "the previous record was mutated in place"


def test_editing_a_volume_is_checked_like_the_loader():
    s = _with_volumes()
    for bad in (dict(part=8), dict(hitzone_row=7), dict(shape="box"),
                dict(radius=-1.0), dict(offset=[1.0, 2.0]), dict(bone=70000),
                dict(colour="red")):
        try:
            s.edit_volume(0, **bad)
        except MF.ManifestError:
            continue
        raise AssertionError("%r was accepted" % (bad,))
    s.edit_volume(0, bone=7, offset=[0.0, 55.0, 0.0], hitzone_row=0, part=1)
    v = s.volumes()[0]
    assert (v.bone, v.offset, v.hitzone_row, v.part) == (7, [0.0, 55.0, 0.0], 0, 1)


def test_keep_only_leaves_one_volume_and_counts_the_rest():
    """The #19 experiment in one call: one sphere, and a hit lands there or nowhere."""
    s = _with_volumes()
    assert s.keep_only(2) == 2
    assert [v.bone for v in s.volumes()] == [4]
    s.remove_volume(0)
    assert s.volumes() == []
    try:
        s.remove_volume(0)
    except MF.ManifestError:
        return
    raise AssertionError("removing from an empty list was accepted")


def test_volume_ops_replace_append_and_delete_by_file_index():
    """An edit to a block the FILE has is a ReplaceBlock at its index; a new one is
    appended; a dropped one is deleted last and from the back so the earlier
    indices stay valid. Checked against the ops, then against the patched text."""
    import tempfile
    text = open(ZINOGRE, encoding="utf-8").read()
    base = _with_volumes()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "z.toml")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(base.preview(text))
        m = MF.load(path)
        assert len(m.hurtboxes) == 3
        s = P.PartSession(m, n_bones=46)
        assert s.pending == 0
        s.scale_volume(0, 2.0)                       # replace [0]
        s.remove_volume(1)                           # delete file index 1
        s.add_volume(MF.Hurtbox(bone=9, radius=10.0, part=3, hitzone_row=4))
        kinds = [type(op).__name__ + (":%d" % op.index if hasattr(op, "index") else "")
                 for op in s.ops()]
        assert kinds == ["ReplaceBlock:0", "AppendBlock", "ReplaceBlock:1"], kinds
        assert s.pending == 3 and s.pending_volumes == 3
        assert s.volume_changed(0) and not s.volume_changed(1) and s.volume_changed(2)
        s.save()
        m2 = MF.load(path)
        assert [(h.bone, h.radius) for h in m2.hurtboxes] == \
            [(2, 200.0), (4, 60.0), (9, 10.0)], m2.hurtboxes
        assert m2.hurtboxes[0].flags == 0 and MF.load(path).hurtboxes[1].part == 2


def test_flags_round_trip_through_the_manifest_as_hex():
    s = _with_volumes()
    text = s.preview(open(ZINOGRE, encoding="utf-8").read())
    assert "flags = 0x101" in text
    m = MF.loads(text)
    assert [h.flags for h in m.hurtboxes] == [0, 0x101, 0]


def test_over_capacity_counts_what_would_not_fit_in_place():
    s = _with_volumes()
    assert s.over_capacity == 0                      # capacity unknown = no check
    s.capacity = 2
    assert s.over_capacity == 1
    s.keep_only(0)
    assert s.over_capacity == 0


def test_fill_row_sets_the_weapon_columns_at_once():
    s = _session()
    s.adopt_grid([_State(GRID), _State(GRID)])
    s.fill_row(0, 0, 255)
    row = s.state(0).rows[0]
    assert row[1:4] == [255, 255, 255] and row[0] == 100 and row[9] == 110
    assert s.state(1).rows[0] == ROW


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
