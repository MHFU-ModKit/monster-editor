"""The attack system's write path: the sets a move hits with, and a record's levers.

`attacks.AttackSession` is the hitbox editor's authoring surface for issue #33, and
like `parts.PartSession` none of it needs a window. What is worth pinning is what the
session refuses and what it reports:

* a lever the record does not have (only `power` / `element` / `volume` are decoded);
* a value outside a byte, a negative set, a capsule with two numbers for `to`;
* `keep_only` keeps the volume's SET's other sets alone — the other sets are other
  attacks;
* adopting a host set REPORTS the bone indices off the port's rig and keeps the
  125/126/127 markers in order — the walker needs them where they were;
* the ops replace/append/delete `[[hitbox]]` and `[[attack]]` blocks by FILE index,
  so a save shows up in `git diff` as exactly the change made.
"""
import os
import shutil
import sys
import tempfile

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

from mhfu_monster_editor import attacks as A
from mhfu_monster_editor import intel as I
from mhfu_monster_editor import manifest as MF

PORTS = os.path.join(_ROOT, "ports")
SHIPPED = os.path.join(PORTS, "zinogre.toml")


def _bare(shipped=SHIPPED):
    """The shipped manifest with every `[[hitbox]]` / `[[attack]]` block stripped,
    as a temp copy — every test starts from nothing authored on the attack side."""
    text = open(shipped, encoding="utf-8").read()
    m = MF.loads(text)
    ops = [MF.ReplaceBlock("hitbox", i, None) for i in range(len(m.hitboxes) - 1, -1, -1)]
    ops += [MF.ReplaceBlock("attack", i, None) for i in range(len(m.attacks) - 1, -1, -1)]
    bare = MF.patch(text, ops) if ops else text
    d = tempfile.mkdtemp(prefix="mhfu_bare_atk_")
    path = os.path.join(d, "zinogre.toml")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(bare)
    m = MF.load(path)
    assert not m.hitboxes and not m.attacks
    return m, path


class _Sphere:
    """Duck-typed like `intel.HitSphere` / `hitzone.Sphere`."""
    def __init__(self, bone, radius, a=(0.0, 0.0, 0.0), b=None, flags=0):
        self.bone, self.radius, self.a, self.b, self.flags = bone, radius, a, b, flags

    @property
    def is_capsule(self):
        return self.b is not None


# em75's set 2 — the Tigrex charge — as hitbox.py reads it: ten records, two of
# them the 125 joiner between the tail spheres
CHARGE = [_Sphere(10, 150.0), _Sphere(18, 150.0), _Sphere(34, 250.0), _Sphere(4, 250.0),
          _Sphere(2, 250.0), _Sphere(41, 250.0), _Sphere(125, 0.0, b=(0.0, 0.0, 0.0)),
          _Sphere(42, 200.0), _Sphere(125, 0.0, b=(0.0, 0.0, 0.0)), _Sphere(43, 120.0)]


def test_a_fresh_session_has_nothing_staged_and_nothing_authored():
    m, _ = _bare()
    s = A.AttackSession(m, n_bones=46)
    assert s.volumes() == [] and s.sets() == [] and s.attacks() == []
    assert s.pending == 0 and s.save() is None


def test_adopting_a_set_copies_the_hosts_records_in_order_and_reports_off_rig():
    m, _ = _bare()
    s = A.AttackSession(m, n_bones=40)          # a rig with no joint 41/42/43
    got = s.adopt_set(2, CHARGE, source="em75")
    assert got.adopted == 10 and got.source == "em75"
    assert got.off_rig == [41, 42, 43], got.off_rig
    assert not got.clean and "3 name a bone" in got.describe()
    vols = s.volumes()
    assert [h.bone for h in vols] == [10, 18, 34, 4, 2, 41, 125, 42, 125, 43]
    assert all(h.set == 2 for h in vols)
    assert vols[6].is_marker and vols[6].is_capsule, "the joiner keeps its shape"
    assert s.sets() == [2] and len(s.volumes_of(2)) == 10
    assert s.pending == 10


def test_adopting_a_set_again_replaces_what_the_port_had_for_it_only():
    m, _ = _bare()
    s = A.AttackSession(m, n_bones=46)
    s.add_volume(MF.Hitbox(bone=1, radius=99.0, set=0, label="my roar"))
    s.adopt_set(2, CHARGE)
    s.adopt_set(2, CHARGE[:3])
    assert len(s.volumes_of(2)) == 3 and len(s.volumes_of(0)) == 1
    assert s.volumes_of(0)[0][1].label == "my roar"
    assert s.drop_set(2) == 3 and s.sets() == [0]


def test_editing_a_volume_is_checked_like_the_loader():
    m, _ = _bare()
    s = A.AttackSession(m, n_bones=46)
    i = s.add_volume(MF.Hitbox(bone=10, radius=150.0, set=2))
    h = s.edit_volume(i, bone=12, radius=170.0, offset=[0.0, 20.0, 0.0])
    assert (h.bone, h.radius, h.offset, h.set) == (12, 170.0, [0.0, 20.0, 0.0], 2)
    assert s.volume_changed(i)
    for bad in (dict(part=1), dict(hitzone_row=2), dict(set=-1), dict(radius=-1.0),
                dict(shape="box"), dict(to=[1.0, 2.0]), dict(bone=70000)):
        try:
            s.edit_volume(i, **bad)
        except MF.ManifestError:
            continue
        raise AssertionError("%r was accepted on a hitbox" % (bad,))
    try:
        s.edit_volume(5, radius=1.0)
    except MF.ManifestError as e:
        assert "no volume 5" in str(e)
    else:
        raise AssertionError("an index off the list was accepted")


def test_scaling_changes_the_radius_and_nothing_else():
    m, _ = _bare()
    s = A.AttackSession(m)
    i = s.add_volume(MF.Hitbox(bone=10, radius=150.0, set=2, offset=[1.0, 2.0, 3.0]))
    h = s.scale_volume(i, 3.0)
    assert h.radius == 450.0 and h.offset == [1.0, 2.0, 3.0] and h.bone == 10


def test_keep_only_is_within_the_set_because_the_other_sets_are_other_attacks():
    m, _ = _bare()
    s = A.AttackSession(m)
    s.adopt_set(2, CHARGE)
    s.add_volume(MF.Hitbox(bone=31, radius=700.0, set=3))
    s.add_volume(MF.Hitbox(bone=31, radius=1200.0, set=4))
    idx = [i for i, h in s.volumes_of(2) if h.bone == 43][0]
    gone = s.keep_only(idx)
    assert gone == 9
    assert [h.bone for _, h in s.volumes_of(2)] == [43]
    assert len(s.volumes_of(3)) == 1 and len(s.volumes_of(4)) == 1
    assert len(s.volumes()) == 3


def test_over_capacity_is_per_set_and_unknown_means_no_check():
    m, _ = _bare()
    s = A.AttackSession(m)
    s.adopt_set(2, CHARGE)
    s.add_volume(MF.Hitbox(bone=1, radius=1.0, set=2))
    s.add_volume(MF.Hitbox(bone=1, radius=1.0, set=5))
    assert s.over_capacity(2) == 0, "no capacity known -> no complaint"
    s.capacities = {2: 10, 5: 1}
    assert s.over_capacity(2) == 1 and s.over_capacity(5) == 0
    assert s.over_capacity_all() == {2: 1}


def test_attack_levers_are_the_three_measured_fields_and_bytes():
    m, _ = _bare()
    s = A.AttackSession(m)
    a = s.set_attack(6, power=40)
    assert (a.id, a.power, a.element, a.volume) == (6, 40, None, None)
    a = s.set_attack(6, element=0x21, label="charge")
    assert (a.power, a.element, a.label) == (40, 0x21, "charge"), "levers merge"
    a = s.set_attack(6, power=None)
    assert a.power is None and a.element == 0x21, "None clears one lever"
    for bad in (dict(kind=2), dict(angle=1), dict(raw=b""), dict(power=256),
                dict(element=-1), dict(volume=999)):
        try:
            s.set_attack(6, **bad)
        except MF.ManifestError:
            continue
        raise AssertionError("%r was accepted on an attack record" % (bad,))
    assert [x.id for x in s.attacks()] == [6] and s.pending_attacks == 1


def test_clearing_a_record_drops_the_ports_block_and_the_hosts_byte_stands():
    m, path = _bare()
    s = A.AttackSession(m)
    s.set_attack(6, power=40)
    s.set_attack(31, volume=5)
    assert s.save() == "saved 2 attack record(s)"
    m2 = MF.load(path)
    assert [a.id for a in m2.attacks] == [6, 31]
    s2 = A.AttackSession(m2)
    s2.clear_attack(6)
    assert s2.attack(6) is None and [a.id for a in s2.attacks()] == [31]
    assert s2.pending_attacks == 1
    s2.set_attack(6, power=41)                   # re-staging un-drops it
    assert s2.attack(6).power == 41 and s2.pending_attacks == 1
    s2.clear_attack(6)
    s2.save()
    assert [a.id for a in MF.load(path).attacks] == [31]


def test_ops_replace_append_and_delete_by_file_index():
    m, path = _bare()
    s = A.AttackSession(m)
    s.adopt_set(2, CHARGE[:3])
    s.set_attack(6, power=64)
    assert s.save() == "saved 3 hitbox change(s), 1 attack record(s)"
    m2 = MF.load(path)
    assert [h.bone for h in m2.hitboxes] == [10, 18, 34]
    s2 = A.AttackSession(m2)
    s2.edit_volume(1, radius=999.0)              # replace file block 1
    s2.remove_volume(0)                          # delete file block 0
    s2.add_volume(MF.Hitbox(bone=44, radius=5.0, set=2))     # append
    ops = s2.ops()
    kinds = [(type(o).__name__, getattr(o, "index", None), getattr(o, "text", "") is None)
             for o in ops]
    assert ("ReplaceBlock", 1, False) in kinds
    assert ("AppendBlock", None, False) in kinds
    assert kinds[-1] == ("ReplaceBlock", 0, True), "deletions last, from the back"
    s2.save()
    m3 = MF.load(path)
    assert [(h.bone, h.radius) for h in m3.hitboxes] == [(18, 999.0), (34, 250.0), (44, 5.0)]
    assert m3.attacks[0].power == 64, "the attack block survived a volume-only save"


def test_a_save_shows_up_as_the_blocks_and_nothing_else():
    m, path = _bare()
    before = open(path, encoding="utf-8").read()
    s = A.AttackSession(m)
    s.adopt_set(2, CHARGE[:2], label="charge")
    s.set_attack(6, power=64, element=0x21)
    s.save()
    after = open(path, encoding="utf-8").read()
    assert after.startswith(before.rstrip("\n")), "the prose above was disturbed"
    tail = after[len(before.rstrip("\n")):]
    assert tail.count("[[hitbox]]") == 2 and tail.count("[[attack]]") == 1
    assert "element = 0x21" in tail and 'label = "charge"' in tail


def test_discard_leaves_the_manifest_as_it_was():
    m, _ = _bare()
    s = A.AttackSession(m)
    s.adopt_set(2, CHARGE)
    s.set_attack(6, power=1)
    assert s.pending == 11
    s.discard()
    assert s.pending == 0 and s.volumes() == [] and s.attacks() == []


def test_the_preview_refuses_a_patch_that_would_not_load_back():
    m, path = _bare()
    s = A.AttackSession(m)
    s.add_volume(MF.Hitbox(bone=1, radius=1.0, set=0))
    # sabotage the block emitter's output through the op list
    s._work[0] = (None, MF.Hitbox(bone=1, radius=1.0, set=0, shape="capsule",
                                  to=[1.0, 2.0]))   # a two-number `to`
    try:
        s.preview(open(path, encoding="utf-8").read())
    except MF.ManifestError:
        return
    raise AssertionError("a preview that would not load back was produced")


def test_a_manifest_with_no_path_says_where_it_would_go_rather_than_guessing():
    m = MF.loads(open(SHIPPED, encoding="utf-8").read())
    s = A.AttackSession(m)
    s.set_attack(6, power=1)
    try:
        s.save()
    except MF.ManifestError as e:
        assert "nowhere to save" in str(e)
        return
    raise AssertionError("saved a pathless manifest somewhere")


def test_the_tigrex_charge_set_adopts_from_the_intel_as_the_engine_walks_it():
    """With `species/em75.json` built: set 2 through the editor layer is the ten
    records `hitbox.py` reads, the Zinogre's 46-joint rig has every index, and the
    session says so — which is not the same as the spheres being on the right joints."""
    si = I.find_intel(75)
    if si is None or not si.attacks.present:
        return
    m, _ = _bare()
    s = A.AttackSession(m, n_bones=46)
    st = si.attacks.set(2)
    got = s.adopt_set(2, st.spheres, source="em75 set 2")
    assert got.adopted == 10 and got.clean, got.describe()
    assert "not the same as pointing at the right joint" in got.describe()
    s.capacities = {2: si.attacks.capacity(2)}
    assert s.over_capacity(2) == 0
    s.add_volume(MF.Hitbox(bone=1, radius=1.0, set=2))
    assert s.over_capacity(2) == 1


def test_summarise_names_the_sets_and_the_tuned_records():
    m, _ = _bare()
    lines = A.summarise(m)
    assert lines[0].startswith("hitboxes: none authored")
    m.hitboxes.append(MF.Hitbox(bone=1, radius=1.0, set=2))
    m.hitboxes.append(MF.Hitbox(bone=1, radius=1.0, set=5))
    m.attacks.append(MF.Attack(id=6, power=40, volume=2))
    lines = A.summarise(m)
    assert "2 volume(s) over set(s) 2, 5" in lines[0]
    assert "6(power,volume)" in lines[1]


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
