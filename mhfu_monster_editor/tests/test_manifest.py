"""The manifest schema: what it accepts, what it refuses, and that it round-trips.

The round trip matters more than it looks. There is no TOML *writer* in the stdlib, so
`manifest.dumps` is hand-rolled — and an emitter that is only ever read back by its own
parser can drift into a private dialect without anyone noticing. Every round trip here
goes out through `dumps` and back in through `tomllib`, so the file stays TOML.
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

from mhfu_monster_editor import manifest as MF

PORTS = os.path.join(_ROOT, "ports")

MINIMAL = """
[port]
name = "test"
host_species = 75
pac = "test.bin"

[source]
model = 5339
"""


def test_minimal_manifest_derives_everything_else():
    m = MF.loads(MINIMAL)
    assert m.name == "test"
    assert m.host_frame == 6185, m.host_frame       # host_species + 6110
    assert m.fid == 6186, m.fid                     # engine asks for species + 6111
    assert m.orig == "file_06185.bin.orig", m.orig
    assert m.source.geo == 5340 and m.source.anim == 5341
    assert m.build.skin == "auto" and m.build.source_skeleton is False
    assert m.schema == MF.SCHEMA


def test_a_typo_in_a_key_is_refused_not_ignored():
    """The whole point of a schema. `groundlift` would silently build unlifted."""
    bad = MINIMAL + '\n[build]\ngroundlift = 165.3\n'
    try:
        MF.loads(bad)
    except MF.ManifestError as e:
        assert "groundlift" in str(e), e
        assert "ground_lift" in str(e), "the message must list the known keys"
    else:
        raise AssertionError("an unknown build key was accepted")


def test_wrong_types_are_refused():
    for body, why in (
            ('[port]\nname = 1\nhost_species = 75\npac = "x"\n[source]\nmodel = 1\n',
             "name must be a string"),
            ('[port]\nname = "x"\nhost_species = "75"\npac = "x"\n[source]\nmodel = 1\n',
             "host_species must be an int"),
            (MINIMAL + '\n[build]\nsource_skeleton = "yes"\n', "bool, not string"),
            (MINIMAL + '\n[build]\nskin = "magic"\n', "skin is an enum")):
        try:
            MF.loads(body)
        except MF.ManifestError:
            pass
        else:
            raise AssertionError("accepted a bad manifest: " + why)


def test_missing_required_keys_name_themselves():
    try:
        MF.loads('[port]\nname = "x"\nhost_species = 75\n[source]\nmodel = 1\n')
    except MF.ManifestError as e:
        assert "pac" in str(e), e
    else:
        raise AssertionError("a manifest with no `pac` was accepted")


def test_cross_references_are_structural():
    """A move pointing at an undeclared clip becomes `mv.clip == nil` in Lua and the
    move silently plays whatever the engine wanted. That is a typo, so it is an error
    here rather than a validator warning."""
    for body, needle in (
            (MINIMAL + '\n[moves.charge]\nmain = 2\nsub = 8\nclip = "nope"\n', "nope"),
            (MINIMAL + '\n[moves.charge]\nmain = 2\nsub = 8\n', "clip"),
            (MINIMAL + '\n[[effect]]\nmove = "nope"\nframe = 4\nid = 42\nbone = 33\n',
             "nope")):
        try:
            MF.loads(body)
        except MF.ManifestError as e:
            assert needle in str(e), (needle, str(e))
        else:
            raise AssertionError("accepted a dangling reference: " + needle)


def test_two_clips_cannot_share_a_slot():
    body = MINIMAL + '\n[clips.a]\nslot = 61\n\n[clips.b]\nslot = 61\n'
    try:
        MF.loads(body)
    except MF.ManifestError as e:
        assert "61" in str(e), e
    else:
        raise AssertionError("two names for one slot were accepted")


def test_a_future_schema_is_refused_rather_than_guessed():
    try:
        MF.loads("schema = 99\n" + MINIMAL)
    except MF.ManifestError as e:
        assert "99" in str(e), e
    else:
        raise AssertionError("a schema this loader does not speak was accepted")


# --------------------------------------------------------------------------- #
# the emitter
# --------------------------------------------------------------------------- #
def _full():
    return MF.loads(MINIMAL + """
[build]
source_skeleton = true
skin = "source"
ground_lift = 165.3
animated = 46
drop_joints = [12, 13]

[clips.charge]
slot = 61
frames = 382
loop = false
label = 'he said "charge" \\\\ then left'

[clips.idle]
slot = 1
frames = 220
loop = true

[moves.charge]
main = 2
sub = 8
clip = "charge"
latch = 2
min_gap = 4

[[hurtbox]]
bone = 10
radius = 150.0
part = 0

[[effect]]
move = "charge"
frame = 4
id = 42
bone = 33
""")


def test_dumps_round_trips_through_tomllib():
    m = _full()
    again = MF.loads(MF.dumps(m))
    assert again == m, "round trip lost something"
    # and it is STABLE: a second pass changes nothing
    assert MF.dumps(again) == MF.dumps(m)


def test_floats_survive_the_round_trip_exactly():
    m = MF.loads(MINIMAL + "\n[build]\nground_lift = 165.3\n")
    assert MF.loads(MF.dumps(m)).build.ground_lift == 165.3


def test_quotes_and_backslashes_in_a_label_survive():
    m = _full()
    again = MF.loads(MF.dumps(m))
    assert again.clips["charge"].label == m.clips["charge"].label
    assert '"' in again.clips["charge"].label


def test_a_control_character_in_a_label_does_not_corrupt_the_file():
    """The editor writes clip labels from typed text (issue #8). A basic TOML string
    may not carry a raw control character, so a label with a newline or a tab in it
    would emit a file `tomllib` then refuses to read — losing the whole manifest, not
    just the label."""
    m = _full()
    m.clips["charge"].label = "shoulder\ncharge\tinto\ra wall\x00end"
    again = MF.loads(MF.dumps(m))
    assert again.clips["charge"].label == m.clips["charge"].label


def test_the_emitter_refuses_a_type_it_cannot_render():
    try:
        MF._atom({"a": 1})
    except TypeError:
        pass
    else:
        raise AssertionError("the emitter guessed at a dict")


# --------------------------------------------------------------------------- #
# the porter interface
# --------------------------------------------------------------------------- #
def test_build_args_resolve_file_ids_to_paths():
    m = _full()
    a = m.build_args("ROOT")
    assert a["model"].endswith(os.path.join("extracted_mhp3", "data_files",
                                            "file_05339.bin")), a["model"]
    assert a["geo"].endswith("file_05340.bin")
    assert a["anim"].endswith("file_05341.bin")
    assert a["frame"].endswith(os.path.join("extracted", "data_files",
                                            "file_06185.bin")), a["frame"]
    assert a["skin"] == "source" and a["source_skeleton"] is True
    assert a["ground_lift"] == 165.3 and a["animated"] == 46
    assert a["drop_joints"] == "12,13"
    # bone_offset is UNSET here, so the porter must be left to read the measured value
    # out of p3rd_anim_map rather than take a number from the manifest.
    assert "anim_bone_offset" not in a


def test_build_argv_is_a_command_you_could_paste():
    argv = _full().build_argv("workspace", out="tmp/x.bin")
    assert "--source-skeleton" in argv
    assert argv[argv.index("--skin") + 1] == "source"
    assert argv[argv.index("--ground-lift") + 1] == "165.3"
    assert argv[argv.index("--out") + 1] == "tmp/x.bin"


# --------------------------------------------------------------------------- #
# the checked-in manifests
# --------------------------------------------------------------------------- #
def test_the_shipped_ports_load():
    ports = {m.name: m for m in MF.discover(PORTS)}
    assert set(ports) == {"zinogre", "brute_tigrex"}, sorted(ports)
    for m in ports.values():
        assert m.host_species == 75            # both ride the Tigrex
        assert m.fid == m.host_frame + 1
        assert m.build.source_skeleton and m.build.skin == "source"
    assert ports["zinogre"].source.em_id == 40
    assert ports["brute_tigrex"].source.em_id == 58
    # 🔴 neither pins bone_offset: p3rd_anim_map's MEASURED value is the single source,
    # and forking it here is how a port builds at one bone map and renders at another.
    for m in ports.values():
        assert m.build.bone_offset is None, m.name


def test_the_shipped_ports_survive_a_canonical_rewrite():
    """Comments are lost (tomllib drops them) — the DATA must not be."""
    for m in MF.discover(PORTS):
        assert MF.loads(MF.dumps(m)) == m, m.name


# --------------------------------------------------------------------------- #
# patch — the editor's write path (issue #8)
# --------------------------------------------------------------------------- #
COMMENTED = """# a header that must survive
schema = 1

[port]
name = "test"      # trailing comments too
host_species = 75
pac = "test.bin"

[source]
model = 5339

# a comment block that introduces the clips
[clips.charge]
slot = 61
frames = 382       # measured against THIS build

[moves.go]
main = 2
sub = 8
clip = "charge"
"""


def _comments(text):
    return [l for l in text.splitlines() if l.strip().startswith("#")]


def test_patch_keeps_every_comment_and_touches_only_the_key_it_was_given():
    """🔴 Why `dumps` is not the editor's write path.

    `tomllib` drops comments, so re-emitting a hand-authored `ports/*.toml` destroys
    the prose in it — and both shipped manifests are about half prose, carrying things
    like "🔴 CLIP IDS ARE PER BUILD". The editor patches lines instead.
    """
    out = MF.patch(COMMENTED, [MF.SetKey("clips.charge", "label", "crazy charge")])
    assert _comments(out) == _comments(COMMENTED), "a comment was lost"
    before, after = COMMENTED.splitlines(), out.splitlines()
    assert [l for l in after if l not in before] == ['label = "crazy charge"']
    assert MF.loads(out).clips["charge"].label == "crazy charge"


def test_patch_replaces_a_value_and_keeps_the_comment_after_it():
    out = MF.patch(COMMENTED, [MF.SetKey("clips.charge", "frames", 264)])
    assert "frames = 264       # measured against THIS build" in out, out
    assert MF.loads(out).clips["charge"].frames == 264


def test_patch_inserts_a_new_table_with_the_others_of_its_kind():
    """A new `[clips.x]` belongs with the clips, not after the moves at the bottom."""
    out = MF.patch(COMMENTED, [MF.SetKey("clips.roar", "slot", 53)])
    lines = out.splitlines()
    assert lines.index("[clips.roar]") < lines.index("[moves.go]"), out
    assert MF.loads(out).clips["roar"].slot == 53


def test_patch_removes_a_key_when_the_value_is_none():
    out = MF.patch(COMMENTED, [MF.SetKey("clips.charge", "frames", None)])
    assert MF.loads(out).clips["charge"].frames is None
    assert _comments(out) == [l for l in _comments(COMMENTED)
                             if "measured against" not in l]


def test_a_rename_moves_the_table_and_the_reference_together():
    out = MF.patch(COMMENTED, [MF.RenameClip("charge", "roar")])
    m = MF.loads(out)
    assert "roar" in m.clips and "charge" not in m.clips
    assert m.moves["go"].clip == "roar", "the move still points at the old name"


def test_a_patch_that_would_not_load_raises_instead_of_landing_on_disk():
    """`patch` re-parses its own output. A dangling reference is caught HERE."""
    try:
        MF.patch(COMMENTED, [MF.SetKey("moves.go", "clip", "nope")])
    except MF.ManifestError as e:
        assert "nope" in str(e), e
    else:
        raise AssertionError("a move pointing at an undeclared clip was written")


def test_strings_with_newlines_and_quotes_survive_the_patch():
    """The editor writes free text: a label typed with a quote must round-trip."""
    ugly = 'he said "spin", then\nfell over\ttwice'
    out = MF.patch(COMMENTED, [MF.SetKey("clips.charge", "label", ugly)])
    assert MF.loads(out).clips["charge"].label == ugly


def test_patch_edits_the_real_shipped_manifests_without_disturbing_them():
    for name in ("zinogre", "brute_tigrex"):
        with open(os.path.join(PORTS, name + ".toml"), encoding="utf-8") as fh:
            src = fh.read()
        clip = sorted(MF.loads(src).clips)[0]
        out = MF.patch(src, [
            MF.SetKey("clips.%s" % clip, "labelled_build", "%s.bin@0badcafe" % name),
            MF.SetKey("clips.newly_named", "slot", 57),
            MF.SetKey("clips.newly_named", "frames", 120)])
        assert _comments(out) == _comments(src), "%s lost a comment" % name
        m, before = MF.loads(out), MF.loads(src)
        assert m.clips[clip].labelled_build.endswith("@0badcafe")
        assert m.clips["newly_named"].slot == 57
        assert m.build == before.build and m.source == before.source
        assert set(m.moves) == set(before.moves)
        assert set(m.clips) - set(before.clips) == {"newly_named"}



# --------------------------------------------------------------------------- #
# the part system (#10) — hurtbox volumes, named parts, the damage grid
# --------------------------------------------------------------------------- #
ROW = [100, 75, 65, 40, 0, 15, 5, 30, 20, 110]
GRID = [ROW] + [[0] * 10 for _ in range(6)]


def _hz(state="normal", rows=None):
    rows = GRID if rows is None else rows
    return ('\n[[hitzone]]\nstate = "%s"\nrows = [\n%s\n]\n'
            % (state, "\n".join("  [%s]," % ", ".join(map(str, r)) for r in rows)))


def test_a_part_and_a_hitzone_row_are_stored_as_different_fields():
    """🔴 The correction #10 rests on. The host's 0x28 record carries BOTH: `part`
    (0..7) is the damage accumulator that breaks, `hitzone_row` (0..6) chooses the
    percentages. A Tigrex wing is part 6 and row 5."""
    m = MF.loads(MINIMAL + """
[parts.wing]
index = 6
hitzone_row = 5

[[hurtbox]]
bone = 12
radius = 90.0
part = 6
hitzone_row = 5
""")
    h = m.hurtboxes[0]
    assert (h.part, h.hitzone_row) == (6, 5)
    assert m.parts["wing"].index == 6 and m.parts["wing"].hitzone_row == 5
    assert MF.loads(MF.dumps(m)) == m


def test_a_capsule_keeps_both_of_its_endpoints():
    m = MF.loads(MINIMAL + """
[[hurtbox]]
bone = 6
radius = 65.0
shape = "capsule"
offset = [35.0, 0.0, 0.0]
to = [330.0, 0.0, 0.0]
""")
    h = m.hurtboxes[0]
    assert h.is_capsule and h.offset == [35.0, 0.0, 0.0] and h.to == [330.0, 0.0, 0.0]
    assert MF.loads(MF.dumps(m)) == m


def test_a_sphere_does_not_emit_a_shape_key():
    """`shape` defaults to "sphere", and writing the default into every record
    would triple the size of a hurtbox block for no information."""
    m = MF.loads(MINIMAL + "\n[[hurtbox]]\nbone = 1\nradius = 10.0\n")
    assert "shape" not in MF.dumps(m)
    assert MF.loads(MF.dumps(m)).hurtboxes[0].shape == "sphere"


def test_a_grid_that_is_not_seven_by_ten_is_refused():
    """A `0x48` block is seven rows of ten. Accepting eight rows would write past
    the block and into the next state's."""
    for rows, why in [(GRID[:6], "six rows"),
                      (GRID + [[0] * 10], "eight rows"),
                      ([[0] * 9] + GRID[1:], "a nine-wide row")]:
        try:
            MF.loads(MINIMAL + _hz(rows=rows))
        except MF.ManifestError:
            continue
        raise AssertionError("%s was accepted" % why)


def test_a_percentage_outside_a_byte_is_refused():
    bad = [list(ROW) for _ in range(7)]
    bad[3][1] = 300
    try:
        MF.loads(MINIMAL + _hz(rows=bad))
    except MF.ManifestError as e:
        assert "percentage" in str(e), e
        return
    raise AssertionError("a hitzone of 300 was accepted")


def test_the_grid_round_trips_through_the_emitter():
    """`dumps` writes the rows as a nested array with a column header comment. The
    header is a comment, so it must not come back as data."""
    m = MF.loads(MINIMAL + _hz())
    text = MF.dumps(m)
    assert "thunder" in text, "the column header did not survive"
    back = MF.loads(text)
    assert back == m
    assert back.hitzones[0].value(0, "thunder") == 30
    assert back.hitzones[0].value(0, "ko") == 110


def test_a_part_outside_the_engines_eight_slots_is_refused():
    """The deposit does `entity+0x3B8[part & 7]`, so part 9 is silently part 1.
    Accepting it would give the author a break bar that shares another one."""
    for text, why in [("[parts.x]\nindex = 9\n", "a part index of 9"),
                      ("[[hurtbox]]\nbone=1\nradius=1.0\npart = 8\n",
                       "a hurtbox part of 8"),
                      ("[[hurtbox]]\nbone=1\nradius=1.0\nhitzone_row = 7\n",
                       "a hitzone row of 7")]:
        try:
            MF.loads(MINIMAL + "\n" + text)
        except MF.ManifestError:
            continue
        raise AssertionError("%s was accepted" % why)


def test_an_unknown_shape_is_refused_rather_than_treated_as_a_sphere():
    try:
        MF.loads(MINIMAL + '\n[[hurtbox]]\nbone=1\nradius=1.0\nshape="box"\n')
    except MF.ManifestError as e:
        assert "sphere, capsule" in str(e), e
        return
    raise AssertionError("shape = \"box\" was accepted")


def test_an_offset_must_be_three_numbers():
    try:
        MF.loads(MINIMAL + "\n[[hurtbox]]\nbone=1\nradius=1.0\noffset=[1.0, 2.0]\n")
    except MF.ManifestError as e:
        assert "not 3" in str(e), e
        return
    raise AssertionError("a two-component offset was accepted")


def test_the_shipped_manifests_still_load_and_round_trip():
    """The part block is additive: adding the schema must not change how the shipped
    ports parse. (The Zinogre carries authored tables since 2026-09-11 — the #19
    experiment — so "declares none" is no longer asserted; the round trip is.)"""
    for m in MF.discover(PORTS):
        assert MF.loads(MF.dumps(m)) == m
        for h in m.hurtboxes:
            assert 0 <= (h.part or 0) < MF.PART_SLOTS
        for hz in m.hitzones:
            assert len(hz.rows) == MF.HITZONE_ROWS



def test_a_block_op_appends_a_hurtbox_without_touching_the_prose():
    """`SetKey` addresses `[table.key]` and cannot reach an array of tables — a file
    has many `[[hurtbox]]` blocks and the name does not say which. The block ops are
    how the editor writes them, and the point of patching at all is that the
    hand-authored comments survive."""
    src = open(os.path.join(PORTS, "zinogre.toml"), encoding="utf-8").read()
    had = len(MF.loads(src).hurtboxes)          # the shipped file may carry some
    h = MF.Hurtbox(bone=10, radius=150.0, part=1, hitzone_row=2, label="head")
    out = MF.patch(src, [MF.AppendBlock(MF.hurtbox_block(h))])
    m = MF.loads(out)
    assert len(m.hurtboxes) == had + 1 and m.hurtboxes[-1].part == 1
    assert m.hurtboxes[-1].hitzone_row == 2
    assert out.count("#") >= src.count("#"), "comments were lost"
    assert src.rstrip() in out.replace("\n\n[[hurtbox]]", "@@").replace("@@", "") \
        or src.splitlines()[0] in out


def test_replacing_a_grid_block_leaves_every_other_byte_alone():
    rows = [[100, 75, 65, 40, 0, 15, 5, 30, 20, 110]] + [[0] * 10 for _ in range(6)]
    src = open(os.path.join(PORTS, "zinogre.toml"), encoding="utf-8").read()
    one = MF.patch(src, [MF.AppendBlock(
        MF.hitzone_block(MF.HitzoneState(name="normal", rows=rows)))])
    hotter = [list(r) for r in rows]
    hotter[0][7] = 45
    two = MF.patch(one, [MF.ReplaceBlock(
        "hitzone", 0, MF.hitzone_block(MF.HitzoneState(name="normal", rows=hotter)))])
    assert MF.loads(two).hitzones[0].value(0, "thunder") == 45
    # everything before the block is byte for byte what it was
    head = two[:two.index("[[hitzone]]")]
    assert head == one[:one.index("[[hitzone]]")]


def test_replacing_a_block_that_is_not_there_raises():
    try:
        MF.patch(MINIMAL, [MF.ReplaceBlock("hitzone", 0, "[[hitzone]]\nstate = \"x\"")])
    except MF.ManifestError as e:
        assert "no [[hitzone]] block" in str(e), e
        return
    raise AssertionError("replacing a nonexistent block was accepted")


def test_a_block_op_that_would_produce_an_unloadable_file_raises_before_disk():
    """`patch` re-parses its own output. A grid block with six rows is legal TOML and
    an illegal manifest, and it must fail at the call, not on the next session."""
    six = "[[hitzone]]\nstate = \"x\"\nrows = [\n" + \
          "\n".join("  [%s]," % ", ".join(["0"] * 10) for _ in range(6)) + "\n]"
    try:
        MF.patch(MINIMAL, [MF.AppendBlock(six)])
    except MF.ManifestError:
        return
    raise AssertionError("a six-row grid was written")


def test_a_hurtbox_block_survives_a_round_trip_through_the_patcher():
    """The emitter used by the patcher and the one used by `dumps` must agree, or the
    editor writes records `dumps` would write differently."""
    h = MF.Hurtbox(bone=6, radius=65.0, part=4, hitzone_row=5, shape="capsule",
                   offset=[35.0, 0.0, 0.0], to=[330.0, 0.0, 0.0], label="left wing")
    m = MF.loads(MF.patch(MINIMAL, [MF.AppendBlock(MF.hurtbox_block(h))]))
    assert m.hurtboxes[0] == h
    assert MF.hurtbox_block(h) in MF.dumps(m).replace("\n\n", "\n")



# --------------------------------------------------------------------------- #
# the attack side (#33): [[hitbox]] is a hurtbox record keyed by SET, [[attack]] the
# three measured levers on a record
# --------------------------------------------------------------------------- #
def test_a_hitbox_is_keyed_by_set_and_carries_no_part_or_row():
    m = MF.loads(MINIMAL + '''
[[hitbox]]
set = 2
bone = 10
radius = 150.0

[[hitbox]]
set = 2
bone = 43
radius = 120.0
shape = "capsule"
offset = [0.0, 0.0, 0.0]
to = [0.0, 0.0, -200.0]
label = "tail tip"
''')
    assert [h.set for h in m.hitboxes] == [2, 2]
    assert m.hitboxes[1].is_capsule and m.hitboxes[1].to == [0.0, 0.0, -200.0]
    assert not hasattr(m.hitboxes[0], "part") and not hasattr(m.hitboxes[0], "hitzone_row")
    try:
        MF.loads(MINIMAL + "\n[[hitbox]]\nset = 2\nbone = 1\nradius = 1.0\npart = 1\n")
    except MF.ManifestError as e:
        assert "part" in str(e), e
    else:
        raise AssertionError("a hitbox with a part field was accepted")
    try:
        MF.loads(MINIMAL + "\n[[hitbox]]\nbone = 1\nradius = 1.0\n")
    except MF.ManifestError as e:
        assert "set" in str(e), e
    else:
        raise AssertionError("a hitbox with no set was accepted")


def test_the_hitbox_marker_bones_are_a_coordinate_space():
    m = MF.loads(MINIMAL + "\n[[hitbox]]\nset = 0\nbone = 126\nradius = 150.0\n"
                 "shape = \"capsule\"\n\n[[hitbox]]\nset = 0\nbone = 125\nradius = 0.0\n"
                 "\n[[hitbox]]\nset = 0\nbone = 12\nradius = 90.0\n")
    a, b, c = m.hitboxes
    assert a.is_marker and a.is_node_space, "126 hangs on the node's own two points"
    assert b.is_marker and not b.is_node_space, "125 is a joiner with no geometry"
    assert not c.is_marker and not c.is_node_space


def test_an_attack_block_names_a_record_and_only_the_levers_it_sets():
    m = MF.loads(MINIMAL + "\n[[attack]]\nid = 6\npower = 40\n\n[[attack]]\nid = 31\n"
                 "element = 0x81\nvolume = 5\nlabel = \"dragon\"\n\n[[attack]]\nid = 9\n")
    a6, a31, a9 = m.attacks
    assert (a6.power, a6.element, a6.volume) == (40, None, None)
    assert (a31.power, a31.element, a31.volume) == (None, 0x81, 5)
    assert a9.is_empty and not a6.is_empty
    for bad in ("power = 300", "element = -1", "volume = 256"):
        try:
            MF.loads(MINIMAL + "\n[[attack]]\nid = 1\n%s\n" % bad)
        except MF.ManifestError:
            continue
        raise AssertionError("%s was accepted on an attack record" % bad)


def test_two_attack_blocks_for_one_record_are_refused():
    try:
        MF.loads(MINIMAL + "\n[[attack]]\nid = 6\npower = 1\n\n[[attack]]\nid = 6\npower = 2\n")
    except MF.ManifestError as e:
        assert "record 6" in str(e), e
        return
    raise AssertionError("the same record twice was accepted — the last write would win")


def test_hitbox_and_attack_blocks_round_trip_through_dumps_and_the_patcher():
    h = MF.Hitbox(bone=43, radius=120.0, set=2, shape="capsule",
                  offset=[0.0, 0.0, 0.0], to=[0.0, 0.0, -200.0], flags=0x101,
                  label="tail tip")
    a = MF.Attack(id=6, power=40, element=0x21, volume=2, label="charge")
    m = MF.loads(MF.patch(MINIMAL, [MF.AppendBlock(MF.hitbox_block(h)),
                                    MF.AppendBlock(MF.attack_block(a))]))
    assert m.hitboxes == [h] and m.attacks == [a]
    text = MF.dumps(m)
    assert MF.hitbox_block(h) in text.replace("\n\n", "\n")
    assert MF.attack_block(a) in text.replace("\n\n", "\n")
    assert "element = 0x21" in text, "a gate byte is written as a mask"
    again = MF.loads(text)
    assert again.hitboxes == [h] and again.attacks == [a]
    # replacing the block by index leaves the head of the file byte for byte
    h2 = MF.Hitbox(bone=44, radius=120.0, set=2, label="tail tip moved")
    two = MF.patch(text, [MF.ReplaceBlock("hitbox", 0, MF.hitbox_block(h2))])
    assert MF.loads(two).hitboxes == [h2]
    assert two[:two.index("[[hitbox]]")] == text[:text.index("[[hitbox]]")]

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
