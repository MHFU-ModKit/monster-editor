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
