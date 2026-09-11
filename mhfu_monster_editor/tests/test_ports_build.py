"""`--manifest` must build the SAME BYTES as the flag soup it replaces.

This is the only claim in issue #2 that cannot be argued, only measured: for each
shipped port, run `tools/build_p3rd_port.py` once through the documented command line
and once through `--manifest`, and diff the two PACs. Anything less — "the arguments
look equivalent" — is how the Brute shipped at the wrong bone offset for months.

The documented commands are transcribed below and are the ground truth side of the
comparison; the manifest side must match them without being told what they were.

⚠️ Needs the game extracts under `workspace/`. Returns early without them, like
`tools/mhfu_model/tests/test_stream_partition.py` — no game data lives in this repo.
"""
import hashlib
import os
import sys
import tempfile

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "tools"))

from mhfu_monster_editor import manifest as MF
from mhfu_monster_editor import validate as V

WORKSPACE = os.path.join(_ROOT, "workspace")
MHFU = os.path.join(WORKSPACE, "extracted", "data_files")
MHP3 = os.path.join(WORKSPACE, "extracted_mhp3", "data_files")


def _f(base, n):
    return os.path.join(base, "file_%05d.bin" % n)


# The build commands as they are written down. Zinogre: docs/agent_session_log.md
# ("Repro (v10)") and .claude/skills/monster-porting/SKILL.md. Brute: the
# source-skeleton route of .claude/skills/monster-porting/references/build-pipeline.md,
# which is what the 2026-08-30 DECISION settled on ("load him as a NEW monster").
DOCUMENTED = {
    "zinogre": ["--source-skeleton", "--skin", "source",
                "--ground-lift", "165.3", "--em-id", "40", "--animated", "46",
                "--model", _f(MHP3, 5339), "--geo", _f(MHP3, 5340),
                "--anim", _f(MHP3, 5341), "--frame", _f(MHFU, 6185)],
    "brute_tigrex": ["--source-skeleton", "--skin", "source",
                     "--model", _f(MHP3, 5248), "--geo", _f(MHP3, 5249),
                     "--anim", _f(MHP3, 5250), "--frame", _f(MHFU, 6185)],
}


def _have_data():
    return all(os.path.exists(p) for p in
               (_f(MHFU, 6185), _f(MHP3, 5339), _f(MHP3, 5248)))


def _build(argv, out):
    import build_p3rd_port
    build_p3rd_port.main(list(argv) + ["--out", out])
    return hashlib.sha256(open(out, "rb").read()).hexdigest()


def test_the_manifest_reproduces_the_documented_build_byte_for_byte():
    if not _have_data():
        return                      # extracts are gitignored; docs/ASSETS.md rebuilds
    ports = {m.name: m for m in MF.discover(os.path.join(_ROOT, "ports"))}
    assert set(ports) == set(DOCUMENTED), (sorted(ports), sorted(DOCUMENTED))
    with tempfile.TemporaryDirectory() as td:
        for name, flags in sorted(DOCUMENTED.items()):
            a = _build(flags, os.path.join(td, name + ".flags.bin"))
            b = _build(["--manifest", os.path.join(_ROOT, "ports", name + ".toml"),
                        "--data-root", WORKSPACE],
                       os.path.join(td, name + ".manifest.bin"))
            assert a == b, "%s: flags %s != manifest %s" % (name, a[:16], b[:16])
            print("      %-14s %s" % (name, a))


def test_an_explicit_flag_still_beats_the_manifest():
    """Additive, not a rewrite: `--manifest X --skin auto` is a one-off variant of a
    recorded build. If the manifest overrode the command line instead, a modder
    experimenting on top of a port would get the recorded build back and no warning."""
    if not _have_data():
        return
    with tempfile.TemporaryDirectory() as td:
        man = os.path.join(_ROOT, "ports", "zinogre.toml")
        base = ["--manifest", man, "--data-root", WORKSPACE]
        a = _build(base, os.path.join(td, "a.bin"))
        b = _build(base + ["--ground-lift", "0"], os.path.join(td, "b.bin"))
        assert a != b, "--ground-lift 0 was ignored: the manifest overrode the flag"


def test_the_out_fallback_never_lands_in_the_current_directory():
    """A built port is Capcom data spliced from two games (docs/ASSETS.md category D).
    `port.pac` is a bare filename, so falling back to it verbatim writes a 750 KB PAC
    into whatever directory you happened to be in — at the repo root that is an
    UNIGNORED file, one `git add -A` from committing game data. It has to land in
    `tmp/`, which is the gitignored home ASSETS.md registers for it."""
    import build_p3rd_port as BP

    man = os.path.join(_ROOT, "ports", "zinogre.toml")
    argv = ["--manifest", man, "--data-root", WORKSPACE]
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as td:
        try:
            os.chdir(td)
            ap = BP._parser()
            a = ap.parse_args(argv)
            BP._apply_manifest(ap, a, argv)
            if a.out is None:            # mirrors main()'s fallback
                raise AssertionError("no --out was derived from the manifest at all")
        finally:
            os.chdir(cwd)
        assert os.path.dirname(a.out) == "tmp", a.out
        assert os.path.basename(a.out) == "zinogre_v10.bin", a.out
        stray = [n for n in os.listdir(td) if n.endswith(".bin")]
        assert not stray, "a PAC path was resolved into the CWD: %s" % stray


def test_the_shipped_manifests_validate_against_their_own_builds():
    """The clip fingerprints and the bone ranges, against the real PAC rather than a
    stub. This is what makes `frames`/`loop` in ports/*.toml worth writing down."""
    if not _have_data():
        return
    with tempfile.TemporaryDirectory() as td:
        for m in MF.discover(os.path.join(_ROOT, "ports")):
            out = os.path.join(td, m.name + ".bin")
            _build(["--manifest", os.path.join(_ROOT, "ports", m.name + ".toml"),
                    "--data-root", WORKSPACE], out)
            issues = V.validate(m, pac=out)
            errs = [i for i in issues if i.level == V.ERROR]
            assert not errs, "%s: %s" % (m.name, [str(e) for e in errs])
            # the two we expect: no census on this machine (#4), and the shipped
            # labels predate #8 so none of them records the build it was written
            # against — plus, since the Zinogre carries authored hit tables (#19,
            # 2026-09-11), the two advisories authoring them always earns.
            # Anything ELSE is a real finding.
            assert all(i.code in ("INTEL_ABSENT", "LABEL_UNKEYED", "HITZONE_SHARED",
                                  "PARTS_UNNAMED") for i in issues), \
                "%s: %s" % (m.name, [str(i) for i in issues])


def test_the_brute_showcase_clip_ids_do_not_survive_this_build():
    """A regression test for the FINDING, not for the code. brute_showcase.lua names
    `trapped = 82` and `break_free = 69`; on the source-skeleton build slot 82 does not
    exist and slot 69 is filler. If a future porter change makes either legitimate,
    this fails and the manifest can gain them back."""
    if not _have_data():
        return
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "brute.bin")
        _build(["--manifest", os.path.join(_ROOT, "ports", "brute_tigrex.toml"),
                "--data-root", WORKSPACE], out)
        table = V.clip_table(open(out, "rb").read())
        assert 82 not in table, "slot 82 now exists — re-check brute_showcase.lua"
        assert table[69] == table[1], "slot 69 is no longer a copy of the idle clip"
        assert table[61] == (382, 0), table.get(61)     # the charge, as the manifest says


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    if not _have_data():
        print("  workspace/ extracts absent — build-comparison tests skipped "
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
