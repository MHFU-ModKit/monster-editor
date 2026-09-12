"""`runtime.py` — the manifest's hit tables as the Lua module `mhfu_port.lua` eats.

The claims worth a test: the record order matches the `0x28` layout the runtime
writer assumes, a capsule keeps its far end and a sphere ships zeros there, the
grid is emitted row by row, the content id changes when the tables do, and the
module is syntactically Lua (checked with `luac` when one is on the box).

Run:  python -m pytest mhfu_monster_editor/tests/test_runtime.py -q
"""
import os
import shutil
import subprocess
import sys
import tempfile

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# the framework's Lua library and stubs live in a sibling checkout upstream; the public
# monster-editor repo does not carry them, and the tests that need them skip
_PORT_LUA = os.path.join(_ROOT, "framework", "prx", "mods", "lua_host", "scripts", "mhfu_port.lua")
_HAVE_FRAMEWORK = os.path.exists(_PORT_LUA)
sys.path.insert(0, _ROOT)

from mhfu_monster_editor import manifest as MF
from mhfu_monster_editor import parts as P
from mhfu_monster_editor import runtime as RT

ZINOGRE = os.path.join(_ROOT, "ports", "zinogre.toml")
ROW = [100, 75, 65, 40, 0, 15, 5, 30, 20, 110]


class _State:
    def __init__(self, rows):
        self.rows = rows


def _authored(tmp):
    """A copy of the Zinogre manifest with two volumes and one maxed grid state."""
    path = os.path.join(tmp, "z.toml")
    shutil.copyfile(ZINOGRE, path)
    # the shipped manifest carries the user's own tables on BOTH sides (it does
    # since 2026-09-11); the fixture is the hurtbox side alone — exactly two
    # volumes and one state, no attack side, so `export` needs no attack tables
    from mhfu_monster_editor import attacks as _A
    a = _A.AttackSession(MF.load(path), n_bones=46)
    for st in list(a.sets()):
        a.drop_set(st)
    for rec in list(a.attacks()):
        a.clear_attack(rec.id)
    a.save()
    s = P.PartSession(MF.load(path), n_bones=46)
    while s.volumes():
        s.remove_volume(0)
    s.add_volume(MF.Hurtbox(bone=2, radius=600.0, part=1, hitzone_row=0,
                            offset=[0.0, 55.0, 0.0], label="one big sphere"))
    s.add_volume(MF.Hurtbox(bone=6, radius=65.0, part=4, hitzone_row=5,
                            shape="capsule", offset=[35.0, 0.0, 0.0],
                            to=[330.0, 0.0, 0.0], flags=0x101))
    s.adopt_grid([_State([list(ROW)] + [[0] * 10 for _ in range(6)])])
    s.fill_row(0, 0, 255)
    s.save()
    return MF.load(path)


def test_a_record_is_twelve_literals_in_record_order():
    h = MF.Hurtbox(bone=2, radius=600.0, part=1, hitzone_row=3, offset=[0.0, 55.0, 0.0])
    row = RT.volume_row(h)
    assert row == ["2", "0", "3", "1", "0x0", "600.0", "0.0", "55.0", "0.0",
                   "0.0", "0.0", "0.0"], row
    cap = MF.Hurtbox(bone=6, radius=65.0, part=4, hitzone_row=5, shape="capsule",
                     offset=[35.0, 0.0, 0.0], to=[330.0, 0.0, 0.0], flags=0x101)
    assert RT.volume_row(cap)[1] == "1" and RT.volume_row(cap)[4] == "0x101"
    assert RT.volume_row(cap)[9:] == ["330.0", "0.0", "0.0"]
    # a sphere's `to`, if someone left one in the file, does NOT ship
    sph = MF.Hurtbox(bone=1, radius=1.0, to=[9.0, 9.0, 9.0])
    assert RT.volume_row(sph)[9:] == ["0.0", "0.0", "0.0"]


def test_the_module_carries_both_tables_and_names_its_source():
    with tempfile.TemporaryDirectory() as d:
        m = _authored(d)
        text = RT.lua_hit_module(m, capacity=42, source="ports/z.toml")
    assert 'mhfu.port.mod("zinogre_hit"' in text
    assert 'P.hit("zinogre", {' in text
    assert "species = 75," in text
    assert "{ 2, 0, 0, 1, 0x0, 600.0, 0.0, 55.0, 0.0, 0.0, 0.0, 0.0 },  -- one big sphere" \
        in text, text
    assert "{ 6, 1, 5, 4, 0x101, 65.0, 35.0, 0.0, 0.0, 330.0, 0.0, 0.0 }," in text
    assert "{ 100, 255, 255, 255, 0, 15, 5, 30, 20, 110 }," in text
    assert text.count("        { ") == 7, "one line per grid row"
    assert "capacity: 42 record(s)." in text
    assert "ports/z.toml" in text


def test_over_capacity_is_written_into_the_file_as_a_fact():
    with tempfile.TemporaryDirectory() as d:
        m = _authored(d)
    text = RT.lua_hit_module(m, capacity=1)
    assert "1 MORE than fit; the runtime truncates" in text


def test_nothing_authored_is_refused_rather_than_shipping_an_empty_call():
    m = MF.load(ZINOGRE)
    if m.hurtboxes or m.hitzones:
        return
    try:
        RT.lua_hit_module(m)
    except MF.ManifestError:
        return
    raise AssertionError("an empty hit module was generated")


def test_the_content_id_follows_the_tables():
    with tempfile.TemporaryDirectory() as d:
        m = _authored(d)
        a = RT.content_id(m)
        s = P.PartSession(m, n_bones=46)
        s.scale_volume(0, 2.0)
        s.save()
        b = RT.content_id(MF.load(m.path))
    assert a != b and len(a) == 8


def test_export_writes_the_module_and_its_embed_header():
    with tempfile.TemporaryDirectory() as d:
        m = _authored(d)
        out = os.path.join(d, "out", "zinogre_hit.lua")
        path = RT.export(m, out=out, root=_ROOT, capacity=42)
        assert str(path) == out and os.path.exists(out)
        text = open(out, encoding="utf-8").read()
        assert text.startswith("-- zinogre_hit.lua — GENERATED")
        # the .lua.h beside it, if tools/embed_lua.py is there
        if os.path.exists(os.path.join(_ROOT, "tools", "embed_lua.py")):
            assert os.path.exists(out + ".h")
            assert "k_lua_zinogre_hit" in open(out + ".h", encoding="utf-8").read()
        luac = shutil.which("luac")
        if luac:
            subprocess.run([luac, "-p", out], check=True)


def test_export_accepts_a_manifest_loaded_by_a_relative_path():
    """The editor loads `ports/zinogre.toml` relative and the repo root comes back
    absolute; the first export refused that pairing with pathlib's "not in the
    subpath" error at the deploy button (2026-09-11)."""
    m = MF.load(ZINOGRE)
    if not (m.hurtboxes or m.hitzones):
        return
    cwd = os.getcwd()
    os.chdir(_ROOT)
    try:
        rel = MF.load(os.path.join("ports", "zinogre.toml"))
        assert not os.path.isabs(str(rel.path))
        # the shipped manifest authors an attack side since 2026-09-11, which the
        # export refuses to place without the host's table addresses
        atk = RT.host_attack_tables(rel)
        if (rel.hitboxes or rel.attacks) and atk is None:
            print("SKIP: no species/em75.json attacks block")
            return
        with tempfile.TemporaryDirectory() as d:
            path = RT.export(rel, out=os.path.join(d, "z_hit.lua"), embed=False,
                             attacks=atk)
            text = open(path, encoding="utf-8").read()
    finally:
        os.chdir(cwd)
    assert "from ports/zinogre.toml" in text, text.splitlines()[0]


def test_deploy_copies_beside_the_other_mods_or_says_there_is_no_memstick():
    with tempfile.TemporaryDirectory() as d:
        m = _authored(d)
        src = RT.export(m, out=os.path.join(d, "zinogre_hit.lua"), root=_ROOT,
                        embed=False)
        mods = os.path.join(d, "mods")
        assert RT.deploy(src, mods_dir=RT.Path(mods)) is None
        os.mkdir(mods)
        dep = RT.deploy(src, mods_dir=RT.Path(mods))
        assert dep is not None and os.path.exists(dep.module)
        assert open(dep.module, "rb").read() == open(src, "rb").read()
        # no library beside a module exported into a temp dir: the repo's own is
        # the fallback, and an empty mods dir is behind it by definition
        assert dep.library is not None and "stale" in dep.describe()
        assert RT.deploy(src, mods_dir=RT.Path(mods)).library is None, \
            "identical copies are not re-copied"


def test_deploy_brings_the_library_along_when_the_memsticks_is_behind():
    """The failure that hid the hitbox editor's first run: the module on the
    memstick carried `attack_sets`, the memstick's mhfu_port.lua was the version
    from before #33 and dropped them, and the log still said HIT TABLES APPLIED."""
    if not _HAVE_FRAMEWORK:
        print("SKIP: no framework checkout beside the editor")
        return
    lib = os.path.join(_ROOT, "framework", "prx", "mods", "lua_host", "scripts",
                       "mhfu_port.lua")
    with tempfile.TemporaryDirectory() as d:
        m = _authored(d)
        src = RT.export(m, out=os.path.join(d, "zinogre_hit.lua"), root=_ROOT, embed=False)
        mods = os.path.join(d, "mods")
        os.mkdir(mods)
        stale = os.path.join(mods, "mhfu_port.lua")
        open(stale, "w").write("-- the version from before\n")
        dep = RT.deploy(src, mods_dir=RT.Path(mods), library=RT.Path(lib))
        assert dep.library is not None and "stale" in dep.describe()
        assert open(stale, "rb").read() == open(lib, "rb").read()
        again = RT.deploy(src, mods_dir=RT.Path(mods), library=RT.Path(lib))
        assert again.library is None, "identical copies are not re-copied"
        # the default: the library beside a module that lives in the scripts dir
        assert RT.sync_library(RT.Path(mods), source=RT.Path(lib)) is None
        os.remove(stale)
        assert RT.sync_library(RT.Path(mods), source=RT.Path(lib)) is not None



def test_the_runtime_writer_lands_the_bytes_the_format_module_describes():
    """The other half of the seam, offline: `mhfu_port.lua`'s P.hit() writer run
    under a fake `mhfu` over the REAL em75 set 0 and Tigrex grid bytes. Record 0
    becomes ours, record 1 the sentinel, record 2 stays the original, the grid
    bytes land and the pad does not, and a change under us is re-applied.
    Needs `lua` on the box and the extracts; skips otherwise."""
    if not _HAVE_FRAMEWORK:
        print("SKIP: no framework checkout beside the editor")
        return
    lua = shutil.which("lua")
    data = os.path.join(_ROOT, "workspace", "extracted", "data_files")
    ovl, gt = os.path.join(data, "file_06108.bin"), os.path.join(data, "file_00070.bin")
    if not lua or not (os.path.exists(ovl) and os.path.exists(gt)):
        print("SKIP: no lua or no extracts")
        return
    port_lua = os.path.join(_ROOT, "framework", "prx", "mods", "lua_host", "scripts",
                            "mhfu_port.lua")
    with tempfile.TemporaryDirectory() as d:
        m = _authored(d)
        s = P.PartSession(m, n_bones=46)
        s.keep_only(0)
        s.edit_volume(0, hitzone_row=2, radius=582.0, offset=[0.0, -30.0, 30.0])
        s.add_state("enraged", rows=[[0] * 10 for _ in range(7)])
        s.fill_row(1, 6, 255)
        s.save()
        m = MF.load(m.path)
        RT.export(m, out=os.path.join(d, "zinogre_hit.lua"), root=_ROOT, embed=False)
        blob = open(ovl, "rb").read()
        off = 0x09D58CD0 - 0x09D1A180
        open(os.path.join(d, "set0.bin"), "wb").write(blob[off:off + 43 * 0x28])
        g = open(gt, "rb").read()
        off = 0x09BC6798 - 0x09A5F200
        open(os.path.join(d, "grid.bin"), "wb").write(g[off:off + 0x48 * 2 + 8])
        run = subprocess.run([lua, os.path.join(os.path.dirname(__file__),
                                                "lua_hit_harness.lua"), d, port_lua],
                             capture_output=True, text=True)
        assert run.returncode == 0, run.stdout + run.stderr
        assert "HARNESS OK" in run.stdout, run.stdout



def test_the_library_walks_a_declared_chain_and_refuses_to_loop_one_pair():
    """`mhfu_port.lua`'s move chain under the fake `mhfu`: a re-entry of the running
    pair is refused (forced through with {force=true}), `after` is walked when the
    engine leaves the pair or `hold_max` runs out, a move with neither is logged
    PARKED once, a declared pair the engine enters itself is painted by the hook
    once per entry, and an `after` the engine already reached is adopted. Needs
    `lua`; skips otherwise."""
    if not _HAVE_FRAMEWORK:
        print("SKIP: no framework checkout beside the editor")
        return
    lua = shutil.which("lua")
    if not lua:
        print("SKIP: no lua")
        return
    port_lua = os.path.join(_ROOT, "framework", "prx", "mods", "lua_host", "scripts",
                            "mhfu_port.lua")
    run = subprocess.run([lua, os.path.join(os.path.dirname(__file__),
                                            "lua_chain_harness.lua"), port_lua],
                         capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr
    assert "HARNESS OK" in run.stdout, run.stdout


def test_the_library_fits_lua_hosts_file_buffer():
    """lua_host reads each memstick script through one scratch buffer
    (`G_FILEBUF_SZ` in framework/prx/mods/lua_host/mod.cpp) and SKIPS a bigger
    file with a single boot-log line. On 2026-09-11 mhfu_port.lua crossed the old
    48 KB and the whole port — the quest swap included — silently did not exist.
    Read the cap out of the source so the two cannot drift apart."""
    if not _HAVE_FRAMEWORK:
        print("SKIP: no framework checkout beside the editor")
        return
    src = open(os.path.join(_ROOT, "framework", "prx", "mods", "lua_host", "mod.cpp"),
               encoding="utf-8").read()
    import re
    m = re.search(r"#define G_FILEBUF_SZ \((\d+) \* 1024\)", src)
    assert m, "G_FILEBUF_SZ not found"
    cap = int(m.group(1)) * 1024 - 1
    scripts = os.path.join(_ROOT, "framework", "prx", "mods", "lua_host", "scripts")
    for name in sorted(os.listdir(scripts)):
        if name.endswith(".lua"):
            size = os.path.getsize(os.path.join(scripts, name))
            assert size <= cap, "%s is %d B, over lua_host's %d B buffer: it would be " \
                "SKIPPED at boot" % (name, size, cap)


def test_the_seam_stubs_assemble_branchless_and_in_bounds():
    """`tools/em_vhook_check.py`: the em_vhook v3 stubs the runtime above relies on,
    assembled with the host cc from the same stubs.h the PRX builds, read back by
    an independent decoder — no branch, frame-free slot 29, one call in slot 32,
    targets and config offsets in range. Needs a C compiler; skips otherwise."""
    if not _HAVE_FRAMEWORK:
        print("SKIP: no framework checkout beside the editor")
        return
    if not shutil.which("cc"):
        print("SKIP: no cc")
        return
    run = subprocess.run([sys.executable, os.path.join(_ROOT, "tools", "em_vhook_check.py")],
                         capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr
    assert "OK:" in run.stdout, run.stdout


def test_the_library_enters_pairs_through_the_native_seam_when_it_is_live():
    """`mhfu_port.lua` over a fake em_vhook v3 seam (`lua_native_harness.lua`): the
    first live tick installs the moves' claims as substitutions and the port's
    rules, slot by slot with the rest cleared; `play()` goes through
    `mhfu.em_request` and is confirmed when the cells show the pair; the
    translator's alternative main is tracked rather than read as "ended"; a
    request the engine declines is reported with the enter-action ring and does
    NOT walk `after`; `{raw=true}` still writes the cells; a redefine re-arms.
    Needs `lua`; skips otherwise."""
    if not _HAVE_FRAMEWORK:
        print("SKIP: no framework checkout beside the editor")
        return
    lua = shutil.which("lua")
    if not lua:
        print("SKIP: no lua")
        return
    port_lua = os.path.join(_ROOT, "framework", "prx", "mods", "lua_host", "scripts",
                            "mhfu_port.lua")
    run = subprocess.run([lua, os.path.join(os.path.dirname(__file__),
                                            "lua_native_harness.lua"), port_lua],
                         capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr
    assert "HARNESS OK" in run.stdout, run.stdout


# --------------------------------------------------------------------------- #
# the attack side (#33)
# --------------------------------------------------------------------------- #
from mhfu_monster_editor import attacks as A                          # noqa: E402

_TABLES = RT.AttackTables(volumes_va=0x09D60768, records_va=0x09D60848, n_sets=56,
                          n_records=107, capacities={0: 5, 2: 10, 5: 1})


def _attack_authored(tmp, *, bad_cap=False):
    """A copy of the Zinogre manifest authoring set 2 and two record levers — or,
    for the negative harness case, set 0 with a cap the live table will not have."""
    path = os.path.join(tmp, "bad.toml" if bad_cap else "z.toml")
    shutil.copyfile(ZINOGRE, path)
    # the shipped manifest carries the user's #19 tables; this fixture is the
    # attack side ALONE, so the harness needs no species row seeded
    ps = P.PartSession(MF.load(path), n_bones=46)
    while ps.volumes():
        ps.remove_volume(0)
    for i in range(len(ps.states())):
        ps._drop_grids.add(i)
    ps.save()
    m = MF.load(path)
    s = A.AttackSession(m, n_bones=46)
    for st in list(s.sets()):
        s.drop_set(st)
    for a in list(s.attacks()):
        s.clear_attack(a.id)
    if bad_cap:
        s.add_volume(MF.Hitbox(bone=1, radius=50.0, set=0))
        s.set_attack(9, power=1)
    else:
        s.add_volume(MF.Hitbox(bone=12, radius=170.0, set=2, offset=[0.0, 20.0, 0.0],
                               label="head"))
        s.add_volume(MF.Hitbox(bone=125, radius=0.0, set=2, shape="capsule",
                               to=[0.0, 0.0, 0.0]))
        s.add_volume(MF.Hitbox(bone=44, radius=100.0, set=2, shape="capsule",
                               to=[0.0, 0.0, -175.0], label="tail tip"))
        s.set_attack(6, power=40, label="the charge, softer")
        s.set_attack(31, volume=5)
    s.save()
    return MF.load(path)


def test_an_attack_volume_is_the_same_twelve_literals_with_row_and_part_zero():
    h = MF.Hitbox(bone=44, radius=100.0, set=2, shape="capsule", offset=[1.0, 2.0, 3.0],
                  to=[0.0, 0.0, -175.0], flags=0x101)
    row = RT.attack_volume_row(h)
    assert row == ["44", "1", "0", "0", "0x101", "100.0", "1.0", "2.0", "3.0",
                   "0.0", "0.0", "-175.0"], row
    assert len(row) == len(RT.volume_row(MF.Hurtbox(bone=1, radius=1.0)))


def test_the_module_carries_the_attack_tables_sets_and_levers():
    with tempfile.TemporaryDirectory() as d:
        m = _attack_authored(d)
        text = RT.lua_hit_module(m, capacity=42, attacks=_TABLES)
        assert "attack_tables = { volumes = 0x09D60768, records = 0x09D60848, " \
               "n_sets = 56, n_records = 107 }" in text
        assert "[2] = { cap = 10, volumes = {" in text
        assert "{ 12, 0, 0, 0, 0x0, 170.0, 0.0, 20.0, 0.0, 0.0, 0.0, 0.0 },  -- head" in text
        assert "{ 125, 1, 0, 0, 0x0, 0.0," in text and "-- marker" in text
        assert "{ id = 6, power = 40, element = nil, volume = nil },  -- the charge" in text
        assert "{ id = 31, power = nil, element = nil, volume = 5 }," in text
        assert RT.sets_of(m) == {2: m.hitboxes}
        luac = shutil.which("luac")
        if luac:
            out = os.path.join(d, "m.lua")
            open(out, "w").write(text)
            subprocess.run([luac, "-p", out], check=True)


def test_over_capacity_per_set_is_written_as_a_fact():
    with tempfile.TemporaryDirectory() as d:
        m = _attack_authored(d)
        text = RT.lua_hit_module(m, attacks=RT.AttackTables(
            volumes_va=1, records_va=2, n_sets=56, n_records=107, capacities={2: 2}))
        assert "[2] = { cap = 2, volumes = {  -- 1 MORE than fit" in text


def test_hitboxes_without_table_addresses_are_refused_not_shipped_blind():
    with tempfile.TemporaryDirectory() as d:
        m = _attack_authored(d)
        try:
            RT.lua_hit_module(m, attacks=None)
        except MF.ManifestError as e:
            assert "species/em75.json" in str(e), e
        else:
            raise AssertionError("shipped attack sets with nowhere to write them")
        for bad in (MF.Hitbox(bone=1, radius=1.0, set=56),):
            m2 = MF.load(m.path)
            m2.hitboxes.append(bad)
            try:
                RT.lua_hit_module(m2, attacks=_TABLES)
            except MF.ManifestError as e:
                assert "0..55" in str(e), e
            else:
                raise AssertionError("a set past the host's table was accepted")
        m3 = MF.load(m.path)
        m3.attacks.append(MF.Attack(id=107, power=1))
        try:
            RT.lua_hit_module(m3, attacks=_TABLES)
        except MF.ManifestError as e:
            assert "107 record(s)" in str(e), e
        else:
            raise AssertionError("a record past the host's table was accepted")


def test_the_content_id_follows_the_attack_tables_too():
    with tempfile.TemporaryDirectory() as d:
        m = _attack_authored(d)
        a = RT.content_id(m)
        m.attacks[0] = MF.Attack(id=6, power=41)
        b = RT.content_id(m)
        m.hitboxes[0] = MF.Hitbox(bone=13, radius=170.0, set=2, offset=[0.0, 20.0, 0.0])
        c = RT.content_id(m)
        assert len({a, b, c}) == 3, (a, b, c)


def test_host_attack_tables_come_from_the_intel_when_it_is_built():
    m = MF.load(ZINOGRE)
    t = RT.host_attack_tables(m)
    if t is None:
        print("SKIP: no species/em75.json with an attacks block")
        return
    assert (t.volumes_va, t.records_va, t.n_sets, t.n_records) == (0x09D60768, 0x09D60848,
                                                                    56, 107)
    assert t.capacities[2] == 10 and t.capacities[0] == 5


def test_the_attack_writer_lands_each_set_in_place_and_refuses_a_cap_mismatch():
    """The attack half of the seam, offline: `mhfu_port.lua`'s P.hit() run under
    the fake `mhfu` over the REAL em75 set-pointer table, sets 0 and 2, and the
    0x18 record array. Set 2 becomes three of ours + sentinel with the original
    record 4 untouched, set 0 stays byte for byte, record 6 gets only its power,
    record 31 only its volume, a change under us is re-applied — and an export
    whose `cap` disagrees with the live count is REFUSED with nothing written.
    Needs `lua` on the box and the extracts; skips otherwise."""
    lua = shutil.which("lua")
    data = os.path.join(_ROOT, "workspace", "extracted", "data_files")
    ovl = os.path.join(data, "file_06108.bin")
    if not lua or not os.path.exists(ovl):
        print("SKIP: no lua or no extracts")
        return
    port_lua = os.path.join(_ROOT, "framework", "prx", "mods", "lua_host", "scripts",
                            "mhfu_port.lua")
    sys.path.insert(0, os.path.join(_ROOT, "tools"))
    from mhfu_model import hitbox as HB, hitzone as HZ
    img = HZ.Image.parse(open(ovl, "rb").read())
    prim = HB.primary_table(HB.tables(img))
    assert prim.volume_table_va == 0x09D60768 and prim.records_va == 0x09D60848
    tables = RT.AttackTables(volumes_va=prim.volume_table_va, records_va=prim.records_va,
                             n_sets=len(prim.volumes), n_records=len(prim.attacks),
                             capacities={v_i: len(v.spheres)
                                         for v_i, v in enumerate(prim.volumes)})
    with tempfile.TemporaryDirectory() as d:
        m = _attack_authored(d)
        RT.export(m, out=os.path.join(d, "zinogre_hit.lua"), root=_ROOT, embed=False,
                  attacks=tables)
        bad = _attack_authored(d, bad_cap=True)
        bad_tables = RT.AttackTables(volumes_va=tables.volumes_va,
                                     records_va=tables.records_va, n_sets=tables.n_sets,
                                     n_records=tables.n_records, capacities={0: 99})
        open(os.path.join(d, "bad_hit.lua"), "w").write(
            RT.lua_hit_module(bad, attacks=bad_tables).replace(
                'mhfu.port.mod("zinogre_hit"', 'mhfu.port.mod("zinogre_hit_bad"'))
        blob = img.data
        ptr_off = img.off(prim.volume_table_va)
        open(os.path.join(d, "atk_ptrs.bin"), "wb").write(
            blob[ptr_off:ptr_off + 4 * len(prim.volumes)])
        for idx in (0, 2):
            v = prim.volumes[idx]
            off = img.off(v.va)
            open(os.path.join(d, "atk_set%d.bin" % idx), "wb").write(
                blob[off:off + (len(v.spheres) + 1) * 0x28])
        roff = img.off(prim.records_va)
        open(os.path.join(d, "atk_recs.bin"), "wb").write(
            blob[roff:roff + len(prim.attacks) * 0x18])
        run = subprocess.run([lua, os.path.join(os.path.dirname(__file__),
                                                "lua_attack_harness.lua"), d, port_lua],
                             capture_output=True, text=True)
        assert run.returncode == 0, run.stdout + run.stderr
        assert "HARNESS OK" in run.stdout, run.stdout

if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("ok  " + fn.__name__)
