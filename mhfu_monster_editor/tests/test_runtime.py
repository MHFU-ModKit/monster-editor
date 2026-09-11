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
    s = P.PartSession(MF.load(path), n_bones=46)
    # the shipped manifest may carry the user's own tables (it does since
    # 2026-09-11); the fixture starts from exactly two volumes and one state
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
        with tempfile.TemporaryDirectory() as d:
            path = RT.export(rel, out=os.path.join(d, "z_hit.lua"), embed=False)
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
        dst = RT.deploy(src, mods_dir=RT.Path(mods))
        assert dst is not None and os.path.exists(dst)
        assert open(dst, "rb").read() == open(src, "rb").read()



def test_the_runtime_writer_lands_the_bytes_the_format_module_describes():
    """The other half of the seam, offline: `mhfu_port.lua`'s P.hit() writer run
    under a fake `mhfu` over the REAL em75 set 0 and Tigrex grid bytes. Record 0
    becomes ours, record 1 the sentinel, record 2 stays the original, the grid
    bytes land and the pad does not, and a change under us is re-applied.
    Needs `lua` on the box and the extracts; skips otherwise."""
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


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("ok  " + fn.__name__)
