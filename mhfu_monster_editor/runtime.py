"""The manifest's hit tables, as the RUNTIME consumes them — a generated Lua module.

`ports/<name>.toml` is what the port WANTS; `mhfu_port.lua` is what runs. Until issue
#17 makes the runtime read the manifest itself, this is the bridge for the two tables
issue #19 is about: every `[[hurtbox]]` and every `[[hitzone]]` state, emitted as one
`P.hit(<port>, {...})` call that `mhfu_port.lua` writes into the running game.

Where they land, and why it is in place (2026-09-11):

* the collision volumes go OVER the host species' own set — the one its
  `game_task.ovl` row points at (`0x09BB8A00 + species*0x1D0`; em75 -> `0x09D58CD0`),
  record by record, then a `bone = 0xFFFF` sentinel. Both engine walkers stop at the
  sentinel, so a shorter list is fine and a longer one is NOT: the records after the
  original sentinel are somebody else's bytes. The runtime truncates at the set's
  original count and says so; `--capacity` lets this tool say so first.
* the damage grid goes over the species' `0x48` state blocks, as many states as the
  manifest authored and the game has (Tigrex: two). This half was already proven live
  on 2026-06-28 (every byte 0xFF -> 411-damage hits); the volumes half is what #19 is
  testing.
* **the attack side (#33)**: every `[[hitbox]]` set goes OVER the host's own volume
  set of that index, reached through the overlay's set-pointer table
  (`attack_tables.volumes + set*4`; em75: `0x09D60768`), record by record, then a
  sentinel — the same in-place rule as the hurtboxes, one set at a time, each with
  its own capacity. Each `[[attack]]` writes only the levers it names into the
  record at `attack_tables.records + id*0x18` (`+0x02` power, `+0x09` element,
  `+0x0A` volume). Both addresses are the overlay's own, static, read from
  `species/emNN.json`; the runtime measures each set's live count on first contact
  and REFUSES a set whose count is not the exported `cap` — the table is then not
  what the export assumed, and nothing is written. The set replacement was proven
  by RAM poke on a native Tigrex (645 -> 152 -> 1381 units), and this generated
  path was validated LIVE 2026-09-11: applied by hot reload into a running quest (the same apply path a boot takes), `HIT TABLES APPLIED … 1 attack set(s)/10 volume(s)`, the Zinogre's lunge connected through the authored r=600 sphere and the power lever took the hit from carting to single digits.

The file is REGENERATED, never edited: it carries the manifest path and a content
hash so a stale copy on the memstick is recognisable in `framework.log`.

    python -m mhfu_monster_editor.runtime ports/zinogre.toml            # -> scripts/
    python -m mhfu_monster_editor.runtime ports/zinogre.toml --deploy   # + memstick

Stdlib only, like `manifest.py` — the same layer.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .manifest import (HITZONE_COLUMNS, HITZONE_ROWS, Hitbox, Hurtbox, HitzoneState,
                       ManifestError, PortManifest, load)

#: where a generated module lives in the repo (beside `mhfu_port.lua`)
SCRIPTS_DIR = Path("framework") / "prx" / "mods" / "lua_host" / "scripts"
#: the local PPSSPP memstick's mods dir — the file the game actually loads
MEMSTICK_MODS = Path.home() / ".config" / "ppsspp" / "PSP" / "PLUGINS" / "mhfu_framework" / "mods"
#: `tools/embed_lua.py` keeps the committed `.lua.h` in step so the Docker build
#: (no python) never has to regenerate it
EMBED_TOOL = Path("tools") / "embed_lua.py"
#: the library the generated module calls into. 🔴 A stale copy on the memstick
#: silently IGNORES fields it does not know: on 2026-09-11 a deployed module
#: carried `attack_sets`, the memstick's pre-#33 mhfu_port.lua dropped them, and
#: `HIT TABLES APPLIED` was still logged for the two tables it did know — which
#: read as "the hitbox editor does nothing". `deploy` keeps the library in step.
LIBRARY = "mhfu_port.lua"

SHAPE_ID = {"sphere": 0, "capsule": 1}


def _f(v: float) -> str:
    """A Lua float literal that round-trips a float32 and reads like a number."""
    s = repr(float(v))
    return s if "." in s or "e" in s else s + ".0"


def volume_row(h: Hurtbox) -> List[str]:
    """One record as its twelve Lua literals, in RECORD order — bone, shape,
    hitzone_row, part, flags, radius, ax, ay, az, bx, by, bz — so the runtime's
    writer is a straight `for i, v in ipairs(rec)` over the `0x28` layout."""
    a = list(h.offset or (0.0, 0.0, 0.0))
    b = list(h.to or (0.0, 0.0, 0.0)) if h.is_capsule else [0.0, 0.0, 0.0]
    return [str(int(h.bone)), str(SHAPE_ID[h.shape]), str(int(h.hitzone_row or 0)),
            str(int(h.part or 0)), "0x%X" % int(h.flags or 0), _f(h.radius),
            _f(a[0]), _f(a[1]), _f(a[2]), _f(b[0]), _f(b[1]), _f(b[2])]


def attack_volume_row(h: Hitbox) -> List[str]:
    """An attack volume in the SAME twelve-literal record order as a hurtbox, with
    `hitzone_row` and `part` zero — so the runtime has one record writer."""
    a = list(h.offset or (0.0, 0.0, 0.0))
    b = list(h.to or (0.0, 0.0, 0.0)) if h.is_capsule else [0.0, 0.0, 0.0]
    return [str(int(h.bone)), str(SHAPE_ID[h.shape]), "0", "0",
            "0x%X" % int(h.flags or 0), _f(h.radius),
            _f(a[0]), _f(a[1]), _f(a[2]), _f(b[0]), _f(b[1]), _f(b[2])]


@dataclass
class AttackTables:
    """Where the host overlay keeps its attack data — the addresses the runtime
    writes through, from `species/emNN.json` (static, the overlay's own)."""
    volumes_va: int                       # the set-pointer table (u32 per set)
    records_va: int                       # the 0x18 record array
    n_sets: int
    n_records: int
    #: set index -> the host's record count = how many fit in place
    capacities: Dict[int, int] = field(default_factory=dict)


def sets_of(m: PortManifest) -> Dict[int, List[Hitbox]]:
    """The authored volumes grouped by set, in file order within a set."""
    out: Dict[int, List[Hitbox]] = {}
    for h in m.hitboxes:
        out.setdefault(int(h.set), []).append(h)
    return dict(sorted(out.items()))


def content_id(m: PortManifest) -> str:
    """Eight hex digits over the tables, so the log can name the version."""
    h = hashlib.sha1()
    for v in m.hurtboxes:
        h.update(",".join(volume_row(v)).encode())
        h.update(b";")
    for st in m.hitzones:
        for row in st.rows:
            h.update(bytes(int(x) & 0xFF for x in row))
    for st, vols in sets_of(m).items():
        h.update(b"set%d:" % st)
        for v in vols:
            h.update(",".join(attack_volume_row(v)).encode())
            h.update(b";")
    for a in m.attacks:
        h.update(("atk%d:%s,%s,%s;" % (a.id, a.power, a.element, a.volume)).encode())
    return h.hexdigest()[:8]


def lua_hit_module(m: PortManifest, *, capacity: Optional[int] = None,
                   source: str = "", attacks: Optional[AttackTables] = None) -> str:
    """The Lua text. ``capacity`` is the host set's record count when known —
    the number the runtime cannot exceed in place — and is written into the file
    as a fact for the reader, whatever the count. ``attacks`` is where the host
    keeps its attack tables; REQUIRED once the manifest authors a `[[hitbox]]` or
    an `[[attack]]`, because without the addresses there is nowhere to write."""
    if not (m.hurtboxes or m.hitzones or m.hitboxes or m.attacks):
        raise ManifestError("%s authors neither [[hurtbox]], [[hitzone]], [[hitbox]] "
                            "nor [[attack]] — there is nothing to ship"
                            % (source or m.name))
    if (m.hitboxes or m.attacks) and attacks is None:
        raise ManifestError("%s authors [[hitbox]]/[[attack]] but the host's attack "
                            "table addresses are unknown — build species/em%02d.json "
                            "(tools/em_intel.py --all) so the runtime knows where to "
                            "write" % (source or m.name, m.host_species))
    for h in m.hitboxes:
        if attacks is not None and not 0 <= int(h.set) < attacks.n_sets:
            raise ManifestError("%s: [[hitbox]] set %d — host em%02d has %d volume "
                                "set(s) (0..%d)" % (source or m.name, h.set,
                                                     m.host_species, attacks.n_sets,
                                                     attacks.n_sets - 1))
    for a in m.attacks:
        if attacks is not None and not 0 <= int(a.id) < attacks.n_records:
            raise ManifestError("%s: [[attack]] id %d — host em%02d has %d record(s)"
                                % (source or m.name, a.id, m.host_species,
                                   attacks.n_records))
        if a.volume is not None and attacks is not None \
                and not 0 <= int(a.volume) < attacks.n_sets:
            raise ManifestError("%s: [[attack]] id %d volume %d — host em%02d has %d "
                                "set(s)" % (source or m.name, a.id, a.volume,
                                            m.host_species, attacks.n_sets))
    for st in m.hitzones:
        if len(st.rows) != HITZONE_ROWS or any(len(r) != len(HITZONE_COLUMNS)
                                                for r in st.rows):
            raise ManifestError("hitzone state %r is not %d rows of %d"
                                % (st.name, HITZONE_ROWS, len(HITZONE_COLUMNS)))
    mod = "%s_hit" % m.name
    cid = content_id(m)
    src = source or ("ports/%s.toml" % m.name)
    over = (len(m.hurtboxes) - capacity) if capacity is not None else 0
    sets = sets_of(m)
    out: List[str] = [
        "-- %s.lua — GENERATED by mhfu_monster_editor.runtime from %s" % (mod, src),
        "-- (%d volume(s), %d grid state(s), %d attack set(s), %d attack record(s), "
        "id %s). Do not edit: re-export from the"
        % (len(m.hurtboxes), len(m.hitzones), len(sets), len(m.attacks), cid),
        "-- editor's Parts panel or `python -m mhfu_monster_editor.runtime %s`." % src,
        "--",
        "-- The port's hit tables as `mhfu_port.lua`'s P.hit() writes them into the",
        "-- running game: the collision volumes IN PLACE over the host species' own set",
        "-- (its game_task.ovl row's +0x240 pointer), sentinel-terminated, and the",
        "-- damage grid over the species' 0x48 state blocks. Both land once the port's",
        "-- entity is live in-area and are re-checked every tick (issue #19).",
        "-- The attack side (#33): each [[hitbox]] set IN PLACE over the host's own set",
        "-- of that index through the overlay's set-pointer table, sentinel-terminated,",
        "-- refused if the live set's count is not `cap`; each [[attack]] writes only",
        "-- the levers it names into the 0x18 record (+0x02 power, +0x09 element,",
        "-- +0x0A volume).",
    ]
    if capacity is not None:
        out.append("-- Host set capacity: %d record(s)%s." % (
            capacity, " — %d MORE than fit; the runtime truncates and logs" % over
            if over > 0 else ""))
    out += [
        "",
        "mhfu.port = mhfu.port or { _queue = {} }",
        "mhfu.port.mod = mhfu.port.mod or function(n, f) mhfu.port._queue[n] = f end",
        "",
        'mhfu.port.mod("%s", function(P)' % mod,
        '  P.hit("%s", {' % m.name,
        "    species = %d," % m.host_species,
        '    id = "%s",' % cid,
    ]
    if m.hurtboxes:
        out.append("    -- bone, shape(0 sphere/1 capsule), hitzone_row, part, flags, "
                   "radius, ax, ay, az, bx, by, bz")
        out.append("    volumes = {")
        for h in m.hurtboxes:
            note = h.label or ("marker" if h.is_marker else "")
            out.append("      { %s },%s" % (", ".join(volume_row(h)),
                                             ("  -- " + note) if note else ""))
        out.append("    },")
    else:
        out.append("    volumes = nil,   -- the host's own set stays as it is")
    if m.hitzones:
        out.append("    -- per state: %d rows x (%s)" % (HITZONE_ROWS,
                                                        " ".join(HITZONE_COLUMNS)))
        out.append("    grid = {")
        for st in m.hitzones:
            out.append("      {  -- %s" % st.name)
            for row in st.rows:
                out.append("        { %s }," % ", ".join(str(int(x)) for x in row))
            out.append("      },")
        out.append("    },")
    else:
        out.append("    grid = nil,      -- the species grid stays as it is")
    if attacks is not None and (sets or m.attacks):
        out.append("    -- where host em%02d keeps its attack data (static, the "
                   "overlay's own)" % m.host_species)
        out.append("    attack_tables = { volumes = 0x%08X, records = 0x%08X, "
                   "n_sets = %d, n_records = %d },"
                   % (attacks.volumes_va, attacks.records_va, attacks.n_sets,
                      attacks.n_records))
    if sets:
        out.append("    -- per set: cap = the host's record count (the in-place "
                   "limit AND the fingerprint);")
        out.append("    -- rows: bone, shape(0 sphere/1 capsule), 0, 0, flags, "
                   "radius, ax, ay, az, bx, by, bz")
        out.append("    attack_sets = {")
        for st, vols in sets.items():
            cap = attacks.capacities.get(st) if attacks is not None else None
            over = (len(vols) - cap) if cap is not None else 0
            out.append("      [%d] = { cap = %s, volumes = {%s"
                       % (st, "nil" if cap is None else str(cap),
                          "  -- %d MORE than fit; the runtime truncates and logs"
                          % over if over > 0 else ""))
            for h in vols:
                note = h.label or ("node-space" if h.is_node_space
                                   else ("marker" if h.is_marker else ""))
                out.append("        { %s },%s" % (", ".join(attack_volume_row(h)),
                                                   ("  -- " + note) if note else ""))
            out.append("      } },")
        out.append("    },")
    elif attacks is not None and m.attacks:
        out.append("    attack_sets = nil,   -- the host's own sets stay as they are")
    if m.attacks:
        out.append("    -- the levers on a record; nil = the host's byte stands")
        out.append("    attacks = {")
        for a in m.attacks:
            out.append("      { id = %d, power = %s, element = %s, volume = %s },%s"
                       % (a.id, "nil" if a.power is None else str(a.power),
                          "nil" if a.element is None else "0x%02X" % a.element,
                          "nil" if a.volume is None else str(a.volume),
                          ("  -- " + a.label) if a.label else ""))
        out.append("    },")
    out += ["  })", "end)", ""]
    return "\n".join(out)


def _repo_root(m: PortManifest, root: Optional[Path]) -> Path:
    if root is not None:
        return Path(root)
    if m.path is not None:
        return Path(m.path).resolve().parent.parent      # ports/x.toml -> repo
    return Path.cwd()


def export(m: PortManifest, *, out: Optional[Path] = None, root: Optional[Path] = None,
           capacity: Optional[int] = None, embed: bool = True,
           attacks: Optional[AttackTables] = None) -> Path:
    """Write the module into the repo's scripts dir (or ``out``). Returns the path.

    Also refreshes the committed `.lua.h` beside it when `tools/embed_lua.py` is
    there, so `make -C framework/prx` — which has no python — stays buildable.
    """
    repo = _repo_root(m, root).resolve()
    # the manifest is usually loaded by a RELATIVE path (`ports/zinogre.toml`) and
    # the repo root is absolute; `relative_to` refuses that pairing outright, so
    # both sides are resolved first
    src = (str(Path(m.path).resolve().relative_to(repo))
           if m.path and _under(m.path, repo) else "")
    text = lua_hit_module(m, capacity=capacity, source=src, attacks=attacks)
    path = Path(out) if out is not None else repo / SCRIPTS_DIR / ("%s_hit.lua" % m.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    tool = repo / EMBED_TOOL
    if embed and tool.exists():
        subprocess.run([sys.executable, str(tool), str(path), str(path) + ".h"],
                       check=True)
    return path


@dataclass
class Deployment:
    """What `deploy` put on the memstick: the module, and the library if it was
    behind (None = already in step, or no repo copy to sync from)."""
    module: Path
    library: Optional[Path] = None

    def describe(self) -> str:
        return "%s%s" % (self.module.name,
                         "" if self.library is None
                         else " + %s (the memstick's was stale)" % self.library.name)


def sync_library(mods_dir: Path = MEMSTICK_MODS, *,
                 source: Optional[Path] = None) -> Optional[Path]:
    """Copy the repo's `mhfu_port.lua` onto the memstick when the two differ.
    Returns the destination when it copied; None when identical, or when there is
    no memstick or no source to copy from."""
    src = Path(source) if source is not None else Path.cwd() / SCRIPTS_DIR / LIBRARY
    if not mods_dir.is_dir() or not src.is_file():
        return None
    dst = mods_dir / LIBRARY
    if dst.is_file() and dst.read_bytes() == src.read_bytes():
        return None
    shutil.copyfile(src, dst)
    return dst


def deploy(path: Path, mods_dir: Path = MEMSTICK_MODS, *,
           library: Optional[Path] = None) -> Optional[Deployment]:
    """Copy a generated module onto the memstick — AND the library it calls into,
    if the memstick's is behind. None if there is no memstick here. A running game
    hot-reloads both; a cold one picks them up at boot.

    ``library`` is the repo's `mhfu_port.lua`; by default the one beside ``path``
    when it was exported into the repo's scripts dir, else the cwd's."""
    if not mods_dir.is_dir():
        return None
    path = Path(path)
    dst = mods_dir / path.name
    shutil.copyfile(path, dst)
    src = library if library is not None else (
        path.parent / LIBRARY if (path.parent / LIBRARY).is_file() else None)
    return Deployment(module=dst, library=sync_library(mods_dir, source=src))


def _under(p: Path, root: Path) -> bool:
    try:
        Path(p).resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def host_capacity(m: PortManifest) -> Optional[int]:
    """The host set's record count from `species/emNN.json`, if it is built."""
    try:
        from .intel import find_intel
        si = find_intel(m.host_species)
    except Exception:                                            # noqa: BLE001
        return None
    if si is None or not si.parts.present:
        return None
    return si.parts.capacity


def host_attack_tables(m: PortManifest) -> Optional[AttackTables]:
    """The host's attack table addresses and per-set capacities from
    `species/emNN.json`, if it is built and carries the attacks block."""
    try:
        from .intel import find_intel
        si = find_intel(m.host_species)
    except Exception:                                            # noqa: BLE001
        return None
    if si is None or not si.attacks.present:
        return None
    t = si.attacks.primary
    if t is None or t.volume_table_va is None:
        return None
    return AttackTables(volumes_va=t.volume_table_va, records_va=t.records_va,
                        n_sets=len(t.sets), n_records=len(t.attacks),
                        capacities={st.index: st.capacity for st in t.sets})


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("manifest", help="ports/<name>.toml")
    ap.add_argument("-o", "--out", help="write here instead of the repo's scripts dir")
    ap.add_argument("--deploy", action="store_true",
                    help="also copy to %s" % MEMSTICK_MODS)
    ap.add_argument("--capacity", type=int, default=None,
                    help="host set record count (default: from species/emNN.json)")
    ap.add_argument("--no-embed", action="store_true",
                    help="do not refresh the .lua.h beside the module")
    ap.add_argument("--print", action="store_true", help="print the module instead")
    a = ap.parse_args(argv)
    m = load(a.manifest)
    cap = a.capacity if a.capacity is not None else host_capacity(m)
    atk = host_attack_tables(m)
    if a.print:
        sys.stdout.write(lua_hit_module(m, capacity=cap, source=a.manifest, attacks=atk))
        return 0
    path = export(m, out=Path(a.out) if a.out else None, capacity=cap,
                  embed=not a.no_embed, attacks=atk)
    sets = sets_of(m)
    print("wrote %s (%d volume(s), %d state(s), %d attack set(s), %d attack record(s), "
          "id %s%s)"
          % (path, len(m.hurtboxes), len(m.hitzones), len(sets), len(m.attacks),
             content_id(m), "" if cap is None else ", host capacity %d" % cap))
    if cap is not None and len(m.hurtboxes) > cap:
        print("⚠️ %d volume(s) exceed the host set's %d — the runtime will truncate"
              % (len(m.hurtboxes) - cap, cap))
    if atk is not None:
        for st, vols in sets.items():
            c = atk.capacities.get(st)
            if c is not None and len(vols) > c:
                print("⚠️ set %d: %d volume(s) but the host's holds %d — the runtime "
                      "will truncate" % (st, len(vols), c))
    if a.deploy:
        dep = deploy(path)
        print("deployed -> %s" % dep.describe() if dep else
              "no memstick mods dir at %s — not deployed" % MEMSTICK_MODS)
    return 0


if __name__ == "__main__":                                     # pragma: no cover
    raise SystemExit(main())
