#!/usr/bin/env python3
"""Which ATTACK a handler spawns — the join from `(main,sub)` to a hitbox, offline.

Issue #33 settled that the attack volume is DATA: a handler asks for an attack by
id, the engine copies `attack_table[id]` (`0x18` bytes) into a node and points
`node+0x34` at `VOLUME_TABLE[record+0x0A]`. `mhfu_model/hitbox.py` reads both tables
out of any overlay. What it cannot know is which id which HANDLER passes — that
literal lives in the handler's MIPS, exactly where `em_effects.py` finds the effect
ids. This is the same scan for the other primitive.

    em<N>.ovl   handler: addiu a2, zero, ID ; jal <species spawner>     <- read HERE
      -> game_task <spawner>(?, entity, id)     allocate the 0xE0 node, construct it
      -> em<N>    initialiser(node, entity, id) em75's is 0x09D4B318
      -> game_task 0x09B674C0(node, ent, id, HANDLE) copy attack_table[id]

## The spawner is PER SPECIES (2026-09-11)

em75's handlers call `0x09B661E8` — 84 sites, every one with a literal `a2`, and pair
(1,4), the Tigrex charge measured live at −72 HP, resolves to ids {6, 31}, both on
volume set 2: the set the live replacement moved the hit with (645 → 152 → 1381 u).
**No other overlay calls that address.** Each species has its OWN node constructor
in game_task, a family at `0x09B63000..0x09B66B00`: em07 → `0x09B633E8` (81 sites,
ids 1..87 against 88 records), em14 → `0x09B649D8`, em40 → `0x09B65238`, em82 →
`0x09B66650`, … `spawner_of` picks it structurally: the game_task function in that
range the overlay calls with the most literal `a2` sites, whose ids are ≥ 3 distinct
values starting at ≤ 2. That rule drops the functions several overlays share
(`0x09B63D30`, always with id 34; `0x09B658B8`) — whatever those are, they are not a
moveset's attack table — and em75's own 2–11-site extras (`0x09B662B0`, `0x09B66A78`,
…), which line up with its four projectile tables and are reported as `extras`,
found but NOT joined to a table.

🔴 **Only em75's join is MEASURED.** For the other 16 the spawner→table binding is an
inference from the id range fitting the biggest table, and `consistency()` says per
overlay whether it does (em02/15/20/55/59: the literal ids run PAST the records
`hitbox.read_attacks` returned — the table read stopped early, and the join is
partial there).

⚠️ The literal is the id the HANDLER passes. em75's initialiser adds +33 for an
entity of species 76 and +70 for 88 (one table, sliced by an offset), so the RECORD
an entity uses is `literal + hitbox.id_offset(overlay_species, entity_species)`.
For the overlay's own species that is 0.

    tools/em_attacks.py file_06108.bin            # per handler, with the volumes
    tools/em_attacks.py --census                  # all 17: spawner, sites, fit

→ `tools/mhfu_model/hitbox.py` (the tables), `tools/em_intel.py` (the per-pair join),
`docs/agent_memory_map.md` "The ATTACK side, decoded".
"""
from __future__ import annotations

import argparse
import collections
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ovl_explore import Overlay                                     # noqa: E402
from mips_dis import decode                                         # noqa: E402
import em_effects as fx                                             # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent / "mhfu_model"))
import hitzone as hz                                                # noqa: E402
import hitbox as hb                                                 # noqa: E402

#: em75's spawner — the one walked live (2026-09-11). Kept as the measured anchor.
SPAWN_ATTACK_EM75 = 0x09B661E8
#: the game_task range holding every species' node constructor
SPAWN_FAMILY = (0x09B63000, 0x09B66B00)
#: the argument register carrying the attack id at every member of the family
ID_REG = "a2"
#: a species spawner is called with at least this many distinct literal ids …
MIN_DISTINCT_IDS = 3
#: … the smallest of which is at most this (a moveset's ids start at 1)
MAX_FIRST_ID = 2


@dataclass
class Spawner:
    fn: int
    sites: list = field(default_factory=list)        # [{site, aid, fn}]

    @property
    def literal_ids(self) -> list:
        return sorted({s["aid"] for s in self.sites if s["aid"] is not None})

    @property
    def n_literal(self) -> int:
        return sum(1 for s in self.sites if s["aid"] is not None)


def family_calls(ov: Overlay) -> dict[int, Spawner]:
    """Every call from this overlay into the spawner family, grouped by target."""
    lo, hi = SPAWN_FAMILY
    out: dict[int, Spawner] = {}
    va = ov.text_va
    while va < ov.text_end:
        ins = decode(ov.word(va), va)
        if ins.op == "jal" and lo <= ins.target < hi:
            sp = out.setdefault(ins.target, Spawner(ins.target))
            # `fx.const_arg` stops at the first control transfer, so an id set before
            # a branch the call is only sometimes reached through comes back None
            # rather than as an attack the handler always spawns
            sp.sites.append(dict(site=va, aid=fx.const_arg(ov, va, ID_REG),
                                 fn=ov.func_start(va)))
        va += 4
    return out


def _looks_like_moveset(sp: Spawner) -> bool:
    ids = sp.literal_ids
    return len(ids) >= MIN_DISTINCT_IDS and ids[0] <= MAX_FIRST_ID


def spawner_of(ov: Overlay) -> Spawner | None:
    """This species' attack spawner, or None (em01 and em33 have no attack table
    and no candidate either — consistent)."""
    cands = [sp for sp in family_calls(ov).values() if _looks_like_moveset(sp)]
    if not cands:
        return None
    return max(cands, key=lambda sp: (sp.n_literal, -sp.fn))


def extras(ov: Overlay, main: Spawner | None) -> list[Spawner]:
    """The other family members this overlay calls — em75's projectile spawners.
    Found, counted, and NOT joined to a table: nothing here knows which."""
    return [sp for fn, sp in sorted(family_calls(ov).items())
            if main is None or fn != main.fn]


def sites(ov: Overlay) -> list[dict]:
    """Every call to this species' spawner, with the literal id when there is one."""
    sp = spawner_of(ov)
    return [] if sp is None else list(sp.sites)


def by_handler(ov: Overlay) -> dict[int, list[int]]:
    """`{handler_fn: sorted unique literal ids}` — computed ids left out."""
    out: dict[int, set[int]] = collections.defaultdict(set)
    for s in sites(ov):
        if s["aid"] is not None:
            out[s["fn"]].add(s["aid"])
    return {fn: sorted(v) for fn, v in out.items()}


def consistency(sp: Spawner | None, primary) -> str:
    """Does the spawner's id range fit the table `hitbox.py` read? One of
    `measured` (em75), `consistent`, `ids_exceed_table`, `no_spawner`, `no_table`."""
    if sp is None:
        return "no_spawner"
    if primary is None:
        return "no_table"
    if sp.fn == SPAWN_ATTACK_EM75:
        return "measured"
    ids = sp.literal_ids
    return "consistent" if ids and ids[-1] < len(primary.attacks) else "ids_exceed_table"


def report(path: Path) -> None:
    ov = Overlay.load_file(path)
    img = hz.Image.parse(Path(path).read_bytes())
    tables = hb.tables(img)
    prim = hb.primary_table(tables)
    sp = spawner_of(ov)
    if sp is None:
        print(f"{path.name}  ({ov.name})  no attack spawner found in the family"
              f" — {len(tables)} table(s)")
        return
    print(f"{path.name}  ({ov.name})  spawner 0x{sp.fn:08X}: {len(sp.sites)} site(s), "
          f"{sp.n_literal} with a literal id, {len(sp.sites) - sp.n_literal} computed; "
          f"table {'-' if prim is None else '%d records' % len(prim.attacks)} "
          f"-> {consistency(sp, prim)}")
    for ex in extras(ov, sp):
        print(f"  extra spawner 0x{ex.fn:08X}: {len(ex.sites)} site(s), ids "
              f"{ex.literal_ids or '-'} (not joined to a table)")
    for fn, ids in sorted(by_handler(ov).items()):
        bits = []
        for i in ids:
            v = prim.volume_for(i) if prim is not None else None
            a = prim.attacks[i] if prim is not None and i < len(prim.attacks) else None
            bits.append("id %d%s" % (
                i, "" if a is None else " (pow %d, set %d%s)" % (
                    a.power, a.volume,
                    "" if v is None else ": " + ", ".join(
                        "b%d r%g" % (s.bone, s.radius) for s in v.spheres[:4])
                    + (", …" if len(v.spheres) > 4 else ""))))
        print("  handler 0x%08X  %s" % (fn, "; ".join(bits)))


def census(paths: list[Path]) -> None:
    print("=== attack spawner per species (the join's provenance) ===")
    for p in paths:
        ov = Overlay.load_file(p)
        prim = hb.primary_table(hb.tables(hz.Image.parse(p.read_bytes())))
        sp = spawner_of(ov)
        ids = [] if sp is None else sp.literal_ids
        print("%-8s %-10s %3d site(s) %3d literal  ids %s..%s  records %3s  %s"
              % (ov.name, "-" if sp is None else "0x%08X" % sp.fn,
                 0 if sp is None else len(sp.sites), 0 if sp is None else sp.n_literal,
                 ids[0] if ids else "-", ids[-1] if ids else "-",
                 "-" if prim is None else len(prim.attacks), consistency(sp, prim)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("overlay", nargs="*", help="em*.ovl file(s); default = all")
    ap.add_argument("--census", action="store_true",
                    help="one line per species instead of per-handler detail")
    a = ap.parse_args()
    paths = [Path(x) if Path(x).exists() else fx.DATA_DIR / x for x in a.overlay]
    if not paths:
        paths = fx.em_overlays()
    if a.census:
        census(paths)
    else:
        for p in paths:
            report(p)
            print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
