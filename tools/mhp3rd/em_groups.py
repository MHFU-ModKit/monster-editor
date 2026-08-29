#!/usr/bin/env python3
"""Name every MHP3rd big-monster file group from its overlay, not by eye.

`docs/MHP3RD_FILE_MONSTER_MAP.md` was built by rendering models and matching
skeletons — slow, and it left several species as "(a Fanged Beast — Lagombi most
likely)". It did not need to be: each group opens with the species' AI overlay
and an `MWo3` header carries the overlay's NAME at file offset 32. `em040m0.ovl`
says em040, and that is an identification rather than a guess.

    python tools/mhp3rd/em_groups.py workspace/extracted_mhp3/data_files

A group runs from one `m0` overlay to the next: some number of overlay copies,
then one or more asset triples.

    [em<N>m0][em<N>m1]([em<N>m2][em<N>m3])?  ( [model+skel][GE geometry][.anim] )+

⚠️ **Two things vary, and assuming either is fixed mis-assigns files.**
  * **The overlay count.** m0/m1 load at `0x09DB3D80`/`0x09DE8C00`; some species
    also ship m2/m3 at `0x09E1DA80`/`0x09E32500` — four concurrent slots for that
    species rather than two. em010 (`file_05181..05187`) has four.
  * **The triple count.** A family shares one AI overlay. em001 spans
    `file_05138..05145`: the overlays, then Rathian (`05140`) AND Rathalos
    (`05143`) as two complete asset sets behind the same code.

So the leading overlays are consumed by NAME, not by count.

⚠️ **m0 and m1 are the same overlay compiled for two different load addresses**
(`0x09DB3D80` and `0x09DE8C00`) — MHP3rd's way of having two big monsters of one
species on screen. They are not two monsters, and the second one is NOT part of
the previous group: `file_05342/05343` sit right after the Zinogre's anim and are
em041's pair, which is why they read as an unexplained identical-size couple.

⚠️ **There is no effect/VFX file in a group.** Five files, all accounted for.
A monster's effects are code in the overlay — see `docs/EFFECTS_AND_VFX.md`.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ovl_explore import Overlay                                      # noqa: E402

M0 = re.compile(r"^em(\d+)m0\.ovl$")


def groups(data_dir: Path):
    """Each em group with every asset triple it owns.

    The boundary is the NEXT m0 overlay, not a fixed stride — see the module
    docstring. The last group runs to the last file present.
    """
    is_ovl, starts = {}, []
    files = sorted(data_dir.glob("file_0*.bin"))
    for p in files:
        head = p.read_bytes()[:64]
        i = int(p.stem[5:])
        if head[:4] != b"MWo3":
            continue
        is_ovl[i] = head[32:64].split(b"\0")[0].decode(errors="replace")
        if M0.match(is_ovl[i]):
            starts.append((int(M0.match(is_ovl[i]).group(1)), i))
    last = max(int(p.stem[5:]) for p in files)
    out = []
    for k, (em, i) in enumerate(starts):
        end = starts[k + 1][1] if k + 1 < len(starts) else last + 1
        # consume the leading overlay copies BY NAME — the count is 2 or 4
        j = i
        ovls = []
        while j < end and j in is_ovl:
            ovls.append(j)
            j += 1
        variants = [(t, t + 1, t + 2) for t in range(j, end, 3) if t + 2 < end]
        # 🔴 Report any remainder rather than dropping it: silently truncating
        # would read as "that group is fully mapped".
        out.append(dict(em=em, ovls=ovls, end=end - 1, variants=variants,
                        unaccounted=(end - j) - 3 * len(variants)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir", nargs="?",
                    default="workspace/extracted_mhp3/data_files")
    ap.add_argument("--markdown", action="store_true")
    a = ap.parse_args()
    g = groups(Path(a.data_dir))
    if a.markdown:
        print("| em id | overlays | model+skel | geometry | moveset |")
        print("|---|---|---|---|---|")
        for r in g:
            for n, (mo, ge, an) in enumerate(r["variants"]):
                print("| **em%03d**%s | %s | `file_%05d` | `file_%05d` | "
                      "`file_%05d` |"
                      % (r["em"], "" if n == 0 else " (variant %d)" % n,
                         "/".join("`file_%05d`" % o for o in r["ovls"])
                         if n == 0 else "(shared)", mo, ge, an))
    else:
        print("%d big-monster groups, %d asset sets"
              % (len(g), sum(len(r["variants"]) for r in g)))
        for r in g:
            print("  em%03d  %d overlay(s) file_%05d..  files %05d..%05d  "
                  "%d variant(s)%s"
                  % (r["em"], len(r["ovls"]), r["ovls"][0], r["ovls"][0],
                     r["end"], len(r["variants"]),
                     "  +%d UNACCOUNTED" % r["unaccounted"]
                     if r["unaccounted"] else ""))
            for mo, ge, an in r["variants"]:
                print("           model=file_%05d geo=file_%05d anim=file_%05d"
                      % (mo, ge, an))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
