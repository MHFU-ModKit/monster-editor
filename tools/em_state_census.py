#!/usr/bin/env python3
"""
Which behaviour states does the engine ACTUALLY use, how long does it hold them,
and do they move the monster?

    python tools/em_state_census.py                     # the default framework.log
    python tools/em_state_census.py --log path --since 4000000 --min-samples 5

🔴 WHY THIS EXISTS. `tools/em_moveset.py <ovl> --states` lists every `(main, sub)`
the species dispatcher can reach — 231 of them for Tigrex. It says nothing about
whether the engine ever GOES there, and that difference is what wrecked the first
working build of the Brute showcase.

The showcase ran its whole pinned loop on `(4,15)` and `(4,8)`, chosen off the
offline table because their handlers ask the executor for trap animations. Main 4
is the damage-reaction bank: its handlers check for a condition that an
undamaged, untrapped monster does not have, and return immediately. Measured on
the live build, **411 of 411 forced moves survived exactly one tick** — the clip
restarted from frame 0 twice a second and never played through, which is exactly
what "no animation plays to the end" looks like on screen. In ~1600 observed
transitions the engine entered `(4,15)` and `(4,8)` **zero times on its own**.

So: pick pairs the engine already uses. A pair with a long mean dwell is one
whose handler is happy to run; a pair the census never saw is one that will bounce
straight back out no matter how good it looks in the dispatcher table.

Input is `[state]` and `[brute] t=` lines, which the observe-only probe in
`brute_dmg.lua` writes. Nothing has to be running — this reads the log on disk.

⚠️ `move/tick` is `?` when a state never occurred on two consecutive ticks with
the monster CO-LOCATED. Cross-section distances are meaningless (every section
has its own world frame), so those samples are dropped rather than guessed at.
"""

import argparse
import collections
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOG = (ROOT / "workspace/ppsspp-home/ppsspp/PSP/PLUGINS"
                      "/mhfu_framework/framework.log")

RE_STATE = re.compile(r"\[state\] main=(\d+) sub=(\d+) \(a1=(\d+)\) t=(\d+)")
RE_TICK = re.compile(r"\[brute\] t=(\d+) sec=(\d+)/(\d+)( SAME)? out=(\d+) "
                     r"in=(\d+) .* d=(\d+)")

MAX_DWELL = 60          # a longer gap means the probe stopped, not a long state
MAX_DIST = 8000         # beyond this a "distance" is almost certainly cross-frame


def census(lines):
    transitions, ticks = [], []
    main = sub = -1
    for line in lines:
        m = RE_STATE.search(line)
        if m:
            main, sub = int(m.group(1)), int(m.group(2))
            transitions.append((int(m.group(4)), main, sub, int(m.group(3))))
            continue
        m = RE_TICK.search(line)
        if m:
            ticks.append((int(m.group(1)), bool(m.group(4)), int(m.group(7)),
                          main, sub))

    dwell = collections.defaultdict(list)
    anims = collections.defaultdict(set)
    for (t, ma, su, a1), (t2, _, _, _) in zip(transitions, transitions[1:]):
        d = t2 - t
        if 0 <= d <= MAX_DWELL:
            dwell[(ma, su)].append(d)
            anims[(ma, su)].add(a1)

    # movement: only between two CONSECUTIVE ticks, in the same state, co-located
    moved = collections.defaultdict(list)
    prev = None
    for row in ticks:
        t, same, dist, ma, su = row
        if (prev and same and prev[1] and prev[3] == ma and prev[4] == su
                and t - prev[0] == 1 and dist < MAX_DIST and prev[2] < MAX_DIST):
            moved[(ma, su)].append(abs(prev[2] - dist))
        prev = row
    return dwell, anims, moved


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=str(DEFAULT_LOG))
    ap.add_argument("--since", type=int, default=0,
                    help="byte offset — framework.log spans many boots")
    ap.add_argument("--min-samples", type=int, default=5)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--state", action="append", default=[],
                    help="also report these pairs however rare, e.g. --state 4,15")
    args = ap.parse_args()

    path = Path(args.log)
    if not path.exists():
        print(f"no log at {path}", file=sys.stderr)
        return 1
    blob = path.read_bytes()[args.since:].decode("utf-8", "replace")
    dwell, anims, moved = census(blob.splitlines())
    if not dwell:
        print("no [state] lines — is the observe-only probe deployed?", file=sys.stderr)
        return 2

    def fmt(k):
        v = dwell.get(k)
        if not v:
            return f"{str(k):8s} {'—':>6} {0:4d} {'—':>10} {'':>12}   NEVER ENTERED"
        sp = moved.get(k)
        mv = sum(sp) / len(sp) if sp else None
        dw = sum(v) / len(v)
        verdict = ("HOLDS + STATIONARY" if mv is not None and mv < 60 and dw >= 8
                   else "HOLDS + MOVES" if dw >= 8
                   else "short — will bounce out")
        return (f"{str(k):8s} {dw:6.1f} {len(v):4d} "
                f"{(f'{mv:10.0f}' if mv is not None else '         ?')} "
                f"{str(sorted(anims[k])):>12s}   {verdict}")

    print(f"{'state':8s} {'dwell':>6} {'n':>4} {'move/tick':>10} {'a1':>12}   "
          f"verdict     (dwell in 2 Hz ticks)")
    rows = sorted(((sum(v) / len(v), k) for k, v in dwell.items()
                   if len(v) >= args.min_samples), reverse=True)
    for _, k in rows[:args.top]:
        print(fmt(k))

    if args.state:
        print("\nasked for explicitly:")
        for spec in args.state:
            ma, su = (int(x) for x in spec.split(","))
            print(fmt((ma, su)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
