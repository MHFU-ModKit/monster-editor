<p align="center">
  <img src="misc/banner.svg" width="720" alt="MONSTER-EDITOR">
</p>

# MHFU ModKit — monster editor

A desktop editor for **ported monsters** in Monster Hunter Freedom Unite. It draws the animal
skinned and textured, plays every clip at the engine's own rate, puts the host action's frame
numbers on the same timeline as your clip, and edits hitboxes, hurtboxes, parts and moves as
data in `ports/<name>.toml` — validated against the game's own tables before anything is
deployed to the [framework](https://github.com/MHFU-ModKit/framework).

> **Status: working, WIP.** Every panel below is used to author the Zinogre port; the layout
> and the manifest schema are still moving.

## What it does

- **Manifest** — `ports/<name>.toml`: schema, loader, validator, dumper. Pure data, stdlib only.
- **Scene** — a render-agnostic view of a monster PAC: bind skeleton, skin palette, mesh groups,
  textures, the clip list, `pose(clip, frame)`.
- **Viewport** — the skinned, textured animal; bone overlay; the collision gizmos on the live
  pose; a **reference actor** (the host monster) playing the action's own animation beside it.
- **Clips** — what is really in each slot, labelling clips *into the manifest*, coverage.
- **Action inspector** — what the host action expects per `(main, sub)`: cursor gates, what ends
  the action, effect spawns, measured dwell — as markers on the timeline and graded findings.
- **Parts and hitzones** — name the eight accumulators, edit the damage grid, adopt the host's
  tables, edit / scale / delete a volume.
- **Attacks** — the volume sets a move hits with, and the levers on an attack record.
- **Runtime export** — the tables as the runtime eats them: `ports/<name>.toml` →
  `<name>_hit.lua`, deployed to the memory stick in place.

## Install and run

```bash
git clone https://github.com/MHFU-ModKit/monster-editor.git
cd monster-editor
python3 -m venv venv && venv/bin/pip install -r mhfu_monster_editor/requirements.txt
venv/bin/python -m mhfu_monster_editor ports/zinogre.toml                 # the app
venv/bin/python -m mhfu_monster_editor ports/zinogre.toml --headless out.png   # one picture, no window
venv/bin/python -m mhfu_monster_editor.core workspace/extracted/data_files/file_06185.bin --clips
```

The dependency floor rises one layer at a time: `manifest`/`intel` are stdlib, `core` adds numpy
(and the format library), `render` adds moderngl, `ui` adds imgui-bundle. Only `ui` needs a
display, so the render tests run on a headless box.

**You bring the game data.** `workspace/extracted/` is your own extract (made with the
[`formats`](https://github.com/MHFU-ModKit/formats) tools), and `species/emNN.json` — the host
species' action intel — is regenerated from it with `python tools/em_intel.py --all`. Both are
gitignored here.

## Layout

```
mhfu_monster_editor/   the package: manifest, intel, align, parts, attacks, clips, validate,
                       runtime, core/ (scene, pose), render/ (GL), ui/ (imgui), tests/
ports/                 the port manifests (also published with example-mods)
tools/mhfu_model/      the format library — a copy of MHFU-ModKit/formats at the same commit
tools/em_*.py          the species-intel analysers that produce species/emNN.json
docs/                  MOD_PORTED_MONSTER.md — the port library the export targets
```

## Tests

```bash
venv/bin/python -m pytest mhfu_monster_editor/tests
```

Tests that need the game data or the framework's Lua library skip when they are absent.

## No game data is included

This repository contains **no game files, no extracted assets, no artwork** — not the ISO, not
decrypted archives, not models or textures, not dumped tables. All of it is gitignored and is
reproduced from your own legally obtained copy of the game (see the `formats` repo's
`docs/ASSETS.md`). Everything targets **MHFU EU (ULES01213)**; addresses will not line up with a
JP or NA build.

## About this repository

`monster-editor` is one of the [MHFU-ModKit](https://github.com/MHFU-ModKit) repositories. They are cut
from one upstream research repository and re-published from it, so they move in lockstep — a
file that appears in two of them is the same file at the same commit. Pull requests are welcome
here; an accepted one is applied upstream and comes back in the next export, which is why
`main` only takes changes through PRs. Issues are welcome for bugs, questions and findings alike.

The siblings:

- [`framework`](https://github.com/MHFU-ModKit/framework) — the runtime mod framework: one PRX, many mods, hot-reloaded Lua
- [`example-mods`](https://github.com/MHFU-ModKit/example-mods) — Lua mods and port manifests to learn from and drop on a memory stick
- [`hud`](https://github.com/MHFU-ModKit/hud) — a live read-only HUD and AI editor over the PPSSPP debugger
- [`blender-addon`](https://github.com/MHFU-ModKit/blender-addon) — import, edit and export big monsters in Blender
- [`formats`](https://github.com/MHFU-ModKit/formats) — the file-format library, ISO extraction and the MHP3rd→MHFU porter

## License

[MIT](LICENSE). Not affiliated with or endorsed by Capcom. Monster Hunter is a trademark of
Capcom Co., Ltd.
