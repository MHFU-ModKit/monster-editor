# mhfu_model

Constraint-aware library for MHFU **big-monster** model PACs (geometry + skeleton +
bind-pose + animation). Single source of truth for the on-disk format and the
engine's rules, shared by the Blender addon, a CLI, and the live PRX injector.
See `specs/002-model-anim-pipeline/tasks.md` and `docs/ANIMATION_FORMAT.md`.

## Status

- **Phase 0 (DONE):** read side + byte-exact PAC container. Parses + round-trips
  byte-identical on all 49 big-monster PACs (`file_06111`–`file_06159`); geometry
  and animation decode for all 49.
- **Phase 1 (DONE):** read-only Blender importer (`../../blender_mhfu`).
- **Phase 2 (DONE):** constraint validator (`constraints.py`) — `validate(...)` +
  CLI `python -m mhfu_model.validate <pac> [species|ref.bin]`. All 49 PACs validate
  clean; each broken rule flags a specific code. Drives the Blender panel + CLI.
- **Phase 3 (DONE, one HITL gate):** encoders (`skeleton.encode`, `anim.encode`,
  `pmo.encode`) + `repack()`. Byte-identical for unedited assets; correct for anim
  value / keyframe-count / new-slot / skeleton bind-pose **and PMO vertex-move**
  edits. PMO geometry is **in-place** (reshape: move verts/normals/UVs at the same
  topology — byte-exact re-encode incl. the parser's native struct-alignment pad
  bytes); adding/removing geometry (a GE-list rebuild) is rejected here and lives in
  `pmo_topology.py` (Phase 5). The in-PPSSPP visual check is the one human step
  (`examples/edit_anim_demo.py`).
- **`pmo_topology.py` — topology-GROW encoder (Phase 5, kept separate from `pmo.py`).**
  Byte-level rebuild of the PMO `geBase` GE-list region to ADD vertices/faces within an
  existing vertex group (inherits its bone/material/VTYPE). `parse` / `grow_group` /
  `grow_group_explicit` / `serialize` re-lay the region 16-byte-aligned, patch each list's
  VADDR/IADDR + each vgroup record's I3/I4/I5, bump header size; everything before `geBase`
  stays byte-identical. CLIs:
  - `python -m mhfu_model.pmo_topology in.bin out.bin -n <verts> --spread <r>` — synthetic
    ring/fan grow (**spread>0 required** — a uniform shift makes degenerate, invisible
    triangles). `--bone <i>` / `-g <vgroup>` choose the target group (bind index ==
    vgroup draw order); `--weight-slot <s>` picks the bone-palette slot the new verts bind
    100% to; `--force-16bit`; `--list` prints the vgroup→bone table.
  - `grow_group_explicit(g, verts, tris, …)` — author-supplied verts+faces (the Blender
    path): positions/UV/normals written ABSOLUTE from the caller.
  **Polish DONE + PROVEN IN-GAME (2026-06-18):** new verts now get **real per-vertex attributes** —
  varied UVs (circular texture patch → textured, not the old one-texel dark look), outward
  normals (catch light), and a **clean single-bone weight** (1.0 on `weight_slot`, default
  the group's primary bone) instead of copying vertex0. **16-bit-index auto-promote**: an
  8-bit group crossing 256 verts promotes its VTYPE index field (`B`→`H`), raising the cap
  to 65536. Positions are bounded by the group's per-axis header scale (existing verts are
  stored as fractions of `scale`, so coords past the bounding box saturate). Tested in
  `tests/test_pmo_topology.py` (50-PMO geometry-preserving round-trip + grow + auto-promote
  + explicit-attrs + hard cap).
- Phase 4 = live in-RAM inject (reshape, same size) · Phase 5 = ADD geometry, delivered
  live via the relocate-source path (`framework/prx` `mhfu.inject_relocate` → a grown PAC
  in xram). Synthetic-grow geometry **PROVEN in-game 2026-06-18**, as are the polished
  attributes + 16-bit promote + the **Blender-authored** add path (extra verts in an existing
  mesh → `pmo_topology`): a Blender-authored textured dome rendered on the live Tigrex's head,
  reacting to light, no garbage.

## Quick use

```python
import sys; sys.path.insert(0, "tools")          # or PYTHONPATH=tools
from mhfu_model import load_pac

mm = load_pac("workspace/extracted/data_files/file_06134.bin")   # Tigrex
mm.skeleton.bone_count                # 25
mm.skeleton.bones[0].bind_pos         # (x, y, z) bind-pose, engine units
mm.model.mesh_groups[0].vertices      # decoded geometry (dicts: x,y,z,u,v,...)
mm.anim.animations[0].tracks          # per-bone channels -> keyframes
mm.pac.to_bytes() == open(...,"rb").read()   # byte-identical round-trip
```

Edit + write back:

```python
from mhfu_model import load_pac, repack
from mhfu_model import constraints as K

mm = load_pac("file_06134.bin")
mm.anim.animations[0].tracks[2].channels[0].keyframes[0].value += 200  # edit
assert K.validate(mm.model, mm.skeleton, mm.anim).ok                   # gate
open("out.bin", "wb").write(repack(mm))                                # write
```

CLIs (all `PYTHONPATH=tools`):
- `python -m mhfu_model.dump <file_0XXXX.bin>` — model/skeleton/anim summary
- `python -m mhfu_model.validate <pac> [species|ref.bin]` — constraint report (exit≠0 on error)
- `python -m mhfu_model.examples.edit_anim_demo <src> <dst> [factor]` — make an edited PAC

Tests: `PYTHONPATH=tools python tools/mhfu_model/tests/test_{roundtrip,convert,encode,constraints}.py`

## Format notes (verified Phase 0)

- **PAC**: `u32 count` + `count × (u32 off, u32 size)` interleaved; subs 16-byte
  aligned; empty trailing slots `(0,0)`; opaque tail; file padded to `0x800`.
  Dual-model-set monsters carry a 2nd skeleton/PMO/TMH in slots 4-6.
- **Skeleton** (sub-0, `0xC0000000`): header 0x1C + bone sections (0x10C);
  tree links are bone indices (`-1`=none); bind scale/rot/pos floats.
- **PMO** (sub-1): header at offset **8**; mesh-table stride **0x20 (legacy) or
  0x18 (small-mon)** — auto-detected; monsters are rigid-skinned.
- **Animation** (sub-3, P3rd pack): header `{0x64, hsize, slot_count}`; offset
  table at **`hsize-4`**; nested `{0x80000000|tag, count, size}` sections chain
  exactly; keyframe = 8B `s16 value, frame, ease_in, ease_out`.
- **Quantization**: rotation `4096=90°`, location `16=1.0`, scale `256=1.0`.

## Encoders (Phase 3) + validator (Phase 2) — gotchas pinned

- **Anim block re-encode is byte-exact** from fields (csz=`0x0C+nkf*8`,
  bsz=`0x0C+Σch`, size=`0x14+Σbsz`; no hidden padding — verified all 49). Anim
  offsets are **4-byte aligned**.
- **Anim slots alias**: multiple slots can point at the SAME offset (Tigrex slots
  0 & 2 share one clip). `anim.encode` patches each *unique* offset; a divergent
  edit on an aliased slot triggers a de-aliasing rebuild.
- **There is a secondary table** in the gap between the offset table and the first
  anim — the in-place path preserves it verbatim; the rebuild path preserves
  `raw[:first_anim]` (a size-changing edit may need it refreshed: known limitation).
- **Validator rules are data-verified, not literal-spec:** anims animate a *subset*
  of joints in index order, so lockstep = all-anims-agree + tracks ≤ joints (NOT
  tracks==bones; `file_06143` = 23 tracks / 26 bones). Channel rule =
  `popcount(mask) == channel count` (60539 tracks). Skeleton parse truncates on
  some dual-set variants — joint count is taken as `max(header, parsed)`.
