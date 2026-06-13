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
  value / keyframe-count / new-slot / skeleton bind-pose edits. PMO *geometry* edit
  emission is the remaining stretch (re-emits source byte-identical today). The
  in-PPSSPP visual check is the one human step (see `examples/edit_anim_demo.py`).
- Phase 4 = live in-RAM inject · Phase 5 = new geometry/monster.

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
