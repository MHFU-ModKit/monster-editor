# mhfu_model

Constraint-aware library for MHFU **big-monster** model PACs (geometry + skeleton +
bind-pose + animation). Single source of truth for the on-disk format and the
engine's rules, shared by the Blender addon, a CLI, and the live PRX injector.
See `specs/002-model-anim-pipeline/tasks.md` and `docs/ANIMATION_FORMAT.md`.

## Status

- **Phase 0 (DONE):** read side + byte-exact PAC container. Parses + round-trips
  byte-identical on all 49 big-monster PACs (`file_06111`–`file_06159`); geometry
  and animation decode for all 49.
- Phase 1 = read-only Blender importer · Phase 2 = validator · Phase 3 = encoders
  · Phase 4 = live in-RAM inject · Phase 5 = new geometry/monster.

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

CLI summary: `PYTHONPATH=tools python -m mhfu_model.dump <file_0XXXX.bin>`
Tests: `PYTHONPATH=tools python tools/mhfu_model/tests/test_roundtrip.py`

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

`encode()` is a lossless raw passthrough until the Phase 3 encoders land; the
round-trip test guards them as they replace it.
