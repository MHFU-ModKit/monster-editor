"""mhfu_model.inject — Phase 4 host side: push an edited PAC to the live game.

The PRX (`framework/prx/src/core/inject.cpp`) watches a memstick *inject dir* and
overwrites the species' loaded buffer in place. This module is the host end: it
validates an edited `MonsterModel`, repacks it, and drops the bytes into that dir
as `file_<id>.bin` — the only "file write" in the loop, and it touches a loose
memstick file, never the game's DATA.BIN / ISO / savestate.

Effect in-game (see `framework/prx/include/mhfu/inject.h`):
  - animation value edits update the LIVE monster next frame;
  - skeleton / geometry edits apply on the next engine rebuild (section re-entry).

CLI:
    python -m mhfu_model.inject <edited_or_source.pac> [--file-id N] [--dir D]
        [--no-validate] [--species NAME|ref.bin]

If `--file-id` is omitted it is parsed from a `file_NNNNN.bin` name. With no
`--dir`, the PPSSPP memstick inject dir is auto-detected.
"""
from __future__ import annotations

import argparse
import os
import re
import sys

# PAC file index = em_id + 0x17AB (em01=file_06060, Tigrex em75=file_06134). The
# engine's loader fileId is exactly this index, so the inject filename carries it.
_NAME_RE = re.compile(r"file_(\d{4,6})\b")

# PPSSPP memstick roots, most-likely first. The PRX sees `ms0:/PSP/...`.
_MEMSTICK_ROOTS = (
    "~/.config/ppsspp/PSP",
    "~/Documents/PPSSPP/PSP",
    "~/Library/Application Support/PPSSPP/PSP",
)
_INJECT_SUBDIR = "PLUGINS/mhfu_framework/inject"


def default_inject_dir(create: bool = True) -> str:
    """Locate (and optionally create) the PPSSPP memstick inject dir."""
    for root in _MEMSTICK_ROOTS:
        base = os.path.expanduser(root)
        if os.path.isdir(base):
            d = os.path.join(base, _INJECT_SUBDIR)
            if create:
                os.makedirs(d, exist_ok=True)
            return d
    raise FileNotFoundError(
        "no PPSSPP memstick found; pass --dir explicitly. Tried: "
        + ", ".join(_MEMSTICK_ROOTS)
    )


def file_id_from_name(name: str) -> int:
    """Parse the engine fileId from a `file_NNNNN.bin` path/name."""
    m = _NAME_RE.search(os.path.basename(name))
    if not m:
        raise ValueError(f"cannot infer file id from {name!r}; pass file_id")
    return int(m.group(1))


def inject_filename(file_id: int) -> str:
    return f"file_{file_id:05d}.bin"


def write_inject_bytes(data: bytes, file_id: int, inject_dir: str | None = None) -> str:
    """Atomically place `data` as the inject file for `file_id`. Returns the path.

    Written to a temp name then `os.replace`d so the PRX (which polls size+mtime)
    never observes a half-written file.
    """
    if inject_dir is None:
        inject_dir = default_inject_dir()
    os.makedirs(inject_dir, exist_ok=True)
    dst = os.path.join(inject_dir, inject_filename(file_id))
    tmp = dst + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, dst)
    return dst


def emit_model(mm, file_id: int | None = None, inject_dir: str | None = None,
               validate: bool = True, species=None) -> str:
    """Validate (optional) -> repack a MonsterModel -> write the inject file.

    Raises ValueError with the validator report if the edit is engine-invalid,
    so a modder can never push an asset the game can't represent.
    """
    from . import repack
    if validate:
        from .constraints import validate as run_validate
        rep = run_validate(mm.model, mm.skeleton,
                           mm.anim.animations if mm.anim else [],
                           target_species=species)
        if not rep:
            raise ValueError("injection blocked — edit is engine-invalid:\n" + str(rep))
    if file_id is None:
        raise ValueError("file_id is required for emit_model")
    return write_inject_bytes(repack(mm), file_id, inject_dir)


def _main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="mhfu_model.inject", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pac", help="edited (or source) big-monster PAC")
    ap.add_argument("--file-id", type=int, default=None,
                    help="engine fileId (default: parsed from file_NNNNN.bin)")
    ap.add_argument("--dir", default=None, help="inject dir (default: auto-detect)")
    ap.add_argument("--no-validate", action="store_true",
                    help="skip the constraint validator (not recommended)")
    ap.add_argument("--species", default=None,
                    help="species template name or reference PAC for validation")
    args = ap.parse_args(argv)

    from . import load_pac
    fid = args.file_id if args.file_id is not None else file_id_from_name(args.pac)
    mm = load_pac(args.pac)

    species = args.species
    if species and os.path.exists(species):
        from .constraints import register_template
        ref = load_pac(species)
        species = os.path.basename(args.species)
        register_template(species, ref.skeleton)

    try:
        path = emit_model(mm, file_id=fid, inject_dir=args.dir,
                          validate=not args.no_validate, species=species)
    except ValueError as e:
        print(e, file=sys.stderr)
        return 2
    print(f"injected file {fid} -> {path}")
    print("PRX picks it up within ~0.5s; anim updates live, skeleton/geom on "
          "next section re-entry.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
