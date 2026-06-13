"""CLI: `python -m mhfu_model.validate <file_0XXXX.bin> [species]`.

Runs the constraint validator against a PAC and prints the report; exit code is
non-zero if any ERROR is present (so it can gate a build/export step). When a
second PAC path or a known species name is given, it is registered as the
template the skeleton is checked against.

Run from the repo's `tools/` dir or with PYTHONPATH=tools.
"""
from __future__ import annotations

import os
import sys

from . import load_pac
from . import constraints as K


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    path = argv[0]
    mm = load_pac(path)

    species = None
    if len(argv) > 1:
        arg = argv[1]
        if os.path.isfile(arg):                      # template from a reference PAC
            species = os.path.splitext(os.path.basename(arg))[0]
            K.register_template(species, load_pac(arg).skeleton)
        else:                                        # name: template from THIS pac
            species = arg
            K.register_template(species, mm.skeleton)

    rep = K.validate(mm.model, mm.skeleton, mm.anim, target_species=species)
    print("validate %s%s" % (os.path.basename(path),
                             " vs template %r" % species if species else ""))
    print(rep)
    print("\n%d error(s), %d warning(s) — %s"
          % (len(rep.errors), len(rep.warnings), "OK" if rep.ok else "FAILED"))
    return 0 if rep.ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
