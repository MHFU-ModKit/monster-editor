"""Constraint validator (scaffold — predicates land in Phase 2).

The validator is the reason this library exists: every edit/export op runs through
it so a modder cannot author something the engine can't represent. Phase 0 defines
only the result types + an empty `validate()`; Phase 2 fills the predicates listed
in specs/002-model-anim-pipeline/tasks.md ("Game constraints the validator MUST
enforce").
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

ERROR = "error"
WARN = "warn"

# s16 quantization domains (engine units)
ROT_UNIT = 4096.0     # = 90 deg
LOC_UNIT = 16.0       # = 1.0
SCL_UNIT = 256.0      # = 1.0
S16_MIN, S16_MAX = -32768, 32767


@dataclass
class Result:
    level: str          # ERROR | WARN
    code: str
    message: str
    fix_hint: str = ""


@dataclass
class Report:
    results: List[Result] = field(default_factory=list)

    def add(self, level, code, message, fix_hint=""):
        self.results.append(Result(level, code, message, fix_hint))

    @property
    def errors(self):
        return [r for r in self.results if r.level == ERROR]

    @property
    def warnings(self):
        return [r for r in self.results if r.level == WARN]

    @property
    def ok(self) -> bool:
        return not self.errors

    def __bool__(self):
        return self.ok


def validate(model=None, skeleton=None, animations=None,
             target_species: Optional[str] = None) -> Report:
    """Run all constraint predicates. Phase 2 implements the checks."""
    return Report()
