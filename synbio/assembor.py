"""Curated yeast promoter/terminator part helpers.

This module integrates the small Assembor-style yeast parts table into
Zhuangzhou without importing the external curation workspace. The CSV keeps the
six-column schema used by the source project:

name,type,bin,sequence,doi,note
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Iterable


FIELDS = ["name", "type", "bin", "sequence", "doi", "note"]
PROMOTER_BINS = {
    "very_strong",
    "strong",
    "medium_high",
    "medium",
    "low",
    "inducible",
    "repressible",
    "unknown",
}
TERMINATOR_BINS = {"high", "standard", "medium", "weak", "short_synthetic", "unknown"}
RESERVED_DOI_VALUES = {"manual:S288C_R64", "database:SGD", "database:Addgene", "unknown"}
DOI_RE = re.compile(r"^10\.\S+/.+")
VALID_SEQUENCE = set("ACGTN")


@dataclass(frozen=True)
class YeastPart:
    name: str
    type: str
    bin: str
    sequence: str
    doi: str
    note: str

    @property
    def has_sequence(self) -> bool:
        return bool(self.sequence)

    @property
    def length(self) -> int:
        return len(self.sequence)

    @property
    def gc_percent(self) -> float | None:
        if not self.sequence:
            return None
        gc = self.sequence.count("G") + self.sequence.count("C")
        return 100.0 * gc / len(self.sequence)

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "type": self.type,
            "bin": self.bin,
            "sequence": self.sequence,
            "doi": self.doi,
            "note": self.note,
            "has_sequence": self.has_sequence,
            "length": self.length,
            "gc_percent": self.gc_percent,
        }


def default_parts_path() -> Path:
    return Path(resources.files("zhuangzhou.synbio").joinpath("data/yeast_parts.csv"))


def load_yeast_parts(path: str | Path | None = None) -> list[YeastPart]:
    csv_path = Path(path) if path else default_parts_path()
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != FIELDS:
            found = ",".join(reader.fieldnames or [])
            raise ValueError(f"invalid yeast parts header: {found}")
        parts = []
        for row in reader:
            parts.append(
                YeastPart(
                    name=(row["name"] or "").strip(),
                    type=(row["type"] or "").strip(),
                    bin=(row["bin"] or "").strip(),
                    sequence="".join((row["sequence"] or "").split()).upper(),
                    doi=(row["doi"] or "").strip(),
                    note=(row["note"] or "").strip(),
                )
            )
    return parts


def validate_yeast_parts(path: str | Path | None = None) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    info: list[str] = []

    try:
        parts = load_yeast_parts(path)
    except Exception as exc:
        return {"valid": False, "errors": [str(exc)], "warnings": warnings, "info": info}

    seen: set[str] = set()
    for index, part in enumerate(parts, start=2):
        if not part.name:
            errors.append(f"row {index}: name is required")
        elif part.name in seen:
            errors.append(f"row {index}: duplicate name {part.name!r}")
        seen.add(part.name)

        if part.type not in {"promoter", "terminator"}:
            errors.append(f"row {index}: invalid type {part.type!r}")
        elif part.type == "promoter" and part.bin not in PROMOTER_BINS:
            errors.append(f"row {index}: invalid promoter bin {part.bin!r}")
        elif part.type == "terminator" and part.bin not in TERMINATOR_BINS:
            errors.append(f"row {index}: invalid terminator bin {part.bin!r}")

        if part.sequence:
            invalid = sorted(set(part.sequence) - VALID_SEQUENCE)
            if invalid:
                errors.append(f"row {index}: invalid bases {''.join(invalid)!r}")
            if part.length < 30 or part.length > 3000:
                warnings.append(f"row {index}: sequence length {part.length} is outside 30-3000 bp")
            gc = part.gc_percent
            if gc is not None and (gc < 20.0 or gc > 80.0):
                warnings.append(f"row {index}: GC content {gc:.1f}% is outside 20-80%")

        if not part.doi:
            errors.append(f"row {index}: doi is required")
        elif part.doi not in RESERVED_DOI_VALUES and not DOI_RE.match(part.doi):
            errors.append(f"row {index}: invalid doi {part.doi!r}")

    promoters = sum(1 for part in parts if part.type == "promoter")
    terminators = sum(1 for part in parts if part.type == "terminator")
    sequenced = sum(1 for part in parts if part.has_sequence)
    info.extend([
        f"rows={len(parts)}",
        f"promoters={promoters}",
        f"terminators={terminators}",
        f"sequenced={sequenced}",
    ])
    return {"valid": not errors, "errors": errors, "warnings": warnings, "info": info}


def query_yeast_parts(
    part_type: str | None = None,
    strength_bin: str | None = None,
    require_sequence: bool = False,
    path: str | Path | None = None,
) -> list[dict]:
    parts = load_yeast_parts(path)
    filtered: Iterable[YeastPart] = parts
    if part_type:
        filtered = (part for part in filtered if part.type == part_type)
    if strength_bin:
        filtered = (part for part in filtered if part.bin == strength_bin)
    if require_sequence:
        filtered = (part for part in filtered if part.has_sequence)
    return [part.as_dict() for part in filtered]


def select_yeast_expression_parts(
    promoter_bin: str = "strong",
    terminator_bin: str = "standard",
    require_sequence: bool = False,
    n_candidates: int = 5,
) -> dict:
    promoters = query_yeast_parts("promoter", promoter_bin, require_sequence)[:n_candidates]
    terminators = query_yeast_parts("terminator", terminator_bin, require_sequence)[:n_candidates]
    if not terminators and terminator_bin != "high":
        terminators = query_yeast_parts("terminator", "high", require_sequence)[:n_candidates]
    return {
        "organism": "saccharomyces_cerevisiae",
        "promoter_bin": promoter_bin,
        "terminator_bin": terminator_bin,
        "require_sequence": require_sequence,
        "promoters": promoters,
        "terminators": terminators,
        "validation": validate_yeast_parts(),
    }
