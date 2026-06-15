"""Remove duplicate synthetic parquet files, keeping the largest file per part+seed.

The synthetic runner can produce names like:

    3dDataset0014_seed0_3090.parquet
    3dDataset0014_seed0_4080.parquet

Both represent the same part and seed, with different PC suffixes.  This script
keeps the largest parquet in each part+seed group and deletes the rest only when
--apply is passed.  Without --apply it prints a dry-run report.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path


DEFAULT_PARQUET_DIR = (
    r"Y:\04_개별폴더\22. 통합과정 오인욱"
    r"\sdf_dataset_synthetic_v2\_ALL_PARQUET_FILES"
)

_PARQUET_NAME_RE = re.compile(r"^(?P<part>.+)_seed(?P<seed>\d+)(?:_(?P<suffix>.*))?$")


@dataclass(frozen=True)
class ParquetEntry:
    path: Path
    part: str
    seed: int
    is_chosen_only: bool
    size_bytes: int
    mtime: float

    @property
    def group_key(self) -> tuple[str, int, bool]:
        return (self.part, self.seed, self.is_chosen_only)


def _parse_entry(path: Path) -> ParquetEntry | None:
    stem = path.stem
    is_chosen_only = False
    chosen_suffix = "_chosen_only"
    if stem.endswith(chosen_suffix):
        is_chosen_only = True
        stem = stem[: -len(chosen_suffix)]

    match = _PARQUET_NAME_RE.match(stem)
    if not match:
        return None

    stat = path.stat()
    return ParquetEntry(
        path=path,
        part=match.group("part"),
        seed=int(match.group("seed")),
        is_chosen_only=is_chosen_only,
        size_bytes=int(stat.st_size),
        mtime=float(stat.st_mtime),
    )


def _pick_keep(entries: list[ParquetEntry]) -> ParquetEntry:
    return max(entries, key=lambda item: (item.size_bytes, item.mtime, item.path.name))


def dedupe(parquet_dir: Path, apply: bool) -> dict[str, int]:
    parquet_dir = parquet_dir.expanduser().resolve()
    if not parquet_dir.exists():
        raise FileNotFoundError(f"Parquet directory not found: {parquet_dir}")
    if not parquet_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {parquet_dir}")

    groups: dict[tuple[str, int, bool], list[ParquetEntry]] = {}
    skipped_unparsed = 0
    for path in sorted(parquet_dir.glob("*.parquet")):
        entry = _parse_entry(path)
        if entry is None:
            skipped_unparsed += 1
            continue
        groups.setdefault(entry.group_key, []).append(entry)

    duplicate_groups = 0
    deleted_files = 0
    bytes_to_delete = 0
    for key, entries in sorted(groups.items(), key=lambda item: item[0]):
        if len(entries) < 2:
            continue
        duplicate_groups += 1
        keep = _pick_keep(entries)
        kind = "chosen_only" if key[2] else "normal"
        print(
            f"[Duplicate] part={key[0]} seed={key[1]} kind={kind} count={len(entries)} "
            f"keep={keep.path.name} size={keep.size_bytes}"
        )
        for entry in sorted(entries, key=lambda item: item.path.name):
            if entry.path == keep.path:
                continue
            bytes_to_delete += entry.size_bytes
            deleted_files += 1
            action = "DELETE" if apply else "DRY-RUN delete"
            print(f"  [{action}] {entry.path.name} size={entry.size_bytes}")
            if apply:
                entry.path.unlink()

    print(
        f"[Summary] groups={len(groups)} duplicate_groups={duplicate_groups} "
        f"files_to_delete={deleted_files} bytes_to_delete={bytes_to_delete} "
        f"skipped_unparsed={skipped_unparsed} apply={apply}"
    )
    return {
        "groups": len(groups),
        "duplicate_groups": duplicate_groups,
        "files_to_delete": deleted_files,
        "bytes_to_delete": bytes_to_delete,
        "skipped_unparsed": skipped_unparsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deduplicate _ALL_PARQUET_FILES by part+seed, keeping the largest parquet."
    )
    parser.add_argument(
        "--parquet-dir",
        default=DEFAULT_PARQUET_DIR,
        help="Path to sdf_dataset_synthetic_v2/_ALL_PARQUET_FILES",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete duplicates. Omit for dry-run.",
    )
    args = parser.parse_args()
    dedupe(Path(args.parquet_dir), apply=bool(args.apply))


if __name__ == "__main__":
    main()
