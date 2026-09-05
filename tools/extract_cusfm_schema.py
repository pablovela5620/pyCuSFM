"""Rebuild `data/cusfm_schema/cusfm_protos.fdset` from the cuSFM binaries.

cuSFM's prebuilt executables embed every `.proto` they were compiled against as
serialised `FileDescriptorProto` blobs in their `protodesc_cold` ELF section.
This tool reads that section out of each binary, recovers the blobs, merges them
across binaries and writes one dependency-ordered `FileDescriptorSet`. That set
is what `colsfm.schema` turns into runtime message classes, so cuSFM protobufs
are parsed by protobuf itself rather than by hand-written guesswork.

Ported from `data/cusfm_re/tools/extract_protodesc.py` (section location) and
`data/cusfm_re/schema/extract_desc.py` (blob boundary scan).

Usage::

    pixi run -e colsfm python tools/extract_cusfm_schema.py
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Final, TypeAlias

import tyro
from google.protobuf import descriptor_pb2, descriptor_pool

from colsfm import REPO_ROOT

DescriptorBlobs: TypeAlias = dict[str, bytes]
"""Serialised `FileDescriptorProto` bytes keyed by the proto file name."""

SECTION_NAME: Final[str] = "protodesc_cold"
"""ELF section holding the embedded descriptors in every cuSFM binary."""

PROTO_NAME_PATTERN: Final[re.Pattern[bytes]] = re.compile(rb"protos/[a-z0-9_/]+\.proto")
"""Every embedded descriptor names itself with this path shape."""

FILE_DESCRIPTOR_WIRE_TYPES: Final[dict[int, int]] = {
    1: 2,
    2: 2,
    3: 2,
    4: 2,
    5: 2,
    6: 2,
    7: 2,
    8: 2,
    9: 2,
    10: 0,
    11: 0,
    12: 2,
    13: 2,
    14: 0,
}
"""Valid top-level `FileDescriptorProto` field numbers mapped to their wire types.

Used to walk a descriptor field by field and find where it ends, since the blobs
sit back to back in the section with no length prefix of their own.
"""


@dataclass(frozen=True)
class ExtractConfig:
    """Where to read cuSFM binaries from and where to write the merged schema."""

    binary_dir: Path = REPO_ROOT / "pycusfm" / "x86_cuda13" / "bin"
    """Directory of unstripped cuSFM executables to scan."""
    output_path: Path = REPO_ROOT / "data" / "cusfm_schema" / "cusfm_protos.fdset"
    """Destination `FileDescriptorSet`, vendored into the repo."""
    extra_descriptor_sets: tuple[Path, ...] = ()
    """Additional `FileDescriptorSet` files to merge in, for descriptors no binary carries."""
    verbose: bool = True
    """Print one line per proto file found."""


def read_elf_section(binary_path: Path, section_name: str) -> bytes:
    """Read one ELF section's bytes out of a binary.

    Args:
        binary_path: Executable to read.
        section_name: Section to extract, e.g. ``protodesc_cold``.

    Returns:
        The raw section bytes, or empty bytes when the section is absent.
    """
    listing: str = subprocess.run(
        ["readelf", "-S", "-W", str(binary_path)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    for line in listing.splitlines():
        match: re.Match[str] | None = re.search(r"\]\s+(\S+)\s+\S+\s+([0-9a-f]+)\s+([0-9a-f]+)\s+([0-9a-f]+)", line)
        if match is None or match.group(1) != section_name:
            continue
        offset: int = int(match.group(3), 16)
        size: int = int(match.group(4), 16)
        return binary_path.read_bytes()[offset : offset + size]
    return b""


def read_varint(data: bytes, index: int) -> tuple[int | None, int]:
    """Decode a protobuf base-128 varint.

    Args:
        data: Buffer to read from.
        index: Offset of the first varint byte.

    Returns:
        The decoded value (None when the varint is truncated or absurd) and the
        offset just past it.
    """
    value: int = 0
    shift: int = 0
    while index < len(data):
        byte: int = data[index]
        index += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, index
        shift += 7
        if shift > 63:
            return None, index
    return None, index


def find_descriptor_end(data: bytes, start: int) -> int | None:
    """Walk `FileDescriptorProto` fields from `start` until one does not parse.

    Args:
        data: Section bytes.
        start: Offset of the descriptor's first tag byte.

    Returns:
        The exclusive end offset of the descriptor, or None when no `name` field
        was seen (so the candidate is not a descriptor at all).
    """
    index: int = start
    seen_name: bool = False
    while index < len(data):
        tag: int | None
        cursor: int
        tag, cursor = read_varint(data, index)
        if tag is None:
            break
        field_number: int = tag >> 3
        wire_type: int = tag & 7
        if FILE_DESCRIPTOR_WIRE_TYPES.get(field_number) != wire_type:
            break
        if field_number == 1:
            seen_name = True
        if wire_type == 2:
            length: int | None
            length, cursor = read_varint(data, cursor)
            if length is None or cursor + length > len(data):
                break
            cursor += length
        else:
            value: int | None
            value, cursor = read_varint(data, cursor)
            if value is None:
                break
        index = cursor
    return index if seen_name else None


def scan_descriptor_blobs(section: bytes) -> DescriptorBlobs:
    """Recover every embedded `FileDescriptorProto` from a `protodesc_cold` section.

    Each descriptor begins with the `name` field, tag byte ``0x0A`` followed by a
    length-delimited path such as ``protos/visual/general/keyframe.proto``, so the
    path itself is the anchor.

    Args:
        section: Raw section bytes.

    Returns:
        Serialised descriptor bytes keyed by proto file name.
    """
    blobs: DescriptorBlobs = {}
    for match in PROTO_NAME_PATTERN.finditer(section):
        name_start: int = match.start()
        name_length: int = match.end() - match.start()
        for prefix_length in (1, 2):
            header_start: int = name_start - prefix_length - 1
            if header_start < 0 or section[header_start] != 0x0A:
                continue
            declared_length: int | None
            cursor: int
            declared_length, cursor = read_varint(section, header_start + 1)
            if declared_length != name_length or cursor != name_start:
                continue
            end: int | None = find_descriptor_end(section, header_start)
            if end is None:
                continue
            blob: bytes = section[header_start:end]
            descriptor: descriptor_pb2.FileDescriptorProto = descriptor_pb2.FileDescriptorProto()
            if descriptor.ParseFromString(blob) != len(blob):
                continue
            if descriptor.name != match.group().decode("ascii"):
                continue
            previous: bytes | None = blobs.get(descriptor.name)
            if previous is None or len(blob) > len(previous):
                blobs[descriptor.name] = blob
            break
    return blobs


def read_descriptor_set(path: Path) -> DescriptorBlobs:
    """Read an existing `FileDescriptorSet` into per-file blobs.

    Args:
        path: `FileDescriptorSet` on disk.

    Returns:
        Serialised descriptor bytes keyed by proto file name.
    """
    file_set: descriptor_pb2.FileDescriptorSet = descriptor_pb2.FileDescriptorSet.FromString(path.read_bytes())
    return {descriptor.name: descriptor.SerializeToString() for descriptor in file_set.file}


def order_by_dependency(blobs: DescriptorBlobs) -> list[descriptor_pb2.FileDescriptorProto]:
    """Sort descriptors so every file follows the files it imports.

    Args:
        blobs: Serialised descriptor bytes keyed by proto file name.

    Returns:
        Descriptors in an order a `DescriptorPool` accepts in one pass.

    Raises:
        ValueError: When the import graph has a cycle or an unsatisfiable edge.
    """
    descriptors: dict[str, descriptor_pb2.FileDescriptorProto] = {
        name: descriptor_pb2.FileDescriptorProto.FromString(blob) for name, blob in blobs.items()
    }
    ordered: list[descriptor_pb2.FileDescriptorProto] = []
    emitted: set[str] = set()
    pending: list[str] = sorted(descriptors)
    while pending:
        progressed: bool = False
        for name in list(pending):
            dependencies: list[str] = [dep for dep in descriptors[name].dependency if dep in descriptors]
            if not all(dep in emitted for dep in dependencies):
                continue
            ordered.append(descriptors[name])
            emitted.add(name)
            pending.remove(name)
            progressed = True
        if not progressed:
            raise ValueError(f"Unresolvable protobuf import order for {pending}")
    return ordered


def build_descriptor_set(blobs: DescriptorBlobs) -> descriptor_pb2.FileDescriptorSet:
    """Assemble a dependency-ordered `FileDescriptorSet` and prove it loads.

    Args:
        blobs: Serialised descriptor bytes keyed by proto file name.

    The proof matters: `colsfm.schema.build_schema` adds the files in file order
    and lets a failure propagate, so the ordering has to be right here rather than
    rediscovered at load time.

    Returns:
        A `FileDescriptorSet` that a fresh `DescriptorPool` accepts in file order.
    """
    file_set: descriptor_pb2.FileDescriptorSet = descriptor_pb2.FileDescriptorSet()
    pool: descriptor_pool.DescriptorPool = descriptor_pool.DescriptorPool()
    for descriptor in order_by_dependency(blobs):
        pool.Add(descriptor)
        file_set.file.add().CopyFrom(descriptor)
    return file_set


def extract(config: ExtractConfig) -> descriptor_pb2.FileDescriptorSet:
    """Scan every binary in `config.binary_dir` and write the merged schema.

    Args:
        config: Input directory, output path and merge extras.

    Returns:
        The written `FileDescriptorSet`.

    Raises:
        FileNotFoundError: When the binary directory does not exist.
        ValueError: When two sources disagree about one proto file's contents.
    """
    if not config.binary_dir.is_dir():
        raise FileNotFoundError(f"No cuSFM binary directory at {config.binary_dir}")
    blobs: DescriptorBlobs = {}
    sources: dict[str, str] = {}
    for binary_path in sorted(config.binary_dir.iterdir()):
        if not binary_path.is_file():
            continue
        section: bytes = read_elf_section(binary_path, SECTION_NAME)
        if not section:
            continue
        found: DescriptorBlobs = scan_descriptor_blobs(section)
        _merge(blobs, sources, found, binary_path.name)
        if config.verbose:
            print(f"{binary_path.name}: {len(found)} proto files")
    for extra_path in config.extra_descriptor_sets:
        _merge(blobs, sources, read_descriptor_set(extra_path), str(extra_path))

    file_set: descriptor_pb2.FileDescriptorSet = build_descriptor_set(blobs)
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_path.write_bytes(file_set.SerializeToString())
    if config.verbose:
        for descriptor in file_set.file:
            print(f"  {descriptor.name}  <- {sources[descriptor.name]}")
    print(f"wrote {config.output_path} with {len(file_set.file)} proto files")
    return file_set


def _merge(blobs: DescriptorBlobs, sources: dict[str, str], found: DescriptorBlobs, source: str) -> None:
    """Fold newly found descriptors into the accumulated set.

    Args:
        blobs: Accumulated descriptor bytes, mutated in place.
        sources: First source per proto file name, mutated in place.
        found: Descriptors from one binary or descriptor set.
        source: Human-readable origin of `found`, for the report.

    Raises:
        ValueError: When a proto file appears twice with different contents.
    """
    for name, blob in found.items():
        previous: bytes | None = blobs.get(name)
        if previous is None:
            blobs[name] = blob
            sources[name] = source
        elif previous != blob:
            raise ValueError(f"{name} differs between {sources[name]} and {source}")


def main() -> None:
    """Run the extractor from the command line."""
    extract(tyro.cli(ExtractConfig))


if __name__ == "__main__":
    main()
