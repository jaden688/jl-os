"""Build a PEWTER-FS volume image.

PEWTER: Persistent Extent Write, Table, Entry, and Retrieval Filesystem.

The kernel can only READ what this writes, and only if every field agrees
exactly -- the superblock checksum, the FNV-1a over each file, the block size
baked into pewter.h. A mismatch here does not produce a mount error at some
convenient later point; it produces a volume the kernel refuses, or worse, one
it accepts with wrong data. So the layout constants below are duplicated from
pewter.h deliberately and asserted, rather than being derived and hoped over.

    python mkpewter.py --out pewter.img --size 32 \
        --add /models/core.nhit=core.nhit --add /etc/motd=motd.txt
"""

from __future__ import annotations

import argparse
import pathlib
import struct

MAGIC = 0x52545750          # "PWTR" little-endian
VERSION = 1
BLOCK = 4096
NAME_MAX = 40
ENTRY_SZ = 64
SUPER_FMT = "<12I"          # 11 fields + checksum
ENTRY_FMT = f"<{NAME_MAX}s6I"


def fnv1a(data: bytes) -> int:
    """Must match pewter_hash() in pewter.c byte for byte."""
    h = 2166136261
    for b in data:
        h ^= b
        h = (h * 16777619) & 0xFFFFFFFF
    return h


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="pewter.img")
    ap.add_argument("--size", type=int, default=32, help="volume size in MiB")
    ap.add_argument("--dir-entries", type=int, default=64)
    ap.add_argument("--add", action="append", default=[],
                    help="/path/on/volume=hostfile")
    args = ap.parse_args()

    total_blocks = (args.size * 1024 * 1024) // BLOCK

    # Layout. Every region is a whole number of blocks so nothing straddles a
    # boundary and the kernel never has to read two blocks for one structure.
    bitmap_bits = total_blocks
    bitmap_blocks = (bitmap_bits + 7) // 8
    bitmap_blocks = (bitmap_blocks + BLOCK - 1) // BLOCK
    dir_blocks = (args.dir_entries * ENTRY_SZ + BLOCK - 1) // BLOCK

    bitmap_start = 1
    dir_start = bitmap_start + bitmap_blocks
    data_start = dir_start + dir_blocks

    image = bytearray(total_blocks * BLOCK)
    bitmap = bytearray(bitmap_blocks * BLOCK)

    def mark(block: int) -> None:
        bitmap[block // 8] |= 1 << (block % 8)

    # Metadata blocks are allocated up front: the allocator must never hand out
    # the superblock or the bitmap that describes it.
    for b in range(data_start):
        mark(b)

    entries = []
    cursor = data_start

    for spec in args.add:
        if "=" not in spec:
            print(f"  bad --add (want /vol/path=hostfile): {spec}")
            return 1
        vpath, host = spec.split("=", 1)
        data = pathlib.Path(host).read_bytes()

        name = vpath.encode("ascii")
        if len(name) >= NAME_MAX:
            print(f"  name too long (max {NAME_MAX - 1}): {vpath}")
            return 1

        blocks = (len(data) + BLOCK - 1) // BLOCK
        if cursor + blocks > total_blocks:
            print(f"  volume full adding {vpath}")
            return 1

        off = cursor * BLOCK
        image[off:off + len(data)] = data      # tail stays zero, no stale bytes
        for b in range(cursor, cursor + blocks):
            mark(b)

        entries.append((name, cursor, len(data), blocks, fnv1a(data)))
        print(f"  {vpath:<28} {len(data):>9,} bytes  "
              f"{blocks:>4} blocks @ {cursor}  fnv 0x{fnv1a(data):08x}")
        cursor += blocks

    # Directory table
    dirbuf = bytearray(dir_blocks * BLOCK)
    for i, (name, start, size, blocks, csum) in enumerate(entries):
        rec = struct.pack(ENTRY_FMT, name, start, size, blocks, csum, 1, 0)
        assert len(rec) == ENTRY_SZ, len(rec)
        dirbuf[i * ENTRY_SZ:(i + 1) * ENTRY_SZ] = rec

    # Superblock. Checksum covers the 11 fields preceding it -- the same range
    # pewter_mount() hashes, which is sizeof(PewterSuper) - 4.
    fields = [MAGIC, VERSION, BLOCK, total_blocks, bitmap_start, bitmap_blocks,
              dir_start, dir_blocks, args.dir_entries, data_start, len(entries)]
    csum = fnv1a(struct.pack("<11I", *fields))
    superblock = struct.pack(SUPER_FMT, *fields, csum)

    image[0:len(superblock)] = superblock
    image[bitmap_start * BLOCK:bitmap_start * BLOCK + len(bitmap)] = bitmap
    image[dir_start * BLOCK:dir_start * BLOCK + len(dirbuf)] = dirbuf

    pathlib.Path(args.out).write_bytes(image)

    used = cursor - data_start
    avail = total_blocks - data_start
    print(f"\n  PEWTER volume {args.out}")
    print(f"    magic        0x{MAGIC:08X} (\"PWTR\")")
    print(f"    block size   {BLOCK} bytes")
    print(f"    total        {total_blocks:,} blocks ({args.size} MiB)")
    print(f"    bitmap       block {bitmap_start} .. {bitmap_start+bitmap_blocks-1}")
    print(f"    directory    block {dir_start} .. {dir_start+dir_blocks-1}"
          f"  ({args.dir_entries} entries)")
    print(f"    data         block {data_start} ..")
    print(f"    files        {len(entries)}")
    print(f"    used         {used * BLOCK / 1024:.0f} KiB of "
          f"{avail * BLOCK / 1024 / 1024:.0f} MiB")
    print(f"    super fnv    0x{csum:08x}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
