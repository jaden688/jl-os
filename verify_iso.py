"""Structural validation of a JL/OS ISO. No xorriso, no GRUB, no Linux.

Checks the bytes a BIOS actually reads, in the order it reads them, and exits
non-zero on the first failure. This exists because the previous image in this
repository was accepted as "an ISO" purely because of its filename: it had no
volume descriptor, no boot catalog, and a Windows executable where the
filesystem should start.

    python verify_iso.py jl-os.iso
"""

from __future__ import annotations

import pathlib
import struct
import sys

SECTOR = 2048


def fail(msg: str) -> None:
    print(f"   FAIL: {msg}")
    sys.exit(1)


def main() -> int:
    if len(sys.argv) < 2:
        fail("usage: verify_iso.py <image.iso>")
    data = pathlib.Path(sys.argv[1]).read_bytes()

    if len(data) % SECTOR:
        fail(f"image is {len(data)} bytes, not a whole number of {SECTOR}-byte sectors")

    # --- the check the old image failed ------------------------------------
    if data[:2] == b"MZ" or data[16 * SECTOR:16 * SECTOR + 2] == b"MZ":
        fail("PE/DOS executable found where the ISO filesystem belongs")

    # --- Primary Volume Descriptor at sector 16 ----------------------------
    pvd = data[16 * SECTOR:17 * SECTOR]
    if pvd[0] != 1 or pvd[1:6] != b"CD001":
        fail(f"no Primary Volume Descriptor at sector 16 (got {pvd[:6]!r})")
    total = struct.unpack_from("<I", pvd, 80)[0]
    if total * SECTOR != len(data):
        fail(f"PVD says {total} sectors, image holds {len(data) // SECTOR}")
    print(f"   PVD ok: 'CD001', volume '{pvd[40:72].decode(errors='replace').strip()}', "
          f"{total} sectors")

    # --- Boot Record Volume Descriptor at sector 17 ------------------------
    brvd = data[17 * SECTOR:18 * SECTOR]
    if brvd[0] != 0 or brvd[1:6] != b"CD001":
        fail("no Boot Record Volume Descriptor at sector 17")
    if not brvd[7:39].startswith(b"EL TORITO SPECIFICATION"):
        fail("boot record is not El Torito")
    cat_lba = struct.unpack_from("<I", brvd, 71)[0]
    print(f"   El Torito boot record ok: catalog at LBA {cat_lba}")

    # --- terminator --------------------------------------------------------
    term = data[18 * SECTOR:19 * SECTOR]
    if term[0] != 0xFF or term[1:6] != b"CD001":
        fail("no Volume Descriptor Set Terminator at sector 18")
    print("   terminator ok")

    # --- boot catalog ------------------------------------------------------
    cat = data[cat_lba * SECTOR:(cat_lba + 1) * SECTOR]
    if cat[0] != 0x01 or cat[30] != 0x55 or cat[31] != 0xAA:
        fail("boot catalog validation entry is malformed")
    words = struct.unpack_from("<16H", cat, 0)
    if sum(words) & 0xFFFF:
        fail(f"validation entry checksum is {sum(words) & 0xFFFF:#06x}, must be 0")
    print("   boot catalog validation entry ok (checksum sums to zero)")

    ent = cat[32:64]
    if ent[0] != 0x88:
        fail(f"default entry not marked bootable (0x{ent[0]:02X})")
    if ent[1] != 0x00:
        fail(f"expected no-emulation boot, got media type 0x{ent[1]:02X}")
    boot_lba = struct.unpack_from("<I", ent, 8)[0]
    nsec = struct.unpack_from("<H", ent, 6)[0]
    print(f"   default entry ok: bootable, no-emulation, image at LBA {boot_lba}, "
          f"{nsec} virtual sectors")

    # --- the boot image itself ---------------------------------------------
    boot = data[boot_lba * SECTOR:(boot_lba + 1) * SECTOR]
    if boot[2046:2048] != b"\x55\xAA":
        fail("boot image lacks the 0x55AA signature")
    if not any(boot):
        fail("boot image is empty")
    print("   stage-1 present with 0x55AA signature")

    # --- the kernel the stage-1 was patched to load ------------------------
    idx = -1
    for i in range(0, len(boot) - 16):
        if boot[i:i + 2] == b"\x10\x00" and struct.unpack_from("<H", boot, i + 6)[0] == 0x1000:
            idx = i
            break
    if idx < 0:
        fail("no Disk Address Packet found in stage-1")
    k_sectors = struct.unpack_from("<H", boot, idx + 2)[0]
    k_lba = struct.unpack_from("<Q", boot, idx + 8)[0]
    if k_lba == 0 or k_sectors == 0:
        fail(f"stage-1 was never patched (LBA={k_lba}, sectors={k_sectors})")
    if (k_lba + k_sectors) * SECTOR > len(data):
        fail(f"stage-1 would read past the end of the image (LBA {k_lba}+{k_sectors})")
    print(f"   stage-1 DAP patched: kernel at LBA {k_lba}, {k_sectors} sectors")

    kern = data[k_lba * SECTOR:(k_lba + k_sectors) * SECTOR]
    hits = [k for k in range(0, min(8192, len(kern)) - 12, 4)
            if struct.unpack_from("<I", kern, k)[0] == 0x1BADB002]
    if not hits:
        fail("no Multiboot 1 magic in the first 8 KiB of the loaded kernel")
    m, f, c = struct.unpack_from("<III", kern, hits[0])
    if (m + f + c) & 0xFFFFFFFF:
        fail(f"multiboot checksum is {(m + f + c) & 0xFFFFFFFF:#010x}, must be 0")
    print(f"   kernel multiboot header ok at +{hits[0]} (magic/flags/checksum sum to zero)")

    print("   ISO STRUCTURE VALID")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
