"""JL/OS ISO builder -- ISO 9660 + El Torito, written from scratch.

Replaces grub-mkrescue. No GRUB, no xorriso, no Linux host tooling: this runs
on any machine with a Python interpreter and produces a genuinely bootable
optical image.

What a BIOS actually requires to boot a CD:

  sector 16   Primary Volume Descriptor    identifies the filesystem
  sector 17   Boot Record Volume Descriptor "EL TORITO SPECIFICATION" + a
                                            pointer to the boot catalog
  sector 18   Volume Descriptor Terminator
  boot catalog  validation entry + default entry describing the boot image
  boot image    our stage-1, loaded to 0x7C00 in real mode (no emulation)

The previous build_iso.js produced none of this. It concatenated bytes into a
file named .iso; at offset 32768, where the Primary Volume Descriptor must
begin with "CD001", it had "MZ" -- the header of a Windows executable. No
firmware would ever have booted it.

The stage-1 loader needs to know where the kernel sits inside the image, which
is only known once the layout is decided, so this patches KERNEL_LBA, the
sector count and the image size into the boot sector after placing it.

    python mkiso.py --boot boot16.bin --kernel jl_kernel.flat --out jl-os.iso
"""

from __future__ import annotations

import argparse
import pathlib
import struct
import sys

SECTOR = 2048


def pad_to_sector(data: bytes) -> bytes:
    rem = len(data) % SECTOR
    return data + b"\x00" * (SECTOR - rem) if rem else data


def both_endian16(v: int) -> bytes:
    """ISO 9660 stores many numbers in both byte orders back to back."""
    return struct.pack("<H", v) + struct.pack(">H", v)


def both_endian32(v: int) -> bytes:
    return struct.pack("<I", v) + struct.pack(">I", v)


def iso_datetime() -> bytes:
    """17-byte dec-datetime. Fixed value keeps builds byte-reproducible."""
    return b"2026010100000000" + bytes([0])


def primary_volume_descriptor(total_sectors: int, root_lba: int,
                              root_size: int) -> bytes:
    pvd = bytearray(SECTOR)
    pvd[0] = 1                                   # type: primary
    pvd[1:6] = b"CD001"                          # standard identifier
    pvd[6] = 1                                   # version
    pvd[8:40] = b"JL/OS".ljust(32, b" ")         # system identifier
    pvd[40:72] = b"JL_OS".ljust(32, b" ")        # volume identifier
    pvd[80:88] = both_endian32(total_sectors)    # volume space size
    pvd[120:124] = both_endian16(1)              # volume set size
    pvd[124:128] = both_endian16(1)              # volume sequence number
    pvd[128:132] = both_endian16(SECTOR)         # logical block size
    pvd[132:140] = both_endian32(root_size)      # path table size
    pvd[156:190] = root_directory_record(root_lba, root_size, is_self=True)
    pvd[190:318] = b" " * 128                    # volume set identifier
    pvd[318:446] = b"JL/OS".ljust(128, b" ")     # publisher
    pvd[813:830] = iso_datetime()                # creation
    pvd[830:847] = iso_datetime()                # modification
    pvd[847:864] = b"0" * 16 + bytes([0])        # expiration (none)
    pvd[864:881] = iso_datetime()                # effective
    pvd[881] = 1                                 # file structure version
    return bytes(pvd)


def root_directory_record(lba: int, size: int, is_self: bool) -> bytes:
    rec = bytearray(34)
    rec[0] = 34                                  # record length
    rec[2:10] = both_endian32(lba)               # extent LBA
    rec[10:18] = both_endian32(size)             # data length
    rec[18:25] = bytes([126, 1, 1, 0, 0, 0, 0])  # recording date, tz
    rec[25] = 0x02                               # flags: directory
    rec[28:32] = both_endian16(1)                # volume sequence number
    rec[32] = 1                                  # identifier length
    rec[33] = 0 if is_self else 1                # 0 = ".", 1 = ".."
    return bytes(rec)


def boot_record_volume_descriptor(catalog_lba: int) -> bytes:
    """Type 0 descriptor that tells the BIOS where the boot catalog lives."""
    brvd = bytearray(SECTOR)
    brvd[0] = 0
    brvd[1:6] = b"CD001"
    brvd[6] = 1
    brvd[7:39] = b"EL TORITO SPECIFICATION".ljust(32, b"\x00")
    brvd[71:75] = struct.pack("<I", catalog_lba)  # little-endian only here
    return bytes(brvd)


def terminator_volume_descriptor() -> bytes:
    tvd = bytearray(SECTOR)
    tvd[0] = 0xFF
    tvd[1:6] = b"CD001"
    tvd[6] = 1
    return bytes(tvd)


def boot_catalog(boot_lba: int, sector_count: int) -> bytes:
    """Validation entry + default entry, 32 bytes each."""
    cat = bytearray(SECTOR)

    # --- validation entry ---
    val = bytearray(32)
    val[0] = 0x01                    # header id
    val[1] = 0x00                    # platform: 80x86
    val[4:28] = b"JL/OS".ljust(24, b"\x00")
    val[30] = 0x55
    val[31] = 0xAA
    # 16-bit words of the entry must sum to zero mod 0x10000.
    total = sum(struct.unpack("<16H", bytes(val)))
    val[28:30] = struct.pack("<H", (-total) & 0xFFFF)
    cat[0:32] = val

    # --- default entry: no emulation, load at 0x7C00 ---
    ent = bytearray(32)
    ent[0] = 0x88                    # bootable
    ent[1] = 0x00                    # no emulation
    ent[2:4] = struct.pack("<H", 0)  # load segment 0 => BIOS default 0x7C00
    ent[4] = 0x00                    # system type
    ent[6:8] = struct.pack("<H", sector_count)   # virtual 512-byte sectors
    ent[8:12] = struct.pack("<I", boot_lba)      # LBA of the boot image
    cat[32:64] = ent
    return bytes(cat)


def patch_stage1(boot: bytes, kernel_lba: int, kernel_sectors: int,
                 kernel_bytes: int) -> bytes:
    """Write the kernel's location into the stage-1 image.

    Patched by signature rather than by hardcoded offset: the assembler is free
    to move things around, and a wrong offset would corrupt the loader silently
    instead of failing loudly.
    """
    b = bytearray(boot)

    # DAP: 0x10, 0x00, <count u16>, <off u16>, <seg u16>, <lba u64>
    sig = bytes([0x10, 0x00])
    idx = -1
    for i in range(0, len(b) - 16):
        if b[i:i + 2] == sig and struct.unpack_from("<H", b, i + 6)[0] == 0x1000:
            idx = i
            break
    if idx < 0:
        sys.exit("ERROR: could not locate the Disk Address Packet in stage-1")

    struct.pack_into("<H", b, idx + 2, kernel_sectors)
    struct.pack_into("<Q", b, idx + 8, kernel_lba)

    # kernel_dwords immediately follows the 16-byte DAP.
    dwords = (kernel_bytes + 3) // 4
    struct.pack_into("<I", b, idx + 16, dwords)

    print(f"  stage1 patched: DAP at +{idx}, LBA={kernel_lba}, "
          f"sectors={kernel_sectors}, dwords={dwords}")
    return bytes(b)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", required=True, help="stage-1 flat binary")
    ap.add_argument("--kernel", required=True, help="kernel flat binary")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    boot = pathlib.Path(args.boot).read_bytes()
    kernel = pathlib.Path(args.kernel).read_bytes()

    if len(boot) > SECTOR:
        sys.exit(f"ERROR: stage-1 is {len(boot)} bytes, must fit one {SECTOR}-byte sector")

    # Fixed layout. Everything downstream is derived from these.
    lba_pvd, lba_brvd, lba_term, lba_catalog, lba_root = 16, 17, 18, 19, 20
    lba_boot = 21
    lba_kernel = 22

    kernel_sectors_2k = (len(kernel) + SECTOR - 1) // SECTOR
    total_sectors = lba_kernel + kernel_sectors_2k

    print(f"JL/OS ISO builder (no GRUB, no xorriso)")
    print(f"  stage-1 : {len(boot):,} bytes  -> LBA {lba_boot}")
    print(f"  kernel  : {len(kernel):,} bytes ({kernel_sectors_2k} sectors) -> LBA {lba_kernel}")

    boot = patch_stage1(boot, lba_kernel, kernel_sectors_2k, len(kernel))

    root_dir = bytearray(SECTOR)
    root_dir[0:34] = root_directory_record(lba_root, SECTOR, is_self=True)
    root_dir[34:68] = root_directory_record(lba_root, SECTOR, is_self=False)

    image = bytearray()
    image += b"\x00" * (SECTOR * 16)                       # system area
    image += primary_volume_descriptor(total_sectors, lba_root, SECTOR)
    image += boot_record_volume_descriptor(lba_catalog)
    image += terminator_volume_descriptor()
    # El Torito counts the boot image in 512-byte "virtual sectors".
    image += boot_catalog(lba_boot, max(1, (len(boot) + 511) // 512))
    image += bytes(root_dir)
    image += pad_to_sector(boot)
    image += pad_to_sector(kernel)

    assert len(image) == total_sectors * SECTOR, (len(image), total_sectors * SECTOR)
    pathlib.Path(args.out).write_bytes(bytes(image))
    print(f"  wrote {args.out}: {len(image):,} bytes, {total_sectors} sectors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
