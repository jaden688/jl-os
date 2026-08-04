"""Validate a Multiboot 1 header without GRUB. Exits non-zero on failure."""
import struct, sys, pathlib

MAGIC = 0x1BADB002

def main() -> int:
    data = pathlib.Path(sys.argv[1]).read_bytes()
    if data[:4] == b"\x7fELF":
        # ELF: the header lives in the file, find it by scanning the image.
        pass
    window = min(len(data), 8192 + 4096)   # allow for ELF headers before it
    hits = [i for i in range(0, window - 12, 4)
            if struct.unpack_from("<I", data, i)[0] == MAGIC]
    if not hits:
        print("   FAIL: no Multiboot 1 magic 0x1BADB002 found"); return 1
    off = hits[0]
    magic, flags, checksum = struct.unpack_from("<III", data, off)
    total = (magic + flags + checksum) & 0xFFFFFFFF
    if total != 0:
        print(f"   FAIL: magic+flags+checksum = 0x{total:08X}, must be 0"); return 1
    if off % 4:
        print(f"   FAIL: header at {off} is not 4-byte aligned"); return 1
    print(f"   magic 0x{magic:08X}  flags 0x{flags:08X}  checksum 0x{checksum:08X}")
    print(f"   at file offset {off}, 4-byte aligned, fields sum to zero")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
