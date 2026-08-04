# JL/OS — bare-metal boot, verification and debugging

## Boot chain

```
firmware (BIOS/CSM)
  -> GRUB 2                         reads /boot/grub/grub.cfg from the ISO
  -> multiboot /boot/jl_kernel.bin  Multiboot 1 (NOT multiboot2)
  -> _start                 (boot.s)  stack, cld, .bss zeroed, EAX/EBX preserved
  -> kmain(magic, mb_info)  (kmain.c) validates 0x2BADB002, serial + VGA
  -> jl_hud_main()          (jl_hud.c) owns display and PS/2, never returns
```

**This is Multiboot 1.** Header magic `0x1BADB002`, loader magic `0x2BADB002`,
GRUB command `multiboot`. Multiboot2 (`0xE85250D6`, `multiboot2`) is not used
anywhere. Earlier comments in `boot.s` claiming Multiboot2 were wrong and have
been corrected.

## Dependencies

```bash
sudo apt install -y build-essential gcc-multilib \
                    grub-pc-bin grub-common xorriso qemu-system-x86
```

A real cross-compiler is preferred and is selected automatically when present:

```bash
i686-elf-gcc  i686-elf-as  i686-elf-ld
```

Without it the build falls back to host `gcc -m32` and prints
`FALLBACK -- i686-elf-gcc not found`. The fallback is freestanding
(`-ffreestanding -nostdlib -fno-builtin`) so no host CRT, startup files or
headers are linked, but a cross-compiler removes that class of risk entirely.

Check which one is active:

```bash
make toolchain-info
```

## Build

```bash
make clean
make kernel      # jl_kernel.bin  — freestanding 32-bit ELF
make iso         # jl-os.iso      — grub-mkrescue image
```

## Verify

```bash
make verify
```

Fails on any of: not a 32-bit x86 ELF `EXEC`; a PE/Windows image in the boot
path; a missing `.multiboot` section; missing linker symbols; `grub-file`
rejecting the header; the ISO not being ISO 9660; or the kernel or `grub.cfg`
absent inside the image.

Expected on success:

```
Type: EXEC   Machine: Intel 80386   Entry point address: 0x101000
[ 1] .multiboot  PROGBITS  00100000  001000  00000c
kernel_start kernel_end bss_start bss_end stack_top : present
grub-file: VALID Multiboot 1 kernel
/boot/jl_kernel.bin and /boot/grub/grub.cfg present in ISO
VERIFY PASSED
```

## Run

```bash
make run              # QEMU, VGA window + COM1 on the terminal
make run-nographic    # serial only, no window
```

Expected serial transcript:

```
[boot] C kernel entered (kmain)
[boot] multiboot magic = 0x2BADB002 expected 0x2BADB002
[boot] multiboot magic validated
[boot] multiboot_info at 0x000.....
[boot] bss 0x00105000 .. 0x0010CF44 cleared by _start
[boot] kernel image 0x00100000 .. 0x0010CF44 (N bytes)
[boot] JL runtime initialized (JlOsState)
[boot] entering HUD (owns display and PS/2 from here)
```

Serial is the authoritative evidence. A blank screen cannot distinguish a
display failure from a kernel that never ran; a line arriving on COM1 proves
the CPU executed the instruction that wrote it.

## Debug with GDB

```bash
make debug         # QEMU starts paused with a GDB stub on tcp::1234
```

In a second terminal:

```bash
gdb jl_kernel.bin
(gdb) target remote localhost:1234
(gdb) break *0x101000          # _start, entry address from the ELF header
(gdb) break kmain
(gdb) continue
(gdb) info registers           # at _start: EAX=0x2BADB002, EBX=mb_info
(gdb) x/8i $eip
(gdb) continue                 # into kmain
(gdb) info args                # magic and mb_info as C arguments
(gdb) print/x magic
(gdb) x/16xb bss_start
(gdb) backtrace
```

`jl_kernel.bin` is built with `-g` and is never stripped, so symbols and line
numbers are available. Break at `_start` by address because it is reached
before any C frame exists.

## Physical hardware

**Not yet attempted.** Nothing below has been validated on a real machine, and
no claim of physical bare-metal success is made.

- **BIOS vs UEFI.** `grub-mkrescue` normally produces an image bootable by
  legacy BIOS/CSM, and by UEFI when `grub-efi-amd64-bin` is installed. The
  kernel is 32-bit and assumes a legacy BIOS boot. On a UEFI-only machine,
  enable CSM/Legacy boot or expect it not to start.
- **Secure Boot must be disabled.** The image is unsigned; firmware will refuse
  it otherwise.
- **Writing to USB is destructive.** `dd` to the wrong device destroys that
  disk. Confirm with `lsblk` immediately before writing, and target the whole
  device, not a partition.

  ```bash
  lsblk
  sudo dd if=jl-os.iso of=/dev/sdX bs=4M status=progress conv=fsync
  ```

- **Use spare hardware**, not a machine holding anything you need.
- **Serial capture.** If the target has a real COM port or a USB-serial
  adapter, capture at 115200 8N1 — on hardware this is often the only way to
  see why a boot failed.
- **Assumptions QEMU hides.** A PS/2 keyboard controller at ports 0x60/0x64
  (USB-only machines may present nothing without BIOS legacy emulation); VGA
  text mode at 0xB8000 (UEFI-native graphics may not provide it); a
  16550-compatible UART at 0x3F8. QEMU supplies all three; real hardware may
  supply none.

## Obsolete — not part of the build

`build_iso.js` concatenates bytes into a file named `.iso` with no ISO 9660
filesystem and no El Torito boot catalog. The image it produced had `MZ` at
offset 32768 where the primary volume descriptor belongs, and could never boot.
The previous `franken_kernel.bin` was a PE32 Windows executable, not a kernel.

Neither is authoritative. Only `make iso` is.
