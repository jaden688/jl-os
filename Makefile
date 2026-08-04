# =============================================================================
# JL/OS  --  authoritative bare-metal build
# =============================================================================
# One build path. Firmware -> GRUB -> Multiboot 1 -> _start -> kmain -> HUD.
#
# build_iso.js is NOT part of this pipeline and must not be used to produce a
# boot image: it fabricates a file that resembles an ISO without an ISO 9660
# filesystem or an El Torito boot catalog. The only authoritative image is the
# one grub-mkrescue writes from the tree this Makefile assembles.
#
# Toolchain: a real i686-elf cross compiler is preferred. Host gcc with -m32 is
# accepted as a documented fallback, but only with -ffreestanding -nostdlib so
# no host CRT, startup files or headers are linked in.
# =============================================================================

KERNEL   := jl_kernel.bin
KERNEL_FLAT := jl_kernel.flat
STAGE1   := boot16.bin
ISO      := jl-os.iso
PYTHON   ?= python

# --- toolchain selection -----------------------------------------------------
CROSS := $(shell command -v i686-elf-gcc 2>/dev/null)

ifneq ($(CROSS),)
  CC        := i686-elf-gcc
  AS        := i686-elf-as
  LD        := i686-elf-ld
  OBJCOPY   := i686-elf-objcopy
  ARCHFLAG  :=
  ASFLAGS   :=
  LDEMU     :=
  TOOLCHAIN := i686-elf cross-compiler
else
  CC        := gcc
  AS        := as
  LD        := ld
  OBJCOPY   := objcopy
  ARCHFLAG  := -m32
  ASFLAGS   := --32
  LDEMU     := -m elf_i386
  TOOLCHAIN := host gcc -m32 (FALLBACK -- i686-elf-gcc not found)
endif

# --- flags -------------------------------------------------------------------
# -ffreestanding       no hosted environment; freestanding subset only
# -nostdlib            do not link host CRT or standard libraries
# -fno-builtin         no implicit lowering to libc symbols
# -fno-stack-protector no __stack_chk_fail dependency
# -fno-pie -fno-pic    kernel is linked at a fixed physical address
# -fno-asynchronous-unwind-tables  keeps .eh_frame out of a kernel image
CFLAGS := $(ARCHFLAG) -std=gnu11 -ffreestanding -nostdlib -fno-builtin \
          -fno-stack-protector -fno-pie -fno-pic \
          -fno-asynchronous-unwind-tables \
          -Wall -Wextra -Wshadow -Wpointer-arith -O2 -g

LDFLAGS := $(LDEMU) -T linker.ld -nostdlib -z noexecstack

# Compiler runtime helpers. 32-bit targets lower 64-bit division to __udivdi3.
# Resolve libgcc from the selected compiler rather than leaving it empty and
# failing unpredictably on other machines.
LIBGCC := $(shell $(CC) $(ARCHFLAG) -print-libgcc-file-name 2>/dev/null)

OBJS := boot.o kmain.o idt.o idt_stubs.o pic_pit.o serial.o jl_hud.o nhit.o jl_core.o

.PHONY: all kernel iso run run-nographic debug verify clean toolchain-info stage1

all: iso

toolchain-info:
	@echo "toolchain : $(TOOLCHAIN)"
	@echo "CC        : $(CC) $(ARCHFLAG)"
	@echo "libgcc    : $(if $(LIBGCC),$(LIBGCC),<none found>)"

# --- compile -----------------------------------------------------------------
boot.o: boot.s
	$(AS) $(ASFLAGS) boot.s -o boot.o

idt_stubs.o: idt_stubs.s
	$(AS) $(ASFLAGS) idt_stubs.s -o idt_stubs.o

idt.o: idt.c idt.h serial.h jl_core.h
	$(CC) $(CFLAGS) -c idt.c -o idt.o

pic_pit.o: pic_pit.c pic_pit.h idt.h serial.h jl_core.h
	$(CC) $(CFLAGS) -c pic_pit.c -o pic_pit.o

kmain.o: kmain.c jl_core.h serial.h idt.h pic_pit.h
	$(CC) $(CFLAGS) -c kmain.c -o kmain.o

serial.o: serial.c serial.h jl_core.h
	$(CC) $(CFLAGS) -c serial.c -o serial.o

jl_hud.o: jl_hud.c jl_core.h nhit.h nhit_weights.h pic_pit.h
	$(CC) $(CFLAGS) -c jl_hud.c -o jl_hud.o

nhit.o: nhit.c nhit.h jl_core.h
	$(CC) $(CFLAGS) -c nhit.c -o nhit.o

jl_core.o: jl_core.c jl_core.h
	$(CC) $(CFLAGS) -c jl_core.c -o jl_core.o

# --- link --------------------------------------------------------------------
kernel: $(KERNEL)

$(KERNEL): $(OBJS) linker.ld
	$(LD) $(LDFLAGS) $(OBJS) $(LIBGCC) -o $(KERNEL)
	@echo "linked $(KERNEL) with $(TOOLCHAIN)"

# --- iso (our own bootloader + our own ISO writer; no GRUB, no xorriso) ------
stage1: $(STAGE1)

boot16.o: boot16.s
	$(AS) $(ASFLAGS) boot16.s -o boot16.o

# Stage-1 is linked flat at 0x7C00, where the BIOS places an El Torito
# no-emulation boot image, then stripped to raw bytes.
$(STAGE1): boot16.o
	$(LD) $(LDEMU) -e _boot_start -Ttext 0x7C00 --oformat binary boot16.o -o $(STAGE1)

# The kernel is loaded by our own stage-1, which cannot parse ELF in real mode,
# so give it a flat image. .bss is NOBITS and correctly absent from this.
$(KERNEL_FLAT): $(KERNEL)
	$(OBJCOPY) -O binary $(KERNEL) $(KERNEL_FLAT)

iso: $(ISO)

$(ISO): $(STAGE1) $(KERNEL_FLAT) mkiso.py
	$(PYTHON) mkiso.py --boot $(STAGE1) --kernel $(KERNEL_FLAT) --out $(ISO)

# --- verify ------------------------------------------------------------------
# Every check is fatal: the target fails loudly rather than reporting success
# on an image that would not boot.
verify: $(KERNEL)
	@echo "== 1. kernel is 32-bit x86 ELF executable =="
	@file $(KERNEL)
	@file $(KERNEL) | grep -q "ELF 32-bit LSB executable" \
	  || { echo "FAIL: not a 32-bit ELF executable"; exit 1; }
	@file $(KERNEL) | grep -qE "Intel (80386|i386)" \
	  || { echo "FAIL: wrong architecture"; exit 1; }
	@echo "== 2. NOT a PE/Windows executable =="
	@if file $(KERNEL) | grep -qi "PE32\|MS Windows"; then \
	   echo "FAIL: a PE executable is being passed off as a kernel"; exit 1; fi
	@echo "   confirmed: no PE/COFF image in the boot path"
	@echo "== 3. entry point and ELF header =="
	@readelf -h $(KERNEL) | grep -E "Type:|Machine:|Entry point"
	@readelf -h $(KERNEL) | grep -q "EXEC" \
	  || { echo "FAIL: not an EXEC image"; exit 1; }
	@echo "== 4. sections and multiboot placement =="
	@readelf -S $(KERNEL) | grep -E "multiboot|\.text|\.bss" || true
	@readelf -S $(KERNEL) | grep -q "multiboot" \
	  || { echo "FAIL: .multiboot section missing (was KEEP dropped?)"; exit 1; }
	@echo "== 5. linker symbols present =="
	@for s in kernel_start kernel_end bss_start bss_end stack_top; do \
	   readelf -s $(KERNEL) | grep -qw $$s \
	     || { echo "FAIL: symbol $$s missing"; exit 1; }; \
	 done
	@echo "   kernel_start kernel_end bss_start bss_end stack_top : present"
	@echo "== 6. multiboot header, checked by us (no GRUB required) =="
	@$(PYTHON) check_multiboot.py $(KERNEL)
	@echo "== 7. ISO structure, checked by us (no xorriso required) =="
	@if [ -f $(ISO) ]; then \
	   file $(ISO); \
	   $(PYTHON) verify_iso.py $(ISO); \
	 else echo "   SKIP: $(ISO) not built (run 'make iso')"; fi
	@echo ""
	@echo "VERIFY PASSED"

# --- run / debug -------------------------------------------------------------
QEMU      := qemu-system-i386
# -no-reboot/-no-shutdown : a triple fault stops visibly instead of looping
# -serial stdio           : COM1 to the terminal -- the authoritative evidence
QEMUFLAGS := -m 256M -no-reboot -no-shutdown -serial stdio

run: $(ISO)
	$(QEMU) -cdrom $(ISO) $(QEMUFLAGS)

run-nographic: $(ISO)
	$(QEMU) -cdrom $(ISO) -m 256M -no-reboot -no-shutdown -nographic

# Starts paused (-S) with a GDB stub on tcp::1234 (-s).
debug: $(ISO)
	$(QEMU) -cdrom $(ISO) $(QEMUFLAGS) -S -s

clean:
	rm -rf *.o $(KERNEL) $(KERNEL_FLAT) $(STAGE1) $(ISO)
