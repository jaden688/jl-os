# JL/OS — architecture and honest maturity

## Execution path

```
firmware (BIOS/CSM)
  -> GRUB 2                          Multiboot 1, `multiboot` command
  -> _start        boot.s            stack, cld, .bss cleared, EAX/EBX preserved
  -> kmain         kmain.c           validates 0x2BADB002, COM1 + VGA evidence
  -> jl_os_init    jl_core.c         MK descriptors, IPC ring
  -> jl_hud_main   jl_hud.c          owns VGA and PS/2; never returns
       └─ nhit_stack  nhit.c         ring convolution on Z/64, integer only
```

No hosted runtime anywhere below GRUB: no libc, no `printf`, no `malloc`, no
files, sockets or threads.

## Microkernel descriptors

MK-1/2/3 are retained. What each currently *is*, not what it is named:

| | current maturity |
|---|---|
| **MK-1 tensor / NHIT** | **Real.** `nhit.c` performs actual integer arithmetic: circular convolution on Z/64, INT4 taps, INT8 activations, INT32 accumulate, shift rescale. Weights in `nhit_weights.h` are genuinely trained by `train_nhit.py`. |
| **MK-2 scheduler** | **Descriptors only.** Load percentages and thread counts in `JlOsState` are fields, not a scheduler. There is no task switching, no preemption, no timer. |
| **MK-3 IPC ring** | **Real data structure, no concurrency.** `jl_dispatch_ipc` writes real messages into a real ring buffer, but there is one thread of control and nothing yet to synchronise. |

## Real vs simulated

**Real kernel mechanisms**
- Multiboot 1 boot, validated loader magic, kernel-owned stack, self-cleared `.bss`
- VGA text output by direct writes to `0xB8000`, double buffered
- PS/2 keyboard input by polling ports `0x60`/`0x64`
- COM1 16550 UART at `0x3F8`, 115200 8N1, with a loopback self-test
- `panic()` — interrupts off, reason to serial and VGA, permanent `hlt`
- `RDTSC` cycle timing around real work

**Real arithmetic**
- NHIT ring convolution, integer only. 192 trained INT4 taps across 3 layers,
  where an equivalent GEMM stack would carry 12,288 weights.

**Simulated — explicitly not an LLM**
- `jl_inference_step` in `jl_core.c` selects tokens from a fixed 20-entry
  vocabulary using a linear congruential generator. It is a placeholder. It
  performs no inference and must not be described as a language model.
- MK-2 "scheduling" and the per-kernel load percentages.

**Model capability, stated plainly.** `nhit_weights.h` records measured numbers,
not aspirational ones:

```
float accuracy 0.2397 / integer path 0.1200 / baseline 0.1172
```

Next-character prediction. The float model reaches roughly 2× the
majority-class baseline; INT4 quantisation removes most of that, leaving the
integer path — the one that actually boots — at approximately baseline. It is
real trained inference executing real integer arithmetic in kernel space, and
it is not yet accurate. 192 INT4 taps is why. Larger N or INT8 taps is the
obvious next step and has not been done.

## Not implemented

No IDT or interrupt handling (interrupts stay masked — enabling them without an
IDT triple-faults). No paging or virtual memory. No physical memory manager;
the Multiboot memory map is received but not parsed. No timer, no preemption,
no processes, no syscalls, no storage, no filesystem, no USB, no networking, no
SMP, no userspace.

## Hardware assumptions

Legacy BIOS/CSM boot, VGA text mode at `0xB8000`, a PS/2 controller at
`0x60`/`0x64`, and a 16550-compatible UART at `0x3F8`. QEMU provides all four.
Modern UEFI-only hardware may provide none — which is why serial exists as
independent evidence.

## Obsolete

`build_iso.js` is not the build path and cannot produce a bootable image; see
`BAREMETAL_BOOT_GUIDE.md`. The previous `franken_kernel.bin` was a PE32 Windows
executable and has been removed along with the hosted `.exe` artifacts, none of
which were evidence of bare-metal execution.
