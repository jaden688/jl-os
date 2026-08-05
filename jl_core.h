/* ============================================================================
 * JL OS: BARE-METAL MULTI-MICROKERNEL NEURAL INFERENCE OPERATING SYSTEM
 * ============================================================================
 * Zero dependencies. No POSIX libc. No Linux kernel. 
 * Bare-metal x86_64 Multiboot2 compliant microkernel architecture.
 * ============================================================================ */

#ifndef JL_CORE_H
#define JL_CORE_H

/* The kernel has no libc, so it defines its own fixed-width types. A HOST build
 * (test_lm_host.c) compiles these same sources against real headers to measure
 * the integer path, and there these typedefs would collide with the platform's
 * own -- size_t in particular is 64-bit on a 64-bit host. Deferring to
 * <stdint.h> under JL_HOST is what lets the harness score the ACTUAL kernel
 * code rather than a reimplementation of it.
 *
 * Kernel builds never define JL_HOST and are unaffected. */
#ifdef JL_HOST
#include <stdint.h>
#include <stddef.h>
#else
typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
typedef unsigned long long uint64_t;
typedef signed char        int8_t;
typedef signed short       int16_t;
typedef signed int         int32_t;
typedef signed long long   int64_t;
typedef uint32_t           size_t;
#endif

#ifndef NULL
#define NULL ((void*)0)
#endif
#define ALIGN16 __attribute__((aligned(16)))

/* ============================================================================
 * HARDWARE & VGA DRIVER CONSTANTS
 * ============================================================================ */
#define VGA_WIDTH  80
#define VGA_HEIGHT 25
#define VGA_MEMORY ((uint16_t*)0xB8000)

typedef enum {
    VGA_BLACK = 0,
    VGA_BLUE = 1,
    VGA_GREEN = 2,
    VGA_CYAN = 3,
    VGA_RED = 4,
    VGA_MAGENTA = 5,
    VGA_BROWN = 6,
    VGA_LIGHT_GREY = 7,
    VGA_DARK_GREY = 8,
    VGA_LIGHT_BLUE = 9,
    VGA_LIGHT_GREEN = 10,
    VGA_LIGHT_CYAN = 11,
    VGA_LIGHT_RED = 12,
    VGA_LIGHT_MAGENTA = 13,
    VGA_LIGHT_BROWN = 14,
    VGA_YELLOW = 14,
    VGA_WHITE = 15,
} VGAColor;

uint8_t vga_entry_color(VGAColor fg, VGAColor bg);
uint16_t vga_entry(unsigned char uc, uint8_t color);

/* ============================================================================
 * TENSOR & QUANTIZED MATRIX STRUCTURES (INT4 / FP32)
 * ============================================================================ */
#define VECTOR_DIM 64
#define MATRIX_ROWS 64
#define MATRIX_COLS 64

// Quantized INT4 Block for zero-copy ultra-fast GEMM matrix math
typedef struct {
    float scale;
    uint8_t qweight[VECTOR_DIM / 2]; // 4-bit packed weights
} ALIGN16 QuantBlock4;

typedef struct {
    uint32_t rows;
    uint32_t cols;
    QuantBlock4 blocks[MATRIX_ROWS];
} ALIGN16 QuantMatrix4;

/* ============================================================================
 * MULTI-MICROKERNEL ARCHITECTURE & ZERO-COPY IPC RING BUS
 * ============================================================================ */
#define MAX_KERNELS 3
#define IPC_RING_SIZE 128

typedef enum {
    MK_TENSOR_CORE  = 0, // Microkernel 1: GEMM & SIMD Vector compute
    MK_ATTN_SCHED   = 1, // Microkernel 2: Paged KV-Cache & Attention
    MK_SYNAPSE_BUS  = 2  // Microkernel 3: Zero-Copy IPC & Memory Routing
} MicrokernelID;

typedef struct {
    uint64_t msg_id;
    MicrokernelID sender;
    MicrokernelID receiver;
    uint32_t opcode; // 0x1: MATMUL, 0x2: ATTENTION_PROJ, 0x3: ROUTE_TOKEN
    float vector_data[16];
    uint64_t timestamp_ticks;
} IPCMessage;

typedef struct {
    IPCMessage buffer[IPC_RING_SIZE];
    uint32_t head;
    uint32_t tail;
    uint64_t total_dispatches;
} IPCRingBus;

typedef struct {
    MicrokernelID id;
    const char* name;
    uint32_t active_tasks;
    uint32_t load_pct;
    uint64_t ops_executed;
} MicrokernelNode;

typedef struct {
    MicrokernelNode nodes[MAX_KERNELS];
    IPCRingBus bus;
    uint64_t total_tokens_generated;
    uint32_t tps_rate;
} JlOsState;

/* ============================================================================
 * FUNCTION DECLARATIONS
 * ============================================================================ */
void k_vga_init(void);
void k_vga_clear(void);
void k_vga_putc(char c, uint8_t color);
void k_vga_puts(const char* str, uint8_t color);
void k_vga_draw_header(void);

// Microkernel math primitives
void gemm_quantized_int4(const float* input, const QuantMatrix4* weight, float* output);
void softmax(float* x, int size);
void jl_os_init(JlOsState* state);
void jl_dispatch_ipc(JlOsState* state, MicrokernelID sender, MicrokernelID receiver, uint32_t opcode, float* payload);
void jl_inference_step(JlOsState* state, char* out_token);

/* Native bare-metal HUD (hud.c). Owns the screen after boot: draws to
 * 0xB8000 directly and reads the PS/2 keyboard. Never returns. */
void jl_hud_main(void);

#endif // JL_CORE_H
