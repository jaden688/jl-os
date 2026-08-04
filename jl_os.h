#ifndef JL_OS_H
#define JL_OS_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <math.h>

#define MAX_NODES 3
#define RING_BUFFER_SIZE 256
#define EMBED_DIM 128
#define VOCAB_SIZE 50

// Kernel IDs
typedef enum {
    MK1_TENSOR_CORE = 0,   // Microkernel 1: Matrix / GEMM dispatch
    MK2_ATTN_CORE   = 1,   // Microkernel 2: Attention & KV-Cache Scheduler
    MK3_SYNAPSE_CORE= 2    // Microkernel 3: Zero-Copy IPC & Neural Router
} KernelID;

// Message structure for Ring Buffer IPC
typedef struct {
    uint64_t msg_id;
    KernelID sender;
    KernelID receiver;
    uint32_t opcode;       // 1: EXEC_GEMM, 2: ATTN_PROJECT, 3: ROUTE_TOKEN
    float data_payload[EMBED_DIM];
    uint64_t timestamp_ns;
} IPCMessage;

// Ring buffer IPC ring for Kernel-to-Kernel Zero-Copy Messaging
typedef struct {
    IPCMessage buffer[RING_BUFFER_SIZE];
    uint32_t head;
    uint32_t tail;
    uint64_t total_messages_routed;
} IPCRingBuffer;

// Microkernel State Struct
typedef struct {
    KernelID id;
    const char *name;
    uint32_t active_threads;
    float cpu_load_pct;
    uint64_t ops_count;
    uint64_t memory_used_kb;
    uint32_t status; // 1: OK, 2: BUSY, 3: OVERCLOCK
} MicrokernelNode;

// Global System Architecture
typedef struct {
    MicrokernelNode kernels[MAX_NODES];
    IPCRingBuffer ipc_bus;
    uint64_t total_tokens_generated;
    double current_tps; // Tokens per second
    float avg_ipc_latency_us;
} JlSystem;

// Function Prototypes
void jl_system_init(JlSystem *sys);
void jl_ipc_send(JlSystem *sys, KernelID sender, KernelID receiver, uint32_t opcode, float *payload);
int jl_ipc_pop(JlSystem *sys, IPCMessage *msg_out);
void jl_run_inference_step(JlSystem *sys, const char *prompt, char *output_token, float *latency_us);

#endif // JL_OS_H
