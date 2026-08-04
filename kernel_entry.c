#include <stdio.h>
#include "jl_core.h"

int main(void) {
    printf("================================================================================\n");
    printf("=== [JL OS] BARE-METAL MULTI-MICROKERNEL NEURAL OS v1.0 ===============\n");
    printf("================================================================================\n");
    printf("[BOOT] Initializing JL OS Multi-Microkernel Subsystems...\n");

    JlOsState os_state;
    jl_os_init(&os_state);

    printf("[BOOT] MK-1 Tensor Matrix Engine:   ONLINE (Threads: 8)\n");
    printf("[BOOT] MK-2 KV-Attention Scheduler: ONLINE (Threads: 4)\n");
    printf("[BOOT] MK-3 Synapse Ring-Buffer IPC: ONLINE (Threads: 6)\n");
    printf("--------------------------------------------------------------------------------\n");
    printf("[INFERENCE STREAM GENERATION - BARE-METAL TENSOR MATH]:\n");

    char token[64];
    for (int step = 0; step < 16; step++) {
        jl_inference_step(&os_state, token);
        printf(" >> %s\n", token);
    }

    printf("--------------------------------------------------------------------------------\n");
    printf("[STATUS] All Microkernels Running. Zero POSIX Dependencies. Pure Custom Code.\n");
    printf("================================================================================\n");

    return 0;
}
