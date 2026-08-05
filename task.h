/* ============================================================================
 * JL OS :: TASK  --  kernel threads and the round-robin scheduler
 * ============================================================================
 * Rung 10, and the end of a piece of dishonesty: MK-2 has been reported as
 * "online (simulated)" since the first boot. This replaces it. Tasks here are
 * real -- each has its own stack, its own saved context, and is preempted by
 * the PIT whether it cooperates or not.
 *
 * These are KERNEL threads: ring 0, one shared address space, no isolation.
 * Separate address spaces and user mode are rungs 11 and 12. What makes this
 * rung real is preemption -- a task that never yields still loses the CPU.
 * ============================================================================ */

#ifndef JL_TASK_H
#define JL_TASK_H

#include "jl_core.h"

#define TASK_STACK_SIZE 8192u
#define TASK_NAME_MAX 16

typedef enum {
    TASK_READY = 0,
    TASK_RUNNING,
    TASK_BLOCKED,
    TASK_DONE
} TaskState;

typedef struct Task {
    /* MUST stay first: switch_context writes through a pointer to this field. */
    uint32_t esp;
    uint32_t id;
    TaskState state;
    uint32_t slices;                /* times actually scheduled -- measured */
    uint32_t *stack_base;           /* kept for kfree on exit */
    char name[TASK_NAME_MAX];
    struct Task *next;
} Task;

/* Adopt the current execution context as task 0. Must run before any spawn:
 * the first switch has to have somewhere to save the context it leaves. */
void sched_init(void);

/* Create a runnable task. Returns 0 if memory is exhausted -- callers check,
 * because there is no failure path below this. */
Task *task_spawn(const char *name, void (*entry)(void));

/* Give up the rest of this slice voluntarily. */
void sched_yield(void);

/* Called from the timer IRQ once preemption is armed. Separate from
 * sched_yield only so the tick accounting stays honest about which switches
 * were forced and which were voluntary. */
void sched_tick(void);

/* Preemption is OFF until this is called. Arming it before there is more than
 * one task would switch the kernel to a task list of one and gain nothing,
 * while adding a context switch to every timer interrupt during early boot. */
void sched_enable_preemption(void);
int sched_preemption_enabled(void);

Task *sched_current(void);
uint32_t sched_task_count(void);
uint32_t sched_switches(void);

/* Walk the task list for reporting. Returns 0 past the end. */
Task *sched_task_at(uint32_t index);

#endif
