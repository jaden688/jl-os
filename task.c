/* ============================================================================
 * JL OS :: TASK  --  kernel threads and scheduler (implementation)
 * ============================================================================ */

#include "task.h"
#include "kheap.h"
#include "serial.h"

extern void switch_context(uint32_t *save_esp_here, uint32_t load_esp);

static Task *current = 0;
static Task *task_list = 0;             /* circular, singly linked */
static uint32_t next_id = 0;
static uint32_t task_count = 0;
static uint32_t switches = 0;
static int preempt = 0;

static void name_copy(char *dst, const char *src) {
    uint32_t i = 0;
    for (; src && src[i] && i < TASK_NAME_MAX - 1; i++) dst[i] = src[i];
    dst[i] = 0;
}

/* Where a task lands when its entry function returns. Without this the `ret`
 * at the end of the entry function would jump to whatever happened to be on
 * the stack -- for a freshly built stack, zero. */
static void task_exit(void) {
    current->state = TASK_DONE;
    serial_write("[task] ");
    serial_write(current->name);
    serial_write(" finished\n");
    /* Never returns: a DONE task is skipped by the scheduler forever. */
    for (;;) sched_yield();
}

void sched_init(void) {
    /* Task 0 is the context already running. Its ESP is not set here -- it gets
     * written by the first switch_context, which saves the real value at the
     * exact point the switch happens. Any value invented now would be wrong. */
    Task *t = (Task *)kmalloc(sizeof(Task));
    if (!t) {
        serial_write("[task] cannot allocate task 0 -- scheduler disabled\n");
        return;
    }
    t->esp = 0;
    t->id = next_id++;
    t->state = TASK_RUNNING;
    t->slices = 1;
    t->stack_base = 0;                  /* boot stack, from linker.ld, not heap */
    name_copy(t->name, "kmain");
    t->next = t;                        /* circular list of one */

    current = t;
    task_list = t;
    task_count = 1;

    serial_write("[task] scheduler initialized, task 0 = kmain\n");
}

Task *task_spawn(const char *name, void (*entry)(void)) {
    if (!current) return 0;

    uint32_t *stack = (uint32_t *)kmalloc(TASK_STACK_SIZE);
    if (!stack) return 0;
    Task *t = (Task *)kmalloc(sizeof(Task));
    if (!t) { kfree(stack); return 0; }

    /* Build, by hand, exactly the frame switch_context expects to pop. Order
     * matters and is the mirror image of the pops in switch.s:
     *
     *   [esp+20] entry        <- where `ret` goes
     *   [esp+16] ebp
     *   [esp+12] ebx
     *   [esp+8]  esi
     *   [esp+4]  edi
     *   [esp+0]  eflags       <- popped first by popfl
     *
     * Above `entry` sits task_exit, so when the entry function returns it
     * returns into the exit path rather than into garbage.
     */
    uint32_t *sp = (uint32_t *)((uint8_t *)stack + TASK_STACK_SIZE);
    sp = (uint32_t *)((uint32_t)sp & ~15u);      /* 16-byte align */

    *(--sp) = (uint32_t)task_exit;      /* return address of `entry`      */
    *(--sp) = (uint32_t)entry;          /* switch_context's ret target    */
    *(--sp) = 0;                        /* ebp                            */
    *(--sp) = 0;                        /* ebx                            */
    *(--sp) = 0;                        /* esi                            */
    *(--sp) = 0;                        /* edi                            */
    /* EFLAGS with IF set and bit 1 (always-one). A task started with
     * interrupts masked would never be preempted, and if it never yielded the
     * whole system would stop. */
    *(--sp) = 0x202;

    t->esp = (uint32_t)sp;
    t->id = next_id++;
    t->state = TASK_READY;
    t->slices = 0;
    t->stack_base = stack;
    name_copy(t->name, name);

    /* Insert after current, keeping the ring intact. Interrupts are masked
     * because the timer may call sched_tick and walk this list. */
    __asm__ __volatile__("cli");
    t->next = current->next;
    current->next = t;
    task_count++;
    __asm__ __volatile__("sti");

    serial_write("[task] spawned ");
    serial_write(t->name);
    serial_write("\n");
    return t;
}

/* Pick the next task that is not DONE and switch to it. Returns with `current`
 * updated. Caller must have interrupts masked. */
static void schedule_locked(void) {
    if (!current || task_count < 2) return;

    Task *prev = current;
    Task *next = current->next;
    uint32_t guard = 0;
    while (next->state == TASK_DONE && guard++ < task_count) next = next->next;

    /* Everything else has exited; keep running what we have rather than
     * switching into a finished task's dead stack. */
    if (next == prev || next->state == TASK_DONE) return;

    if (prev->state == TASK_RUNNING) prev->state = TASK_READY;
    next->state = TASK_RUNNING;
    next->slices++;
    current = next;
    switches++;

    switch_context(&prev->esp, next->esp);
    /* Execution resumes here when `prev` is scheduled again. */
}

void sched_yield(void) {
    __asm__ __volatile__("cli");
    schedule_locked();
    __asm__ __volatile__("sti");
}

void sched_tick(void) {
    if (!preempt) return;
    /* Already inside the IRQ0 handler with interrupts masked by the gate, and
     * the EOI has been sent, so the next tick can arrive for whichever task we
     * switch to. No cli/sti here -- adding sti would re-enter this handler on
     * the outgoing task's stack. */
    schedule_locked();
}

void sched_enable_preemption(void) {
    preempt = 1;
    serial_write("[task] preemption armed -- PIT now drives the scheduler\n");
}

int sched_preemption_enabled(void) { return preempt; }
Task *sched_current(void) { return current; }
uint32_t sched_task_count(void) { return task_count; }
uint32_t sched_switches(void) { return switches; }

Task *sched_task_at(uint32_t index) {
    if (!task_list || index >= task_count) return 0;
    Task *t = task_list;
    for (uint32_t i = 0; i < index; i++) t = t->next;
    return t;
}
