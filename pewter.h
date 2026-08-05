/* ============================================================================
 * JL OS :: PEWTER-FS
 * ============================================================================
 * Persistent Extent Write, Table, Entry, and Retrieval Filesystem.
 *
 * Rung 14. The layer that takes raw metal -- sectors and blocks -- and turns it
 * into named, usable structure.
 *
 * Deliberately tiny, and the omissions are the design:
 *
 *   no journaling      A journal protects against a crash between two writes.
 *                      PEWTER's writes are allocate-then-commit against a
 *                      superblock that is rewritten last, so a crash loses the
 *                      new file rather than the volume.
 *   no permissions     There is one address space and no user mode yet
 *                      (rung 11). Permission bits with nobody to enforce them
 *                      are decoration.
 *   no fragmentation   Files are CONTIGUOUS extents: start block plus length.
 *                      A model is written once and read once; the cost of
 *                      contiguity is wasted space on a full volume, and the
 *                      benefit is that reading a 171 KiB file is one seek and
 *                      one sequential run.
 *   flat namespace     Names are full path strings ("/models/core.nhit") in a
 *                      single table. It reads hierarchical and costs no tree.
 *
 * Layout:
 *
 *   block 0                     superblock
 *   bitmap_start .. +blocks     allocation bitmap, one bit per block
 *   dir_start    .. +blocks     directory table, fixed 64-byte entries
 *   data_start   ..             file extents
 *
 * Blocks are 4096 bytes -- eight ATA sectors, and exactly PAGE_SIZE, so a file
 * loaded from disk lands on page boundaries with no copy to realign it.
 * ============================================================================ */

#ifndef JL_PEWTER_H
#define JL_PEWTER_H

#include "jl_core.h"

#define PEWTER_MAGIC     0x52545750u    /* "PWTR" little-endian */
#define PEWTER_VERSION   1u
#define PEWTER_BLOCK     4096u
#define PEWTER_NAME_MAX  40u
#define PEWTER_ENTRY_SZ  64u

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t block_size;
    uint32_t total_blocks;
    uint32_t bitmap_start;
    uint32_t bitmap_blocks;
    uint32_t dir_start;
    uint32_t dir_blocks;
    uint32_t dir_entries;
    uint32_t data_start;
    uint32_t file_count;
    uint32_t checksum;              /* FNV-1a over the 11 fields above */
} __attribute__((packed)) PewterSuper;

typedef struct {
    char     name[PEWTER_NAME_MAX];
    uint32_t start_block;
    uint32_t size;                  /* bytes */
    uint32_t blocks;                /* extent length in blocks */
    uint32_t checksum;              /* FNV-1a over the file's bytes */
    uint32_t flags;                 /* bit 0 = in use */
    uint32_t reserved;
} __attribute__((packed)) PewterEntry;

#define PEWTER_F_USED 1u

/* Read and validate the superblock. Non-zero on success. Requires ata_init. */
int pewter_mount(void);

int pewter_mounted(void);

uint32_t pewter_file_count(void);
uint32_t pewter_free_bytes(void);
uint32_t pewter_total_bytes(void);

/* Directory walk for listing. Returns 0 past the last used entry. */
const PewterEntry *pewter_entry_at(uint32_t index);

const PewterEntry *pewter_find(const char *name);

/* Read a whole file into buf. Returns bytes read, or 0 on failure.
 *
 * The stored checksum is VERIFIED and a mismatch fails the read. A model whose
 * weights are silently corrupt still produces confident output, so this is the
 * difference between a visible failure and a wrong answer. */
uint32_t pewter_read(const char *name, void *buf, uint32_t max);

/* Create a file from a contiguous run of free blocks. Returns bytes written.
 * The superblock is rewritten LAST, so an interrupted write leaves the volume
 * exactly as it was rather than half-committed. */
uint32_t pewter_write(const char *name, const void *buf, uint32_t size);

/* FNV-1a, exposed because mkpewter.py must compute identical values. */
uint32_t pewter_hash(const void *data, uint32_t len);

#endif
