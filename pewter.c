/* ============================================================================
 * JL OS :: PEWTER-FS  (implementation)
 * ============================================================================ */

#include "pewter.h"
#include "ata.h"
#include "serial.h"
#include "kheap.h"

#define SECTORS_PER_BLOCK (PEWTER_BLOCK / ATA_SECTOR_SIZE)   /* 8 */

static PewterSuper sb;
static int mounted = 0;

/* One block of scratch. Static rather than automatic: 4 KiB is half the kernel
 * stack and a quarter of a task stack, and a filesystem read must not be the
 * thing that overflows either. */
static uint8_t blockbuf[PEWTER_BLOCK];

/* Allocation bitmap, cached in memory for the life of the mount. See the
 * commentary above bitmap_test for why this is not read from disk per query. */
static uint8_t *bitmap = 0;

uint32_t pewter_hash(const void *data, uint32_t len) {
    /* FNV-1a. Chosen because the host tool must reproduce it exactly and this
     * is four lines in any language -- a CRC table would be another thing to
     * keep in sync between mkpewter.py and here. */
    const uint8_t *p = (const uint8_t *)data;
    uint32_t h = 2166136261u;
    for (uint32_t i = 0; i < len; i++) {
        h ^= p[i];
        h *= 16777619u;
    }
    return h;
}

static int read_block(uint32_t block, void *buf) {
    return ata_read(block * SECTORS_PER_BLOCK, SECTORS_PER_BLOCK, buf)
           == SECTORS_PER_BLOCK;
}

static int write_block(uint32_t block, const void *buf) {
    return ata_write(block * SECTORS_PER_BLOCK, SECTORS_PER_BLOCK, buf)
           == SECTORS_PER_BLOCK;
}

static int name_eq(const char *a, const char *b) {
    uint32_t i = 0;
    for (; i < PEWTER_NAME_MAX; i++) {
        if (a[i] != b[i]) return 0;
        if (!a[i]) return 1;
    }
    return 1;
}

int pewter_mount(void) {
    mounted = 0;
    if (!ata_present()) {
        serial_write("[pewter] no disk -- nothing to mount\n");
        return 0;
    }
    if (!read_block(0, blockbuf)) {
        serial_write("[pewter] cannot read block 0\n");
        return 0;
    }

    const PewterSuper *s = (const PewterSuper *)blockbuf;
    if (s->magic != PEWTER_MAGIC) {
        serial_write("[pewter] bad magic -- not a PEWTER volume\n");
        return 0;
    }
    if (s->version != PEWTER_VERSION) {
        serial_write("[pewter] unsupported version\n");
        return 0;
    }
    if (s->block_size != PEWTER_BLOCK) {
        /* The block size is baked into SECTORS_PER_BLOCK and blockbuf, so a
         * mismatch cannot be handled by reading harder. */
        serial_write("[pewter] block size mismatch\n");
        return 0;
    }
    uint32_t want = pewter_hash(s, sizeof(PewterSuper) - 4);
    if (want != s->checksum) {
        serial_write("[pewter] superblock checksum mismatch -- refusing mount\n");
        return 0;
    }

    for (uint32_t i = 0; i < sizeof(PewterSuper); i++)
        ((uint8_t *)&sb)[i] = blockbuf[i];

    bitmap = (uint8_t *)kmalloc(sb.bitmap_blocks * PEWTER_BLOCK);
    if (!bitmap) {
        serial_write("[pewter] cannot allocate bitmap cache -- mount refused\n");
        return 0;
    }
    for (uint32_t b = 0; b < sb.bitmap_blocks; b++) {
        if (!read_block(sb.bitmap_start + b, bitmap + b * PEWTER_BLOCK)) {
            serial_write("[pewter] bitmap read failed -- mount refused\n");
            kfree(bitmap); bitmap = 0;
            return 0;
        }
    }
    mounted = 1;

    /* Computed BEFORE taking the lock: the free-space scan reads the whole
     * bitmap off disk, and holding interrupts off across disk I/O would drop
     * timer ticks. Only the printing needs to be atomic. */
    uint32_t mib_free = pewter_free_bytes() / (1024u * 1024u);

    uint32_t f = serial_lock();
    serial_write("[pewter] volume mounted\n");
    serial_write("[pewter] ");
    serial_dec(mib_free);
    serial_write(" MiB free, ");
    serial_dec(sb.file_count);
    serial_write(" files\n");
    serial_unlock(f);
    return 1;
}

const PewterEntry *pewter_entry_at(uint32_t index) {
    if (!mounted || index >= sb.dir_entries) return 0;

    /* Entries per block, and which block this index lives in. The directory is
     * read on demand rather than cached: it is a handful of blocks and caching
     * it would mean invalidating on every write. */
    uint32_t per = PEWTER_BLOCK / PEWTER_ENTRY_SZ;
    uint32_t blk = sb.dir_start + index / per;
    if (!read_block(blk, blockbuf)) return 0;

    const PewterEntry *e =
        (const PewterEntry *)(blockbuf + (index % per) * PEWTER_ENTRY_SZ);
    return (e->flags & PEWTER_F_USED) ? e : 0;
}

const PewterEntry *pewter_find(const char *name) {
    if (!mounted) return 0;
    for (uint32_t i = 0; i < sb.dir_entries; i++) {
        const PewterEntry *e = pewter_entry_at(i);
        if (e && name_eq(e->name, name)) return e;
    }
    return 0;
}

uint32_t pewter_read(const char *name, void *buf, uint32_t max) {
    const PewterEntry *e = pewter_find(name);
    if (!e) {
        serial_write("[pewter] not found: ");
        serial_write(name);
        serial_write("\n");
        return 0;
    }
    /* Copy the entry out before reading data: pewter_entry_at returned a
     * pointer INTO blockbuf, which the data reads below are about to overwrite. */
    PewterEntry ent = *e;

    if (ent.size > max) {
        serial_write("[pewter] buffer too small for ");
        serial_write(name);
        serial_write("\n");
        return 0;
    }

    uint8_t *out = (uint8_t *)buf;
    uint32_t remaining = ent.size;
    for (uint32_t b = 0; b < ent.blocks && remaining; b++) {
        if (!read_block(ent.start_block + b, blockbuf)) return 0;
        uint32_t n = remaining < PEWTER_BLOCK ? remaining : PEWTER_BLOCK;
        for (uint32_t i = 0; i < n; i++) *out++ = blockbuf[i];
        remaining -= n;
    }

    /* Verified, not assumed. Corrupt model weights do not crash -- they produce
     * confident nonsense, which is the failure mode hardest to notice. */
    uint32_t got = pewter_hash(buf, ent.size);
    if (got != ent.checksum) {
        serial_write("[pewter] CHECKSUM MISMATCH on ");
        serial_write(name);
        serial_write(" -- data is corrupt, read rejected\n");
        return 0;
    }
    return ent.size;
}

/* --- allocation ---------------------------------------------------------- */

/* The bitmap is CACHED in memory for the life of the mount.
 *
 * The first implementation read the containing block from disk on every single
 * bit query, which made find_extent do one 4 KiB disk read per candidate block
 * -- thousands of reads to scan a 32 MiB volume, slow enough that a write never
 * finished inside a 25-second boot. Scanning an allocation bitmap is inherently
 * a per-bit operation, so the fix is to stop going to the disk for it, not to
 * scan more cleverly.
 *
 * A 32 MiB volume needs 1 KiB of bitmap; 4 GiB would need 128 KiB. Allocated
 * from the kernel heap, which is why this could not have been done before
 * rung 9. */

static int bitmap_test(uint32_t block) {
    if (!bitmap) return 1;                         /* no cache = treat as used */
    return (bitmap[block / 8] >> (block % 8)) & 1u;
}

/* Set bits in the cache, then flush only the blocks the change touched. */
static int bitmap_mark(uint32_t start, uint32_t count) {
    if (!bitmap) return 0;
    uint32_t first_byte = start / 8;
    uint32_t last_byte = (start + count - 1) / 8;

    for (uint32_t b = start; b < start + count; b++)
        bitmap[b / 8] |= (uint8_t)(1u << (b % 8));

    for (uint32_t blk = first_byte / PEWTER_BLOCK;
         blk <= last_byte / PEWTER_BLOCK; blk++) {
        if (!write_block(sb.bitmap_start + blk, bitmap + blk * PEWTER_BLOCK))
            return 0;
    }
    return 1;
}

/* First contiguous run of `count` free blocks. Contiguity is the whole point of
 * an extent filesystem: it is why a read is one sequential run and why there is
 * no block-pointer indirection to walk. */
static uint32_t find_extent(uint32_t count) {
    uint32_t run = 0;
    for (uint32_t b = sb.data_start; b < sb.total_blocks; b++) {
        if (bitmap_test(b)) { run = 0; continue; }
        if (++run == count) return b - count + 1;
    }
    return 0;
}

uint32_t pewter_write(const char *name, const void *buf, uint32_t size) {
    if (!mounted || !size) return 0;

    uint32_t blocks = (size + PEWTER_BLOCK - 1) / PEWTER_BLOCK;
    uint32_t start = find_extent(blocks);
    if (!start) {
        serial_write("[pewter] no contiguous extent large enough for ");
        serial_write(name);
        serial_write("\n");
        return 0;
    }

    /* Find a free directory slot before writing any data, so a full directory
     * fails before it has consumed blocks it cannot reference. */
    uint32_t per = PEWTER_BLOCK / PEWTER_ENTRY_SZ;
    uint32_t slot = 0xFFFFFFFFu;
    for (uint32_t i = 0; i < sb.dir_entries && slot == 0xFFFFFFFFu; i++) {
        if (!read_block(sb.dir_start + i / per, blockbuf)) return 0;
        PewterEntry *e = (PewterEntry *)(blockbuf + (i % per) * PEWTER_ENTRY_SZ);
        if (!(e->flags & PEWTER_F_USED)) slot = i;
    }
    if (slot == 0xFFFFFFFFu) {
        serial_write("[pewter] directory full\n");
        return 0;
    }

    const uint8_t *in = (const uint8_t *)buf;
    uint32_t remaining = size;
    for (uint32_t b = 0; b < blocks; b++) {
        uint32_t n = remaining < PEWTER_BLOCK ? remaining : PEWTER_BLOCK;
        for (uint32_t i = 0; i < PEWTER_BLOCK; i++)
            blockbuf[i] = (i < n) ? *in++ : 0;      /* zero the tail, no leak */
        if (!write_block(start + b, blockbuf)) return 0;
        remaining -= n;
    }

    if (!bitmap_mark(start, blocks)) return 0;

    if (!read_block(sb.dir_start + slot / per, blockbuf)) return 0;
    PewterEntry *e = (PewterEntry *)(blockbuf + (slot % per) * PEWTER_ENTRY_SZ);
    for (uint32_t i = 0; i < PEWTER_NAME_MAX; i++)
        e->name[i] = (i < PEWTER_NAME_MAX && name[i]) ? name[i] : 0;
    e->start_block = start;
    e->size = size;
    e->blocks = blocks;
    e->checksum = pewter_hash(buf, size);
    e->flags = PEWTER_F_USED;
    e->reserved = 0;
    if (!write_block(sb.dir_start + slot / per, blockbuf)) return 0;

    /* Superblock LAST. Until this lands the file is invisible, so an
     * interrupted write leaves a consistent volume with some blocks marked
     * used -- leaked, not corrupt. That is the trade instead of a journal. */
    sb.file_count++;
    sb.checksum = pewter_hash(&sb, sizeof(PewterSuper) - 4);
    for (uint32_t i = 0; i < PEWTER_BLOCK; i++) blockbuf[i] = 0;
    for (uint32_t i = 0; i < sizeof(PewterSuper); i++)
        blockbuf[i] = ((const uint8_t *)&sb)[i];
    if (!write_block(0, blockbuf)) return 0;

    return size;
}

int pewter_mounted(void) { return mounted; }
uint32_t pewter_file_count(void) { return mounted ? sb.file_count : 0; }
uint32_t pewter_total_bytes(void) {
    return mounted ? (sb.total_blocks - sb.data_start) * PEWTER_BLOCK : 0;
}

uint32_t pewter_free_bytes(void) {
    if (!mounted) return 0;
    uint32_t free_blocks = 0;
    for (uint32_t b = sb.data_start; b < sb.total_blocks; b++)
        if (!bitmap_test(b)) free_blocks++;
    return free_blocks * PEWTER_BLOCK;
}
