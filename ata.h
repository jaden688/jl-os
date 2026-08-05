/* ============================================================================
 * JL OS :: ATA  --  PIO disk driver
 * ============================================================================
 * Rung 13, and the one that matters most for scale.
 *
 * Stage-1 reads the kernel with INT 13h, but that is BIOS code running in real
 * mode. The instant CR0.PE is set the BIOS is unreachable, so from kmain
 * onwards the kernel has had NO way to touch a disk at all. That is why the
 * model ships inside the boot image: not because the model is small, but
 * because the boot image was the only storage the kernel could reach.
 *
 * This talks to the IDE controller directly, in 28-bit LBA PIO mode. PIO is
 * slow -- one 16-bit word at a time through a port, no DMA -- and that is a
 * deliberate trade: DMA needs bus mastering, a PRD table and physical memory
 * below 4 GiB that stays pinned. PIO needs a status register and patience, and
 * a model loaded once at boot does not care.
 * ============================================================================ */

#ifndef JL_ATA_H
#define JL_ATA_H

#include "jl_core.h"

#define ATA_SECTOR_SIZE 512u

/* Probe the primary bus master. Non-zero if a disk answered IDENTIFY. */
int ata_init(void);

/* Read `count` sectors starting at LBA into buf (count * 512 bytes).
 * Returns the number of sectors actually read. Callers must check: a short
 * read here becomes silent data corruption several layers up. */
uint32_t ata_read(uint32_t lba, uint8_t count, void *buf);

/* Write `count` sectors. Returns sectors written. */
uint32_t ata_write(uint32_t lba, uint8_t count, const void *buf);

/* Sector count reported by IDENTIFY, or 0 if no disk. */
uint32_t ata_sectors(void);

/* Model string from IDENTIFY, so the boot log names the device it found
 * rather than asserting one is present. */
const char *ata_model(void);

int ata_present(void);

#endif
