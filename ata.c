/* ============================================================================
 * JL OS :: ATA  --  PIO disk driver (implementation)
 * ============================================================================ */

#include "ata.h"
#include "serial.h"

/* Primary IDE channel. The command block is at 0x1F0 and the control block at
 * 0x3F6 -- two separate ranges, a legacy of the ISA card these registers came
 * from. */
#define ATA_DATA        0x1F0
#define ATA_ERROR       0x1F1
#define ATA_SECCOUNT    0x1F2
#define ATA_LBA_LO      0x1F3
#define ATA_LBA_MID     0x1F4
#define ATA_LBA_HI      0x1F5
#define ATA_DRIVE       0x1F6
#define ATA_STATUS      0x1F7
#define ATA_COMMAND     0x1F7
#define ATA_CONTROL     0x3F6

#define ATA_SR_BSY      0x80
#define ATA_SR_DRDY     0x40
#define ATA_SR_DRQ      0x08
#define ATA_SR_ERR      0x01

#define ATA_CMD_READ    0x20
#define ATA_CMD_WRITE   0x30
#define ATA_CMD_FLUSH   0xE7
#define ATA_CMD_IDENTIFY 0xEC

static int present = 0;
static uint32_t total_sectors = 0;
static char model_str[41];

static inline uint8_t inb(uint16_t p) {
    uint8_t v; __asm__ __volatile__("inb %1,%0" : "=a"(v) : "Nd"(p)); return v;
}
static inline void outb(uint16_t p, uint8_t v) {
    __asm__ __volatile__("outb %0,%1" :: "a"(v), "Nd"(p));
}
static inline uint16_t inw(uint16_t p) {
    uint16_t v; __asm__ __volatile__("inw %1,%0" : "=a"(v) : "Nd"(p)); return v;
}
static inline void outw(uint16_t p, uint16_t v) {
    __asm__ __volatile__("outw %0,%1" :: "a"(v), "Nd"(p));
}

/* The controller needs ~400ns after a drive select before STATUS is valid.
 * Reading the alternate status port four times is the canonical way to spend
 * it: each read is an ISA bus cycle and has no side effects, unlike reading
 * the real status register, which acknowledges the interrupt. */
static void ata_delay(void) {
    for (int i = 0; i < 4; i++) (void)inb(ATA_CONTROL);
}

/* Bounded wait. An unbounded spin here hangs the whole kernel on a drive that
 * never becomes ready, with no console output to say why. */
static int ata_wait(uint8_t mask, uint8_t want, uint32_t limit) {
    while (limit--) {
        uint8_t s = inb(ATA_STATUS);
        if (s & ATA_SR_ERR) return -1;
        if ((s & mask) == want) return 0;
    }
    return -2;
}

static int ata_wait_ready(void) {
    return ata_wait(ATA_SR_BSY, 0, 4000000u);
}

int ata_init(void) {
    /* Float check: an absent channel reads 0xFF on every port, because nothing
     * is driving the bus. Issuing IDENTIFY to that would wait forever. */
    if (inb(ATA_STATUS) == 0xFF) {
        serial_write("[disk] no ATA controller on the primary channel\n");
        return 0;
    }

    outb(ATA_DRIVE, 0xA0);              /* master, CHS/LBA bits irrelevant here */
    ata_delay();
    outb(ATA_SECCOUNT, 0);
    outb(ATA_LBA_LO, 0);
    outb(ATA_LBA_MID, 0);
    outb(ATA_LBA_HI, 0);
    outb(ATA_COMMAND, ATA_CMD_IDENTIFY);
    ata_delay();

    if (inb(ATA_STATUS) == 0) {
        serial_write("[disk] no drive attached to primary master\n");
        return 0;
    }
    if (ata_wait(ATA_SR_BSY, 0, 4000000u) < 0) {
        serial_write("[disk] IDENTIFY did not complete\n");
        return 0;
    }
    /* A non-zero LBA_MID/HI after IDENTIFY means the device is ATAPI (our own
     * boot CD is one) and does not speak the ATA read command set. */
    if (inb(ATA_LBA_MID) || inb(ATA_LBA_HI)) {
        serial_write("[disk] primary master is ATAPI, not a hard disk\n");
        return 0;
    }
    if (ata_wait(ATA_SR_DRQ, ATA_SR_DRQ, 4000000u) < 0) {
        serial_write("[disk] IDENTIFY returned no data\n");
        return 0;
    }

    uint16_t id[256];
    for (int i = 0; i < 256; i++) id[i] = inw(ATA_DATA);

    /* Words 60-61 hold the 28-bit LBA sector count. */
    total_sectors = ((uint32_t)id[61] << 16) | id[60];

    /* Model string is words 27..46, byte-swapped within each word. */
    for (int i = 0; i < 20; i++) {
        model_str[i * 2] = (char)(id[27 + i] >> 8);
        model_str[i * 2 + 1] = (char)(id[27 + i] & 0xFF);
    }
    model_str[40] = 0;
    for (int i = 39; i >= 0 && (model_str[i] == ' ' || !model_str[i]); i--)
        model_str[i] = 0;

    present = 1;
    serial_write("[disk] ATA controller online: ");
    serial_write(model_str);
    serial_write(", ");
    serial_dec(total_sectors / 2048);
    serial_write(" MiB\n");
    return 1;
}

uint32_t ata_read(uint32_t lba, uint8_t count, void *buf) {
    if (!present || !count) return 0;
    uint16_t *out = (uint16_t *)buf;

    if (ata_wait_ready() < 0) return 0;
    /* 0xE0 selects master in LBA mode; the low nibble carries LBA bits 24-27,
     * which is what makes this 28-bit and caps addressing at 128 GiB. */
    outb(ATA_DRIVE, 0xE0 | ((lba >> 24) & 0x0F));
    ata_delay();
    outb(ATA_SECCOUNT, count);
    outb(ATA_LBA_LO, (uint8_t)(lba & 0xFF));
    outb(ATA_LBA_MID, (uint8_t)((lba >> 8) & 0xFF));
    outb(ATA_LBA_HI, (uint8_t)((lba >> 16) & 0xFF));
    outb(ATA_COMMAND, ATA_CMD_READ);

    for (uint32_t s = 0; s < count; s++) {
        /* DRQ is raised once PER SECTOR, so this must be inside the loop --
         * checking once and then reading count*256 words works on fast
         * emulators and corrupts data on real hardware. */
        if (ata_wait(ATA_SR_BSY, 0, 4000000u) < 0) return s;
        if (ata_wait(ATA_SR_DRQ, ATA_SR_DRQ, 4000000u) < 0) return s;
        for (int i = 0; i < 256; i++) *out++ = inw(ATA_DATA);
        ata_delay();
    }
    return count;
}

uint32_t ata_write(uint32_t lba, uint8_t count, const void *buf) {
    if (!present || !count) return 0;
    const uint16_t *in = (const uint16_t *)buf;

    if (ata_wait_ready() < 0) return 0;
    outb(ATA_DRIVE, 0xE0 | ((lba >> 24) & 0x0F));
    ata_delay();
    outb(ATA_SECCOUNT, count);
    outb(ATA_LBA_LO, (uint8_t)(lba & 0xFF));
    outb(ATA_LBA_MID, (uint8_t)((lba >> 8) & 0xFF));
    outb(ATA_LBA_HI, (uint8_t)((lba >> 16) & 0xFF));
    outb(ATA_COMMAND, ATA_CMD_WRITE);

    for (uint32_t s = 0; s < count; s++) {
        if (ata_wait(ATA_SR_BSY, 0, 4000000u) < 0) return s;
        if (ata_wait(ATA_SR_DRQ, ATA_SR_DRQ, 4000000u) < 0) return s;
        for (int i = 0; i < 256; i++) outw(ATA_DATA, *in++);
        ata_delay();
    }

    /* Without FLUSH CACHE the drive may acknowledge writes it has only
     * buffered, and a reset loses them. */
    outb(ATA_COMMAND, ATA_CMD_FLUSH);
    ata_wait_ready();
    return count;
}

uint32_t ata_sectors(void) { return total_sectors; }
const char *ata_model(void) { return present ? model_str : "none"; }
int ata_present(void) { return present; }
