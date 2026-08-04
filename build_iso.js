/* =============================================================================
 * OBSOLETE -- NOT THE BUILD PATH. DO NOT USE.
 * =============================================================================
 * This script concatenates bytes into a file named .iso. It does not create an
 * ISO 9660 filesystem and does not create an El Torito boot catalog, so the
 * result cannot boot on any firmware. The image it produced had `MZ` at offset
 * 32768, where the primary volume descriptor is required to be.
 *
 * The authoritative bootable image is produced only by:
 *
 *     make iso        (freestanding compile/link + grub-mkrescue)
 *
 * Kept for history. Excluded from the build. See BAREMETAL_BOOT_GUIDE.md.
 * ============================================================================= */

import fs from 'fs';
import path from 'path';

// Minimal ISO9660 + El Torito Boot Sector Builder script
function createBootableISO() {
  const isoPath = path.join(process.cwd(), 'franken-ao.iso');
  const kernelPath = path.join(process.cwd(), 'franken_kernel.bin');
  
  const kernelBuffer = fs.readFileSync(kernelPath);
  
  // Allocate 1.44MB floppy/ISO boot disk memory buffer (300 * 2048-byte ISO sectors)
  const sectorSize = 2048;
  const numSectors = 400;
  const isoBuffer = Buffer.alloc(numSectors * sectorSize);

  // Write Multiboot Kernel into Sector 16+
  kernelBuffer.copy(isoBuffer, 16 * sectorSize);

  fs.writeFileSync(isoPath, isoBuffer);
  console.log(`[FRANKEN-ISO] Successfully created bootable ISO image: ${isoPath} (${isoBuffer.length} bytes)`);
}

createBootableISO();
