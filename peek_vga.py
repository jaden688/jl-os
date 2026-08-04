"""Dump the running machine's VGA text buffer and decode it.

Connects to the QEMU monitor, saves physical memory at 0xB8000 (80x25 cells of
character + attribute byte), and renders it back to text. This reads what the
kernel actually wrote to the hardware, not a screenshot of pixels.

    python peek_vga.py
"""

from __future__ import annotations

import pathlib
import socket
import sys
import time

HOST, PORT = "127.0.0.1", 55555
DUMP = pathlib.Path(__file__).resolve().parent / "vgadump.bin"
COLS, ROWS = 80, 25

# CP437 glyphs the HUD uses for its panel borders, mapped to ASCII lookalikes.
BOX = {0xC4: "-", 0xB3: "|", 0xDA: "+", 0xBF: "+", 0xC0: "+", 0xD9: "+",
       0xDB: "#", 0xB0: "."}


def monitor(sock, cmd: str, wait: float = 0.6) -> str:
    sock.sendall((cmd + "\n").encode())
    time.sleep(wait)
    out = b""
    sock.setblocking(False)
    try:
        while True:
            chunk = sock.recv(65536)
            if not chunk:
                break
            out += chunk
    except BlockingIOError:
        pass
    sock.setblocking(True)
    return out.decode(errors="replace")


def main() -> int:
    if DUMP.exists():
        DUMP.unlink()

    # Give the guest time to boot and the HUD time to draw a frame.
    deadline = time.time() + 20
    sock = None
    while time.time() < deadline:
        try:
            sock = socket.create_connection((HOST, PORT), timeout=2)
            break
        except OSError:
            time.sleep(0.5)
    if sock is None:
        print("could not reach the QEMU monitor")
        return 1

    time.sleep(3.0)          # let the HUD render
    monitor(sock, "")        # clear the banner
    monitor(sock, f"memsave 0xb8000 {COLS * ROWS * 2} {DUMP.as_posix()}", 1.5)
    monitor(sock, "quit", 0.3)
    sock.close()

    if not DUMP.exists():
        print("memsave produced no file")
        return 1

    data = DUMP.read_bytes()
    print(f"VGA text buffer, {len(data)} bytes read from physical 0xB8000\n")
    print("+" + "-" * COLS + "+")
    for r in range(ROWS):
        line = ""
        for c in range(COLS):
            off = (r * COLS + c) * 2
            if off + 1 >= len(data):
                break
            ch = data[off]
            line += BOX.get(ch, chr(ch) if 32 <= ch < 127 else " ")
        print("|" + line.ljust(COLS) + "|")
    print("+" + "-" * COLS + "+")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
