"""Drive JL/OS: type a prompt over emulated PS/2, then read back video memory.

`sendkey` injects scancodes into the emulated keyboard controller, so the
kernel's own kbd_poll() on port 0x60 is what receives them -- the same path a
physical keyboard would use. Pressing ENTER makes the HUD call run_nhit(),
which executes the trained INT4 ring convolution and times it with RDTSC.

Then memsave lifts the VGA text buffer straight out of physical memory, so the
result is what the kernel wrote to hardware rather than a screenshot.

    python drive_jlos.py
"""

from __future__ import annotations

import pathlib
import socket
import sys
import time

HOST, PORT = "127.0.0.1", 55555
HERE = pathlib.Path(__file__).resolve().parent
DUMP = HERE / "vgadump.bin"
COLS, ROWS = 80, 25
PROMPT = "nhit"

BOX = {0xC4: "-", 0xB3: "|", 0xDA: "+", 0xBF: "+", 0xC0: "+", 0xD9: "+",
       0xDB: "#", 0xB0: "."}


def send(sock, cmd: str, wait: float = 0.25) -> None:
    sock.sendall((cmd + "\n").encode())
    time.sleep(wait)
    sock.setblocking(False)
    try:
        while sock.recv(65536):
            pass
    except (BlockingIOError, OSError):
        pass
    sock.setblocking(True)


def render(data: bytes) -> None:
    print("+" + "-" * COLS + "+")
    for r in range(ROWS):
        line = ""
        for c in range(COLS):
            o = (r * COLS + c) * 2
            if o + 1 >= len(data):
                break
            ch = data[o]
            line += BOX.get(ch, chr(ch) if 32 <= ch < 127 else " ")
        print("|" + line.ljust(COLS) + "|")
    print("+" + "-" * COLS + "+")


def main() -> int:
    if DUMP.exists():
        DUMP.unlink()

    deadline = time.time() + 25
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

    time.sleep(3.0)                      # let the HUD reach its render loop
    send(sock, "")

    print(f"typing '{PROMPT}' into the guest over emulated PS/2 ...")
    for ch in PROMPT:
        send(sock, f"sendkey {ch}", 0.20)
    print("pressing ENTER -> HUD calls run_nhit() ...")
    send(sock, "sendkey ret", 1.2)
    print("pressing ENTER again -> second run, code already JIT-translated ...")
    send(sock, "sendkey ret", 1.2)

    send(sock, f"memsave 0xb8000 {COLS * ROWS * 2} {DUMP.as_posix()}", 1.5)
    try:
        send(sock, "quit", 0.2)
    except OSError:
        pass                             # QEMU closes the socket on quit
    sock.close()

    if not DUMP.exists():
        print("memsave produced no file")
        return 1
    data = DUMP.read_bytes()
    print(f"\nVGA text buffer, {len(data)} bytes from physical 0xB8000\n")
    render(data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
