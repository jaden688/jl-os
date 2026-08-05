"""Freeze the training corpus to a single committed file.

Every accuracy number this session was measured against a MOVING target:
`load_text()` globbed the project's own *.md/*.c/*.h, and those files changed as
work progressed. Baselines drifted 0.1172 -> 0.1304 -> 0.1191 -> 0.1194 across
runs, so no two results were strictly comparable.

This walks the source repositories once, writes a single corpus file, and
records its SHA-256. From here every training run reads that file, so a change
in accuracy means a change in the model rather than a change in the data.

    python freeze_corpus.py
"""

from __future__ import annotations

import hashlib
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "corpus_frozen.txt"

ROOTS = [
    pathlib.Path("C:/Users/J_lin/Desktop/franken-ao"),
    pathlib.Path("C:/Users/J_lin/Desktop/jl-wave-fabric"),
    pathlib.Path("C:/Users/J_lin/Desktop/JL_Engine-SB.Omni"),
    pathlib.Path("D:/pitreader_omega_milestone"),
]
SKIP = (".git", ".venv", "node_modules", "__pycache__", "worktrees",
        "site-packages", "dist", "corpus_frozen.txt")
EXTS = (".md", ".c", ".h", ".py", ".jl", ".txt", ".s")


def main() -> int:
    chunks, files = [], 0
    for root in ROOTS:
        if not root.exists():
            print(f"  skip (absent): {root}")
            continue
        # Sorted for determinism: the same tree must always produce the same
        # corpus, or the freeze is not actually a freeze.
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix not in EXTS:
                continue
            if any(s in path.as_posix() for s in SKIP):
                continue
            try:
                chunks.append(path.read_text(encoding="utf-8", errors="ignore"))
            except OSError:
                continue
            files += 1

    text = "\n".join(chunks)
    OUT.write_text(text, encoding="utf-8")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()

    print(f"\n  files    {files:,}")
    print(f"  chars    {len(text):,}")
    print(f"  sha256   {digest}")
    print(f"  wrote    {OUT.name}")
    print(f"\n  every training run must now read this file and nothing else")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
