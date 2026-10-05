#!/usr/bin/env python3
"""Mutation smoke-fuzz for the GGUF reader.

Usage: fuzz_gguf.py DUCKDB EXTENSION SEED_FILE.gguf ITERATIONS

Each iteration flips bytes, overwrites 8-byte fields with extreme values, or
truncates a seed file, then queries all four GGUF views in a fresh DuckDB
process. SQL errors are expected; a crash, sanitizer report or signal fails.
"""
import os
import random
import subprocess
import sys
import tempfile

duckdb, extension, seed_path, iterations = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
seed = open(seed_path, "rb").read()
rng = random.Random(1234)
extremes = [0, 1, 0x7F, 0xFF, 0xFFFF, 0xFFFFFFFF, 1 << 31, 1 << 63, (1 << 64) - 1, 2048, 4096]
work = tempfile.mkdtemp()
sql = "LOAD '{ext}';\n" + "".join(
    f"SELECT count(*) FROM {fn};\n" for fn in (
        "gguf_header('{p}')", "gguf_metadata('{p}')", "gguf_tensors('{p}')",
        "gguf_metadata_array('{p}', 't.tokens')", "gguf_metadata_array('{p}', 't.nested')"))
for i in range(iterations):
    data = bytearray(seed)
    for _ in range(rng.randint(1, 4)):
        roll = rng.random()
        if roll < 0.45:
            data[rng.randrange(len(data))] = rng.randrange(256)
        elif roll < 0.85:
            at = rng.randrange(max(1, len(data) - 8))
            data[at:at + 8] = rng.choice(extremes).to_bytes(8, "little")
        else:
            del data[rng.randrange(len(data)):]
            if not data:
                data = bytearray(b"G")
    path = os.path.join(work, f"case{i}.gguf")
    open(path, "wb").write(data)
    run = subprocess.run([duckdb, "-unsigned"], input=sql.format(ext=extension, p=path),
                         capture_output=True, text=True, errors="replace", timeout=60)
    if run.returncode < 0 or "Sanitizer" in run.stderr or "runtime error" in run.stderr:
        print(f"CRASH on {path} (exit {run.returncode})\n{run.stderr[-2000:]}")
        sys.exit(1)
    os.unlink(path)
print(f"ok {iterations} mutations of {seed_path}: no crash or sanitizer report")
