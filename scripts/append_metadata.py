#!/usr/bin/env python3
"""Append DuckDB extension metadata to a shared library.

DuckDB reads eight 32-byte fields backwards from the end of the file, after a
fixed signature marker, followed by a 256-byte (unsigned) signature block.
"""
import argparse
import os
import shutil

parser = argparse.ArgumentParser()
for name in ("library", "output", "platform", "duckdb-version", "extension-version"):
    parser.add_argument("--" + name, required=True)
parser.add_argument("--abi-type", default="C_STRUCT")
args = parser.parse_args()


def field(text):
    data = text.encode()
    if len(data) > 32:
        raise SystemExit(f"metadata field exceeds 32 bytes: {text}")
    return data.ljust(32, b"\0")


temporary = args.output + ".tmp"
shutil.copyfile(args.library, temporary)
with open(temporary, "ab") as out:
    out.write(bytes([0, 147, 4, 16]) + b"duckdb_signature" + bytes([128, 4]))
    for text in ("", "", "", args.abi_type, args.extension_version,
                 args.duckdb_version, args.platform, "4"):
        out.write(field(text))
    out.write(bytes(256))
os.replace(temporary, args.output)
