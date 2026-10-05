#!/usr/bin/env python3
"""Differential audit: DuckSemantics GGUF views against gguf-py's GGUFReader.

Usage: audit_gguf.py EXTENSION FILE.gguf [...]

For every file, compares header geometry, every metadata key/type/value, every
element of every array, and every tensor's name, ggml type, dims, absolute
offset, element count and byte size. Exits non-zero on the first mismatch.
"""
import sys

import duckdb
import numpy as np
from gguf import GGUFReader, GGUFValueType

extension, paths = sys.argv[1], sys.argv[2:]
con = duckdb.connect(config={"allow_unsigned_extensions": "true"})
con.execute(f"LOAD '{extension}'")

TYPE_NAMES = {
    GGUFValueType.UINT8: "uint8", GGUFValueType.INT8: "int8",
    GGUFValueType.UINT16: "uint16", GGUFValueType.INT16: "int16",
    GGUFValueType.UINT32: "uint32", GGUFValueType.INT32: "int32",
    GGUFValueType.FLOAT32: "float32", GGUFValueType.BOOL: "bool",
    GGUFValueType.STRING: "string", GGUFValueType.ARRAY: "array",
    GGUFValueType.UINT64: "uint64", GGUFValueType.INT64: "int64",
    GGUFValueType.FLOAT64: "float64",
}


def text(value_type, value):
    if value_type == GGUFValueType.FLOAT32:
        return "%.9g" % float(np.float32(value))
    if value_type == GGUFValueType.FLOAT64:
        return "%.17g" % float(value)
    if value_type == GGUFValueType.BOOL:
        return "true" if value else "false"
    return str(value)


def check(label, ours, theirs):
    if ours != theirs:
        raise SystemExit(f"MISMATCH {label}: ours={ours!r} reference={theirs!r}")


for path in paths:
    ref = GGUFReader(path)
    q = lambda sql: con.execute(sql, [path]).fetchall()
    header = q("SELECT tensor_count, metadata_count, alignment, data_offset FROM gguf_header(?)")[0]
    # gguf-py exposes GGUF.version/tensor_count/kv_count as pseudo-fields.
    pseudo = {"GGUF.version", "GGUF.tensor_count", "GGUF.kv_count"}
    fields = {k: v for k, v in ref.fields.items() if k not in pseudo}
    check(f"{path} tensor_count", header[0], len(ref.tensors))
    check(f"{path} metadata_count", header[1], len(fields))
    check(f"{path} alignment", header[2], ref.alignment)
    check(f"{path} data_offset", header[3], ref.data_offset)

    rows = q("SELECT key, value_type, array_type, array_length, value FROM gguf_metadata(?)")
    check(f"{path} metadata keys", [r[0] for r in rows], list(fields))
    elements = 0
    for key, value_type, array_type, array_length, value in rows:
        field = fields[key]
        check(f"{path} {key} type", value_type, TYPE_NAMES[field.types[0]])
        if field.types[0] == GGUFValueType.ARRAY:
            item = field.types[1] if len(field.types) > 1 else None
            contents = field.contents()
            check(f"{path} {key} array_length", array_length, len(contents))
            if item is not None:
                check(f"{path} {key} array_type", array_type, TYPE_NAMES[item])
            ours = [r[0] for r in q(
                f"SELECT value FROM gguf_metadata_array(?, '{key}') ORDER BY array_index")]
            theirs = [None if item == GGUFValueType.ARRAY else text(item, v) for v in contents]
            check(f"{path} {key} elements", ours, theirs)
            elements += len(ours)
        else:
            check(f"{path} {key} value", value, text(field.types[0], field.contents()))

    tensors = q("SELECT name, ggml_type_id, dims, n_elements, file_offset, n_bytes "
                "FROM gguf_tensors(?) ORDER BY tensor_index")
    for (name, type_id, dims, n, offset, n_bytes), t in zip(tensors, ref.tensors):
        check(f"{path} tensor name", name, t.name)
        check(f"{path} {name} type", type_id, int(t.tensor_type))
        check(f"{path} {name} dims", dims, [int(d) for d in t.shape])
        check(f"{path} {name} n_elements", n, int(t.n_elements))
        check(f"{path} {name} offset", offset, int(t.data_offset))
        check(f"{path} {name} n_bytes", n_bytes, int(t.n_bytes))
    print(f"ok {path}: {len(rows)} keys, {elements} array elements, {len(tensors)} tensors")
