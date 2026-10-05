#!/usr/bin/env python3
"""Write the small GGUF fixtures used by test/run.sh.

valid.gguf exercises every metadata value type, a nested array, a non-default
alignment and three tensors (f32, q8_0 and an unknown ggml type id). Every
other file is one deliberate defect that the reader must reject cleanly.
"""
import os
import struct
import sys

out = sys.argv[1] if len(sys.argv) > 1 else "build/fixtures"
os.makedirs(out, exist_ok=True)

U8, I8, U16, I16, U32, I32, F32, BOOL, STR, ARR, U64, I64, F64 = range(13)
SCALAR = {U8: "<B", I8: "<b", U16: "<H", I16: "<h", U32: "<I", I32: "<i",
          F32: "<f", BOOL: "<?", U64: "<Q", I64: "<q", F64: "<d"}


def s(text):
    data = text if isinstance(text, bytes) else text.encode()
    return struct.pack("<Q", len(data)) + data


def value(kind, v):
    if kind == STR:
        return s(v)
    if kind == ARR:
        item, items = v
        return struct.pack("<IQ", item, len(items)) + b"".join(value(item, x) for x in items)
    return struct.pack(SCALAR[kind], v)


def gguf(kvs, tensors, payload=b"", version=3, alignment=32, magic=b"GGUF",
         tensor_count=None, kv_count=None):
    head = magic + struct.pack("<IQQ", version,
                               len(tensors) if tensor_count is None else tensor_count,
                               len(kvs) if kv_count is None else kv_count)
    body = b"".join(s(k) + struct.pack("<I", kind) + value(kind, v) for k, kind, v in kvs)
    for name, dims, ggml_type, offset in tensors:
        body += s(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims)
        body += struct.pack("<IQ", ggml_type, offset)
    blob = head + body
    blob += b"\0" * ((alignment - len(blob) % alignment) % alignment)
    return blob + payload


def write(name, data):
    with open(os.path.join(out, name), "wb") as f:
        f.write(data)


ALIGN = 64
kvs = [
    ("general.architecture", STR, "fixture"),
    ("general.alignment", U32, ALIGN),
    ("general.name", STR, "Ünïcode ✓ model"),
    ("t.u8", U8, 255), ("t.i8", I8, -128), ("t.u16", U16, 65535), ("t.i16", I16, -32768),
    ("t.u32", U32, 4294967295), ("t.i32", I32, -2147483648),
    ("t.f32", F32, 0.1), ("t.bool", BOOL, True),
    ("t.u64", U64, 18446744073709551615), ("t.i64", I64, -9223372036854775808),
    ("t.f64", F64, 2.5e-300),
    ("t.tokens", ARR, (STR, ["<s>", "▁hello", "", "✓"])),
    ("t.scores", ARR, (F32, [0.0, -1.5, 3.25])),
    ("t.empty", ARR, (I32, [])),
    ("t.nested", ARR, (ARR, [(U8, [1, 2]), (U8, [3])])),
]
# f32 [4, 2] = 32 bytes; q8_0 [32] = one 34-byte block; type 99 is unknown.
f32 = struct.pack("<8f", *range(8))
q8 = b"\x00\x3c" + bytes(range(32))
payload = f32 + b"\0" * (ALIGN - 32) + q8 + b"\0" * (ALIGN - 34) + b"\0" * ALIGN
tensors = [("w.f32", [4, 2], 0, 0), ("w.q8_0", [32], 8, ALIGN), ("w.future", [7], 99, 2 * ALIGN)]
write("valid.gguf", gguf(kvs, tensors, payload, alignment=ALIGN))
write("empty.gguf", gguf([], []))

ok_kv = [("general.architecture", STR, "fixture")]
one_f32 = [("w", [2], 0, 0)]
pay = struct.pack("<2f", 1, 2)
write("bad_magic.gguf", gguf(ok_kv, [], magic=b"GGML"))
write("version1.gguf", gguf(ok_kv, [], version=1))
write("truncated.gguf", gguf(ok_kv, one_f32, pay)[:-4])
write("huge_counts.gguf", gguf(ok_kv, [], kv_count=1 << 60))
write("string_overrun.gguf", gguf([], [], kv_count=1)[:24] + s("k") + struct.pack("<IQ", STR, 100) + b"abc")
write("duplicate_key.gguf", gguf(ok_kv + ok_kv, []))
write("duplicate_tensor.gguf", gguf(ok_kv, one_f32 + [("w", [2], 0, 32)], pay + b"\0" * 24 + pay))
write("misaligned.gguf", gguf(ok_kv, [("w", [2], 0, 4)], pay + pay))
write("payload_overrun.gguf", gguf(ok_kv, [("w", [64], 0, 0)], pay))
write("too_many_dims.gguf", gguf(ok_kv, [("w", [1, 1, 1, 1, 1], 0, 0)], pay))
write("bad_alignment.gguf", gguf([("general.alignment", U32, 3)], []))
write("bad_utf8.gguf", gguf([(b"bad\xffkey", U8, 1)], []))
write("partial_block.gguf", gguf(ok_kv, [("w", [16], 8, 0)], b"\0" * 34))
write("unknown_value_type.gguf", gguf(ok_kv, [])[:24] + s("k") + struct.pack("<I", 77) + b"\0" * 8)
# More rows than one DuckDB vector (2048), so scans must resume across chunks.
many_kv = [("general.alignment", U32, 4)] + [(f"k.{i:05d}", U32, i) for i in range(3000)]
many_t = [(f"t.{i:05d}", [1, 1, 1][: 1 + i % 3], 0, 4 * i) for i in range(3000)]
write("many.gguf", gguf(many_kv, many_t, struct.pack("<3000f", *range(3000)), alignment=4))
print(f"wrote fixtures to {out}")
