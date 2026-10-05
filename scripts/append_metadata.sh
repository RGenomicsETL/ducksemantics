#!/bin/sh
# append_metadata.sh LIBRARY OUTPUT PLATFORM DUCKDB_VERSION EXTENSION_VERSION [ABI_TYPE]
#
# DuckDB reads eight 32-byte fields backwards from the end of an extension,
# after a fixed signature marker, followed by a 256-byte (unsigned) signature.
set -eu
[ $# -ge 5 ] || { echo "usage: $0 LIBRARY OUTPUT PLATFORM DUCKDB_VERSION EXTENSION_VERSION [ABI_TYPE]" >&2; exit 2; }
library=$1 output=$2 platform=$3 duckdb_version=$4 extension_version=$5 abi=${6:-C_STRUCT}

field() {
    [ ${#1} -le 32 ] || { echo "metadata field exceeds 32 bytes: $1" >&2; exit 1; }
    printf '%s' "$1"
    head -c $((32 - ${#1})) /dev/zero
}

temporary=$output.tmp
cp "$library" "$temporary"
{
    printf '\000\223\004\020duckdb_signature\200\004'
    field ""; field ""; field ""
    field "$abi"; field "$extension_version"; field "$duckdb_version"; field "$platform"; field 4
    head -c 256 /dev/zero
} >> "$temporary"
mv "$temporary" "$output"
