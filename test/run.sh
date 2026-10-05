#!/bin/sh
# Runs every test/sql/*.sql script (assertions raise error() on failure) and
# every expected-error case in test/errors.tsv against one extension artifact.
set -u
extension=$(cd "$(dirname "$1")" && pwd)/$(basename "$1")
duckdb=${DUCKDB:-duckdb}
cd "$(dirname "$0")/.." || exit 1
failed=0

session() {
    { printf "LOAD '%s';\n" "$extension"; cat; } | "$duckdb" -unsigned -bail -noheader -list
}

for script in test/sql/*.sql; do
    if output=$(session < "$script" 2>&1); then
        echo "ok   $script"
    else
        echo "FAIL $script"
        printf '%s\n' "$output" | tail -5
        failed=1
    fi
done

cases=0
while IFS='	' read -r sql expected; do
    case $sql in ''|'#'*) continue ;; esac
    cases=$((cases + 1))
    if output=$(printf '%s\n' "$sql" | session 2>&1); then
        echo "FAIL expected an error: $sql"
        failed=1
    elif ! printf '%s' "$output" | grep -qF -- "$expected"; then
        echo "FAIL wrong error for: $sql"
        printf '  expected: %s\n  got: %s\n' "$expected" "$(printf '%s' "$output" | head -3)"
        failed=1
    fi
done < test/errors.tsv
[ $failed -eq 0 ] && echo "ok   test/errors.tsv ($cases cases)"
exit $failed
