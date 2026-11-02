#!/bin/sh
# Runs tokenizer, SQL, and expected-error cases against one built artifact.
# Each invocation owns its mutable fixtures; the build workspace is read-only.
set -u
extension=$(cd "$(dirname "$1")" && pwd)/$(basename "$1")
root=$(cd "$(dirname "$0")/.." && pwd) || exit 1
duckdb=${DUCKDB:-duckdb}
case $duckdb in
    /*) ;;
    */*) duckdb=$(pwd)/$duckdb ;;
esac
tokenizer=${TOKENIZER_TEST:-build/tokenizer_test}
case $tokenizer in
    /*) ;;
    *) tokenizer=$root/$tokenizer ;;
esac
work=$(mktemp -d "${TMPDIR:-/tmp}/ducksemantics-test.XXXXXX") || exit 1
trap 'rm -rf "$work"' 0
trap 'exit 130' INT
trap 'exit 143' TERM
mkdir "$work/build" &&
    cp -R "$root/build/fixtures" "$work/build/fixtures" &&
    cp -R "$root/test" "$work/test" || exit 1
cd "$work" || exit 1
failed=0

if "$tokenizer" build/fixtures; then
    echo "ok   test/tokenizer_test.c"
else
    failed=1
fi

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
