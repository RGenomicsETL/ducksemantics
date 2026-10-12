#!/bin/sh
# Stage the exact released host used by native distribution tests.
set -eu
platform=${1:-linux_amd64}
case "$platform" in
    linux_amd64)
        asset=duckdb_cli-linux-amd64.zip
        digest=08c0ca117111fcede14239d0093792352befdc174218c344d232c13279643d05 ;;
    linux_arm64)
        asset=duckdb_cli-linux-arm64.zip
        digest=02163197027a42149147364d31fa67cac82108517a4be43304a1cc226eaef07a ;;
    *) printf 'Unsupported distribution test platform: %s\n' "$platform" >&2; exit 1 ;;
esac
mkdir -p configure/cli
archive=configure/cli.zip
if [ ! -f "$archive" ] || ! printf '%s  %s\n' "$digest" "$archive" | sha256sum -c - >/dev/null 2>&1; then
    curl -fL --retry 3 "https://github.com/duckdb/duckdb/releases/download/v1.5.5/$asset" -o "$archive"
fi
printf '%s  %s\n' "$digest" "$archive" | sha256sum -c -
(cd configure/cli && cmake -E tar xf ../cli.zip)
chmod +x configure/cli/duckdb
configure/cli/duckdb -c 'SELECT version(); PRAGMA platform;'
