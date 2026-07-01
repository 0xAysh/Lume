#!/usr/bin/env bash
set -euo pipefail

VERSION="0.1.10-alpha.4"
TMPDIR="$(mktemp -d)"
LOG="$TMPDIR/build.log"

cleanup() {
  rm -rf "$TMPDIR"
}
trap cleanup EXIT

cat >"$TMPDIR/Cargo.toml" <<EOF
[package]
name = "lume-sqlite-vec-m0-probe"
version = "0.0.0"
edition = "2021"

[dependencies]
rusqlite = "0.40.1"
sqlite-vec = "=$VERSION"
EOF

mkdir -p "$TMPDIR/src"
cat >"$TMPDIR/src/lib.rs" <<'EOF'
#[cfg(test)]
mod tests {
    use rusqlite::{ffi::sqlite3_auto_extension, Connection};

    #[test]
    fn sqlite_vec_loads() {
        unsafe {
            sqlite3_auto_extension(Some(std::mem::transmute(
                sqlite_vec::sqlite3_vec_init as *const (),
            )));
        }
        let conn = Connection::open_in_memory().unwrap();
        let version: String = conn.query_row("select vec_version()", [], |row| row.get(0)).unwrap();
        assert_eq!(version, "v0.1.10-alpha.4");
    }
}
EOF

if cargo test --manifest-path "$TMPDIR/Cargo.toml" >"$LOG" 2>&1; then
  echo "UNEXPECTED: sqlite-vec $VERSION compiled successfully."
  exit 1
fi

if ! grep -q "sqlite-vec-diskann.c.*file not found" "$LOG"; then
  echo "UNEXPECTED: sqlite-vec $VERSION failed for a different reason."
  cat "$LOG"
  exit 1
fi

SOURCE="$(
  find "${CARGO_HOME:-$HOME/.cargo}/registry/src" \
    -path "*/sqlite-vec-$VERSION/sqlite-vec.c" \
    -print \
    -quit
)"

if [[ -z "$SOURCE" ]]; then
  echo "UNEXPECTED: could not find downloaded sqlite-vec $VERSION source."
  exit 1
fi

if grep -Eq "SQLITE_VEC_ELEMENT_TYPE_FLOAT16|float16" "$SOURCE"; then
  echo "UNEXPECTED: sqlite-vec $VERSION source appears to mention float16."
  exit 1
fi

echo "M0 sqlite-vec probe:"
echo "- pinned crate: sqlite-vec $VERSION"
echo "- Rust crate build: FAILS; packaged source includes missing sqlite-vec-diskann.c"
echo "- fp16 support: NOT PRESENT in the pinned C source"
