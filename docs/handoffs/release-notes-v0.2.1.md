# Release Notes - v0.2.1

> Date: 2026-08-31. Patch release over v0.2.0 fixing Windows defects found by the
> new cross-platform CI (Linux, Windows and macOS arm64 are all green).

## Fixes

- **Windows kb creation** (access denied when finalizing the kb directory): SQLite
  connections are now closed explicitly instead of relying on garbage collection,
  and the final directory move retries on Windows permission errors.
- **Windows kb deletion**: collection drop now retries while Qdrant releases file
  handles asynchronously (transient "Access is denied" server errors).
- **Job system shutdown** drains in-flight jobs instead of abandoning them
  (`JobManager.close`), preventing interrupted-job artifacts from leaking across
  restarts.
- **Deterministic document ordering**: document search and browse queries now
  tie-break on `doc_id`, making pagination stable across platforms.

## Verified baseline

`pixi run lint && pixi run typecheck && pixi run test` are green; the GitHub Actions
matrix passes on `ubuntu-latest`, `windows-latest` and `macos-latest` (arm64).

## Upgrade from v0.2.0

No action needed. Note that v0.2.0 on Windows is broken for kb creation; upgrading
to v0.2.1 is strongly recommended on Windows.
