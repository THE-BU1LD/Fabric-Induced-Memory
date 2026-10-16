# Memory recovery and ordered execution — 13 October 2026

**Blocked: administrator cache purge unavailable; no safe owned worker to stop.**

The process inventory at 2026-10-13T15:41:36.846569+00:00 reports approximately
3.87 GiB resident memory
across Codex/ChatGPT processes and 2.97
GiB across Chrome. The largest individual consumers are active Codex and Chrome
renderers. RSS can double-count shared pages and excludes swapped/compressed
allocations; these are diagnostic estimates, not exclusive physical totals.
Earlier vm_stat showed approximately 5.37 GiB occupied by the compressor and
4.29 GiB swap use. The host has 16 GiB physical RAM. No FIM numerical worker or
study-owned idle workload was identified for termination. Active unrelated apps,
research daemons, browser tabs and indexing processes were left untouched.

## Safe reclaim attempt

The local macOS purge manual states that purge flushes disk buffers and does not
free anonymous application allocations. It can temporarily slow subsequent file
reads but leaves apps running. The direct command failed with Operation not
permitted; non-interactive sudo failed because a password is required. Both
exact failure receipts are preserved. No cache purge succeeded, no process was
killed, and no files were deleted. No permission failure was worked around.
The user was asked to authenticate `sudo /usr/sbin/purge` in their own Terminal;
no password was requested in chat. A purge is not guaranteed to free enough RAM
or satisfy stable pageouts; normal guards must be rechecked afterward.

The final observed admission was 108.094 / 144.406 MiB free plus speculative
against the unchanged 512 MiB floor, with pageouts increasing by
56 and swap at 4375.31 MiB.
The system memory_pressure percentage is not a substitute for the project’s
specific free-plus-speculative and stable-pageout conditions.

## Execution and preservation

V2 reproduction, V3 numerical tests, campaign and raw verification were not
launched. All 440 protected files remain unchanged; development-2 source freeze
matches; the one-campaign budget remains unconsumed. Previous failed attempts
and the two new cache-recovery failures remain on disk. No guard was weakened.

Resume in the user-requested order after actual resource admission: v2 isolated
reproduction, then v3 numerical tests, then the single frozen campaign, then raw
verification. Use fresh outputs. If cache reclamation is insufficient, freeing
application memory requires the user to identify disposable inactive sessions;
current evidence does not authorize terminating unrelated work.
