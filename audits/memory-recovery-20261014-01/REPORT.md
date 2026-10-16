# Memory recovery and guarded retry — 14 October 2026

**BLOCKED_RESOURCE. No numerical worker launched.**

Host inventory shows 16 GiB physical RAM. Codex/ChatGPT summed RSS was 5584.55 MiB (5.45 GiB), Chrome 2913.77 MiB (2.85 GiB). The largest individual process was a Codex renderer at 849.67 MiB. The kernel reported 480363 compressor pages at 16384 bytes each (7.33 GiB). RSS includes shared pages and excludes compressed/swapped allocations; these categories overlap and are not additive physical totals. Initial swap use was 791.50 MiB; replay-time swap was 1581.25 MiB.

No FIM runner or local study lock was found. No process was classified as safely disposable. Unrelated applications, browser tabs, other research and system services were left running. No files were deleted: deleting disk caches would not directly release anonymous application memory.

The app-preserving disk-buffer purge was attempted directly and failed with `Operation not permitted`. Noninteractive `sudo -n /usr/sbin/purge` failed with `a password is required`. No cache reclaim succeeded. These were operating-system permission failures, not an automatic approval-review rejection. Administrator authentication must be performed by the user; no password was requested or collected.

The v2 replay was retried in a new directory. Its first admission passed at 531.67/531.73 MiB, allowing only the isolated source copy. Its mandatory pre-worker recheck failed at 516.28/463.39 MiB, below the unchanged 512 MiB floor. Pageouts and swap were stable within each pair. The retained receipt has an empty steps list: no model import, runtime test or experiment launched. The copied workspace and failed receipt remain intact.

V3 numerical tests, campaign and raw verification remain withheld behind v2. All 440 protected predecessor files remain unchanged; development-2 matches; the one-campaign budget is unconsumed. No guard or frozen result changed.

## Next step

Safe automatic reclaim is exhausted under the instruction not to interrupt unrelated work. The user can authenticate `sudo /usr/sbin/purge` in their own Terminal for disk-buffer reclamation, or identify a disposable inactive application/session to close. Cache purge may not release enough application memory. After any recovery, use a fresh output directory and require all existing admissions to pass before v2, then v3.

Receipts: INVENTORY.json, OWNERSHIP.json, CACHE_RECLAIM.json, SUMMARY.json. Replay receipt: `successor/itfm_v2/publication/itfm-v2-paper-20261011/replays/attempt-20261014-memory-01/REPLAY_RECEIPT.json`.
