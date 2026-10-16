# Durable evidence extension

The renewed implementation request authorizes the previously proposed three
additional repair rounds and three concept reviews. This extends the same
workstream to six cumulative repair rounds and eight concept reviews; it does
not reset any counter. The original 43-file package and all predecessor evidence
remain unchanged. See `extension_20261015/CONTRACT.json` for the recorded scope.

`DurableRecorder` implements the existing recorder interface, so a newly created
instance can be passed directly as `recorder=` to `SafeOnlineAdapter`. Prediction,
target and failure events are committed to disk before their calls return. The
scorer target still goes exclusively to `record_target`. The bridge with this
recorder has not been tensor-tested: the resource gate remains closed.

`DurableObservationLedger` implements the fresh observation ledger interface.
Create it with the frozen source hash, split and model-state hash, and pass it as
`ledger=`. Only forecast metadata and explicitly revealed coordinates enter its
journal. Saved base predictions and keys survive for audit. Forecast state hashes
must match the declared model hash. This does not authenticate observation times
or prove that a subsequent tensor-bank write succeeded.

Each journal record has a sequence number, predecessor hash, event and SHA-256.
Publication writes and fsyncs a temporary file, links it without replacement,
and fsyncs the directory. An exclusive advisory POSIX file lock prevents two
cooperating writers. The lock file stays in place to avoid inode-replacement
races. Directory and record symlinks are refused. These are local-filesystem
semantics, not a distributed or hostile-writer security protocol.

Keep the current `anchor` (record count and tip SHA-256) in a separately retained
run receipt after each successful operation. Recovery requires that exact
anchor and the expected plan hash. A hash chain alone cannot detect removal of
its final records, authenticate its author, or detect a rewritten chain when
the expected anchor is also replaced. An operation that commits before its
anchor reaches the caller has an ambiguous outcome: preserve and inspect both
histories rather than automatically deleting or retrying it. Pending temporary
files remain available for that inspection and are never replayed as commits.

`DurableRecorder.resume` replays the original plan and events through the
unchanged `PredictionRecorder` validation rules. It can accept scorer truth for
already committed predictions, record failures and produce an export. It
refuses new predictions and attachment to a fresh model. This avoids silently
continuing from an empty bank after a crash. Finalization first seals the exact
payload; if publication fails, recovery can export that same payload to a new
path. Missing and failed rows continue to fail evidence integrity.

`audit_observations` returns a scalar replay report: the final ledger checksum
and completed residuals with their actual arrival times. It returns no live
ledger or model handle. Tensor model, optimizer and bank checkpoint restoration
remain unimplemented and require admitted end-to-end tests before any scientific
run can resume. Journal replay is not an independent scientific reproduction.

Journals are capped at 16,384 records and 32 MiB of committed JSON; each record
and the original recorder's raw input/export have an 8 MiB limit. Operations
copy the bounded scalar state to validate before committing. This trades speed
and transient memory for simple transactional semantics. It is not a full-shape
memory profile, and full-size usage has not been admitted or benchmarked.

`verify_export.py` accepts `--bundle`, `--plan`, `--expected-plan-sha256` and
`--receipt`. It checks the exact frozen plan, source validator identity, schema,
evidence checksum and claim boundary, then reruns the existing analysis on the
actual raw evidence. It refuses a changed stored analysis and retains a failure
receipt for malformed inputs. A successfully reproduced negative verdict stays
negative; `PASS_REANALYSIS` is not a positive model result.

Run `run_recovery_tests.py --receipt PATH` through `rtk proxy python3` for the
combined standard-library acceptance suite. This has a 30-second alarm and
retains a receipt with source hashes, logs, elapsed time and peak process RSS.
The tests include process termination, disk failure before and after publication,
truncation, tampering, concurrent writers, incomplete evidence, delayed reveals,
sealed-export recovery and real command-line reanalysis. No torch import is
needed. Existing tensor tests remain separately guarded by `run_tensor_tests.py`.
