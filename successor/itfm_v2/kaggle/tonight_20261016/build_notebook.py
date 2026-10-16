"""Build the portable private notebook from an explicit, hash-checked allowlist."""
import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCE = ROOT / "audits/kaggle-readiness-20261016/fim-kaggle-preflight.zip"
EXPECTED = "9454e9cef6fba5ccf6057545ade46cd566167ad47570753623d789b5ffadfc46"


def build():
    source_bytes = SOURCE.read_bytes()
    assert hashlib.sha256(source_bytes).hexdigest() == EXPECTED
    helpers = ["budget.py", "workflow.py", "device_profile.py", "analysis.py", "protocol_candidate.json",
               "test_budget.py", "test_analysis.py", "test_workflow.py", "SCIENCE_SCOPE.md"]
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        def add_helper(name, data):
            info = zipfile.ZipInfo(name, date_time=(2026, 10, 16, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data)

        inventory = {}
        for name in helpers:
            data = (HERE / name).read_bytes()
            add_helper(name, data)
            inventory[name] = hashlib.sha256(data).hexdigest()
        add_helper("HELPER_MANIFEST.json", json.dumps(inventory, sort_keys=True))
    helper_bytes = stream.getvalue()
    cells = []

    def md(text):
        cells.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)})

    def code(text):
        ast.parse(text)
        cells.append({"cell_type": "code", "metadata": {}, "source": text.splitlines(True),
                      "execution_count": None, "outputs": []})

    md("""# FIM private GPU launch preparation — prepared 16 October 2026
**Current status: BLOCKED BEFORE EXECUTION.** This notebook has not run on Kaggle.
Only synthetic DEVELOPMENT evidence is admitted by the preserved implementation.
The user requested preparation for a **later GPU launch; no execution now**.
The original **22:00 IST / 16:30 UTC on 16 October 2026** completion target is retained
as historical planning context. It is not the deadline for a later launch.
At first account-ledger creation, the operator must explicitly enter a future
completion deadline with timezone. There is no default. The stored deadline governs
admission and cannot be silently extended by rerunning cells or reopening this notebook.

Use one **interactive** notebook, initially **Accelerator: None**, private sharing,
environment pinned to its original image. Do not use Run All or Save & Run All.
Run cells in order, resolving the explicit platform gates. The notebook contains
the verified source ZIP and helpers; it does not depend on a Mac path or credentials.

GPU account observations and session stop are operator duties. Code cannot promise
a hard account limit. Every allocated GPU counts conservatively, including idle
time; using CUDA device0 does not release a second allocated GPU.
""")
    md("## 1. Verify and extract the recorded source snapshot")
    code('''import base64, hashlib, io, json, os, sys, zipfile
from pathlib import Path
BASE = Path('/kaggle/working/fim_20261016')
SRC, SUPPORT, WORK = BASE/'source', BASE/'support', BASE/'evidence'
for directory in (SRC, SUPPORT, WORK): directory.mkdir(parents=True, exist_ok=True)
def unpack(encoded, expected, destination, manifest_name):
    raw = base64.b64decode(encoded)
    assert hashlib.sha256(raw).hexdigest() == expected, 'embedded archive changed'
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        for info in archive.infolist():
            name = Path(info.filename)
            assert not name.is_absolute() and '..' not in name.parts
            assert (info.external_attr >> 16) & 0o170000 != 0o120000
            data = archive.read(info)
            target = destination/name
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists(): assert target.read_bytes() == data, f'existing source differs: {name}'
            else: target.write_bytes(data)
    manifest = json.loads((destination/manifest_name).read_text())
    records = manifest.get('files', manifest)
    for name, expected_digest in records.items():
        assert hashlib.sha256((destination/name).read_bytes()).hexdigest() == expected_digest
''' + f"unpack({base64.b64encode(source_bytes).decode()!r}, {EXPECTED!r}, SRC, 'BUNDLE_MANIFEST.json')\n"
         + f"unpack({base64.b64encode(helper_bytes).decode()!r}, {hashlib.sha256(helper_bytes).hexdigest()!r}, SUPPORT, 'HELPER_MANIFEST.json')\n"
         + '''sys.path[:0] = [str(SUPPORT), str(SRC/'src')]
os.environ['PYTHONPATH'] = str(SRC/'src')
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
from fim_sota.engineering import source_identity, resolve, validate_completed, digest, data_identity
from workflow import SOURCE_SHA256, environment, cpu_gate, create_probe, verify_retrieval, require_gates, gpu_job, freeze, run_next, archive, write, now, sha, validate_profile, supervised
from budget import AccountBudget, quota_policy, utc_timestamp
SESSION = 'fim-20261016-interactive'
def observe_platform(budget):
    q = float(input('Observed remaining Kaggle GPU quota-hours: '))
    observed = input('Quota observation ISO timestamp with timezone: ').strip()
    sessions = json.loads(input('Observed active GPU session IDs as JSON list (include this session if active): '))
    session_time = input('Active Events observation ISO timestamp with timezone: ').strip()
    budget.observe(quota_hours=q, observed_at=observed, active_sessions=sessions,
                   sessions_observed_at=session_time)
assert source_identity()['sha256'] == SOURCE_SHA256
write(WORK/'source.json', source_identity())
print('Verified source', SOURCE_SHA256)
''')
    md("""## 2. Minimal compatible dependencies
Use the packages already installed in the pinned Kaggle environment. Record exact
torch/numpy/pytest versions. Missing packages or torch<2.6 stop on CPU; do not
replace the runtime blindly. No pip install or internet connection is needed if
these required packages are present. The resulting lock file records this run's environment.
""")
    code("env = environment(SRC, WORK)\nprint(json.dumps(env, indent=2))\n")
    md("""## 3. Data generation and split integrity
No external dataset is attached. The maintained generator makes bounded periodic
fields with separate train/validation/test generator streams and owner IDs.
The full numerical CPU suite verifies causality, owner reset, raw-metric replay,
and exact trainer recovery before training is admitted. Source-bound data IDs below
are specifications; generated tensor hashes are retained by actual runs.
""")
    code("""candidate = json.loads((SUPPORT/'protocol_candidate.json').read_text())
for seed in candidate['outcome_access']['candidate_seeds']:
    cfg = resolve({**candidate['candidate_common_config'], 'seed': seed})
    print(seed, data_identity(cfg))
write(WORK/'candidate_protocol.json', candidate)
""")
    md("""## 4. Credentials
None are required for this synthetic fixture. Do not paste Kaggle keys, account
tokens, environment dumps, or unrelated repository contents into this notebook.
The existing private notebook output mechanism is the selected artifact route.
""")
    md("## 5. Resolved source, configuration, and environment manifests")
    code("""print(json.dumps({'source': SOURCE_SHA256, 'status': candidate['status'],
 'historical_planning_completion_target': candidate['historical_planning_completion_time_utc'],
 'completion_deadline_utc_timestamp': (AccountBudget(WORK/'account_budget.json').snapshot()['deadline_utc_timestamp']
     if (WORK/'account_budget.json').exists() else None),
 'deadline_policy': 'Explicit future completion deadline at first ledger creation; existing deadline is preserved',
 'precision': 'fp32',
 'config': candidate['candidate_common_config']}, indent=2))
""")
    md("""## 6. CPU preflight and persistence gate
Keep accelerator **None**. This runs the maintained numerical tests and one tiny
CPU smoke experiment (test supervisor340s; smoke120s), followed by the candidate's
full-shape CPU component profile and a separate three-epoch CPU DEVELOPMENT pilot.
The latter measures validation/evaluation cost as well as training; component and
pilot supervisors are335s and930s. These are measurements, not assumed CPU speeds.
Reruns validate successful receipts and refuse incomplete attempts. Do not erase
failure records to retry or allocate a GPU after an unresolved CPU failure.
""")
    code("""cpu_result = cpu_gate(SRC, WORK)
print(json.dumps(cpu_result, indent=2))
cpu_pilot_cfg = resolve({**candidate['candidate_common_config'], 'device':'cpu', 'seed':101,
 'epochs':3, 'wall_seconds':900, 'checkpoint_seconds':300})
cpu_config_path = WORK/'cpu_pilot.config.json'
if cpu_config_path.exists():
    assert json.loads(cpu_config_path.read_text()) == cpu_pilot_cfg, 'CPU pilot configuration changed'
else:
    write(cpu_config_path, cpu_pilot_cfg)
def cpu_stage(run_id, command, seconds, validator):
    timing_path, logfile = WORK/(run_id+'.timing.json'), WORK/(run_id+'.log')
    if timing_path.exists():
        saved = json.loads(timing_path.read_text())
        assert saved['exit_code'] == 0 and saved['class'] == 'DEVELOPMENT'
        assert saved['validated_evidence'] == validator(), 'CPU stage artifacts changed'
        return saved
    if logfile.exists():
        raise RuntimeError('Incomplete CPU attempt is preserved; explicit review required: '+run_id)
    measured = supervised(command, logfile, seconds)
    verified = validator()
    assert verified.get('source_sha256', verified.get('source',{}).get('sha256')) == SOURCE_SHA256
    result = {**measured, 'validated_evidence':verified, 'class':'DEVELOPMENT', 'time':now()}
    write(timing_path, result)
    return result
cpu_profile_output = WORK/'cpu_representative_profile'
def verify_cpu_profile():
    return validate_profile(SRC, cpu_profile_output/'profile.json', cpu_pilot_cfg)
print(cpu_stage('cpu-components',
 [sys.executable,'-u',str(SUPPORT/'device_profile.py'),'--source-root',str(SRC),
  '--config',str(cpu_config_path),'--output',str(cpu_profile_output)], 335, verify_cpu_profile))
def verify_cpu_pilot():
    receipt, _ = validate_completed(WORK/'cpu_pilot', cpu_pilot_cfg, SOURCE_SHA256)
    return receipt
print(cpu_stage('cpu_pilot',
 [sys.executable,'-u',str(SRC/'scripts/kaggle.py'),'run','--config',str(cpu_config_path),
  '--output',str(WORK/'cpu_pilot')], 930, verify_cpu_pilot))
print(create_probe(WORK))
""")
    md("""### Verify a durable private output before any GPU allocation
In **Save Version**, select **Quick Save**, expand **Advanced settings**, and choose
**Save output for this version when creating a Quick Save**. The observed default
was **Never save output when creating a Quick Save**. Save while sharing remains Private.
Open that saved version and confirm the probe exists in Output. Attach the saved
private notebook version through **Add Input → notebook outputs**; select the exact version.
Enter its retrieved `/kaggle/input/.../persistence_probe.json` path below. If unavailable,
stop: neither a ZIP in working nor Files-only session persistence passes this gate.

Enable **Session options → Persistence → Files only** as an additional best-effort
convenience before changing accelerator; retain the saved private version as the verified copy.
""")
    code("""retrieved_probe = Path(input('Retrieved /kaggle/input/.../persistence_probe.json path: ').strip())
version_url = input('Actual private saved notebook version URL: ').strip()
print(verify_retrieval(WORK/'persistence_probe.json', retrieved_probe,
 WORK/'persistence_verified.json', private_version_url=version_url))
""")
    md("""## 7. Account admission, CUDA preflight, and representative profiling
**Before allocating**, read GPU remaining quota and Active Events in Kaggle. Other
active GPU sessions require coordination; this notebook will not terminate them.
Q0 is never30 by default. Preferred cap=min(6,max(0,Q0−24)); expanded cap is informational
only. This launch requests at most4 quota-hours, within the preferred cap, and retains10%
for variance/export/shutdown. GPU readiness and profiling have a separate20% ceiling
of the authorized quota budget. The guard conservatively counts actual allocation
elapsed time plus the proposed bounded job timeout before admitting another readiness
or profiling job; this ceiling does not consume or replace the10% export/shutdown reserve.
All projects must share this account ledger and coordinate
access; independent copies in separate runtimes are not a synchronized global service.

Enter the actual observation time with timezone, e.g. the UTC time when you read the UI.
For a new ledger, explicitly enter the intended future completion deadline as an ISO
timestamp with timezone. No deadline is inferred from the preparation date or original
planning target. An existing ledger retains its recorded deadline and allowance; an
expired ledger must not be reset or silently extended to admit more work.
Only if this reservation passes, enable the chosen accelerator in Kaggle. This account
offered T4×2 on inspection; select one GPU if available, otherwise count both allocated
T4s conservatively while using only CUDA device0. No dual-GPU speedup is assumed.
""")
    code("""require_gates(WORK)
cpu_pilot_cfg = resolve({**candidate['candidate_common_config'], 'device':'cpu', 'seed':101,
 'epochs':3, 'wall_seconds':900, 'checkpoint_seconds':300})
validate_profile(SRC, WORK/'cpu_representative_profile/profile.json', cpu_pilot_cfg)
cpu_completion, _ = validate_completed(WORK/'cpu_pilot', cpu_pilot_cfg, SOURCE_SHA256)
cpu_timing = json.loads((WORK/'cpu_pilot.timing.json').read_text())
assert cpu_timing['exit_code'] == 0 and cpu_timing['validated_evidence'] == cpu_completion
ledger_path = WORK/'account_budget.json'
if ledger_path.exists():
    budget = AccountBudget(ledger_path)
    observe_platform(budget)
else:
    q0 = float(input('Actual Q0 from Kaggle before allocation: '))
    observed = input('Q0 observation ISO timestamp with timezone: ').strip()
    sessions = json.loads(input('Observed active GPU session IDs as JSON list: '))
    session_time = input('Active Events observation ISO timestamp with timezone: ').strip()
    launch_deadline = input('Explicit future completion deadline ISO timestamp with timezone (no default): ').strip()
    if not launch_deadline or utc_timestamp(launch_deadline) <= utc_timestamp(now()):
        raise ValueError('Completion deadline must be explicitly supplied and in the future')
    policy = quota_policy(q0)
    budget = AccountBudget.create(ledger_path, account_id='ryangomez010', q0_hours=q0,
      observed_at=observed, active_sessions=sessions, sessions_observed_at=session_time,
      deadline_utc=launch_deadline,
      authorized_hours=min(4.0, policy['preferred_budget_hours']))
allocated_count = int(input('Physical GPUs in the accelerator allocation you will select (T4x2 = 2): '))
if SESSION not in budget.snapshot()['sessions']:
    budget.reserve_session(SESSION, 'FIM', allocated_count)
else:
    assert budget.snapshot()['sessions'][SESSION]['allocated_device_count'] == allocated_count, 'existing allocation count differs; do not replace the ledger'
print(json.dumps(budget.snapshot(), indent=2))
print('Reservation only. Now enable accelerator in the UI; this code has not allocated a GPU.')
""")
    md("""After the UI confirms the GPU session is active, run the next cell. If changing
accelerator restarted the kernel, rerun sections **1–5** to reconstruct imports,
paths, candidate and observation helper, then continue with the cell immediately below.
Do not rerun the CPU gate on a GPU allocation or create another allowance. The persisted
ledger and successful CPU/retrieval receipts must still exist; otherwise stop and recover
the exact saved evidence first. Re-enter fresh platform observations. Count time
from the reservation conservatively. Never claim the first CUDA call started billing.
""")
    code("""budget = AccountBudget(WORK/'account_budget.json')
observe_platform(budget)
if budget.snapshot()['sessions'][SESSION]['state'] == 'PENDING_ALLOCATION':
    budget.confirm_allocation(SESSION, observed_at=input('Observed GPU allocation start ISO timestamp: ').strip())
import subprocess
inventory = subprocess.run(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],
                           check=True, capture_output=True, text=True, timeout=10)
device_names = [line.strip() for line in inventory.stdout.splitlines() if line.strip()]
expected_count = budget.snapshot()['sessions'][SESSION]['allocated_device_count']
assert device_names and len(device_names) == expected_count, 'driver device count differs from reserved physical allocation; stop and reconcile'
write(WORK/'gpu_driver_inventory.json', {'observed_at':now(), 'driver_device_names':device_names,
      'driver_visible_count':len(device_names), 'reserved_allocated_device_count':expected_count})
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['FIM_TEST_DEVICE'] = 'cuda'
gpu_config_path = SRC/'kaggle/smoke-cuda.json'
gpu_config = resolve(json.loads(gpu_config_path.read_text()))
def verify_gpu_environment():
    value = json.loads((WORK/'gpu_environment.json').read_text())
    assert value['status'] == 'PASS_RUNTIME_ENVIRONMENT' and value['runtime_executed'] is True
    assert value['source'] == source_identity() and value['device'] == 'cuda' and value['precision'] == 'fp32'
    assert value['config_sha256'] == digest(gpu_config)
    assert value['packages'] == {key:env['packages'][key] for key in ('torch','numpy')}
    return value
gpu_job(SRC, WORK, budget, SESSION, 'gpu-preflight',
 [sys.executable,'-u',str(SRC/'scripts/kaggle.py'),'preflight','--config',str(gpu_config_path),
  '--runtime','--receipt',str(WORK/'gpu_environment.json')], 120,
 verify_artifacts=verify_gpu_environment)
print(json.dumps(verify_gpu_environment(), indent=2))
""")
    md("Run only three targeted FP32 CUDA tests; the full CPU suite is not rerun on GPU.")
    code("""import xml.etree.ElementTree as ET
test_names = ['tests/test_engineering_runtime.py::test_requested_device_training_evaluation',
 'tests/test_engineering_runtime.py::test_train_select_checkpoint_and_replay_requested_cuda',
 'tests/test_training_recovery.py::test_cuda_fp32_tiny_training']
junit = WORK/'cuda_selected_tests.xml'
def verify_cuda_tests():
    tree = ET.parse(junit)
    cases = tree.findall('.//testcase')
    assert len(cases) == 3
    assert {c.attrib['name'] for c in cases} == {n.split('::')[-1] for n in test_names}
    assert not any(c.find(tag) is not None for c in cases for tag in ('failure','error','skipped'))
    value = {'status': 'PASS', 'requested_device': 'cuda', 'precision': 'fp32',
             'tests': test_names, 'junit_sha256': sha(junit), 'source_sha256': SOURCE_SHA256,
             'test_sources':{name:sha(SRC/'tests'/name) for name in
                             ('test_training_recovery.py','test_engineering_runtime.py')}}
    write(WORK/'cuda_selected_tests.json', value)
    return value
observe_platform(budget)
gpu_job(SRC, WORK, budget, SESSION, 'cuda-tests',
 [sys.executable,'-m','pytest','-q',*[str(SRC/n.split('::')[0])+'::'+n.split('::')[1] for n in test_names],
  '--junitxml',str(junit)], 300, verify_artifacts=verify_cuda_tests)
""")
    md("""The profile retains the full candidate batch/resolution/context shapes and FP32.
It warms up then measures component timings. A separate3-epoch DEVELOPMENT pilot
includes all validation, evaluation variants and88-step base rollout. There is no
batch/precision search until measured evidence justifies one.
""")
    code("""pilot_cfg = resolve({**candidate['candidate_common_config'], 'seed':101,
 'epochs':3, 'wall_seconds':900, 'checkpoint_seconds':300})
write(WORK/'pilot.config.json', pilot_cfg)
observe_platform(budget)
profile_output = WORK/'representative_profile'
profile_command = [sys.executable,'-u',str(SUPPORT/'device_profile.py'),
 '--source-root',str(SRC),'--config',str(WORK/'pilot.config.json'),
 '--output',str(profile_output)]
def verify_profile():
    return validate_profile(SRC, profile_output/'profile.json', pilot_cfg)
gpu_job(SRC, WORK, budget, SESSION, 'components', profile_command, 335, verify_artifacts=verify_profile)
observe_platform(budget)
def verify_pilot():
    receipt, _ = validate_completed(WORK/'pilot', pilot_cfg, SOURCE_SHA256)
    return receipt
gpu_job(SRC, WORK, budget, SESSION, 'pilot',
 [sys.executable,'-u',str(SRC/'scripts/kaggle.py'),'run','--config',str(WORK/'pilot.config.json'),
 '--output',str(WORK/'pilot')], 930, verify_artifacts=verify_pilot)
""")
    md("""## 8. Freeze the complete finite DEVELOPMENT campaign
The freeze uses measured end-to-end pilot cost, doubles a whole-run epoch-scaled
estimate, adds startup margin, and admits all five cells or none. It does not inspect
candidate test outcomes. If the matrix does not fit, stop and revise the still-unrun
question explicitly; do not drop a seed or shorten a frozen training cell.
""")
    code("""observe_platform(budget)
frozen = freeze(SUPPORT/'protocol_candidate.json', WORK/'pilot', WORK, budget, SESSION)
print(json.dumps(frozen, indent=2))
""")
    md("""## 9. Live logs and one bounded cell at a time
Refresh the real quota and Active Events, execute one cell, then Quick Save its output
and verify retrieval before starting the next. This deliberately avoids unattended
launches on stale account data. Repeat this cell until ALL_REQUIRED_CELLS_VERIFIED.
Completed cells are hash-validated and reused; attempted incomplete cells are retained
and block silent retries. Training JSONL, stdout and30-second budget heartbeats are real.
""")
    code("""observe_platform(budget)
print(run_next(SRC, WORK, budget, SESSION))
print((WORK/'run_ledger.json').read_text())
""")
    md("""## 10. Restore and resume from saved private outputs
At each safe boundary, preserve an archive using section11 and Quick Save with outputs.
In a fresh session, attach that saved private version read-only, verify its SHA256,
then extract into a new directory using the helper below. Never overwrite an existing
campaign or start a fresh account allowance. Recover the shared account ledger and
reconcile actual platform sessions/quota before any new allocation.

On **Accelerator: None**, rerun sections1–5, then use the concrete restore API:
```python
from workflow import restore
restored_work = BASE/'restored_evidence'  # must not already exist
restore(RETRIEVED_ARCHIVE_PATH, restored_work, RECORDED_ARCHIVE_SHA256, source_root=SRC)
WORK = restored_work
cpu_gate(SRC, WORK)  # rebind current paths and verify the saved CPU receipts/environment
budget = AccountBudget(WORK/'account_budget.json')
observe_platform(budget)
```
Supply the archive path and checksum from the saved export receipt. Reconcile the
previous session's real stop state before reserving any new platform allocation.
Do not resume by blindly rerunning all notebook cells or resetting the ledger.

Full-state trainer resume is available through the maintained command below. It
requires identical source/config/data/device/precision and the same dependency versions.
The original attempted cell remains FAILED/INTERRUPTED; record a new continuation
ID and parent-checkpoint hash, reserve its budget before launch, and do not pool it
automatically. A stopped mid-evaluation run restarts evaluation from the saved model;
the per-owner bank is replayed, not restored from an invented mid-stream state.
""")
    code("""from workflow import restore
print('Restore helper available:', restore.__name__, '(input_archive, NEW_destination, expected_sha256)')
print('Exact maintained continuation command (run only through a reserved bounded wrapper):')
print(f'{sys.executable} -u {SRC}/scripts/kaggle.py run --config ORIGINAL_CONFIG.json '
      '--output NEW_UNIQUE_RUN --resume RETRIEVED_PARENT/checkpoints/latest.pt')
""")
    md("""## 11. Analyze, export, and verify private retrieval
Analysis accepts only complete identity-verified cells. It reports every missing/failed
cell and seed-level paired differences. Five seeds cannot attain a two-sided exact
sign-enumeration p<.05. Timesteps are not independent replicates. This package does not
infer equivalence, novelty, public benchmark superiority or paper acceptance.
""")
    code("""from analysis import run as analyze
analysis_path = WORK/('analysis_'+now().replace(':','-'))
print(analyze(WORK, WORK/'frozen_manifest.json', analysis_path))
archive_path = BASE/('fim_evidence_'+now().replace(':','-')+'.zip')
export_receipt = archive(WORK, archive_path)
write(WORK/'latest_export.json', export_receipt)
print(export_receipt)
""")
    md("""Quick Save with outputs; confirm the archive appears in that private saved version.
Attach it through Add Input and run the following check. Save the resulting retrieval
receipt too. A local readback alone is not remote persistence. The conservative10GiB
archive ceiling stays below the documented20GB notebook output cap, but also check the
total working output size including source and prior archives.
""")
    code("""retrieved_archive = Path(input('Retrieved /kaggle/input/.../fim_evidence_*.zip path: ').strip())
saved_url = input('Exact private saved notebook version URL: ').strip()
print(verify_retrieval(archive_path, retrieved_archive, WORK/'final_retrieval.json',
                      private_version_url=saved_url))
""")
    md("""## 12. Explicit platform shutdown and final quota reconciliation
Stop new work, save/retrieve outputs with margin, then **Stop session** in Kaggle.
Check **Active Events** to confirm this interactive allocation is gone and that no
batch session was started. A Python exit is not session shutdown. Do not stop unrelated
jobs. Read and record the final remaining GPU allowance and observation time.

Run request_stop before clicking Stop session. Because the kernel stops too, complete
the reconciliation on CPU or in the local downloaded ledger. This code does not stop
the platform and never labels a stop-request as a confirmed stop.
""")
    code("""budget.request_stop(SESSION)
print(json.dumps(budget.snapshot(), indent=2))
print('Now save outputs and STOP this Kaggle session in the platform UI; verify Active Events.')
""")
    md("""After observing the stop, on CPU use:
```python
budget.confirm_session_stopped(SESSION, observed_at=ACTUAL_PLATFORM_STOP_ISO_TIME)
budget.observe(quota_hours=ACTUAL_FINAL_BALANCE, observed_at=ACTUAL_QUOTA_ISO_TIME,
               active_sessions=ACTUAL_ACTIVE_GPU_SESSION_IDS,
               sessions_observed_at=ACTUAL_EVENTS_ISO_TIME)
print(budget.snapshot())
```
These observations must be real. Preserve the final ledger outside the stopped runtime.
The source snapshot remains scientific_claim=NOT_EVALUATED; a completed matrix supports
only the stated bounded development diagnostic. Expert scientific and venue-specific
review remain necessary before a research manuscript claim.
""")
    notebook = {"nbformat":4,"nbformat_minor":5,"metadata":{
        "kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
        "language_info":{"name":"python"}, "kaggle":{"accelerator":"none","isInternetEnabled":False},
        "fim":{"source_sha256":"97a31bff8d785c37b0f2af6bfc374900670ed66bd554e17bb1d6525253a719f2",
               "status":"UNEXECUTED_LAUNCH_PACKAGE"}}, "cells":cells}
    for index, cell in enumerate(cells):
        cell['id'] = f'fim-{index:03d}'
    path = HERE/'FIM_tonight_20261016.ipynb'
    path.write_text(json.dumps(notebook, indent=1)+'\n')
    return {"path":str(path),"cells":len(cells),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()}


if __name__ == '__main__':
    print(json.dumps(build(), indent=2))
