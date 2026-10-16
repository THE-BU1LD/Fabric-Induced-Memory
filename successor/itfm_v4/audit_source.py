"""Bounded adversarial audit of retained source. No Torch or model execution.

All constructed rows are invalid-input test fixtures, never research outcomes.
Source functions are selected through AST from the existing public stack without
importing that stack. The original files are read only.
"""
from __future__ import annotations

import argparse
import ast
import datetime
import hashlib
import json
import math
import random
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PUBLIC = ROOT / 'successor/itfm_v2/src/fim_sota'


def selected_functions(path, names, namespace):
    text = path.read_text()
    tree = ast.parse(text)
    definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in definitions} != set(names):
        raise RuntimeError('Requested source function missing')
    unit = ast.Module(body=definitions, type_ignores=[])
    exec(compile(unit, str(path), 'exec'), namespace)
    return hashlib.sha256(text.encode()).hexdigest()


def audit():
    class VerificationError(RuntimeError):
        pass

    ns = {'Mapping': Mapping, 'Sequence': Sequence, 'Any': Any,
          'Path': Path, 'json': json, 'VerificationError': VerificationError}
    source_hash = selected_functions(PUBLIC / 'verification.py',
        ['decide_sota', 'verify_metric_claims', 'recompute_scalar_metrics'], ns)
    protocol = json.loads((ROOT / 'successor/itfm_v2/protocols/sota_confirmation_v1.json').read_text())
    primary = [{'cell_id': 'one-cell-repeated', 'win': True, 'relative_regression': 0.,
                'seed_win_fraction': 1., 'relative_improvement': .2}] * 36
    secondary = [{'family': 'one-family-repeated', 'win': True,
                  'relative_improvement': .2, 'candidate_error': .05,
                  'relative_regression': 0.}] * 5
    duplicate = ns['decide_sota'](protocol=protocol, primary_cells=primary,
        secondary_families=secondary, independently_reproduced=True)
    unique_primary = [dict(primary[0], cell_id=f'{dataset}/{mask}/{train}/{test}')
        for dataset in protocol['primary_track']['datasets']
        for mask in protocol['primary_track']['mask_patterns']
        for train in protocol['primary_track']['train_missing_percent']
        for test in protocol['primary_track']['test_missing_percent_by_train'][str(train)]]
    unique_secondary = [dict(secondary[0], family=family)
        for family in protocol['secondary_track']['families']]
    kwargs = {'protocol': protocol, 'primary_cells': unique_primary,
              'secondary_families': unique_secondary, 'independently_reproduced': True}
    benign = ns['decide_sota'](**kwargs)
    harmful_secondary = [dict(row) for row in unique_secondary]
    # One genuine family loss is offset in the average by four large wins.
    # The protocol still forbids this 100 percent family-level regression.
    harmful_secondary[0].update(win=False, relative_improvement=-1.,
                                relative_regression=1., candidate_error=.1)
    for row in harmful_secondary[1:]:
        row.update(relative_improvement=.5, candidate_error=.05)
    regression = ns['decide_sota'](**dict(kwargs, secondary_families=harmful_secondary))
    controls = {
        'benign_summary_fixture_accepted': benign['broad_sota_allowed'],
        'short_matrix_rejected': not ns['decide_sota'](**dict(kwargs,
            primary_cells=unique_primary[:-1]))['broad_sota_allowed'],
        'false_independence_rejected': not ns['decide_sota'](**dict(kwargs,
            independently_reproduced=False))['broad_sota_allowed'],
    }
    try:
        ns['verify_metric_claims']({'mse': 2.}, {'mse': 1.})
        controls['ordinary_metric_mismatch_rejected'] = False
    except VerificationError:
        controls['ordinary_metric_mismatch_rejected'] = True
    nan_passes = []
    for claimed, recomputed in [(float('nan'), 1.), (1., float('nan')),
                                (float('inf'), float('inf'))]:
        try:
            ns['verify_metric_claims']({'mse': claimed}, {'mse': recomputed})
            nan_passes.append(True)
        except VerificationError:
            nan_passes.append(False)
    with tempfile.TemporaryDirectory(prefix='fim-source-audit-') as tmp:
        p = Path(tmp) / 'predictions.json'
        p.write_text(json.dumps({'rows': [{'prediction': [0.], 'target': [1.]},
                                          {'prediction': [10.], 'target': [10.]}]}))
        pooled = ns['recompute_scalar_metrics'](p)['relative_l2']
    mean_sample = (1. + 0.) / 2
    stats_ns = {'Mapping': Mapping, 'Sequence': Sequence, 'Any': Any, 'EvaluationCell': Any,
                'math': math, 'random': random, 'defaultdict': defaultdict}
    stats_hash = selected_functions(PUBLIC / 'statistics.py',
        ['aggregate_within_trajectory', 'paired_values', 'family_cluster_bootstrap'], stats_ns)
    from types import SimpleNamespace
    rows = []
    for cell, count, error in [('cell-a', 1, 0.), ('cell-b', 9, 1.)]:
        for index in range(count):
            for method, value in [('candidate', error), ('baseline', 1.)]:
                rows.append(SimpleNamespace(method=method, seed=1, family='family-a',
                    trajectory=f'{cell}-{index}', cell=cell, value=value, status='PASS'))
    weighted = stats_ns['family_cluster_bootstrap'](rows, candidate='candidate',
        baseline='baseline', repetitions=100, confidence=.95, seed=1)
    broken_rows = [SimpleNamespace(**vars(row)) for row in rows]
    broken_rows[0].status = 'FAILED'
    paired_before = len(stats_ns['paired_values'](rows, 'candidate', 'baseline')[2])
    paired_after = len(stats_ns['paired_values'](broken_rows, 'candidate', 'baseline')[2])
    adapter_source = (PUBLIC / 'adapter.py').read_text()
    adapter_tree = ast.parse(adapter_source)
    writer = next(n for n in ast.walk(adapter_tree) if isinstance(n, ast.FunctionDef) and n.name == 'write_observed')
    attributes = {n.attr for n in ast.walk(writer) if isinstance(n, ast.Attribute)}
    findings = [
        {'id': 'V01', 'finding': 'Duplicated matrix rows can obtain broad_sota_allowed.',
         'observed': duplicate['broad_sota_allowed'], 'defect_confirmed': duplicate['broad_sota_allowed'],
         'details': '36 copies of one cell and 5 copies of one family; independence supplied as a bare boolean.'},
        {'id': 'V02', 'finding': 'Secondary 100 percent regression is ignored by the decision.',
         'observed': regression['secondary_track_pass'], 'defect_confirmed': regression['secondary_track_pass'],
         'details': 'Five unique protocol families: one loses 100 percent, four win 50 percent; macro gain 20 percent and mean error .06. The forbidden family regression still passes.'},
        {'id': 'V03', 'finding': 'NaN and same-sign infinite metric pairs can pass comparison.',
         'observed': nan_passes, 'defect_confirmed': all(nan_passes)},
        {'id': 'V04', 'finding': 'Pooled relative L2 differs from mean per-sample relative L2.',
         'observed': {'source_verifier': pooled, 'declared_sample_average': mean_sample},
         'defect_confirmed': not math.isclose(pooled, mean_sample, abs_tol=1e-12)},
        {'id': 'V05', 'finding': 'Bootstrap point estimate weights trajectories instead of equal cells.',
         'observed': {'source_estimate': weighted['estimate'], 'equal_cell_estimate': .5},
         'defect_confirmed': not math.isclose(weighted['estimate'], .5, abs_tol=1e-12)},
        {'id': 'V06', 'finding': 'Pairing silently drops an unmatched failed row.',
         'observed': {'paired_before': paired_before, 'paired_after': paired_after},
         'defect_confirmed': paired_after == paired_before - 1,
         'details': 'A complete-matrix gate must run before this statistic; this finding alone does not prove a final scientific false pass.'},
        {'id': 'V07', 'finding': 'Writer never reads target_observation_mask.',
         'observed': 'target_observation_mask' in attributes,
         'defect_confirmed': 'target_observation_mask' not in attributes,
         'details': 'Static AST evidence; no tensor leakage experiment is claimed.'},
    ]
    from fractions import Fraction as F
    identities = []
    for e in [F(-2), F(0), F(3)]:
        for correction in [F(-1), F(0), F(2)]:
            for scale in [F(0), F(1, 2), F(1)]:
                identities.append(e*e-(e-scale*correction)**2 == 2*scale*e*correction-scale*scale*correction*correction)
    if not all(item['defect_confirmed'] for item in findings) or not all(identities) or not all(controls.values()):
        raise RuntimeError('Audit assumptions changed; inspect source and retained evidence')
    hashes = {'verification.py': source_hash, 'statistics.py': stats_hash,
              'adapter.py': hashlib.sha256(adapter_source.encode()).hexdigest()}
    for name, expected in hashes.items():
        if hashlib.sha256((PUBLIC / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Source changed during audit: ' + name)
    return {'at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'audit_revision': 2,
            'status': 'CONFIRMED_SOURCE_DEFECTS_NOT_MODEL_EFFICACY',
            'fixture_boundary': 'All generated rows are adversarial software fixtures; not scientific data or experiments.',
            'source_sha256': hashes,
            'audit_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'control_checks': controls,
            'findings': findings, 'rational_identity_checks_passed': sum(identities),
            'rational_identity_checks_total': len(identities),
            'proof_scope': 'Finite exact-arithmetic sanity checks accompany a general algebraic derivation; no empirical success or novelty claim.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit()
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': result['status'], 'confirmed_findings': len(result['findings']),
                      'arithmetic_checks': result['rational_identity_checks_passed']}))
