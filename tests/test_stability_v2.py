"""Analytic diagnostics only: no model, checkpoint or retained result is loaded."""
import importlib.util
import math
import os
from pathlib import Path

import pytest
import torch

PATH = Path(os.environ.get('FIM_STABILITY_MODULE', Path(__file__).parents[1] / 'fim/core/stability_v2.py'))
spec = importlib.util.spec_from_file_location('tested_stability', PATH)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def x(values):
    return torch.tensor(values, dtype=torch.float64)


@pytest.mark.parametrize('scale', [1e-300, 1e-100, 1., 1e100, 1e300])
def test_bounded_matches_analytic_3_4_5_norm(scale):
    assert m.is_bounded(x([3*scale, 4*scale]), threshold=6*scale)
    assert not m.is_bounded(x([3*scale, 4*scale]), threshold=4*scale)


@pytest.mark.parametrize('scale', [1e-300, 1e-100, 1., 1e100, 1e300])
def test_growth_matches_log_differences_at_extreme_scales(scale):
    seq = [x([scale]), x([2*scale]), x([4*scale])]
    eps = 1e-320
    want = (math.log(4*scale+eps) - math.log(scale+eps)) / 2
    assert m.estimate_growth_rate(seq, eps=eps) == pytest.approx(want, abs=1e-12)


@pytest.mark.parametrize('a,b', [(1e-300, 1e300), (1e300, 1e-300), (0., 1e300), (1e300, 0.)])
def test_growth_does_not_form_overflowing_ratios(a, b):
    eps = 1e-320
    want = math.log(b+eps) - math.log(a+eps)
    assert m.estimate_growth_rate([x([a]), x([b])], eps=eps) == pytest.approx(want, abs=1e-11)


def test_zero_and_short_sequences():
    assert m.estimate_growth_rate([]) == 0
    assert m.estimate_growth_rate([x([1])]) == 0
    assert m.estimate_growth_rate([x([0]), x([0])]) == 0
    assert m.is_bounded(x([0]), threshold=0)
    assert not m.is_bounded(x([1e-300]), threshold=0)


@pytest.mark.parametrize('scale,bound', [(1e300, 5.), (1e300, 1e-300), (1e-200, 1e-250), (1., 2.)])
def test_clip_preserves_direction_and_representable_nonzero_result(scale, bound):
    t = x([3*scale, 4*scale]); old_ptr = t.data_ptr()
    result = m.clamp_norm_(t, bound, eps=0)
    assert result is t and old_ptr == t.data_ptr()
    assert torch.allclose(t / bound, x([.6,.8]), rtol=1e-14, atol=1e-14)


def test_clip_float32_norm_can_exceed_float32_max():
    t = torch.full((2,), 3e38, dtype=torch.float32)
    m.clamp_norm_(t, 4e38, eps=0)
    assert torch.isfinite(t).all()
    assert float(t[0]) == pytest.approx(4e38 / math.sqrt(2), rel=2e-7)


def test_clip_retains_epsilon_semantics_and_noop():
    t=x([3.,4.]); m.clamp_norm_(t, 2., eps=1.)
    assert torch.allclose(t, x([1.,4/3]), rtol=1e-14, atol=1e-14)
    t=x([.2,.1]); old=t.clone(); assert m.clamp_norm_(t,1) is t
    assert torch.equal(t,old)
    assert torch.equal(m.clamp_norm_(x([3.,4.]), 0), x([0.,0.]))


def test_clip_leaf_tensor_and_rng():
    t=x([3.,4.]).requires_grad_(); rng=torch.random.get_rng_state().clone()
    m.clamp_norm_(t,1.,eps=0)
    assert t.requires_grad and t.grad is None
    assert torch.equal(rng,torch.random.get_rng_state())
    assert torch.allclose(t.detach(),x([.6,.8]))


@pytest.mark.parametrize('bad', [-1., float('nan'), float('inf'), True, '2'])
def test_invalid_clip_bound_preserves_tensor(bad):
    t=x([3.,4.]); before=t.clone()
    with pytest.raises((ValueError,TypeError)):
        m.clamp_norm_(t,bad)
    assert torch.equal(t,before)


@pytest.mark.parametrize('eps', [0., -1., float('nan'), float('inf'), True])
def test_growth_invalid_epsilon_is_rejected(eps):
    with pytest.raises((ValueError,TypeError)):
        m.estimate_growth_rate([x([1.]),x([2.])],eps=eps)


@pytest.mark.parametrize('bad', [x([]), x([float('nan')]), x([float('inf')]), torch.tensor([1]), torch.tensor([1j])])
def test_invalid_fields_are_rejected(bad):
    with pytest.raises((ValueError,TypeError)):
        m.is_bounded(bad,threshold=1.)


def test_interior_invalid_frame_cannot_cancel_out():
    with pytest.raises(ValueError):
        m.estimate_growth_rate([x([1]),x([float('nan')]),x([1])])


def test_shape_and_dtype_changes_rejected():
    with pytest.raises(ValueError):
        m.estimate_growth_rate([x([1]),x([1,1])])
    with pytest.raises(ValueError):
        m.estimate_growth_rate([x([1]),torch.tensor([2.],dtype=torch.float32)])


def test_noncontiguous_read_supported_but_mutation_rejected():
    t = x([[1.,2.],[3.,4.]]).T; before=t.clone()
    assert m.is_bounded(t,threshold=6)
    with pytest.raises(ValueError):
        m.clamp_norm_(t,1.)
    assert torch.equal(t,before)


def test_same_dtype_graph_does_not_change_for_readonly_diagnostics():
    t=x([3.,4.]).requires_grad_(); before=t.detach().clone()
    m.is_bounded(t,threshold=6)
    m.estimate_growth_rate([t, t*2])
    assert t.grad is None and torch.equal(t.detach(),before)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_low_precision_large_collection_does_not_overflow_reduction(dtype):
    t=torch.ones(70000,dtype=dtype)
    assert m.is_bounded(t,threshold=300)
    m.clamp_norm_(t,100,eps=0)
    assert float(torch.linalg.vector_norm(t.double())) == pytest.approx(100,rel=.004)
