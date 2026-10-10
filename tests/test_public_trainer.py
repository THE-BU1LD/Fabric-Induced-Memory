"""Executable contracts for the public one-step Trainer, using artificial data."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from fim.models.fim_model import FIMModel
from fim.training.optimizer import EMA
from fim.training.trainer import Trainer


class StructuredStepModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.register_buffer("memory_count", torch.tensor(0, dtype=torch.long))
        self.register_buffer("memory_value", torch.tensor(5.0))
        self.calls = 0
        self.fail = False
        self.observed_weight = None

    def step(self, x):
        self.calls += 1
        self.observed_weight = float(self.weight.detach())
        if self.fail:
            raise RuntimeError("artificial step failure")
        pred = self.weight * x
        return pred, SimpleNamespace(
            prediction=pred, state=pred,
            pre_retrieval_state=pred - 2.0,
            retrieval_context=pred.flatten(1),
            stored_mask=torch.ones(x.shape[0], device=x.device, dtype=torch.bool),
        )


def test_current_fim_model_trains_evaluates_and_schedules():
    torch.manual_seed(31)
    model = FIMModel(
        in_channels=1, out_channels=1, latent_channels=8, trace_dim=4,
        hidden_channels=8, num_layers=1, max_traces=8,
    )
    trainer = Trainer(model, device="cpu", use_amp=False, use_ema=True)
    trainer.set_scheduler(total_steps=4, warmup_steps=1)
    x = torch.randn(1, 1, 6, 8)
    target = 0.5 * x
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    train_stats = trainer.train_step((x, target))
    assert all(torch.isfinite(torch.tensor(value)) for value in vars(train_stats).values())
    assert any(not torch.equal(before[name], p) for name, p in model.named_parameters())
    assert trainer.scheduler.last_epoch == 1
    assert trainer.ema.shadow.keys() == {
        name for name, p in model.named_parameters() if p.requires_grad
    }
    raw = {name: p.detach().clone() for name, p in model.named_parameters()}
    model.reset_state(clear_memory=True)
    eval_stats = trainer.eval_step((x, target))
    assert all(torch.isfinite(torch.tensor(value)) for value in vars(eval_stats).values())
    assert model.training
    for name, p in model.named_parameters():
        torch.testing.assert_close(p, raw[name], rtol=0, atol=0)


def test_trainer_uses_distinct_pre_and_post_retrieval_states():
    trainer = Trainer(StructuredStepModel(), use_amp=False, use_ema=False)
    stats = trainer.train_step((torch.ones(1, 1, 2, 2), torch.zeros(1, 1, 2, 2)))
    assert stats.cons_loss == pytest.approx(1.5)  # smooth L1 for a difference of 2
    assert stats.pred_loss == pytest.approx(1.0)


def test_ema_averages_weights_without_integer_or_episode_buffer_updates():
    model = StructuredStepModel()
    ema = EMA(model, decay=0.5)
    with torch.no_grad():
        model.weight.fill_(3.0)
        model.memory_count.fill_(7)
        model.memory_value.fill_(11.0)
    ema.update(model)
    assert set(ema.shadow) == {"weight"}
    torch.testing.assert_close(ema.shadow["weight"], torch.tensor(2.0))
    with ema.average_parameters(model):
        assert model.weight.item() == 2.0
        assert model.memory_count.item() == 7
        assert model.memory_value.item() == 11.0
    assert model.weight.item() == 3.0


@pytest.mark.parametrize("was_training", [True, False])
@pytest.mark.parametrize("fail", [True, False])
def test_eval_restores_raw_weights_and_mode_even_when_forward_fails(was_training, fail):
    model = StructuredStepModel()
    trainer = Trainer(model, use_amp=False, use_ema=True)
    with torch.no_grad():
        model.weight.fill_(3.0)
    model.train(was_training)
    model.fail = fail
    batch = (torch.ones(1, 1, 2, 2), torch.zeros(1, 1, 2, 2))
    if fail:
        with pytest.raises(RuntimeError, match="artificial step failure"):
            trainer.eval_step(batch)
    else:
        assert trainer.eval_step(batch).pred_loss == pytest.approx(1.0)
    assert model.observed_weight == 1.0
    assert model.weight.item() == 3.0
    assert model.training is was_training


def test_shape_broadcasting_is_rejected_before_optimizer_update():
    model = StructuredStepModel()
    trainer = Trainer(model, use_amp=False)
    with pytest.raises(ValueError, match="prediction shape"):
        trainer.train_step((torch.ones(1, 1, 2, 2), torch.zeros(1, 1, 1, 1)))
    assert model.weight.item() == 1.0
    assert not trainer.optimizer.state


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_target_is_rejected_before_model_execution(invalid):
    model = StructuredStepModel()
    trainer = Trainer(model, use_amp=False)
    with pytest.raises(ValueError, match="target must be finite"):
        trainer.train_step((torch.ones(1, 1, 2, 2), torch.full((1, 1, 2, 2), invalid)))
    assert model.calls == 0
    assert not trainer.optimizer.state


def test_nonfinite_gradients_cannot_update_raw_or_ema_weights():
    model = StructuredStepModel()
    trainer = Trainer(model, use_amp=False)
    model.weight.register_hook(lambda gradient: torch.full_like(gradient, float("inf")))
    with pytest.raises(RuntimeError, match="non-finite"):
        trainer.train_step((torch.ones(1, 1, 2, 2), torch.zeros(1, 1, 2, 2)))
    assert model.weight.item() == 1.0
    assert trainer.ema.shadow["weight"].item() == 1.0
    assert not trainer.optimizer.state


@pytest.mark.parametrize("decay", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_ema_decay_is_rejected(decay):
    with pytest.raises(ValueError, match="EMA decay"):
        EMA(StructuredStepModel(), decay=decay)
