from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

import torch

from fim.training.losses import fim_total_loss
from fim.training.optimizer import EMA, build_optimizer
from fim.training.scheduler import build_scheduler


@dataclass
class TrainStats:
    loss: float
    pred_loss: float
    mem_loss: float
    stab_loss: float
    retr_loss: float
    cons_loss: float


class Trainer:
    """One-step training for FIMModel's current structured step output.

    The caller owns episode boundaries: call ``model.reset_state(clear_memory=True)``
    between independent trajectories. This helper does not execute or replace the
    separately frozen trajectory-isolated experiment runner.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        lr: float = 3e-4,
        weight_decay: float = 1e-2,
        device: str | torch.device = "cpu",
        grad_clip: float = 1.0,
        use_amp: bool = True,
        use_ema: bool = True,
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.grad_clip = float(grad_clip)
        if not 0.0 < self.grad_clip < float("inf"):
            raise ValueError("grad_clip must be finite and positive")
        self.use_amp = use_amp and self.device.type == "cuda"
        self.optimizer = build_optimizer(
            self.model.parameters(), lr=lr, weight_decay=weight_decay,
        )
        self.scheduler = None
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.ema = EMA(self.model) if use_ema else None

    def set_scheduler(self, total_steps: int, warmup_steps: int = 0) -> None:
        self.scheduler = build_scheduler(
            self.optimizer, total_steps=total_steps, warmup_steps=warmup_steps,
        )

    def _batch(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = batch
        if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
            raise TypeError("batch must contain input and target tensors")
        x, y = x.to(self.device), y.to(self.device)
        for name, value in (("input", x), ("target", y)):
            if not value.is_floating_point() or value.numel() == 0:
                raise ValueError(f"{name} must be a nonempty floating-point tensor")
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} must be finite")
        return x, y

    @staticmethod
    def _state_tensor(value):
        return value if isinstance(value, torch.Tensor) else getattr(value, "tensor", None)

    def _loss(self, result, target):
        if isinstance(result, tuple):
            if len(result) != 2:
                raise TypeError("model.step must return (prediction, output)")
            pred, out = result
        else:
            out = result
            pred = getattr(out, "prediction", None)
        state = self._state_tensor(getattr(out, "state", None))
        if not isinstance(pred, torch.Tensor) or not isinstance(state, torch.Tensor):
            raise TypeError("step output must contain tensor prediction and state")
        if pred.shape != target.shape:
            raise ValueError(f"prediction shape {tuple(pred.shape)} != target shape {tuple(target.shape)}")
        if not torch.isfinite(pred).all() or not torch.isfinite(state).all():
            raise ValueError("step prediction and state must be finite")

        # Current FIMStepOutput fields are tensors, not the retired FabricState
        # wrapper. Use the actual retrieval states so consistency is measurable.
        retrieval = getattr(out, "retrieval_context", None)
        if retrieval is None:
            retrieval = getattr(out, "retrieved", None)
        if retrieval is None:
            retrieval = pred.new_zeros((pred.shape[0], 1))
        mask = getattr(out, "stored_mask", None)
        if mask is None:
            salience = getattr(out, "salience", None)
            mask = (salience > 0.5).to(pred.dtype) if salience is not None else pred.new_zeros(pred.shape[0])
        pre = self._state_tensor(getattr(out, "pre_retrieval_state", None))
        post = self._state_tensor(getattr(out, "post_retrieval_state", None))
        pre = state if pre is None else pre
        post = state if post is None else post
        if pre.shape != post.shape:
            raise ValueError("pre/post retrieval state shapes must match")

        loss = fim_total_loss(
            pred=pred, target=target, state=state,
            stored_mask=mask, retrieval_context=retrieval,
            pre_retrieval_state=pre, post_retrieval_state=post,
            w_pred=1.0, w_mem=0.05, w_stab=0.05,
            w_sparse=0.05, w_retr=0.1, w_cons=0.05,
        )
        for name, value in vars(loss).items():
            if value.numel() != 1 or not torch.isfinite(value).all():
                raise ValueError(f"{name} loss must be a finite scalar")
        return loss

    @staticmethod
    def _stats(loss) -> TrainStats:
        return TrainStats(
            loss=float(loss.total.detach().item()),
            pred_loss=float(loss.prediction.detach().item()),
            mem_loss=float(loss.memory.detach().item()),
            stab_loss=float(loss.stability.detach().item()),
            retr_loss=float(loss.retrieval.detach().item()),
            cons_loss=float(loss.consistency.detach().item()),
        )

    def train_step(self, batch) -> TrainStats:
        x, y = self._batch(batch)
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
            loss = self._loss(self.model.step(x), y)
        self.scaler.scale(loss.total).backward()
        self.scaler.unscale_(self.optimizer)
        try:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.grad_clip, error_if_nonfinite=True,
            )
        except RuntimeError:
            # unscale_ has advanced the scaler state even when clipping rejects
            # the gradients. Finish that attempt without an optimizer/scheduler
            # or EMA update, so the next admitted batch can use this Trainer.
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            raise
        self.scaler.step(self.optimizer)
        self.scaler.update()
        if self.scheduler is not None:
            self.scheduler.step()
        if self.ema is not None:
            self.ema.update(self.model)
        return self._stats(loss)

    @torch.no_grad()
    def eval_step(self, batch) -> TrainStats:
        x, y = self._batch(batch)
        modes = [(module, module.training) for module in self.model.modules()]
        averages = self.ema.average_parameters(self.model) if self.ema is not None else nullcontext()
        try:
            self.model.eval()
            with averages:
                loss = self._loss(self.model.step(x), y)
                return self._stats(loss)
        finally:
            # A recursive train(previous) would overwrite deliberately mixed
            # submodule modes, for example a frozen evaluation-only encoder.
            for module, was_training in modes:
                module.training = was_training
