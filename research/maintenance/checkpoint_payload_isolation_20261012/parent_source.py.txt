from __future__ import annotations

import copy
import math
import os
import tempfile
from numbers import Integral
from pathlib import Path
from typing import Any, Dict, Optional

import torch


def _epoch(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError("checkpoint epoch must be a nonnegative integer")
    return int(value)


def _finite_tree(value: Any, label: str) -> None:
    """Admit restricted tensor/primitive state before publishing or applying it."""
    if isinstance(value, torch.Tensor):
        if value.layout != torch.strided or value.device.type == "meta":
            raise ValueError(f"{label} must contain materialized dense tensors")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{label} must contain finite tensors")
    elif isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, (str, int)):
                raise TypeError(f"{label} has an unsupported mapping key")
            _finite_tree(item, f"{label}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _finite_tree(item, f"{label}[{index}]")
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{label} must be finite")
    elif value is not None and not isinstance(value, (str, int, bool)):
        raise TypeError(f"{label} has unsupported checkpoint value {type(value).__name__}")


def _model_state(model: torch.nn.Module, state: Any) -> None:
    current = model.state_dict()
    if not isinstance(state, dict) or state.keys() != current.keys():
        raise ValueError("checkpoint model keys must match exactly")
    aliases: dict[tuple, str] = {}
    for name, target in current.items():
        value = state[name]
        if not isinstance(value, torch.Tensor) or not isinstance(target, torch.Tensor):
            raise TypeError("public checkpoints require tensor-valued model state")
        if value.shape != target.shape or value.dtype != target.dtype:
            raise ValueError(f"checkpoint model shape/dtype mismatch: {name}")
        _finite_tree(value, f"model.{name}")
        if target.layout != torch.strided or target.device.type == "meta":
            raise ValueError("target model must have materialized dense state")
        # The same registered parameter can occur under several module paths.
        # Conflicting aliases would otherwise silently become last-key-wins.
        identity = (target.device, target.untyped_storage().data_ptr(),
                    target.storage_offset(), tuple(target.shape), tuple(target.stride()), target.dtype)
        previous = aliases.get(identity)
        if previous is not None and not torch.equal(value.cpu(), state[previous].cpu()):
            raise ValueError(f"checkpoint model alias conflict: {previous} / {name}")
        aliases[identity] = name


def _optimizer_state(optimizer: torch.optim.Optimizer, state: Any) -> None:
    if type(optimizer) not in (torch.optim.Adam, torch.optim.AdamW, torch.optim.SGD):
        raise TypeError("public checkpoint resume supports Adam, AdamW, and SGD optimizers")
    if not isinstance(state, dict) or set(state) != {"state", "param_groups"}:
        raise ValueError("checkpoint optimizer state is malformed")
    groups = state["param_groups"]
    if not isinstance(groups, list) or len(groups) != len(optimizer.param_groups):
        raise ValueError("checkpoint optimizer parameter groups must match")
    parameters = {}
    groups_by_id = {}
    for saved, current in zip(groups, optimizer.param_groups):
        if not isinstance(saved, dict) or not isinstance(saved.get("params"), list):
            raise ValueError("checkpoint optimizer parameter group is malformed")
        if len(saved["params"]) != len(current["params"]):
            raise ValueError("checkpoint optimizer parameter groups must match")
        if saved.keys() != current.keys():
            raise ValueError("checkpoint optimizer group fields must match")
        for name in saved.keys() - {"params"}:
            _same_structure(saved[name], current[name], f"optimizer.{name}")
        for name in ("lr", "eps", "weight_decay", "momentum", "dampening"):
            if name in saved and (isinstance(saved[name], bool) or not isinstance(saved[name], (int, float))
                                  or saved[name] < 0):
                raise ValueError(f"checkpoint optimizer {name} must be nonnegative")
        if "betas" in saved and (len(saved["betas"]) != 2 or any(not 0 <= beta < 1 for beta in saved["betas"])):
            raise ValueError("checkpoint optimizer betas must be in [0,1)")
        for key, parameter in zip(saved["params"], current["params"]):
            if isinstance(key, bool) or not isinstance(key, int) or key in parameters:
                raise ValueError("checkpoint optimizer parameter identifiers must be unique integers")
            parameters[key] = parameter
            groups_by_id[key] = saved
    if not isinstance(state["state"], dict) or not state["state"].keys() <= parameters.keys():
        raise ValueError("checkpoint optimizer refers to an unknown parameter")
    _finite_tree(state, "optimizer")
    for key, values in state["state"].items():
        if not isinstance(values, dict):
            raise ValueError("checkpoint optimizer per-parameter state must be a mapping")
        if isinstance(optimizer, (torch.optim.Adam, torch.optim.AdamW)) and values:
            required = {"step", "exp_avg", "exp_avg_sq"}
            if groups_by_id[key].get("amsgrad", False):
                required.add("max_exp_avg_sq")
            if set(values) != required:
                raise ValueError("checkpoint Adam state must contain its complete moments and step")
            if not isinstance(values["step"], torch.Tensor):
                raise ValueError("checkpoint Adam step must be a scalar tensor")
            for name in required - {"step"}:
                value = values[name]
                if not isinstance(value, torch.Tensor) or value.dtype != parameters[key].dtype:
                    raise ValueError(f"checkpoint optimizer moment dtype mismatch: {name}")
                if name.endswith("avg_sq") and bool((value < 0).any()):
                    raise ValueError("checkpoint Adam squared moments must be nonnegative")
        elif isinstance(optimizer, torch.optim.SGD):
            if not set(values) <= {"momentum_buffer"}:
                raise ValueError("checkpoint SGD state has unsupported fields")
            if "momentum_buffer" in values:
                momentum = values["momentum_buffer"]
                if not isinstance(momentum, torch.Tensor) or momentum.dtype != parameters[key].dtype:
                    raise ValueError("checkpoint SGD momentum buffer must have the parameter tensor dtype")
        for name, value in values.items():
            if isinstance(value, torch.Tensor):
                if name == "step":
                    if value.ndim != 0 or value.item() < 0 or value.item() != int(value.item()):
                        raise ValueError("checkpoint optimizer step must be a nonnegative integer scalar")
                elif value.shape != parameters[key].shape:
                    raise ValueError(f"checkpoint optimizer tensor shape mismatch: {name}")
    # Native admission also catches optimizer-specific mandatory group fields.
    candidate = copy.deepcopy(optimizer)
    candidate.load_state_dict(copy.deepcopy(state))
    _finite_tree(candidate.state_dict(), "loaded optimizer")


def _same_structure(saved: Any, current: Any, label: str) -> None:
    """Native scheduler loaders update attributes without checking their types."""
    if type(saved) is not type(current):
        raise ValueError(f"checkpoint {label} has an incompatible value type")
    if isinstance(current, dict):
        if saved.keys() != current.keys():
            raise ValueError(f"checkpoint {label} has incompatible fields")
        for key in current:
            _same_structure(saved[key], current[key], f"{label}.{key}")
    elif isinstance(current, (tuple, list)):
        if len(saved) != len(current):
            raise ValueError(f"checkpoint {label} has incompatible length")
        for index, (left, right) in enumerate(zip(saved, current)):
            _same_structure(left, right, f"{label}[{index}]")


def _scheduler_snapshot(scheduler):
    """Preserve composite scheduler objects, not their serialized replacements."""
    snapshots = []
    seen = set()

    def visit(current):
        if id(current) in seen:
            return
        seen.add(id(current))
        fields = {}
        for key, value in vars(current).items():
            if isinstance(value, torch.optim.Optimizer):
                fields[key] = value
            elif isinstance(value, torch.optim.lr_scheduler.LRScheduler):
                fields[key] = value
                visit(value)
            elif isinstance(value, list) and any(isinstance(item, torch.optim.lr_scheduler.LRScheduler) for item in value):
                fields[key] = list(value)
                for child in value:
                    if isinstance(child, torch.optim.lr_scheduler.LRScheduler):
                        visit(child)
            else:
                fields[key] = copy.deepcopy(value)
        snapshots.append((current, fields))

    visit(scheduler)
    return snapshots


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    epoch: int = 0,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Replace the destination only after a complete, flushed serialization.

    This public helper is distinct from the frozen experiment checkpoint code.
    It stores model/optimizer/scheduler state; it does not capture RNG, EMA objects
    outside the model, data-loader cursors, or unregistered episode state.
    """
    payload: Dict[str, Any] = {"model": model.state_dict(), "epoch": _epoch(epoch)}
    _model_state(model, payload["model"])
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
        _optimizer_state(optimizer, payload["optimizer"])
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if extra is not None:
        payload["extra"] = extra
    _finite_tree(payload, "checkpoint")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w+b", dir=path.parent, prefix=f".{path.name}.",
                                         suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    map_location: str | torch.device = "cpu",
):
    """Validate every requested component before changing caller-owned state.

    Model keys, tensor shapes/dtypes and shared aliases are exact. Passing an
    optimizer or scheduler requests a resume of that component, so a missing
    component is an error. No unrestricted-pickle fallback is attempted.
    """
    payload = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(payload, dict) or "model" not in payload or "epoch" not in payload:
        raise ValueError("checkpoint requires model and epoch fields")
    _epoch(payload["epoch"])
    _finite_tree(payload, "checkpoint")
    _model_state(model, payload["model"])
    if optimizer is not None:
        if "optimizer" not in payload:
            raise ValueError("requested optimizer state is missing from checkpoint")
        _optimizer_state(optimizer, payload["optimizer"])
    if scheduler is not None:
        state = payload.get("scheduler")
        if not isinstance(state, dict) or state.keys() != scheduler.state_dict().keys():
            raise ValueError("requested scheduler state is missing or incompatible")
        _same_structure(state, scheduler.state_dict(), "scheduler")
        candidate = copy.deepcopy(scheduler)
        candidate.load_state_dict(copy.deepcopy(state))

    original_model = {name: value.detach().clone() for name, value in model.state_dict().items()}
    original_optimizer_state = None
    original_optimizer_groups = None
    if optimizer is not None:
        # Keep the original Parameter keys/objects; deepcopy(state_dict()) uses
        # integer identifiers and would require invoking load hooks on rollback.
        original_optimizer_state = copy.copy(optimizer.state)
        for parameter, state in optimizer.state.items():
            original_optimizer_state[parameter] = copy.deepcopy(state)
        original_optimizer_groups = [
            {**copy.deepcopy({key: value for key, value in group.items() if key != "params"}),
             "params": list(group["params"])}
            for group in optimizer.param_groups
        ]
    original_scheduler = _scheduler_snapshot(scheduler) if scheduler is not None else []
    try:
        model.load_state_dict(payload["model"], strict=True)
        if optimizer is not None:
            optimizer.load_state_dict(payload["optimizer"])
        if scheduler is not None:
            scheduler.load_state_dict(payload["scheduler"])
    except BaseException:
        # Restore tensor bytes directly so a failing model load hook is not
        # invoked a second time. Parameter objects and existing gradients remain.
        with torch.no_grad():
            for name, value in model.state_dict().items():
                value.copy_(original_model[name])
        if optimizer is not None:
            optimizer.state = original_optimizer_state
            optimizer.param_groups = original_optimizer_groups
        for original, fields in original_scheduler:
            original.__dict__.clear()
            original.__dict__.update(fields)
        raise
    return payload
