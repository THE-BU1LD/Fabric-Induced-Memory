"""Noisy, drifting continuous event streams. Not a physical PDE simulator."""

import random

import torch


def generate(protocol, condition, seed, split):
    cfg = protocol["data"]
    split_index = {"train": 1, "validation": 2, "evaluation": 3}[split]
    g = torch.Generator().manual_seed(seed * 1009 + split_index * 100003)
    # Distinct streams; cue dictionary shared across splits, innovations are not.
    common = torch.Generator().manual_seed(seed)
    count = cfg[
        {
            "train": "train_owners",
            "validation": "validation_owners",
            "evaluation": "evaluation_owners",
        }[split]
    ]
    T, K, D = cfg["steps"], condition["anchors"], cfg["value_dim"]
    anchors = torch.randn(K, cfg["cue_dim"], generator=common)
    anchors = anchors / anchors.square().mean(-1, keepdim=True).sqrt()
    mixing = torch.randn(cfg["cue_dim"], D, generator=common) / cfg["cue_dim"] ** 0.5
    cue = torch.empty(count, T, cfg["cue_dim"])
    target = torch.empty(count, T, D)
    labels = torch.empty(count, T, dtype=torch.long)
    previous = torch.full((count, T), -1, dtype=torch.long)
    # Opaque IDs use a separate RNG and encode no latent regime or outcome.
    ids = random.Random(seed * 37 + split_index).sample(
        range(split_index * 10**8, (split_index + 1) * 10**8), count
    )
    for owner in range(count):
        residuals = torch.randn(K, D, generator=g) * cfg["residual_scale"]
        seen = {}
        for t in range(T):
            # Stationary AR(1) innovations; noise scales are explicitly synthetic.
            drift = condition["drift"]
            residuals = (1 - drift**2) ** 0.5 * residuals + drift * cfg[
                "residual_scale"
            ] * torch.randn(K, D, generator=g)
            distractor = bool(torch.rand((), generator=g) < condition["distractors"])
            key = int(torch.randint(K, (), generator=g))
            if distractor:
                key = K + t  # a truly one-off cue, never a repeated distractor ID
                clean = torch.randn(cfg["cue_dim"], generator=g)
                residual = torch.randn(D, generator=g) * cfg["residual_scale"]
            else:
                clean, residual = anchors[key], residuals[key]
            cue[owner, t] = clean + condition["cue_noise"] * torch.randn(
                cfg["cue_dim"], generator=g
            )
            target[owner, t] = (
                torch.tanh(clean @ mixing)
                + residual
                + cfg["observation_noise"] * torch.randn(D, generator=g)
            )
            labels[owner, t] = key
            previous[owner, t] = seen.get(key, -1)
            seen[key] = t
    times = torch.arange(T)[None, :]
    primary = (
        (previous >= 0)
        & (times - previous > protocol["training"]["history"])
        & (times >= cfg["warmup"])
    )
    return {
        "cue": cue,
        "target": target,
        "anchor": labels,
        "previous": previous,
        "primary": primary,
        "owners": ids,
        "split": split,
    }


def causal_features(cue, residual):
    """At t, target[t] is unavailable. Only y[t-1] enters the input."""
    zeros_c = torch.zeros_like(cue[:, :1])
    zeros_v = torch.zeros_like(residual[:, :1])
    return torch.cat(
        (
            cue,
            torch.cat((zeros_c, cue[:, :-1]), 1),
            torch.cat((zeros_v, residual[:, :-1]), 1),
        ),
        dim=-1,
    )
