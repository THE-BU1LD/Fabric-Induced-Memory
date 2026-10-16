import torch
from torch import nn


class Base(nn.Module):
    def __init__(self, cue_dim, value_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cue_dim, 64), nn.Tanh(), nn.Linear(64, value_dim)
        )

    def forward(self, x):
        return self.net(x)


class SequenceModel(nn.Module):
    def __init__(self, kind, input_dim, value_dim, width, history):
        super().__init__()
        self.kind, self.history = kind, history
        if kind == "finite_history":
            self.net = nn.Sequential(
                nn.Linear(input_dim * history, width),
                nn.GELU(),
                nn.Linear(width, value_dim),
            )
        elif kind == "gru":
            self.net = nn.GRU(input_dim, width, batch_first=True)
            self.head = nn.Linear(width, value_dim)
        elif kind == "attention":
            self.embed = nn.Linear(input_dim, width)
            self.layer = nn.TransformerEncoderLayer(
                width, 4, width * 2, dropout=0.0, batch_first=True
            )
            self.head = nn.Linear(width, value_dim)
        else:
            raise ValueError(kind)

    def forward(self, x):
        if self.kind == "finite_history":
            padded = torch.nn.functional.pad(x, (0, 0, self.history - 1, 0))
            # unfold orders dimensions B,T,F,H; flatten preserves fixed feature-time ordering.
            windows = padded.unfold(1, self.history, 1).flatten(2)
            return self.net(windows)
        if self.kind == "gru":
            return self.head(self.net(x)[0])
        T = x.shape[1]
        positions = torch.arange(T, device=x.device)
        delta = positions[:, None] - positions[None, :]
        mask = (delta < 0) | (delta >= self.history)
        z = self.embed(x)
        # Deterministic sinusoidal position features, no target or owner metadata.
        frequencies = torch.exp(
            torch.arange(0, z.shape[-1], 2, device=x.device)
            * (-9.210340372 / z.shape[-1])
        )
        pe = torch.zeros_like(z[0])
        pe[:, 0::2] = torch.sin(positions[:, None] * frequencies)
        pe[:, 1::2] = torch.cos(positions[:, None] * frequencies)
        return self.head(self.layer(z + pe, src_mask=mask))


def retrieve(
    cues,
    residuals,
    capacity,
    topk,
    temperature,
    minimum_age=1,
    maximum_age=None,
    mode="tensor",
    labels=None,
):
    """Independent per-owner FIFO banks. Read only indices < t, then observe target.

    Source indices are retained for independent ownership/time/error verification.
    Labels are prohibited except in the privileged diagnostic oracle.
    """
    if labels is not None and mode != "oracle":
        raise ValueError("anchor metadata may only enter the diagnostic oracle")
    if mode == "oracle" and labels is None:
        raise ValueError("oracle requires evaluator-side anchor IDs")
    if minimum_age < 1 or (maximum_age is not None and maximum_age < minimum_age):
        raise ValueError("empty or noncausal age interval")
    B, T, D = residuals.shape
    output = torch.zeros_like(residuals)
    sources = torch.full((B, T, topk), -1, dtype=torch.long)
    donor = torch.zeros_like(residuals)
    permuted = torch.zeros(B, T, dtype=torch.bool)
    for t in range(T):
        begin = max(0, t - capacity)
        if maximum_age is not None:
            begin = max(begin, t - maximum_age)
        end = t - minimum_age + 1
        if begin >= end:
            continue
        times = torch.arange(begin, end)
        distances = (cues[:, t, None] - cues[:, begin:end]).square().mean(-1)
        if mode == "oracle":
            match = labels[:, begin:end] == labels[:, t, None]
            distances = torch.where(match, -times.float().expand(B, -1), float("inf"))
        elif mode == "last_value":
            distances = -times.float().expand(B, -1)
        # Stable sort is deterministic: equal distances use earlier FIFO position.
        k = min(topk, len(times))
        chosen = torch.argsort(distances, dim=-1, stable=True)[:, :k]
        selected = distances.gather(1, chosen)
        valid = torch.isfinite(selected[:, 0])
        selected = torch.where(valid[:, None], selected, torch.zeros_like(selected))
        weights = (
            torch.ones_like(selected) / k
            if mode == "knn4"
            else torch.softmax(-selected / temperature, dim=-1)
        )
        absolute = chosen + begin
        values = residuals.gather(1, absolute[:, :, None].expand(-1, -1, D))
        correction = (weights[:, :, None] * values).sum(1)
        if mode == "value_permutation":
            # Rotate VALUES only within the same eligible owner bank. Keys/scores unchanged.
            donor_absolute = (chosen + 1) % len(times) + begin
            donor_values = residuals.gather(
                1, donor_absolute[:, :, None].expand(-1, -1, D)
            )
            correction = (weights[:, :, None] * donor_values).sum(1)
            donor[:, t] = correction
            permuted[:, t] = valid & (len(times) > 1)
        elif mode == "spatial_mean":
            correction = correction.mean(-1, keepdim=True).expand_as(correction)
        elif mode == "sign_flip":
            correction = -correction
        elif mode == "erase":
            correction = torch.zeros_like(correction)
        output[:, t] = torch.where(
            valid[:, None], correction, torch.zeros_like(correction)
        )
        sources[:, t, :k] = torch.where(valid[:, None], absolute, -1)
    return {
        "correction": output,
        "sources": sources,
        "donor": donor,
        "permuted": permuted,
    }
