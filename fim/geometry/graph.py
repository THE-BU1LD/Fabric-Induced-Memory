import torch
import torch.nn as nn

class GridGraph:
    """
    Converts HxW grid into graph adjacency (4-neighbor or 8-neighbor)
    """

    def __init__(self, H, W, connectivity=4):
        if H < 1 or W < 1:
            raise ValueError("grid dimensions must be positive")
        if connectivity not in (4, 8):
            raise ValueError("connectivity must be 4 or 8")
        self.H = H
        self.W = W
        self.N = H * W
        self.connectivity = connectivity

        self.edge_index = self._build_edges()

    def _node_id(self, i, j):
        return i * self.W + j

    def _build_edges(self):
        edges = []

        for i in range(self.H):
            for j in range(self.W):
                u = self._node_id(i, j)

                neighbors = [
                    (i-1, j), (i+1, j),
                    (i, j-1), (i, j+1)
                ]

                if self.connectivity == 8:
                    neighbors += [
                        (i-1, j-1), (i-1, j+1),
                        (i+1, j-1), (i+1, j+1)
                    ]

                for ni, nj in neighbors:
                    if 0 <= ni < self.H and 0 <= nj < self.W:
                        v = self._node_id(ni, nj)
                        edges.append((u, v))

        # Keep the [2, E] contract even for a single node with no neighbors.
        edge_index = torch.tensor(edges, dtype=torch.long).reshape(-1, 2).t().contiguous()
        return edge_index


class GraphPropagation(nn.Module):
    """
    Message passing over graph
    replaces convolution-based diffusion
    """

    def __init__(self, channels, connectivity=4):
        super().__init__()
        if connectivity not in (4, 8):
            raise ValueError("connectivity must be 4 or 8")
        self.connectivity = connectivity
        self.lin = nn.Linear(channels, channels)
        self.register_buffer("_grid_edges", torch.empty((2, 0), dtype=torch.long), persistent=False)
        self._grid_shape = None

    def forward(self, F, edge_index=None):
        """
        F: [B, C, H, W]
        edge_index: optional [2, E] long tensor. If omitted, derive a grid
        topology from F for the single-argument FIMModel propagation hook.
        Incoming messages are summed in F's dtype, including repeated edges.
        """

        if F.ndim != 4:
            raise ValueError("F must have shape [B, C, H, W]")
        B, C, H, W = F.shape
        if C != self.lin.in_features:
            raise ValueError("F channel count must match propagation channels")
        if H < 1 or W < 1:
            raise ValueError("F spatial dimensions must be positive")
        N = H * W

        if edge_index is None:
            grid_shape = (H, W, self.connectivity)
            if (
                self._grid_shape != grid_shape
                or self._grid_edges.device != F.device
                or torch.is_inference(self._grid_edges)
            ):
                # Cached indices outlive this call and may be saved by a later
                # autograd pass, even when their first use is inference-only.
                with torch.inference_mode(False):
                    self._grid_edges = GridGraph(H, W, self.connectivity).edge_index.to(F.device)
                self._grid_shape = grid_shape
            edge_index = self._grid_edges
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, E]")
        if edge_index.dtype != torch.long:
            raise TypeError("edge_index must have dtype torch.long")
        if edge_index.numel() and (edge_index.min().item() < 0 or edge_index.max().item() >= N):
            raise ValueError("edge_index contains an out-of-range node")
        edge_index = edge_index.to(F.device)

        x = F.reshape(B, C, N).permute(0, 2, 1)  # [B, N, C]

        src, dst = edge_index
        messages = self.lin(x[:, src])  # [B, E, C]

        out = torch.zeros_like(x)
        # Advanced-index assignment loses contributions for repeated dst ids.
        # Autocast may change the linear output's dtype; index_add_ requires
        # equal dtypes, so retain the field dtype for accumulation and output.
        out.index_add_(1, dst, messages.to(dtype=out.dtype))

        return out.permute(0, 2, 1).reshape(B, C, H, W)
