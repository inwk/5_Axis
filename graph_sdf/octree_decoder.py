"""Octree-based decoder for volumetric shape transition prediction.

Architecture overview
─────────────────────
Given the face-graph node embeddings and an action context, this decoder
predicts the occupancy of every octree leaf node produced during data
collection:

    1.  Scale-aware positional encoding
            (x, y, z, normalised_depth) → Fourier features [4*(1+2L)]
        The normalised_depth ∈ [0,1] encodes the octree level so the
        network understands the spatial scale of each query cell.

    2.  Input projection MLP → hidden_dim

    3.  Depth embedding  Embedding(max_depth+1, hidden_dim)
        Added to query features to give an explicit per-level bias.

    4.  Stack of _OctreeDecoderBlock × octree_decoder_layers
            a) Cross-attention  query → face-graph node_embeddings
               (each octree cell gathers information from nearby faces)
            b) FiLM conditioning from action_context
               (γ, β scale/shift based on the chosen action)
            c) FFN + residual

    5.  Output head  → 1 logit per octree node
        sigmoid(logit) = P(node centre is inside remaining material)

Inference
─────────
At inference time, call ``extract_mesh`` or ``extract_mesh_adaptive``.

``extract_mesh``:
    Evaluates decoder on a regular R³ grid (all at normalised_depth=1)
    then runs Marching Cubes.  Fast, uniform resolution.

``extract_mesh_adaptive``:
    Starts at coarse_depth, evaluates → refines boundary cells →
    repeats to fine_depth.  Fewer queries, surface-accurate.
    Returns the finest-depth cells as the Marching Cubes input.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import GraphSdfModelConfig


# ─────────────────────────────────────────────────────────────────────────────
# Scale-aware Fourier positional encoding
# ─────────────────────────────────────────────────────────────────────────────

class ScaleAwarePosEncoding(nn.Module):
    """Fourier encoding for (x, y, z, normalised_depth).

    The normalised_depth = depth / max_depth ∈ [0, 1] encodes the octree
    level (0 = root/coarse, 1 = leaf/fine).

    Output dimension: 4 * (1 + 2*L)
    For L=6: 4 * 13 = 52
    """

    def __init__(self, num_bands: int = 6) -> None:
        super().__init__()
        self.num_bands = num_bands
        freqs = 2.0 ** torch.arange(num_bands).float()   # [L]
        self.register_buffer("freqs", freqs)

    @property
    def output_dim(self) -> int:
        """Returns encoding output dimension."""
        return 4 * (1 + 2 * self.num_bands)

    def forward(self, xyzd: torch.Tensor) -> torch.Tensor:
        """Encodes [..., 4] → [..., 4*(1+2L)].

        Args:
            xyzd: Tensor with last dim = 4: (x, y, z, normalised_depth).
        """
        freqs = self.freqs                                # [L]
        # [..., 4, 1] * [L] → [..., 4, L]
        x = xyzd.unsqueeze(-1) * freqs * math.pi         # [..., 4, L]
        sincos = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)  # [..., 4, 2L]
        sincos = sincos.flatten(-2)                       # [..., 4*2L]
        return torch.cat([xyzd, sincos], dim=-1)          # [..., 4*(1+2L)]


# ─────────────────────────────────────────────────────────────────────────────
# Single residual decoder block: cross-attention + FiLM-FFN
# ─────────────────────────────────────────────────────────────────────────────

class _OctreeDecoderBlock(nn.Module):
    """Residual block: cross-attention over face-graph nodes + FiLM + FFN."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.cross_attn_norm = nn.LayerNorm(hidden_dim)
        self.node_norm       = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.film_norm      = nn.LayerNorm(hidden_dim)
        self.film_condition = nn.Linear(hidden_dim, hidden_dim * 2)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query_features: torch.Tensor,                          # [B, K, H]
        node_embeddings: torch.Tensor,                         # [B, N, H]
        action_context: torch.Tensor,                          # [B, H]
        node_key_padding_mask: Optional[torch.Tensor] = None,  # [B, N]
    ) -> torch.Tensor:
        # ── Cross-attention: octree nodes → face-graph nodes ──────────────
        q = self.cross_attn_norm(query_features)
        k = self.node_norm(node_embeddings)
        attended, _ = self.cross_attn(
            query=q, key=k, value=k,
            key_padding_mask=node_key_padding_mask,
            need_weights=False,
        )
        query_features = query_features + self.dropout(attended)

        # ── FiLM conditioning from action context ─────────────────────────
        gamma, beta = self.film_condition(action_context).chunk(2, dim=-1)  # [B, H]
        normed      = self.film_norm(query_features)                         # [B, K, H]
        conditioned = normed * (1.0 + gamma[:, None, :]) + beta[:, None, :]
        query_features = query_features + self.dropout(self.ffn(conditioned))
        return query_features


# ─────────────────────────────────────────────────────────────────────────────
# OctreeDecoder
# ─────────────────────────────────────────────────────────────────────────────

class OctreeDecoder(nn.Module):
    """Predicts occupancy at adaptive octree node positions.

    Inputs (forward):
        node_embeddings: [B, N, H]  – face-graph encoder output
        action_context:  [B, H]     – from ActionEmbedding
        octree_centers:  [B, K, 3]  – octree node centres (normalised coords)
        octree_depths:   [B, K]     – integer depth of each node (0=root)
        node_mask:       [B, N]     – True for padded face nodes (optional)

    Output:
        occ_logits: [B, K]  – raw logit; sigmoid → P(inside material after op)
    """

    def __init__(self, config: GraphSdfModelConfig) -> None:
        super().__init__()
        config.validate()
        hidden = config.hidden_dim
        self.max_depth = config.octree_fine_depth

        # ── Positional encoding ───────────────────────────────────────────
        self.pos_enc  = ScaleAwarePosEncoding(num_bands=config.octree_fourier_bands)
        pos_dim       = self.pos_enc.output_dim            # 4*(1+2L)

        # ── Input projection: pos_encoding → hidden ───────────────────────
        self.input_proj = nn.Sequential(
            nn.Linear(pos_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
        )

        # ── Depth embedding: one learnable vector per octree level ─────────
        self.depth_embedding = nn.Embedding(self.max_depth + 2, hidden)

        # ── Cross-attn + FiLM decoder blocks ─────────────────────────────
        self.decoder_blocks = nn.ModuleList([
            _OctreeDecoderBlock(
                hidden_dim=hidden,
                num_heads=config.octree_cross_attn_heads,
                dropout=config.octree_dropout,
            )
            for _ in range(config.octree_decoder_layers)
        ])

        # ── Output head ───────────────────────────────────────────────────
        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(
        self,
        node_embeddings: torch.Tensor,                         # [B, N, H]
        action_context:  torch.Tensor,                         # [B, H]
        octree_centers:  torch.Tensor,                         # [B, K, 3]
        octree_depths:   torch.Tensor,                         # [B, K]  int
        node_mask: Optional[torch.Tensor] = None,              # [B, N]
    ) -> torch.Tensor:
        """Returns occupancy logits [B, K]."""

        # Build (x, y, z, normalised_depth) ← all ∈ [-1,1] or [0,1]
        norm_depth = octree_depths.float() / float(max(self.max_depth, 1))  # [B, K]
        xyzd = torch.cat([octree_centers, norm_depth.unsqueeze(-1)], dim=-1)  # [B, K, 4]

        # Fourier-encode + project
        pos_encoded     = self.pos_enc(xyzd)                   # [B, K, pos_dim]
        query_features  = self.input_proj(pos_encoded)          # [B, K, H]

        # Add depth embedding
        depth_clamped = octree_depths.long().clamp(0, self.max_depth)
        query_features = query_features + self.depth_embedding(depth_clamped)  # [B, K, H]

        # Cross-attend to face-graph nodes with FiLM conditioning
        for block in self.decoder_blocks:
            query_features = block(
                query_features=query_features,
                node_embeddings=node_embeddings,
                action_context=action_context,
                node_key_padding_mask=node_mask,
            )

        return self.output_head(query_features).squeeze(-1)     # [B, K]

    # ── Mesh extraction helpers ────────────────────────────────────────────

    @torch.no_grad()
    def extract_mesh(
        self,
        node_embeddings: torch.Tensor,           # [1, N, H]
        action_context:  torch.Tensor,           # [1, H]
        node_mask: Optional[torch.Tensor] = None,
        grid_resolution: int = 64,
        occupancy_threshold: float = 0.5,
        bbox_min: tuple[float, float, float] = (-1.0, -1.0, -1.0),
        bbox_max: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ):
        """Evaluate on a regular R³ grid (depth=max_depth) → Marching Cubes.

        Returns (vertices [V,3], faces [F,3]) as CPU tensors, or None.
        """
        device = node_embeddings.device
        R = grid_resolution

        xs = torch.linspace(bbox_min[0], bbox_max[0], R, device=device)
        ys = torch.linspace(bbox_min[1], bbox_max[1], R, device=device)
        zs = torch.linspace(bbox_min[2], bbox_max[2], R, device=device)
        gx, gy, gz = torch.meshgrid(xs, ys, zs, indexing="ij")
        centers_flat = torch.stack([gx, gy, gz], dim=-1).reshape(1, -1, 3)   # [1, R³, 3]
        depths_flat  = torch.full((1, R * R * R), self.max_depth,
                                  dtype=torch.long, device=device)

        chunk, logit_parts = 16384, []
        for s in range(0, centers_flat.shape[1], chunk):
            c = centers_flat[:, s:s + chunk, :]
            d = depths_flat[:, s:s + chunk]
            logit_parts.append(self.forward(node_embeddings, action_context, c, d, node_mask))
        probs = torch.sigmoid(torch.cat(logit_parts, dim=1).squeeze(0))
        probs = probs.reshape(R, R, R).cpu().numpy()

        return _marching_cubes(probs, occupancy_threshold, R, bbox_min, bbox_max)

    @torch.no_grad()
    def extract_mesh_adaptive(
        self,
        node_embeddings: torch.Tensor,           # [1, N, H]
        action_context:  torch.Tensor,           # [1, H]
        node_mask: Optional[torch.Tensor] = None,
        coarse_depth: int = 2,
        occupancy_threshold: float = 0.5,
        boundary_lo: float = 0.25,
        boundary_hi: float = 0.75,
        bbox_min: tuple[float, float, float] = (-1.0, -1.0, -1.0),
        bbox_max: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ):
        """Adaptive octree inference: refine boundary cells to max_depth.

        At each level, cells whose predicted occupancy falls in
        [boundary_lo, boundary_hi] are subdivided into 8 children.
        Cells that are clearly full or empty become leaves.

        Returns (vertices [V,3], faces [F,3]) CPU tensors, or None.
        Requires skimage or mcubes.
        """
        import numpy as np
        device = node_embeddings.device

        bbox_min_t = torch.tensor(bbox_min, device=device, dtype=torch.float32)
        bbox_max_t = torch.tensor(bbox_max, device=device, dtype=torch.float32)

        # Start: uniform grid at coarse_depth
        n = 2 ** coarse_depth
        lin = [torch.linspace(bbox_min[i], bbox_max[i], n + 1, device=device)
               for i in range(3)]
        # Cell centres at coarse_depth
        cx = 0.5 * (lin[0][:-1] + lin[0][1:])
        cy = 0.5 * (lin[1][:-1] + lin[1][1:])
        cz = 0.5 * (lin[2][:-1] + lin[2][1:])
        gx, gy, gz = torch.meshgrid(cx, cy, cz, indexing="ij")
        active_centers = torch.stack([gx, gy, gz], dim=-1).reshape(1, -1, 3)   # [1, n³, 3]
        cell_size = (bbox_max_t - bbox_min_t) / n                               # [3]
        half_size = cell_size / 2                                                # [3]

        # All cells at this depth
        active_depths = torch.full((1, active_centers.shape[1]), coarse_depth,
                                   dtype=torch.long, device=device)

        finest_centers = []
        finest_probs   = []

        for depth in range(coarse_depth, self.max_depth + 1):
            if active_centers.shape[1] == 0:
                break

            logits = self.forward(node_embeddings, action_context,
                                  active_centers, active_depths, node_mask)  # [1, M]
            probs  = torch.sigmoid(logits.squeeze(0))                         # [M]

            if depth == self.max_depth:
                finest_centers.append(active_centers.squeeze(0))
                finest_probs.append(probs)
                break

            # Partition into leaf cells (clear inside/outside) vs boundary
            is_boundary = (probs > boundary_lo) & (probs < boundary_hi)   # [M]

            # Keep non-boundary cells as leaves
            leaf_idx = (~is_boundary).nonzero(as_tuple=True)[0]
            if len(leaf_idx) > 0:
                finest_centers.append(active_centers[0, leaf_idx])
                finest_probs.append(probs[leaf_idx])

            # Subdivide boundary cells
            bnd_idx = is_boundary.nonzero(as_tuple=True)[0]
            if len(bnd_idx) == 0:
                break

            parent_centers = active_centers[0, bnd_idx]   # [B2, 3]
            child_half = half_size / 2
            offsets = torch.tensor(
                [[dx, dy, dz] for dx in [-1, 1] for dy in [-1, 1] for dz in [-1, 1]],
                device=device, dtype=torch.float32,
            )  # [8, 3]
            # [B2, 8, 3] → [B2*8, 3]
            children = (parent_centers[:, None, :] + offsets[None, :, :] * child_half[None, None, :])
            children = children.reshape(1, -1, 3)
            active_centers = children
            active_depths  = torch.full((1, children.shape[1]), depth + 1,
                                        dtype=torch.long, device=device)
            half_size = child_half

        if not finest_centers:
            return None

        all_centers = torch.cat(finest_centers, dim=0).cpu().numpy()   # [total, 3]
        all_probs   = torch.cat(finest_probs, dim=0).cpu().numpy()      # [total]

        # Rasterise the finest-level cells onto a regular grid for Marching Cubes
        R = 2 ** self.max_depth
        grid = np.zeros((R, R, R), dtype=np.float32)
        bmin = np.array(bbox_min)
        bmax = np.array(bbox_max)
        extent = bmax - bmin
        # Map centres → voxel indices
        ijk = np.floor(((all_centers - bmin) / extent) * R).astype(int).clip(0, R - 1)
        for (i, j, k), p in zip(ijk, all_probs):
            grid[i, j, k] = p

        return _marching_cubes(grid, occupancy_threshold, R, bbox_min, bbox_max)


# ─────────────────────────────────────────────────────────────────────────────
# Shared Marching Cubes helper
# ─────────────────────────────────────────────────────────────────────────────

def _marching_cubes(
    probs,               # numpy [R, R, R] float32 in [0,1]
    threshold: float,
    R: int,
    bbox_min: tuple,
    bbox_max: tuple,
):
    """Run Marching Cubes and scale vertices to part-coordinate space."""
    import numpy as np

    try:
        from skimage.measure import marching_cubes          # type: ignore
        verts, faces, _, _ = marching_cubes(probs, level=threshold)
    except Exception:
        try:
            import mcubes                                   # type: ignore
            verts, faces = mcubes.marching_cubes(probs, threshold)
        except Exception:
            return None

    if verts is None or len(verts) == 0:
        return None

    # Scale grid indices → normalised part coordinates
    scale  = torch.tensor(
        [(bbox_max[i] - bbox_min[i]) / (R - 1) for i in range(3)],
        dtype=torch.float32,
    )
    offset = torch.tensor(bbox_min, dtype=torch.float32)
    verts_t = torch.from_numpy(np.asarray(verts, dtype=np.float32)) * scale + offset
    faces_t = torch.from_numpy(np.asarray(faces).astype("int64"))
    return verts_t, faces_t
