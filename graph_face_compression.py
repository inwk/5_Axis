"""Graph reduction utilities for face-level CAM graphs."""

from __future__ import annotations

import numpy as np
import networkx as nx
from typing import List, Dict, Tuple, Optional

__all__ = [
    "compress_graph_by_area",
    "farthest_point_sampling",
    "reduce_visibility_by_groups",
]


# ---------------------------- helpers ----------------------------


def farthest_point_sampling(points: np.ndarray, k: int) -> np.ndarray:
    """
    Simple Farthest Point Sampling (FPS) for (N,3) array.
    If N<=k, returns the original points (copy).

    Parameters
    ----------
    points : np.ndarray
        Shape (N, 3).
    k : int
        Number of points to sample.

    Returns
    -------
    np.ndarray
        Shape (min(N, k), 3).
    """
    N = len(points)
    if N == 0:
        return points
    if N <= k:
        return points.copy()

    idxs = np.zeros(k, dtype=int)
    # pick a random initial point
    idxs[0] = np.random.randint(0, N)
    dists = np.full(N, np.inf)
    last = points[idxs[0]]
    dists = np.minimum(dists, np.linalg.norm(points - last, axis=1))

    for i in range(1, k):
        idxs[i] = int(np.argmax(dists))
        last = points[idxs[i]]
        dists = np.minimum(dists, np.linalg.norm(points - last, axis=1))

    return points[idxs]


def _downsample_points_concat(point_list: List[np.ndarray], target: int = 100) -> np.ndarray:
    """
    Concatenate multiple (Mi,3) point arrays and downsample to `target` points using FPS.
    Pads with zeros if needed.

    Parameters
    ----------
    point_list : List[np.ndarray]
        List of arrays, each (Mi, 3).
    target : int
        Target number of points.

    Returns
    -------
    np.ndarray
        (target, 3) float32.
    """
    if len(point_list) == 0:
        return np.zeros((target, 3), dtype=np.float32)
    pts = np.concatenate(point_list, axis=0)  # (sum Mi, 3)
    pts = farthest_point_sampling(pts, target).astype(np.float32)
    if len(pts) < target:
        pad = np.zeros((target - len(pts), 3), dtype=np.float32)
        pts = np.vstack([pts, pad])
    return pts


def reduce_visibility_by_groups(
    visible_orig: np.ndarray,  # shape (N,), {0,1}
    groups: List[List[int]],
) -> np.ndarray:
    """
    Aggregate original per-face visibility into grouped-node visibility (logical OR).

    Parameters
    ----------
    visible_orig : np.ndarray
        Shape (N,), binary 0/1 per original face.
    groups : List[List[int]]
        Each group contains the indices of original nodes merged into that group.

    Returns
    -------
    np.ndarray
        Shape (K,), binary 0/1 per reduced node.
    """
    K = len(groups)
    out = np.zeros(K, dtype=int)
    for k, idxs in enumerate(groups):
        vals = visible_orig[idxs]
        out[k] = 1 if np.any(vals) else 0
    return out


# ---------------------------- main API ----------------------------

def compress_graph_by_area(
    g: nx.Graph,
    face_areas: np.ndarray,           # shape (N,)
    face_points: List[np.ndarray],    # len=N, each (Pi,3)
    face_visible: Optional[np.ndarray] = None,  # shape (N,), {0,1}
    max_nodes: int = 512,
    area_threshold: Optional[float] = None,
    target_points_per_node: int = 100,
    seed: int = 42,
) -> Tuple[
    nx.Graph,
    np.ndarray,        # areas_new  (K,)
    np.ndarray,        # points_new (K, target_points_per_node, 3)
    np.ndarray,        # visible_new (K,)
    Dict[int, int],    # old2new mapping
    List[List[int]],   # groups (list of original node lists)
]:
    """
    Compress an over-sized face graph into <= max_nodes by merging the smallest faces
    with neighbors. Updates edges and aggregates embeddings:

    - area: sum
    - point cloud: concatenate all member clouds then FPS to target points
    - visibility: OR

    The algorithm keeps merging while (#nodes > max_nodes) or (min_area < area_threshold).

    Parameters
    ----------
    g : nx.Graph
        Original graph with N nodes. Node ids should be arbitrary; they will be
        relabeled to 0..N-1 internally.
    face_areas : np.ndarray
        (N,) float area per face.
    face_points : List[np.ndarray]
        List of (Pi,3) point clouds per face.
    face_visible : Optional[np.ndarray]
        (N,) binary 0/1 visibility per face (default zeros).
    max_nodes : int
        Target maximum number of nodes after compression.
    area_threshold : Optional[float]
        If provided, continue merging while the minimum area is below this threshold.
    target_points_per_node : int
        Number of points to keep in each reduced node via FPS.
    seed : int
        Random seed for deterministic FPS initialization.

    Returns
    -------
    G_new : nx.Graph
        Reduced graph with K nodes.
    areas_new : np.ndarray
        (K,)
    points_new : np.ndarray
        (K, target_points_per_node, 3)
    visible_new : np.ndarray
        (K,)
    old2new : Dict[int,int]
        Mapping from old (0..N-1 internal) node index to new (0..K-1) node index.
    groups : List[List[int]]
        For each new node k, the list of original node indices merged into k (0..N-1 internal).
    """
    np.random.seed(seed)

    N = g.number_of_nodes()
    if face_visible is None:
        face_visible = np.zeros(N, dtype=int)

    # Sanity checks
    assert N == len(face_areas) == len(face_points) == len(face_visible), \
        "Input sizes mismatch"

    # Relabel nodes to dense range 0..N-1 (internal indexing)
    nodes_sorted = sorted(g.nodes())
    mapping_to_dense = {old: i for i, old in enumerate(nodes_sorted)}
    g = nx.relabel_nodes(g, mapping_to_dense, copy=True)

    # Initialize per-group attributes
    groups: Dict[int, List[int]] = {i: [i] for i in range(N)}
    grp_area = {i: float(face_areas[i]) for i in range(N)}
    grp_points = {i: [face_points[i].astype(np.float32)] for i in range(N)}
    grp_visible = {i: int(face_visible[i]) for i in range(N)}

    alive = set(range(N))

    def pick_smallest_group() -> int:
        """Returns the currently alive group with the smallest area."""
        return min(alive, key=lambda k: grp_area[k])

    def pick_merge_target(u: int) -> Optional[int]:
        """Returns the smallest-area alive neighbor for a given group id."""
        neigh = [v for v in g.neighbors(u) if v in alive and v != u]
        if not neigh:
            return None
        return min(neigh, key=lambda v: grp_area[v])

    def merge_u_into_v(u: int, v: int):
        """Merges source group ``u`` into target group ``v`` and rewires edges."""
        groups[v].extend(groups[u])
        grp_area[v] += grp_area[u]
        grp_points[v].extend(grp_points[u])
        grp_visible[v] = int(bool(grp_visible[v] or grp_visible[u]))

        # Rewire edges: neighbors of u now connect to v
        for w in list(g.neighbors(u)):
            if w == v:
                continue
            g.add_edge(v, w)

        # Mark u as removed; avoid self loops
        alive.discard(u)
        if g.has_edge(v, v):
            g.remove_edge(v, v)

    def need_more_merge() -> bool:
        """Returns True while node count/area constraints require more merges."""
        if len(alive) > max_nodes:
            return True
        if area_threshold is not None:
            min_area = min(grp_area[i] for i in alive)
            if min_area < area_threshold:
                return True
        return False

    # Merge loop
    while need_more_merge():
        if len(alive) == 1:
            break
        u = pick_smallest_group()
        v = pick_merge_target(u)
        if v is None:
            # No neighbors: attach u to the smallest other alive node
            candidates = [x for x in alive if x != u]
            if not candidates:
                break
            v = min(candidates, key=lambda x: grp_area[x])
        # Merge the smaller u into v
        merge_u_into_v(u, v)

    # Build mapping and groups output
    alive_list = sorted(alive)
    new_index_of_alive = {old: i for i, old in enumerate(alive_list)}
    old2new: Dict[int, int] = {}
    final_groups: List[List[int]] = []
    for old_v in alive_list:
        final_groups.append(groups[old_v])               # internal 0..N-1 indices
        for old_node in groups[old_v]:
            old2new[old_node] = new_index_of_alive[old_v]

    K = len(alive_list)

    # New graph: connect groups if any original edge crossed groups
    G_new = nx.Graph()
    G_new.add_nodes_from(range(K))
    for (u, v) in g.edges():
        if u not in alive or v not in alive:
            continue
        u2 = new_index_of_alive[u]
        v2 = new_index_of_alive[v]
        if u2 != v2:
            G_new.add_edge(u2, v2)

    # Aggregate features
    areas_new = np.zeros(K, dtype=np.float32)
    points_new = np.zeros((K, target_points_per_node, 3), dtype=np.float32)
    visible_new = np.zeros(K, dtype=np.int64)

    for old_v in alive_list:
        k = new_index_of_alive[old_v]
        areas_new[k] = float(grp_area[old_v])
        points_new[k] = _downsample_points_concat(grp_points[old_v], target=target_points_per_node)
        visible_new[k] = int(grp_visible[old_v])

    return G_new, areas_new, points_new, visible_new, old2new, final_groups, nodes_sorted
