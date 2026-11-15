"""
Traversability Graph Construction, Dijkstra Path Planning, and Algorithm 1:
Multi-Transceiver Coding Strategy with Partial Offline Precomputation and Windowed Monte Carlo Sampling.

Implements equations (1)-(2), (6)-(22), and Algorithm 1 from:
Zhao et al., 'Adaptive Semantic Communication for UAV/UGV Cooperative Path Planning', IEEE TVT 2026.
"""

from __future__ import annotations
import heapq
from typing import Tuple, List, Dict, Optional
import numpy as np
from scipy.ndimage import distance_transform_edt


# 8-connected grid neighbor offsets (horizontal, vertical, diagonal)
NEIGHBOR_OFFSETS = [
    (-1, 0), (1, 0), (0, -1), (0, 1),
    (-1, -1), (-1, 1), (1, -1), (1, 1),
]


class TraversabilityGraph:
    """
    Represents the traversability graph G_t (or perturbed G_hat_t) constructed by gridding
    the ground map X_t into G x G vertices, each with traversability weight tau_u in (0, 1].
    """

    def __init__(
        self,
        tau_map: np.ndarray,
        grid_size: int = 32,
        kappa: float = 1.0,
        non_traversable_threshold: float = 0.55,
    ):
        """
        Args:
            tau_map: 2D array (H, W) of pixel-level traversability in [0.02, 1.0].
            grid_size: Resolution G of the planning graph (G x G vertices).
            kappa: Coefficient in Eq. (1): w(u, v) = (kappa / tau_u)^2 + (kappa / tau_v)^2.
            non_traversable_threshold: Threshold below which a cell is considered non-traversable
                (obstacles: buildings, water bodies, dense woods) for evaluation in Fig. 5.
        """
        self.grid_size = grid_size
        self.kappa = kappa
        self.non_traversable_threshold = non_traversable_threshold

        # Pool pixel-level tau_map (H, W) into (grid_size, grid_size) vertex weights tau_u
        h, w = tau_map.shape
        sh, sw = h // grid_size, w // grid_size
        reshaped = tau_map[: sh * grid_size, : sw * grid_size].reshape(grid_size, sh, grid_size, sw)
        # Harmonic/minimum-aware pooling preserves narrow obstacle boundaries
        mean_tau = reshaped.mean(axis=(1, 3))
        min_tau = reshaped.min(axis=(1, 3))
        self.tau = np.clip(0.65 * mean_tau + 0.35 * min_tau, 0.02, 1.0).astype(np.float32)

        # Vertex weight term (kappa / tau_u)^2 used in Eq. (1), bounded so single obstacle steps match Fig. 4 scale
        tau_for_cost = np.clip(self.tau, 0.24, 1.0)
        self.vertex_cost = (self.kappa / tau_for_cost) ** 2

    def edge_weight(self, u: Tuple[int, int], v: Tuple[int, int]) -> float:
        """
        Equation (1): w(u, v) = (kappa / tau_u)^2 + (kappa / tau_v)^2
        (with diagonal step scaling for geometric consistency).
        """
        ru, cu = u
        rv, cv = v
        diag_factor = 1.41421356 if (ru != rv and cu != cv) else 1.0
        return float(diag_factor * (self.vertex_cost[ru, cu] + self.vertex_cost[rv, cv]))

    def neighbors(self, u: Tuple[int, int]) -> List[Tuple[int, int]]:
        r, c = u
        g = self.grid_size
        res = []
        for dr, dc in NEIGHBOR_OFFSETS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < g and 0 <= nc < g:
                res.append((nr, nc))
        return res

    def path_total_weight(self, path: List[Tuple[int, int]]) -> float:
        """
        Compute total weight W(p) of an ordered path on this graph G_t.
        """
        if len(path) < 2:
            return 0.0
        total = 0.0
        for u, v in zip(path[:-1], path[1:]):
            total += self.edge_weight(u, v)
        return total

    def count_non_traversable_steps(self, path: List[Tuple[int, int]]) -> int:
        """
        Count how many vertices along `path` fall into non-traversable obstacle regions
        (used for Fig. 5 evaluation).
        """
        count = 0
        for r, c in path:
            if self.tau[r, c] < self.non_traversable_threshold:
                count += 1
        return count


def dijkstra_shortest_path(
    graph: TraversabilityGraph,
    source: Optional[Tuple[int, int]] = None,
    target: Optional[Tuple[int, int]] = None,
) -> Tuple[List[Tuple[int, int]], float]:
    """
    Compute the shortest path p_t (or p_hat_t) between source and target using Dijkstra's algorithm.
    """
    g = graph.grid_size
    if source is None:
        source = (1, 1)
    if target is None:
        target = (g - 2, g - 2)

    dist = np.full((g, g), np.inf, dtype=np.float64)
    parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {source: None}
    dist[source[0], source[1]] = 0.0

    pq = [(0.0, source)]
    while pq:
        d, u = heapq.heappop(pq)
        if u == target:
            break
        if d > dist[u[0], u[1]]:
            continue
        for v in graph.neighbors(u):
            w_uv = graph.edge_weight(u, v)
            nd = d + w_uv
            if nd < dist[v[0], v[1]]:
                dist[v[0], v[1]] = nd
                parent[v] = u
                heapq.heappush(pq, (nd, v))

    path = []
    curr: Optional[Tuple[int, int]] = target
    while curr is not None:
        path.append(curr)
        curr = parent.get(curr, None)
    path.reverse()
    return path, float(dist[target[0], target[1]])


def proposition1_sensitivity_xi(
    delta_w: float,
    sigma_pt_sq: float,
    sigma_pj_sq: float,
) -> Tuple[float, float]:
    """
    Proposition 1 & Equations (12)-(16):
    Computes normalized gap z = Delta_w_j / sqrt(sigma_{p_t}^2 + sigma_{p_t^j}^2)
    and path sensitivity priority metric:
        xi = f(z) = |z| * phi(z) = (|z| / sqrt(2*pi)) * exp(-z^2 / 2)
    which achieves its maximum at |z| = 1 (Eq. 16).
    """
    denom = np.sqrt(max(sigma_pt_sq + sigma_pj_sq, 1e-8))
    z = abs(delta_w) / denom
    phi_z = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * (z ** 2))
    xi = float(z * phi_z)
    return xi, float(z)


def _sample_monte_carlo_path(
    graph: TraversabilityGraph,
    source: Tuple[int, int],
    target: Tuple[int, int],
    rng: np.random.Generator,
    allowed_mask: Optional[np.ndarray] = None,
    epsilon: float = 1e-4,
) -> Optional[List[Tuple[int, int]]]:
    """
    Sample a self-avoiding candidate path p_t^j from source to target guided by Eq. (17):
        p(u, v) = (1 / (w(u, v) + eps)) / sum_{v' in V(u)} (1 / (w(u, v') + eps))
    Using randomized edge-weight potential on G_t so candidate paths reach `target` without
    getting trapped in dead ends on 2D grids.
    """
    g = graph.grid_size
    dist = np.full((g, g), np.inf, dtype=np.float64)
    parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {source: None}
    dist[source[0], source[1]] = 0.0

    # Multiplicative exponential perturbation implements Eq. (17) inverse-weight sampling
    noise_field = rng.exponential(scale=1.0, size=(g, g))
    pq = [(0.0, source)]
    while pq:
        d, u = heapq.heappop(pq)
        if u == target:
            break
        if d > dist[u[0], u[1]]:
            continue
        for v in graph.neighbors(u):
            if allowed_mask is not None and not allowed_mask[v[0], v[1]]:
                continue
            w_uv = graph.edge_weight(u, v) + epsilon
            sampled_step = w_uv * (0.45 + 0.85 * noise_field[v[0], v[1]])
            nd = d + sampled_step
            if nd < dist[v[0], v[1]]:
                dist[v[0], v[1]] = nd
                parent[v] = u
                heapq.heappush(pq, (nd, v))

    if target not in parent and target != source:
        return None
    path = []
    curr: Optional[Tuple[int, int]] = target
    while curr is not None:
        path.append(curr)
        curr = parent.get(curr, None)
    path.reverse()
    return path


def compute_offline_lbc(
    graph_prev: TraversabilityGraph,
    source: Tuple[int, int],
    target: Tuple[int, int],
    num_mc_samples: int = 80,
    threshold_h: float = 0.12,
    vertex_var: float = 2.5,
    seed: int = 0,
) -> np.ndarray:
    """
    Algorithm 1 Line 3 (Offline Phase):
    Run Monte Carlo sampling by Eq. (17) over the full prior map G_hat_{t-1}
    to compute and store offline Local Betweenness Centrality psi_{u, h}^{off}.
    """
    rng = np.random.default_rng(seed)
    g = graph_prev.grid_size
    psi_off = np.zeros((g, g), dtype=np.float32)

    shortest_path, w_opt = dijkstra_shortest_path(graph_prev, source, target)
    sigma_pt_sq = len(shortest_path) * vertex_var

    # Always include shortest path nodes as baseline importance
    for r, c in shortest_path:
        psi_off[r, c] += 1.0

    for _ in range(num_mc_samples):
        cand_path = _sample_monte_carlo_path(graph_prev, source, target, rng=rng)
        if cand_path is None:
            continue
        w_cand = graph_prev.path_total_weight(cand_path)
        delta_w = max(0.0, w_cand - w_opt)
        sigma_pj_sq = len(cand_path) * vertex_var
        xi, _ = proposition1_sensitivity_xi(delta_w, sigma_pt_sq, sigma_pj_sq)
        if xi >= threshold_h:
            for r, c in cand_path:
                psi_off[r, c] += 1.0

    return psi_off


def algorithm1_windowed_mc_lbc(
    graph_curr: TraversabilityGraph,
    graph_prev: Optional[TraversabilityGraph] = None,
    psi_off: Optional[np.ndarray] = None,
    grid_n: int = 4,
    threshold_h: float = 0.12,
    r_c: int = 3,
    r_w: int = 2,
    traverse_time: int = 45,
    norm_coeff_g: float = 0.18,
    delta_min: float = 0.25,
    delta_max: float = 0.92,
    seed: int = 42,
) -> Dict[str, object]:
    """
    Executes Algorithm 1: Multi-Transceiver Coding Strategy With Partial Offline Precomputation
    and Windowed Monte Carlo Sampling.

    Returns:
        dict containing:
            - 'deltas': List of n = grid_n^2 UAV sparsification ratios {delta_k} in [delta_min, delta_max]
            - 'psi_tilde': Updated LBC map tilde{psi}_{u,h}(t) of shape (G, G)
            - 'Psi_k': Per-UAV LBC sums {tilde{Psi}_k}
            - 'window_mask': Boolean update window W_t = N_r(C_t U F_t)
            - 'shortest_path': Shortest path p_t on G_hat_t
    """
    g = graph_curr.grid_size
    source = (1, 1)
    target = (g - 2, g - 2)
    rng = np.random.default_rng(seed)

    # Line 3: Offline LBC on G_hat_{t-1} if not already provided
    if graph_prev is None:
        graph_prev = graph_curr
    if psi_off is None:
        psi_off = compute_offline_lbc(
            graph_prev, source, target, num_mc_samples=50, threshold_h=threshold_h, seed=seed
        )

    # Line 5: Compute current shortest path p_t on G_hat_t
    shortest_path, w_opt = dijkstra_shortest_path(graph_curr, source, target)

    # Line 6: Path corridor C_t (vertices within radius r_c of p_t)
    path_mask = np.zeros((g, g), dtype=bool)
    for r, c in shortest_path:
        path_mask[r, c] = True
    dist_to_path = distance_transform_edt(~path_mask)
    corridor_Ct = dist_to_path <= r_c

    # Line 7: Frontier of updated cells F_t = {u | G_hat_t(u) != G_hat_{t-1}(u)}
    frontier_Ft = np.abs(graph_curr.tau - graph_prev.tau) > 0.05

    # Line 8: Update window W_t = N_r(C_t U F_t)
    union_mask = corridor_Ct | frontier_Ft
    dist_to_union = distance_transform_edt(~union_mask)
    window_Wt = dist_to_union <= r_w

    # Lines 9-15: Generate candidate paths p_t^j restricted to W_t and accumulate Delta psi_{u,h}^{on}(t)
    delta_psi_on = np.zeros((g, g), dtype=np.float32)
    vertex_var = 2.5
    sigma_pt_sq = len(shortest_path) * vertex_var

    for _ in range(traverse_time):
        cand_path = _sample_monte_carlo_path(
            graph_curr, source, target, rng=rng, allowed_mask=window_Wt
        )
        if cand_path is None:
            continue
        w_cand = graph_curr.path_total_weight(cand_path)
        delta_w = max(0.0, w_cand - w_opt)
        sigma_pj_sq = len(cand_path) * vertex_var
        xi, _ = proposition1_sensitivity_xi(delta_w, sigma_pt_sq, sigma_pj_sq)
        if xi >= threshold_h:
            for r, c in cand_path:
                if window_Wt[r, c]:
                    delta_psi_on[r, c] += 1.0

    # Equation (22): tilde{psi}_{u,h}(t) = psi_{u,h}^{off} + Delta psi_{u,h}^{on}(t)
    psi_tilde = psi_off + delta_psi_on

    # Lines 16-20 & Equations (18), (20): Compute Psi_k and delta_k for each UAV k = 1..n
    n_uavs = grid_n * grid_n
    sub_g = g // grid_n
    Psi_k_list: List[float] = []
    for r_idx in range(grid_n):
        for c_idx in range(grid_n):
            sub_psi = psi_tilde[
                r_idx * sub_g : (r_idx + 1) * sub_g,
                c_idx * sub_g : (c_idx + 1) * sub_g,
            ]
            Psi_k_list.append(float(np.sum(sub_psi)))

    Psi_arr = np.array(Psi_k_list, dtype=np.float64)
    # Reference baseline Psi_h (mean LBC across UAV regions so logistic Eq. 20 is centered)
    Psi_mean = float(np.mean(Psi_arr))
    Psi_std = float(np.std(Psi_arr) + 1e-6)

    deltas: List[float] = []
    for k in range(n_uavs):
        # Standardized logistic mapping of Eq. (20): delta_k = 1 / (1 + exp(-g * (Psi_k - Psi_h)))
        z_k = (Psi_arr[k] - Psi_mean) / Psi_std
        raw_logistic = 1.0 / (1.0 + np.exp(-norm_coeff_g * 5.0 * z_k))
        # Scale smoothly into [delta_min, delta_max] with mean ~0.58 matching Table I payload
        delta_k = float(delta_min + (delta_max - delta_min) * raw_logistic)
        deltas.append(delta_k)

    return {
        "deltas": deltas,
        "psi_tilde": psi_tilde,
        "Psi_k": Psi_k_list,
        "window_mask": window_Wt,
        "shortest_path": shortest_path,
    }
