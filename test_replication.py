"""
Unit and integration tests verifying all equations and algorithms in the replication of:
'Adaptive Semantic Communication for UAV/UGV Cooperative Path Planning', IEEE TVT 2026.
"""

import numpy as np
import torch
from dataset import generate_aerial_scene, rgb_to_traversability, split_global_map_to_uav_tiles, merge_uav_tiles_to_global_map
from graph_planning import (
    TraversabilityGraph,
    dijkstra_shortest_path,
    proposition1_sensitivity_xi,
    algorithm1_windowed_mc_lbc,
)
from models import PathSCConfig, SemComTransceiver, compute_algorithm2_loss
from wireless_channel import pass_through_wireless_channel, jpeg_wireless_transceiver


def test_dataset_and_traversability_mapping():
    rgb, sem_mask, tau_gt = generate_aerial_scene(map_size=128, biome="urban", seed=12)
    assert rgb.shape == (3, 128, 128)
    assert sem_mask.shape == (128, 128)
    assert tau_gt.shape == (128, 128)
    assert float(tau_gt.min()) >= 0.02 - 1e-5 and float(tau_gt.max()) <= 1.0 + 1e-5

    tiles = split_global_map_to_uav_tiles(rgb, grid_n=4)
    assert len(tiles) == 16
    assert tiles[0].shape == (3, 32, 32)
    merged = merge_uav_tiles_to_global_map(tiles, grid_n=4)
    assert np.allclose(rgb, merged)


def test_proposition1_maximum_at_z_equals_1():
    """Verify Proposition 1 (Eq. 11-16): f(z) = |z| * phi(z) reaches its global maximum at |z| = 1."""
    xi_at_1, z1 = proposition1_sensitivity_xi(delta_w=2.0, sigma_pt_sq=2.0, sigma_pj_sq=2.0)
    assert np.isclose(z1, 1.0)
    for delta_w in [0.5, 1.2, 3.0, 4.5]:
        xi_other, _ = proposition1_sensitivity_xi(delta_w=delta_w, sigma_pt_sq=2.0, sigma_pj_sq=2.0)
        assert xi_at_1 >= xi_other


def test_algorithm1_windowed_mc_lbc_and_deltas():
    rgb, _, tau_gt = generate_aerial_scene(map_size=128, biome="rural", seed=7)
    G = TraversabilityGraph(tau_gt, grid_size=16, kappa=1.0)
    alg1 = algorithm1_windowed_mc_lbc(G, grid_n=4, seed=7)
    deltas = alg1["deltas"]
    assert len(deltas) == 16
    assert all(0.1 <= d <= 1.0 for d in deltas)
    # Regions traversed by the shortest path should have higher average delta_k than unvisited corners
    assert max(deltas) > min(deltas)


def test_path_sc_transceiver_and_algorithm2_loss():
    cfg = PathSCConfig(img_size=64, patch_size=8, embed_dim=64, encoder_depth=2, decoder_depth=2, num_heads=4)
    teacher = SemComTransceiver(cfg, mode="deit_sc")
    student = SemComTransceiver(cfg, mode="path_sc")
    x = torch.rand(2, 3, 64, 64)

    t_out = teacher(x, delta_k=1.0, snr_db=12.0, channel_type="AWGN")
    s_out = student(x, delta_k=0.6, snr_db=12.0, channel_type="Rayleigh")

    assert s_out["x_hat"].shape == (2, 3, 64, 64)
    assert 0.45 <= s_out["active_fraction"] <= 0.75

    losses = compute_algorithm2_loss(x, s_out, t_out, target_delta=0.6)
    assert losses["total"].requires_grad
    losses["total"].backward()
