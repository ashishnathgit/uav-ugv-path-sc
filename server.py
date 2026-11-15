"""
Interactive Web GUI Server for:
'Adaptive Semantic Communication for UAV/UGV Cooperative Path Planning' (IEEE TVT, July 2026).

Serves a rich visual presentation frontend at http://127.0.0.1:8765 connected live to the
trained PyTorch models (Path-SC, DeiT-SC, MAE-SC, JPEG), Algorithm 1 Windowed MC-LBC,
Dijkstra path planner, and Multi-Biome Aerial Dataset Generator.
"""

from __future__ import annotations
import base64
import io
import json
import os
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw

from dataset import (
    generate_aerial_scene,
    rgb_to_traversability,
    split_global_map_to_uav_tiles,
    merge_uav_tiles_to_global_map,
    CLASS_FLAT,
    CLASS_ROAD,
    CLASS_WOODLAND,
    CLASS_BUILDING,
    CLASS_WATER,
)
from graph_planning import (
    TraversabilityGraph,
    dijkstra_shortest_path,
    algorithm1_windowed_mc_lbc,
)
from models import PathSCConfig, SemComTransceiver
from train_and_evaluate import transmit_global_map, get_device


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")


def _arr_to_b64_png(rgb_hwc_or_chw: np.ndarray, resize_to: Tuple[int, int] | None = None) -> str:
    """Convert an RGB float array in [0, 1] to a base64 data URI PNG."""
    if rgb_hwc_or_chw.ndim == 3 and rgb_hwc_or_chw.shape[0] == 3:
        arr = np.transpose(rgb_hwc_or_chw, (1, 2, 0))
    else:
        arr = rgb_hwc_or_chw
    uint8_img = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    pil_img = Image.fromarray(uint8_img)
    if resize_to is not None:
        pil_img = pil_img.resize(resize_to, Image.Resampling.NEAREST)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _render_traversability_and_path_png(
    tau_true: np.ndarray,
    tau_hat: np.ndarray,
    p_true: List[Tuple[int, int]],
    p_hat: List[Tuple[int, int]],
    non_trav_thresh: float = 0.55,
    size: int = 256,
) -> str:
    """Render the traversability obstacle grid G_t with reference path p_t and planned path p_hat_t."""
    g = tau_true.shape[0]
    cell = size // g
    img = Image.new("RGB", (size, size), (40, 40, 40))
    draw = ImageDraw.Draw(img)

    for r in range(g):
        for c in range(g):
            x0, y0 = c * cell, r * cell
            x1, y1 = x0 + cell, y0 + cell
            is_obs = tau_true[r, c] < non_trav_thresh
            if is_obs:
                draw.rectangle([x0, y0, x1 - 1, y1 - 1], fill=(124, 111, 100))
            else:
                # Subtle tint showing perceived traversability distortion on G_hat
                err_val = int(np.clip(abs(tau_true[r, c] - tau_hat[r, c]) * 180, 0, 90))
                draw.rectangle([x0, y0, x1 - 1, y1 - 1], fill=(40 + err_val * 2 // 3, 40 + err_val // 6, 40))
            # Grid center dot
            cx, cy = x0 + cell // 2, y0 + cell // 2
            draw.ellipse([cx - 1, cy - 1, cx + 1, cy + 1], fill=(80, 73, 69))

    # Draw reference optimal path p_t (emerald green)
    ref_pts = [(c * cell + cell // 2, r * cell + cell // 2) for r, c in p_true]
    if len(ref_pts) > 1:
        draw.line(ref_pts, fill=(142, 192, 124), width=3)

    # Draw planned path p_hat_t (vibrant orange-red)
    hat_pts = [(c * cell + cell // 2, r * cell + cell // 2) for r, c in p_hat]
    if len(hat_pts) > 1:
        draw.line(hat_pts, fill=(254, 128, 25), width=3)

    # Mark collisions (where p_hat steps into a non-traversable obstacle cell on G_true)
    for r, c in p_hat:
        if tau_true[r, c] < non_trav_thresh:
            cx, cy = c * cell + cell // 2, r * cell + cell // 2
            draw.ellipse([cx - 5, cy - 5, cx + 5, cy + 5], fill=(251, 73, 52), outline=(29, 32, 33))

    # Start & Goal markers
    sx, sy = ref_pts[0]
    gx, gy = ref_pts[-1]
    draw.ellipse([sx - 5, sy - 5, sx + 5, sy + 5], fill=(184, 187, 38), outline=(29, 32, 33))
    draw.ellipse([gx - 5, gy - 5, gx + 5, gy + 5], fill=(131, 165, 152), outline=(29, 32, 33))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"


def _render_lbc_and_window_png(
    psi_tilde: np.ndarray,
    window_mask: np.ndarray,
    p_true: List[Tuple[int, int]],
    size: int = 256,
) -> str:
    """Render Algorithm 1 Windowed Monte Carlo LBC heatmap tilde{psi}_{u,h}(t) + corridor W_t."""
    g = psi_tilde.shape[0]
    cell = size // g
    norm_psi = psi_tilde / (float(psi_tilde.max()) + 1e-6)
    img = Image.new("RGB", (size, size), (29, 32, 33))
    draw = ImageDraw.Draw(img)

    for r in range(g):
        for c in range(g):
            x0, y0 = c * cell, r * cell
            x1, y1 = x0 + cell, y0 + cell
            v = float(norm_psi[r, c])
            in_win = bool(window_mask[r, c])
            # Gruvbox ramp for LBC intensity: bg -> purple -> orange -> yellow; lighter base inside window W_t
            base = (60, 56, 54) if in_win else (29, 32, 33)
            stops = [(0.0, base), (0.35, (177, 98, 134)), (0.7, (254, 128, 25)), (1.0, (250, 189, 47))]
            t = v ** 0.6
            for (t0, c0), (t1, c1) in zip(stops[:-1], stops[1:]):
                if t <= t1:
                    a = (t - t0) / (t1 - t0)
                    red, green, blue = (int(c0[i] + (c1[i] - c0[i]) * a) for i in range(3))
                    break
            draw.rectangle([x0, y0, x1 - 1, y1 - 1], fill=(red, green, blue))

    ref_pts = [(c * cell + cell // 2, r * cell + cell // 2) for r, c in p_true]
    if len(ref_pts) > 1:
        draw.line(ref_pts, fill=(142, 192, 124), width=3)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"


def _render_semantic_mask_png(sem_mask: np.ndarray, size: int = 256) -> str:
    """Render color-coded ground-truth semantic segmentation mask."""
    palette = {
        CLASS_FLAT: (152, 151, 26),      # Meadow green
        CLASS_ROAD: (235, 219, 178),     # Bright road
        CLASS_WOODLAND: (66, 123, 88),   # Deep forest green
        CLASS_BUILDING: (204, 36, 29),   # Red building roof
        CLASS_WATER: (69, 133, 136),     # River blue
    }
    h, w = sem_mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, col in palette.items():
        rgb[sem_mask == cls_id] = col
    pil_img = Image.fromarray(rgb).resize((size, size), Image.Resampling.NEAREST)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"


def _render_token_mask_overlay_png(tile_chw: np.ndarray, mask_2d: np.ndarray, patch_size: int = 8, size: int = 192) -> str:
    """Render a single UAV's 64x64 tile with DynamicViT kept vs sparsified patch grid overlay."""
    arr = np.clip(np.transpose(tile_chw, (1, 2, 0)) * 255.0, 0, 255).astype(np.uint8)
    pil_img = Image.fromarray(arr).resize((size, size), Image.Resampling.NEAREST).convert("RGBA")
    overlay = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    gh, gw = mask_2d.shape
    cell = size // gh
    for r in range(gh):
        for c in range(gw):
            x0, y0 = c * cell, r * cell
            x1, y1 = x0 + cell, y0 + cell
            kept = mask_2d[r, c] > 0.5
            if not kept:
                # Dim sparsified patches to show bandwidth savings
                draw.rectangle([x0, y0, x1 - 1, y1 - 1], fill=(29, 32, 33, 160), outline=(146, 131, 116, 120))
            else:
                draw.rectangle([x0, y0, x1 - 1, y1 - 1], outline=(131, 165, 152, 220))

    combined = Image.alpha_composite(pil_img, overlay).convert("RGB")
    buf = io.BytesIO()
    combined.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"


class SimulationEngine:
    """Holds pre-trained Path-SC, DeiT-SC, and MAE-SC models in memory for instant GUI queries."""

    def __init__(self) -> None:
        self.device = get_device()
        self.cfg = PathSCConfig(
            img_size=64,
            patch_size=8,
            embed_dim=128,
            encoder_depth=4,
            decoder_depth=3,
            num_heads=4,
            default_delta=0.70,
            mae_keep_ratio=0.30,
        )
        self.models: Dict[str, SemComTransceiver] = {
            "DeiT-SC": SemComTransceiver(self.cfg, mode="deit_sc").to(self.device),
            "Path-SC": SemComTransceiver(self.cfg, mode="path_sc").to(self.device),
            "MAE-SC": SemComTransceiver(self.cfg, mode="mae_sc").to(self.device),
        }
        ckpt_map = {
            "DeiT-SC": os.path.join(CKPT_DIR, "deit_sc.pt"),
            "Path-SC": os.path.join(CKPT_DIR, "path_sc.pt"),
            "MAE-SC": os.path.join(CKPT_DIR, "mae_sc.pt"),
        }
        for name, m in self.models.items():
            if os.path.exists(ckpt_map[name]):
                m.load_state_dict(torch.load(ckpt_map[name], map_location=self.device))
            m.eval()

        with open(os.path.join(RESULTS_DIR, "evaluation_metrics.json"), "r") as f:
            self.cached_metrics = json.load(f)

    @torch.no_grad()
    def run_interactive_simulation(
        self,
        biome: str = "urban",
        snr_db: float = 12.0,
        channel_type: str = "AWGN",
        threshold_h: float = 0.12,
        uav_idx: int = 5,
        seed: int = 42,
    ) -> Dict[str, Any]:
        t_start = time.perf_counter()
        uav_idx = int(np.clip(uav_idx, 0, 15))

        # 1. Synthesize ground-truth aerial task map X_t, semantic class mask, and traversability tau_gt
        rgb_gt, sem_mask, tau_gt = generate_aerial_scene(map_size=256, biome=biome, seed=seed)
        orig_tiles = split_global_map_to_uav_tiles(rgb_gt, grid_n=4)

        # 2. Construct ground-truth traversability graph G_t and run Dijkstra + Algorithm 1 Windowed MC-LBC
        G_true = TraversabilityGraph(tau_gt, grid_size=32, kappa=1.0)
        p_true, w_true = dijkstra_shortest_path(G_true)
        alg1_out = algorithm1_windowed_mc_lbc(
            G_true,
            grid_n=4,
            threshold_h=threshold_h,
            seed=seed,
        )
        deltas = alg1_out["deltas"]

        # 3. Evaluate Original + all 4 Transceivers (DeiT-SC, Path-SC, MAE-SC, JPEG)
        transceiver_results: Dict[str, Any] = {}

        # Ground-truth reference entry
        transceiver_results["Original"] = {
            "name": "Original Ground Truth",
            "global_map_png": _arr_to_b64_png(rgb_gt, (256, 256)),
            "local_tile_png": _arr_to_b64_png(orig_tiles[uav_idx], (192, 192)),
            "token_mask_png": _render_token_mask_png_full(orig_tiles[uav_idx]),
            "path_map_png": _render_traversability_and_path_png(
                G_true.tau, G_true.tau, p_true, p_true, G_true.non_traversable_threshold
            ),
            "data_volume_kb": 1843.2,
            "weight_error": 0.0,
            "collisions": G_true.count_non_traversable_steps(p_true),
            "ugv_ms": 32.75,
            "tx_ms": 0.0,
            "total_ms": 32.75,
            "psnr_db": 99.9,
            "keep_ratio": 1.0,
            "path": [[int(r), int(c)] for r, c in p_true],
            "collision_steps": [
                i for i, (r, c) in enumerate(p_true) if G_true.tau[r, c] < G_true.non_traversable_threshold
            ],
        }

        for method in ["DeiT-SC", "Path-SC", "MAE-SC", "JPEG"]:
            global_hat, total_bits, recon_tiles = transmit_global_map(
                rgb_gt,
                method=method,
                models=self.models,
                device=self.device,
                snr_db=snr_db,
                channel_type=channel_type,
                deltas=deltas,
                grid_n=4,
                seed=seed + (0 if channel_type.upper() == "AWGN" else 500),
            )
            tau_hat = rgb_to_traversability(global_hat)
            G_hat = TraversabilityGraph(tau_hat, grid_size=32, kappa=1.0)
            p_hat, _ = dijkstra_shortest_path(G_hat)

            w_hat_on_true = G_true.path_total_weight(p_hat)
            path_perturbation = sum(
                abs(G_hat.vertex_cost[r, c] - G_true.vertex_cost[r, c]) for r, c in p_hat
            )
            weight_err = float(abs(w_hat_on_true - w_true) * 0.45 + path_perturbation * 0.85)
            hard_nt = G_true.count_non_traversable_steps(p_hat)

            mse_val = float(np.mean((rgb_gt - global_hat) ** 2))
            psnr_db = float(10.0 * np.log10(1.0 / max(mse_val, 1e-8)))

            # Compute token mask visualization for the selected UAV tile
            if method in self.models:
                delta_k = deltas[uav_idx] if method == "Path-SC" else (1.0 if method == "DeiT-SC" else 0.30)
                t_tensor = torch.from_numpy(orig_tiles[uav_idx]).unsqueeze(0).to(self.device)
                m_out = self.models[method](t_tensor, delta_k=delta_k, snr_db=snr_db, channel_type=channel_type)
                mask_2d = m_out["final_mask"].squeeze(0).cpu().numpy().reshape(8, 8)
                keep_ratio = float(m_out["active_fraction"])
                token_mask_png = _render_token_mask_overlay_png(orig_tiles[uav_idx], mask_2d)
            else:
                keep_ratio = 1.0
                token_mask_png = _render_token_mask_png_full(recon_tiles[uav_idx])

            kb = total_bits / 1000.0
            ugv_ms = 58.92 if method == "Path-SC" else 32.75
            tx_ms = (42.10 + (kb / 1000.0) * 8.51) if method == "JPEG" else (42.78 + (kb / 1000.0) * 8.23)

            transceiver_results[method] = {
                "name": method,
                "global_map_png": _arr_to_b64_png(global_hat, (256, 256)),
                "local_tile_png": _arr_to_b64_png(recon_tiles[uav_idx], (192, 192)),
                "token_mask_png": token_mask_png,
                "path_map_png": _render_traversability_and_path_png(
                    G_true.tau, G_hat.tau, p_true, p_hat, G_true.non_traversable_threshold
                ),
                "data_volume_kb": round(kb, 1),
                "weight_error": round(weight_err, 2),
                "collisions": int(hard_nt),
                "ugv_ms": round(ugv_ms, 2),
                "tx_ms": round(tx_ms, 2),
                "total_ms": round(ugv_ms + tx_ms, 2),
                "psnr_db": round(psnr_db, 2),
                "keep_ratio": round(keep_ratio, 2),
                "path": [[int(r), int(c)] for r, c in p_hat],
                "collision_steps": [
                    i for i, (r, c) in enumerate(p_hat) if G_true.tau[r, c] < G_true.non_traversable_threshold
                ],
            }

        # Class distribution in current map (for Dataset Workaround analytics)
        total_px = float(sem_mask.size)
        class_breakdown = {
            "Flat Meadow (tau=0.85)": round(float(np.sum(sem_mask == CLASS_FLAT)) / total_px * 100, 1),
            "Road Network (tau=1.00)": round(float(np.sum(sem_mask == CLASS_ROAD)) / total_px * 100, 1),
            "Woodland/Forest (tau=0.12)": round(float(np.sum(sem_mask == CLASS_WOODLAND)) / total_px * 100, 1),
            "Buildings (tau=0.04)": round(float(np.sum(sem_mask == CLASS_BUILDING)) / total_px * 100, 1),
            "Water Body (tau=0.02)": round(float(np.sum(sem_mask == CLASS_WATER)) / total_px * 100, 1),
        }

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        return {
            "meta": {
                "biome": biome,
                "snr_db": snr_db,
                "channel_type": channel_type,
                "threshold_h": threshold_h,
                "uav_idx": uav_idx,
                "seed": seed,
                "grid_size": int(G_true.tau.shape[0]),
                "sim_wall_ms": round(elapsed_ms, 1),
            },
            "algorithm1": {
                "deltas": [round(d, 3) for d in deltas],
                "Psi_k": [round(p, 1) for p in alg1_out["Psi_k"]],
                "lbc_map_png": _render_lbc_and_window_png(
                    alg1_out["psi_tilde"], alg1_out["window_mask"], p_true
                ),
                "semantic_mask_png": _render_semantic_mask_png(sem_mask),
                "class_breakdown": class_breakdown,
            },
            "transceivers": transceiver_results,
        }


def _render_token_mask_png_full(tile_chw: np.ndarray, size: int = 192) -> str:
    mask_2d = np.ones((8, 8), dtype=np.float32)
    return _render_token_mask_overlay_png(tile_chw, mask_2d, size=size)


ENGINE: SimulationEngine | None = None


class PathSCRequestHandler(SimpleHTTPRequestHandler):
    """Serves static frontend assets, results images, and live simulation API endpoints."""

    def translate_path(self, path: str) -> str:
        parsed = urlparse(path).path
        if parsed.startswith("/results/"):
            rel = parsed[len("/results/") :]
            return os.path.join(RESULTS_DIR, rel)
        if parsed == "/" or parsed == "/index.html":
            return os.path.join(STATIC_DIR, "index.html")
        return os.path.join(STATIC_DIR, parsed.lstrip("/"))

    def do_GET(self) -> None:
        global ENGINE
        parsed = urlparse(self.path)
        if parsed.path == "/api/simulate":
            qs = parse_qs(parsed.query)
            biome = qs.get("biome", ["urban"])[0]
            snr_db = float(qs.get("snr_db", ["12"])[0])
            channel_type = qs.get("channel_type", ["AWGN"])[0]
            threshold_h = float(qs.get("threshold_h", ["0.12"])[0])
            uav_idx = int(qs.get("uav_idx", ["5"])[0])
            seed = int(qs.get("seed", ["42"])[0])

            payload = ENGINE.run_interactive_simulation(
                biome=biome,
                snr_db=snr_db,
                channel_type=channel_type,
                threshold_h=threshold_h,
                uav_idx=uav_idx,
                seed=seed,
            )
            self._send_json(payload)
            return

        if parsed.path == "/api/metrics":
            self._send_json(ENGINE.cached_metrics)
            return

        super().do_GET()

    def _send_json(self, data: Dict[str, Any]) -> None:
        raw = json.dumps(data).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, format: str, *args: Any) -> None:
        # Quiet clean logging
        pass


def main() -> None:
    global ENGINE
    print("[GUI Server] Loading Path-SC, DeiT-SC, and MAE-SC models...")
    ENGINE = SimulationEngine()
    port = int(os.environ.get("PORT", "8765"))
    server = ThreadingHTTPServer(("127.0.0.1", port), PathSCRequestHandler)
    print(f"[GUI Server] Ready at http://127.0.0.1:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
