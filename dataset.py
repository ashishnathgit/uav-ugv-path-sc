"""
Dataset module for replicating 'Adaptive Semantic Communication for UAV/UGV Cooperative Path Planning'.

Addresses dataset limitations in the original paper:
1. LandCover.ai (2.5+ GB of 9000x9500 GeoTIFFs) is heavy and predominantly rural (only 5/41 urban maps).
2. Raw RGB images lack direct ground-truth traversability maps unless inferred via segmentation heuristics [24].

Workarounds implemented:
- Drop-in loader for real LandCover.ai images if present in `data/landcover_ai/`.
- Procedural Multi-Biome Aerial Orthophoto & Semantic Map Generator that synthesizes realistic
  Rural/Forest and Dense Urban aerial scenes with exact semantic classes:
    0: Flat Ground / Meadow (traversable, tau = 0.85)
    1: Road Network (highly traversable, tau = 1.00)
    2: Woodland / Forest (low traversability / obstacle, tau = 0.12)
    3: Building / Urban Structure (non-traversable obstacle, tau = 0.03)
    4: Water Body / River (non-traversable obstacle, tau = 0.02)
- Implements Borges et al. [24] RGB-to-Traversability estimation so reconstructed images from any
  transceiver (Path-SC, DeiT-SC, MAE-SC, JPEG) are converted into traversability graphs G_hat_t.
"""

from __future__ import annotations
import os
import glob
from typing import Tuple, List, Dict, Optional
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.ndimage import gaussian_filter, binary_dilation


# Semantic class definitions matching LandCover.ai
CLASS_FLAT = 0
CLASS_ROAD = 1
CLASS_WOODLAND = 2
CLASS_BUILDING = 3
CLASS_WATER = 4

CLASS_TRAVERSABILITY = {
    CLASS_FLAT: 0.85,
    CLASS_ROAD: 1.00,
    CLASS_WOODLAND: 0.12,
    CLASS_BUILDING: 0.04,
    CLASS_WATER: 0.02,
}

# Distinct, realistic aerial RGB palettes (in [0, 1]) so RGB -> traversability [24] is well-conditioned
CLASS_RGB_MEAN = {
    CLASS_FLAT: np.array([0.58, 0.72, 0.44], dtype=np.float32),      # Light meadow green-tan
    CLASS_ROAD: np.array([0.82, 0.80, 0.76], dtype=np.float32),      # Bright asphalt/concrete road
    CLASS_WOODLAND: np.array([0.14, 0.34, 0.15], dtype=np.float32),  # Deep dark forest canopy green
    CLASS_BUILDING: np.array([0.74, 0.28, 0.22], dtype=np.float32),  # Terracotta / red-brown roof
    CLASS_WATER: np.array([0.12, 0.28, 0.62], dtype=np.float32),     # Deep river/lake blue
}


def _fbm_noise(h: int, w: int, rng: np.random.Generator, scales: List[float] = [16.0, 8.0, 4.0]) -> np.ndarray:
    """Generate smooth multi-scale fractal noise normalized to [0, 1]."""
    acc = np.zeros((h, w), dtype=np.float32)
    weights = [0.5, 0.3, 0.2]
    for s, wt in zip(scales, weights):
        raw = rng.standard_normal((h, w)).astype(np.float32)
        smooth = gaussian_filter(raw, sigma=s)
        acc += wt * smooth
    acc = (acc - acc.min()) / (acc.max() - acc.min() + 1e-8)
    return acc


def generate_aerial_scene(
    map_size: int = 256,
    biome: str = "rural",
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Synthesize a complete ground task map X_t of shape (3, map_size, map_size) in [0, 1],
    along with its semantic class mask (map_size, map_size) and ground-truth traversability (map_size, map_size).

    Guarantees a valid traversable corridor between top-left source and bottom-right destination
    while placing realistic obstacles (water bodies, forests, buildings) that challenge path planning.
    """
    rng = np.random.default_rng(seed)
    h = w = map_size
    sem_mask = np.full((h, w), CLASS_FLAT, dtype=np.int64)

    # 1. Generate woodland / vegetation clusters via fractal noise
    veg_noise = _fbm_noise(h, w, rng, scales=[map_size / 10.0, map_size / 22.0, map_size / 45.0])
    veg_thresh = 0.54 if biome == "rural" else 0.68
    sem_mask[veg_noise > veg_thresh] = CLASS_WOODLAND

    # 2. Add a winding water body / river or lake obstacle
    if rng.random() < 0.85:
        river_vertical = rng.random() < 0.5
        t = np.linspace(0, 1, h)
        freq = rng.uniform(1.2, 2.2)
        phase = rng.uniform(0, 2 * np.pi)
        center_offset = rng.uniform(0.32, 0.68) * w
        amp = rng.uniform(0.08, 0.16) * w
        width = int(max(4, map_size * rng.uniform(0.035, 0.055)))
        curve = (center_offset + amp * np.sin(2 * np.pi * freq * t + phase)).astype(int)
        for idx in range(h):
            c = np.clip(curve[idx], width, w - width - 1)
            if river_vertical:
                sem_mask[idx, c - width : c + width] = CLASS_WATER
            else:
                sem_mask[c - width : c + width, idx] = CLASS_WATER

    # 3. Carve primary road networks (including bridges over water and traversable corridors)
    road_width = max(3, map_size // 42)
    num_roads = 3 if biome == "rural" else 5

    # Ensure at least one winding road network connects start region (top-left) to goal region (bottom-right)
    waypoints = [
        (int(0.08 * h), int(0.08 * w)),
        (int(rng.uniform(0.25, 0.45) * h), int(rng.uniform(0.45, 0.75) * w)),
        (int(rng.uniform(0.55, 0.78) * h), int(rng.uniform(0.25, 0.55) * w)),
        (int(0.92 * h), int(0.92 * w)),
    ]
    for (r0, c0), (r1, c1) in zip(waypoints[:-1], waypoints[1:]):
        steps = max(abs(r1 - r0), abs(c1 - c0)) * 2 + 1
        rs = np.linspace(r0, r1, steps).astype(int)
        cs = np.linspace(c0, c1, steps).astype(int)
        for r, c in zip(rs, cs):
            r_min, r_max = max(0, r - road_width), min(h, r + road_width + 1)
            c_min, c_max = max(0, c - road_width), min(w, c + road_width + 1)
            sem_mask[r_min:r_max, c_min:c_max] = CLASS_ROAD
            # Clear a small flat shoulder around main road
            sr_min, sr_max = max(0, r - road_width * 2), min(h, r + road_width * 2 + 1)
            sc_min, sc_max = max(0, c - road_width * 2), min(w, c + road_width * 2 + 1)
            shoulder = sem_mask[sr_min:sr_max, sc_min:sc_max]
            shoulder[shoulder != CLASS_ROAD] = CLASS_FLAT

    # Secondary roads
    for _ in range(num_roads - 1):
        if rng.random() < 0.5:
            r_pos = int(rng.uniform(0.18, 0.82) * h)
            sem_mask[max(0, r_pos - road_width // 2) : min(h, r_pos + road_width // 2 + 1), :] = CLASS_ROAD
        else:
            c_pos = int(rng.uniform(0.18, 0.82) * w)
            sem_mask[:, max(0, c_pos - road_width // 2) : min(w, c_pos + road_width // 2 + 1)] = CLASS_ROAD

    # 4. Add rectangular building blocks (dense in urban biome, sparse farmsteads in rural biome)
    num_buildings = rng.integers(38, 60) if biome == "urban" else rng.integers(8, 16)
    for _ in range(num_buildings):
        bh = int(rng.integers(max(5, map_size // 36), max(9, map_size // 16)))
        bw = int(rng.integers(max(5, map_size // 36), max(9, map_size // 16)))
        br = int(rng.integers(int(0.06 * h), int(0.92 * h) - bh))
        bc = int(rng.integers(int(0.06 * w), int(0.92 * w) - bw))
        # Avoid blocking the main road or start/goal corners
        patch_sem = sem_mask[br : br + bh, bc : bc + bw]
        if np.any(patch_sem == CLASS_ROAD) or np.any(patch_sem == CLASS_WATER):
            continue
        if (br < 0.16 * h and bc < 0.16 * w) or (br + bh > 0.84 * h and bc + bw > 0.84 * w):
            continue
        sem_mask[br : br + bh, bc : bc + bw] = CLASS_BUILDING

    # Clear start & goal safe zones
    pad = max(6, map_size // 20)
    sem_mask[:pad, :pad] = CLASS_ROAD
    sem_mask[h - pad :, w - pad :] = CLASS_ROAD

    # 5. Synthesize textured RGB aerial image (3, H, W)
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    fine_tex = _fbm_noise(h, w, rng, scales=[2.5, 1.2, 0.6]) - 0.5  # [-0.5, 0.5]
    micro_tex = rng.normal(0.0, 0.025, size=(h, w, 3)).astype(np.float32)

    for cls_id, base_color in CLASS_RGB_MEAN.items():
        mask = sem_mask == cls_id
        if not np.any(mask):
            continue
        tex_scale = 0.14 if cls_id in (CLASS_WOODLAND, CLASS_FLAT) else 0.07
        for c in range(3):
            rgb[:, :, c][mask] = base_color[c] + tex_scale * fine_tex[mask]

    # Add subtle building edge shadows for realism
    bldg_mask = sem_mask == CLASS_BUILDING
    if np.any(bldg_mask):
        dilated = binary_dilation(bldg_mask, iterations=1)
        border = dilated & (~bldg_mask)
        rgb[border] *= 0.72

    rgb = np.clip(rgb + micro_tex, 0.0, 1.0)
    rgb_chw = np.transpose(rgb, (2, 0, 1)).astype(np.float32)

    # 6. Compute ground-truth traversability map tau in [0, 1]
    tau_gt = rgb_to_traversability(rgb_chw)

    return rgb_chw, sem_mask, tau_gt


def rgb_to_traversability(rgb_chw: np.ndarray) -> np.ndarray:
    """
    Estimate traversability map tau in [0.02, 1.0] from an aerial RGB image (3, H, W)
    following the aerial image classification & traversability mapping methodology of Borges et al. [24].

    Uses nearest-prototype soft assignment in RGB feature space to map reconstructed pixels
    to traversability values, so distortions/blurring/artifacts in reconstructed images X_hat_t
    directly perturb the traversability graph G_hat_t = G_t + E_t (Eq. 2).
    """
    if rgb_chw.ndim == 3 and rgb_chw.shape[0] == 3:
        img = np.transpose(rgb_chw, (1, 2, 0))
    else:
        img = rgb_chw

    h, w, _ = img.shape
    flat_img = img.reshape(-1, 3)

    prototypes = np.stack([CLASS_RGB_MEAN[c] for c in range(5)], axis=0)  # (5, 3)
    tau_vals = np.array([CLASS_TRAVERSABILITY[c] for c in range(5)], dtype=np.float32)  # (5,)

    # Squared Euclidean distance to each land-cover class color prototype
    dists = np.sum((flat_img[:, None, :] - prototypes[None, :, :]) ** 2, axis=-1)  # (H*W, 5)
    # Softmax temperature weighting (sharp enough to distinguish classes, smooth enough for Taylor Eq. 8)
    temperature = 0.015
    logits = -dists / temperature
    logits -= np.max(logits, axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= np.sum(probs, axis=1, keepdims=True) + 1e-8

    tau = (probs @ tau_vals).reshape(h, w)
    return np.clip(tau, 0.02, 1.0).astype(np.float32)


def split_global_map_to_uav_tiles(global_img: np.ndarray, grid_n: int = 4) -> List[np.ndarray]:
    """
    Partition a global map X_t of shape (3, H, W) into n = grid_n * grid_n UAV subregions
    {x_t^1, ..., x_t^n}, each of shape (3, H // grid_n, W // grid_n).
    """
    _, h, w = global_img.shape
    th, tw = h // grid_n, w // grid_n
    tiles = []
    for r in range(grid_n):
        for c in range(grid_n):
            tiles.append(global_img[:, r * th : (r + 1) * th, c * tw : (c + 1) * tw].copy())
    return tiles


def merge_uav_tiles_to_global_map(tiles: List[np.ndarray], grid_n: int = 4) -> np.ndarray:
    """
    Fuse n = grid_n * grid_n reconstructed UAV tiles {x_hat_t^1, ..., x_hat_t^n}
    into the global reconstructed map X_hat_t of shape (3, H, W).
    """
    c, th, tw = tiles[0].shape
    global_img = np.zeros((c, th * grid_n, tw * grid_n), dtype=tiles[0].dtype)
    idx = 0
    for r in range(grid_n):
        for col in range(grid_n):
            global_img[:, r * th : (r + 1) * th, col * tw : (col + 1) * tw] = tiles[idx]
            idx += 1
    return global_img


class AerialPathPlanningDataset(Dataset):
    """
    PyTorch Dataset providing UAV subregion images x_t^k and full global maps X_t.
    Automatically loads real LandCover.ai images if present in `data_dir`, and seamlessly
    falls back to (or augments with) our Multi-Biome Procedural Aerial Orthophoto Generator.
    """

    def __init__(
        self,
        num_maps: int = 45,
        global_map_size: int = 256,
        grid_n: int = 4,
        urban_ratio: float = 0.25,
        data_dir: Optional[str] = None,
        seed: int = 42,
    ):
        super().__init__()
        self.global_map_size = global_map_size
        self.grid_n = grid_n
        self.tile_size = global_map_size // grid_n

        self.global_maps: List[Dict[str, object]] = []
        self.uav_tiles: List[np.ndarray] = []

        # Check if real LandCover.ai images exist on disk
        loaded_real = 0
        if data_dir and os.path.isdir(data_dir):
            img_paths = sorted(
                glob.glob(os.path.join(data_dir, "*.tif"))
                + glob.glob(os.path.join(data_dir, "*.png"))
                + glob.glob(os.path.join(data_dir, "*.jpg"))
            )
            for p in img_paths[:num_maps]:
                try:
                    from PIL import Image
                    pil_img = Image.open(p).convert("RGB").resize((global_map_size, global_map_size))
                    rgb_chw = np.transpose(np.asarray(pil_img, dtype=np.float32) / 255.0, (2, 0, 1))
                    tau_gt = rgb_to_traversability(rgb_chw)
                    sem_mask = (tau_gt < 0.25).astype(np.int64) * CLASS_WOODLAND
                    self.global_maps.append(
                        {
                            "rgb": rgb_chw,
                            "sem_mask": sem_mask,
                            "tau_gt": tau_gt,
                            "biome": "rural",
                            "source": os.path.basename(p),
                        }
                    )
                    loaded_real += 1
                except Exception:
                    continue

        # Generate remaining maps via multi-biome synthesis (including balanced rural & urban scenes)
        remaining = max(0, num_maps - loaded_real)
        num_urban = max(5, int(round(remaining * urban_ratio)))
        for i in range(remaining):
            biome = "urban" if i < num_urban else "rural"
            rgb_chw, sem_mask, tau_gt = generate_aerial_scene(
                map_size=global_map_size,
                biome=biome,
                seed=seed + i * 101,
            )
            self.global_maps.append(
                {
                    "rgb": rgb_chw,
                    "sem_mask": sem_mask,
                    "tau_gt": tau_gt,
                    "biome": biome,
                    "source": f"synthetic_{biome}_{i:03d}",
                }
            )

        for gmap in self.global_maps:
            tiles = split_global_map_to_uav_tiles(gmap["rgb"], grid_n=self.grid_n)
            self.uav_tiles.extend(tiles)

    def __len__(self) -> int:
        return len(self.uav_tiles)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.from_numpy(self.uav_tiles[idx])
