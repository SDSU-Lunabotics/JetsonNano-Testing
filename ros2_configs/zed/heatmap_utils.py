import numpy as np


def _safe_ratio(num, den):
    out = np.zeros_like(num, dtype=np.float32)
    np.divide(num, den, out=out, where=den > 1e-6)
    return np.clip(out, 0.0, 1.0)


def compute_heat_score(free_counts, occ_counts, hole_counts, mode="risk"):
    total = free_counts + occ_counts + hole_counts
    mode = str(mode).lower()
    if mode == "risk":
        return _safe_ratio(occ_counts + hole_counts, total)
    if mode == "obstacle":
        return _safe_ratio(occ_counts, total)
    if mode == "hole":
        return _safe_ratio(hole_counts, total)
    if mode == "free":
        return _safe_ratio(free_counts, total)
    if mode == "evidence":
        vis = np.log1p(total.astype(np.float32))
        vmax = float(vis.max())
        return vis / vmax if vmax > 0.0 else vis
    raise ValueError(f"Unsupported heatmap mode: {mode}")


def _colorize_green_to_red(score):
    score = np.clip(score, 0.0, 1.0).astype(np.float32)
    red = (score * 255.0).astype(np.uint8)
    green = ((1.0 - score) * 255.0).astype(np.uint8)
    blue = np.zeros_like(red, dtype=np.uint8)
    return np.stack((blue, green, red), axis=2)


def _colorize_black_to_green(score):
    score = np.clip(score, 0.0, 1.0).astype(np.float32)
    green = (score * 255.0).astype(np.uint8)
    z = np.zeros_like(green, dtype=np.uint8)
    return np.stack((z, green, z), axis=2)


def _colorize_gray(score):
    score = np.clip(score, 0.0, 1.0).astype(np.float32)
    gray = (score * 255.0).astype(np.uint8)
    return np.stack((gray, gray, gray), axis=2)


def render_heatmap(occ_map, mode="risk", min_evidence=1.0):
    score = compute_heat_score(
        occ_map.free_counts,
        occ_map.occ_counts,
        occ_map.hole_counts,
        mode=mode,
    )
    mode = str(mode).lower()
    if mode == "free":
        heat = _colorize_black_to_green(score)
    elif mode == "evidence":
        heat = _colorize_gray(score)
    else:
        heat = _colorize_green_to_red(score)
    evidence = occ_map.free_counts + occ_map.occ_counts + occ_map.hole_counts
    heat[evidence < float(min_evidence)] = (0, 0, 0)
    return heat


def blend_with_map(map_vis, heat_vis, alpha=0.35):
    if map_vis.shape != heat_vis.shape:
        raise ValueError("Map and heatmap shapes must match for blending")
    a = float(np.clip(alpha, 0.0, 1.0))
    if a <= 0.0:
        return map_vis
    out = (1.0 - a) * map_vis.astype(np.float32) + a * heat_vis.astype(np.float32)
    return np.clip(out, 0.0, 255.0).astype(np.uint8)
