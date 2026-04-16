import os
import numpy as np


class OccupancyMap:
    def __init__(
        self,
        map_res_m,
        map_width_m,
        map_height_m,
        map_z_min,
        decay=0.97,
        free_decay=None,
        occ_decay=None,
        hole_decay=None,
        free_confirm_hits=8.0,
        free_decay_unconfirmed=None,
        free_decay_confirmed=1.0,
        free_downgrade_factor=0.6,
        free_confirm_ratio=1.2,
    ):
        self.map_res_m = map_res_m
        self.map_width_m = map_width_m
        self.map_height_m = map_height_m
        self.map_z_min = map_z_min
        self.map_decay = decay
        self.free_decay = decay if free_decay is None else free_decay
        self.occ_decay = decay if occ_decay is None else occ_decay
        self.hole_decay = decay if hole_decay is None else hole_decay
        self.free_confirm_hits = float(free_confirm_hits)
        self.free_decay_unconfirmed = (
            self.free_decay if free_decay_unconfirmed is None else float(free_decay_unconfirmed)
        )
        self.free_decay_confirmed = float(free_decay_confirmed)
        self.free_downgrade_factor = float(free_downgrade_factor)
        self.free_confirm_ratio = float(free_confirm_ratio)

        self.x_min = -map_width_m / 2.0
        self.x_max = map_width_m / 2.0
        self.z_min = map_z_min
        self.z_max = map_z_min + map_height_m

        self.grid_w = int(map_width_m / map_res_m)
        self.grid_h = int(map_height_m / map_res_m)

        self.free_counts = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        self.occ_counts = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        self.hole_counts = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)

    def meta(self):
        return {
            "map_res_m": self.map_res_m,
            "map_width_m": self.map_width_m,
            "map_height_m": self.map_height_m,
            "map_z_min": self.map_z_min,
        }

    def load(self, path):
        data = np.load(path, allow_pickle=True)
        loaded_free = data["free_counts"].astype(np.float32)
        loaded_occ = data["occ_counts"].astype(np.float32)
        loaded_hole = data["hole_counts"].astype(np.float32) if "hole_counts" in data else None
        if (
            "map_res_m" in data
            and "map_width_m" in data
            and "map_height_m" in data
            and "map_z_min" in data
        ):
            meta = {
                "map_res_m": float(data["map_res_m"]),
                "map_width_m": float(data["map_width_m"]),
                "map_height_m": float(data["map_height_m"]),
                "map_z_min": float(data["map_z_min"]),
            }
        elif "meta" in data:
            meta_val = data["meta"]
            if hasattr(meta_val, "item"):
                meta_val = meta_val.item()
            meta = meta_val if isinstance(meta_val, dict) else {}
        else:
            meta = {}
        same_meta = (
            np.isclose(float(meta.get("map_res_m", np.nan)), float(self.map_res_m), atol=1e-6)
            and np.isclose(float(meta.get("map_width_m", np.nan)), float(self.map_width_m), atol=1e-6)
            and np.isclose(float(meta.get("map_height_m", np.nan)), float(self.map_height_m), atol=1e-6)
            and np.isclose(float(meta.get("map_z_min", np.nan)), float(self.map_z_min), atol=1e-6)
        )
        if same_meta:
            if loaded_free.shape == self.free_counts.shape and loaded_occ.shape == self.occ_counts.shape:
                self.free_counts[:] = loaded_free
                self.occ_counts[:] = loaded_occ
                if loaded_hole is not None and loaded_hole.shape == self.hole_counts.shape:
                    self.hole_counts[:] = loaded_hole
                return True, "Loaded map"
            return False, "Map size mismatch; starting with empty map."
        return False, "Map settings differ; starting with empty map."

    def save(self, path):
        save_dir = os.path.dirname(path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        np.savez_compressed(
            path,
            free_counts=self.free_counts,
            occ_counts=self.occ_counts,
            hole_counts=self.hole_counts,
            map_res_m=np.float32(self.map_res_m),
            map_width_m=np.float32(self.map_width_m),
            map_height_m=np.float32(self.map_height_m),
            map_z_min=np.float32(self.map_z_min),
            meta=self.meta(),
        )

    def update(self, x, z, ground_mask, obstacle_mask, hole_mask=None):
        in_bounds = (x >= self.x_min) & (x < self.x_max) & (z >= self.z_min) & (z < self.z_max)
        if not np.any(in_bounds):
            return
        x = x[in_bounds]
        z = z[in_bounds]
        gmask = ground_mask[in_bounds]
        omask = obstacle_mask[in_bounds]
        hmask = None
        if hole_mask is not None:
            hmask = hole_mask[in_bounds]

        ix = ((x - self.x_min) / self.map_res_m).astype(np.int32)
        iz = ((z - self.z_min) / self.map_res_m).astype(np.int32)
        # Flip Z so forward is "up" in the image.
        row = self.grid_h - 1 - iz
        col = ix

        # Two-speed free-space decay:
        # - low-confidence free cells decay faster
        # - confirmed free cells can be held near static (e.g. decay=1.0)
        confirmed_free = self.free_counts >= self.free_confirm_hits
        self.free_counts[~confirmed_free] *= self.free_decay_unconfirmed
        self.free_counts[confirmed_free] *= self.free_decay_confirmed
        self.occ_counts *= self.occ_decay
        self.hole_counts *= self.hole_decay

        def _unique_cells(rr, cc):
            if rr.size == 0:
                return rr, cc
            flat = np.ravel_multi_index((rr, cc), (self.grid_h, self.grid_w))
            uniq = np.unique(flat)
            ur, uc = np.unravel_index(uniq, (self.grid_h, self.grid_w))
            return ur, uc

        if np.any(gmask):
            g_r, g_c = _unique_cells(row[gmask], col[gmask])
            self.free_counts[g_r, g_c] += 1.0
        if np.any(omask):
            occ_r, occ_c = _unique_cells(row[omask], col[omask])
            self.occ_counts[occ_r, occ_c] += 1.0
            # New obstacle evidence should degrade prior free-space confidence.
            if self.free_downgrade_factor < 1.0:
                self.free_counts[occ_r, occ_c] *= self.free_downgrade_factor
        if hmask is not None and np.any(hmask):
            hole_r, hole_c = _unique_cells(row[hmask], col[hmask])
            self.hole_counts[hole_r, hole_c] += 1.0
            # Keep holes informational (blue) without making them non-traversable by
            # degrading free-space confidence. Obstacle evidence still controls blocking.

    def render(self, whole_mode=False):
        # Visualize: white = mapped/known ground, red = confirmed obstacle, blue = hole,
        # black = unknown. When whole_mode is enabled, the map renders all known ground
        # as white and highlights only well-established obstacle/hole evidence.
        free_vis = np.log1p(self.free_counts)
        occ_vis = np.log1p(self.occ_counts)
        hole_vis = np.log1p(self.hole_counts)
        fmax = free_vis.max()
        omax = occ_vis.max()
        hmax = hole_vis.max()
        if fmax > 0:
            free_vis = free_vis / fmax
        if omax > 0:
            occ_vis = occ_vis / omax
        if hmax > 0:
            hole_vis = hole_vis / hmax

        map_vis = np.zeros((self.grid_h, self.grid_w, 3), dtype=np.uint8)

        known = self.known_mask(min_evidence=1.0)
        if np.any(known):
            if whole_mode:
                map_vis[known] = (240, 240, 240)
            else:
                free_shade = (free_vis[known] * 160.0 + 80.0).astype(np.uint8)
                map_vis[known, 1] = free_shade
                map_vis[known, 0] = np.minimum(map_vis[known, 0], (free_shade // 2).astype(np.uint8))
                map_vis[known, 2] = np.minimum(map_vis[known, 2], (free_shade // 2).astype(np.uint8))

        # Show red obstacles only when we have strong evidence.
        strong_occ = self.obstacle_mask(min_occ_count=3.0, min_occ_ratio=2.0, min_occ_advantage=1.0)
        if np.any(strong_occ):
            occ_intensity = (occ_vis[strong_occ] * 255.0).astype(np.uint8)
            occ_intensity = np.maximum(occ_intensity, 120)
            map_vis[strong_occ, 2] = occ_intensity
            map_vis[strong_occ, 1] = np.minimum(map_vis[strong_occ, 1], 40)
            map_vis[strong_occ, 0] = np.minimum(map_vis[strong_occ, 0], 40)

        # Show holes as blue when there is hole evidence.
        hole_cells = (self.hole_counts > 0) & ~strong_occ
        if np.any(hole_cells):
            hole_intensity = (hole_vis[hole_cells] * 255.0).astype(np.uint8)
            hole_intensity = np.maximum(hole_intensity, 80)
            map_vis[hole_cells, 0] = hole_intensity
            map_vis[hole_cells, 1] = np.minimum(map_vis[hole_cells, 1], 140)
            map_vis[hole_cells, 2] = np.minimum(map_vis[hole_cells, 2], 60)

        # Gentle halo around confirmed free-space to improve readability.
        confirmed_free = self.free_counts >= self.free_confirm_hits
        if np.any(confirmed_free):
            near_confirmed_free = inflate_mask(confirmed_free, radius_cells=1)
            evidence = self.free_counts + self.occ_counts + self.hole_counts

            halo_unknown = near_confirmed_free & (evidence < 1.0)
            if np.any(halo_unknown):
                map_vis[halo_unknown] = (220, 220, 220)

        return map_vis

    def world_to_grid(self, x, z):
        if x < self.x_min or x >= self.x_max or z < self.z_min or z >= self.z_max:
            return None
        col = int((x - self.x_min) / self.map_res_m)
        row = int(self.grid_h - 1 - ((z - self.z_min) / self.map_res_m))
        if row < 0 or row >= self.grid_h or col < 0 or col >= self.grid_w:
            return None
        return row, col

    def grid_to_world(self, row, col):
        if row < 0 or row >= self.grid_h or col < 0 or col >= self.grid_w:
            return None
        x = self.x_min + (col + 0.5) * self.map_res_m
        z = self.z_min + (self.grid_h - 1 - row + 0.5) * self.map_res_m
        return x, z

    def obstacle_mask(self, min_occ_count=3.0, min_occ_ratio=1.5, min_occ_advantage=0.0):
        # Mark as obstacle only if we have enough occupied evidence
        # and it significantly outweighs free evidence.
        occ = self.occ_counts
        free = self.free_counts
        ratio_ok = occ >= (free * float(min_occ_ratio))
        adv_ok = (occ - free) >= float(min_occ_advantage)
        return (occ >= float(min_occ_count)) & ratio_ok & adv_ok

    def known_mask(self, min_evidence=1.0):
        evidence = self.free_counts + self.occ_counts + self.hole_counts
        return evidence >= float(min_evidence)


def astar_path(start_rc, goal_rc, obstacle_mask, connectivity=8, traversal_cost_map=None):
    import heapq
    import math

    if start_rc is None or goal_rc is None:
        return None
    if obstacle_mask[goal_rc[0], goal_rc[1]]:
        return None

    h, w = obstacle_mask.shape
    sr, sc = start_rc
    gr, gc = goal_rc
    if sr < 0 or sr >= h or sc < 0 or sc >= w:
        return None
    if gr < 0 or gr >= h or gc < 0 or gc >= w:
        return None

    extra_cost = None
    if traversal_cost_map is not None:
        if traversal_cost_map.shape != obstacle_mask.shape:
            return None
        extra_cost = traversal_cost_map

    connectivity = int(connectivity)
    if connectivity not in (4, 8):
        connectivity = 8

    def heuristic(r, c):
        dr = abs(r - gr)
        dc = abs(c - gc)
        if connectivity == 4:
            return dr + dc
        # Octile distance for 8-connected grids.
        dmin = min(dr, dc)
        dmax = max(dr, dc)
        return (dmax - dmin) + (math.sqrt(2.0) * dmin)

    neighbors = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0)]
    if connectivity == 8:
        diag = math.sqrt(2.0)
        neighbors += [(1, 1, diag), (1, -1, diag), (-1, 1, diag), (-1, -1, diag)]

    open_set = []
    heapq.heappush(open_set, (heuristic(sr, sc), 0, (sr, sc)))
    came_from = {}
    cost = { (sr, sc): 0 }

    while open_set:
        _, g, current = heapq.heappop(open_set)
        if current == (gr, gc):
            # Reconstruct
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        r, c = current
        for dr, dc, step_cost in neighbors:
            nr, nc = r + dr, c + dc
            if nr < 0 or nr >= h or nc < 0 or nc >= w:
                continue
            if obstacle_mask[nr, nc]:
                continue
            # Prevent cutting corners through blocked cells on diagonal moves.
            if dr != 0 and dc != 0 and connectivity == 8:
                if obstacle_mask[r, nc] or obstacle_mask[nr, c]:
                    continue
            add_cost = 0.0 if extra_cost is None else float(extra_cost[nr, nc])
            ng = g + step_cost + max(0.0, add_cost)
            if (nr, nc) not in cost or ng < cost[(nr, nc)]:
                cost[(nr, nc)] = ng
                came_from[(nr, nc)] = (r, c)
                heapq.heappush(open_set, (ng + heuristic(nr, nc), ng, (nr, nc)))
    return None


def inflate_mask(mask, radius_cells):
    if radius_cells <= 0:
        return mask
    h, w = mask.shape
    r = int(radius_cells)
    # Fast square dilation with integral image (O(H*W)).
    src = mask.astype(np.uint8)
    ii = np.pad(src, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)

    row_lo = np.clip(np.arange(h) - r, 0, h)
    row_hi = np.clip(np.arange(h) + r + 1, 0, h)
    col_lo = np.clip(np.arange(w) - r, 0, w)
    col_hi = np.clip(np.arange(w) + r + 1, 0, w)

    sum_window = (
        ii[row_hi[:, None], col_hi[None, :]]
        - ii[row_lo[:, None], col_hi[None, :]]
        - ii[row_hi[:, None], col_lo[None, :]]
        + ii[row_lo[:, None], col_lo[None, :]]
    )
    return sum_window > 0


def clear_mask_circle(mask, center_rc, radius_cells):
    if center_rc is None or radius_cells <= 0:
        return mask
    h, w = mask.shape
    rr0, cc0 = center_rc
    rr, cc = np.ogrid[:h, :w]
    circle = (rr - rr0) * (rr - rr0) + (cc - cc0) * (cc - cc0) <= int(radius_cells) * int(radius_cells)
    out = mask.copy()
    out[circle] = False
    return out
