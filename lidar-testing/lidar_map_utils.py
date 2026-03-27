import numpy as np


class OccupancyMap:
    def __init__(
        self,
        map_res_m,
        map_width_m,
        map_height_m,
        map_forward_min,
        decay=0.98,
        free_decay=None,
        occ_decay=None,
        hole_decay=None,
    ):
        self.map_res_m = float(map_res_m)
        self.map_width_m = float(map_width_m)
        self.map_height_m = float(map_height_m)
        self.map_forward_min = float(map_forward_min)
        self.map_decay = float(decay)
        self.free_decay = self.map_decay if free_decay is None else float(free_decay)
        self.occ_decay = self.map_decay if occ_decay is None else float(occ_decay)
        self.hole_decay = self.map_decay if hole_decay is None else float(hole_decay)

        self.lat_min = -self.map_width_m / 2.0
        self.lat_max = self.map_width_m / 2.0
        self.fwd_min = self.map_forward_min
        self.fwd_max = self.map_forward_min + self.map_height_m

        self.grid_w = int(np.round(self.map_width_m / self.map_res_m))
        self.grid_h = int(np.round(self.map_height_m / self.map_res_m))

        self.free_counts = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        self.occ_counts = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        self.hole_counts = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)

    def meta(self):
        return {
            "map_res_m": self.map_res_m,
            "map_width_m": self.map_width_m,
            "map_height_m": self.map_height_m,
            "map_forward_min": self.map_forward_min,
        }

    def load(self, path):
        data = np.load(path, allow_pickle=True)
        meta = data["meta"].item()
        if (
            float(meta.get("map_res_m", -1.0)) != self.map_res_m
            or float(meta.get("map_width_m", -1.0)) != self.map_width_m
            or float(meta.get("map_height_m", -1.0)) != self.map_height_m
            or float(meta.get("map_forward_min", -1.0e9)) != self.map_forward_min
        ):
            return False, "Map settings differ; starting with empty map"

        free = data["free_counts"].astype(np.float32)
        occ = data["occ_counts"].astype(np.float32)
        hole = data["hole_counts"].astype(np.float32) if "hole_counts" in data else None
        if free.shape != self.free_counts.shape or occ.shape != self.occ_counts.shape:
            return False, "Map shape mismatch; starting with empty map"

        self.free_counts[:] = free
        self.occ_counts[:] = occ
        if hole is not None and hole.shape == self.hole_counts.shape:
            self.hole_counts[:] = hole
        return True, "Loaded map"

    def save(self, path):
        np.savez_compressed(
            path,
            free_counts=self.free_counts,
            occ_counts=self.occ_counts,
            hole_counts=self.hole_counts,
            meta=self.meta(),
        )

    def world_to_grid(self, lat, fwd):
        if lat < self.lat_min or lat >= self.lat_max or fwd < self.fwd_min or fwd >= self.fwd_max:
            return None
        col = int((lat - self.lat_min) / self.map_res_m)
        row = int(self.grid_h - 1 - ((fwd - self.fwd_min) / self.map_res_m))
        if row < 0 or row >= self.grid_h or col < 0 or col >= self.grid_w:
            return None
        return row, col

    def update(self, lat, fwd, ground_mask, obstacle_mask, hole_mask):
        in_bounds = (
            (lat >= self.lat_min)
            & (lat < self.lat_max)
            & (fwd >= self.fwd_min)
            & (fwd < self.fwd_max)
        )
        if not np.any(in_bounds):
            return

        lat = lat[in_bounds]
        fwd = fwd[in_bounds]
        gmask = ground_mask[in_bounds]
        omask = obstacle_mask[in_bounds]
        hmask = hole_mask[in_bounds]

        col = ((lat - self.lat_min) / self.map_res_m).astype(np.int32)
        iz = ((fwd - self.fwd_min) / self.map_res_m).astype(np.int32)
        row = self.grid_h - 1 - iz

        self.free_counts *= self.free_decay
        self.occ_counts *= self.occ_decay
        self.hole_counts *= self.hole_decay

        if np.any(gmask):
            np.add.at(self.free_counts, (row[gmask], col[gmask]), 1.0)
        if np.any(omask):
            np.add.at(self.occ_counts, (row[omask], col[omask]), 1.0)
        if np.any(hmask):
            np.add.at(self.hole_counts, (row[hmask], col[hmask]), 1.0)

    def render(self):
        free_vis = np.log1p(self.free_counts)
        occ_vis = np.log1p(self.occ_counts)
        hole_vis = np.log1p(self.hole_counts)

        if free_vis.max() > 0:
            free_vis /= free_vis.max()
        if occ_vis.max() > 0:
            occ_vis /= occ_vis.max()
        if hole_vis.max() > 0:
            hole_vis /= hole_vis.max()

        out = np.zeros((self.grid_h, self.grid_w, 3), dtype=np.uint8)
        out[:, :, 1] = (free_vis * 255.0).astype(np.uint8)
        out[:, :, 2] = (occ_vis * 255.0).astype(np.uint8)
        out[:, :, 0] = (hole_vis * 255.0).astype(np.uint8)
        return out
