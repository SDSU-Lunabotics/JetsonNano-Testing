import numpy as np


class OccupancyMap:
    def __init__(self, map_res_m, map_width_m, map_height_m, map_z_min, decay=0.97):
        self.map_res_m = map_res_m
        self.map_width_m = map_width_m
        self.map_height_m = map_height_m
        self.map_z_min = map_z_min
        self.map_decay = decay

        self.x_min = -map_width_m / 2.0
        self.x_max = map_width_m / 2.0
        self.z_min = map_z_min
        self.z_max = map_z_min + map_height_m

        self.grid_w = int(map_width_m / map_res_m)
        self.grid_h = int(map_height_m / map_res_m)

        self.free_counts = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        self.occ_counts = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)

    def meta(self):
        return {
            "map_res_m": self.map_res_m,
            "map_width_m": self.map_width_m,
            "map_height_m": self.map_height_m,
            "map_z_min": self.map_z_min,
        }

    def load(self, path):
        data = np.load(path)
        loaded_free = data["free_counts"].astype(np.float32)
        loaded_occ = data["occ_counts"].astype(np.float32)
        meta = data["meta"].item()
        if (
            meta.get("map_res_m") == self.map_res_m
            and meta.get("map_width_m") == self.map_width_m
            and meta.get("map_height_m") == self.map_height_m
            and meta.get("map_z_min") == self.map_z_min
        ):
            if loaded_free.shape == self.free_counts.shape and loaded_occ.shape == self.occ_counts.shape:
                self.free_counts[:] = loaded_free
                self.occ_counts[:] = loaded_occ
                return True, "Loaded map"
            return False, "Map size mismatch; starting with empty map."
        return False, "Map settings differ; starting with empty map."

    def save(self, path):
        np.savez_compressed(
            path,
            free_counts=self.free_counts,
            occ_counts=self.occ_counts,
            meta=self.meta(),
        )

    def update(self, x, z, ground_mask, obstacle_mask):
        in_bounds = (x >= self.x_min) & (x < self.x_max) & (z >= self.z_min) & (z < self.z_max)
        if not np.any(in_bounds):
            return
        x = x[in_bounds]
        z = z[in_bounds]
        gmask = ground_mask[in_bounds]
        omask = obstacle_mask[in_bounds]

        ix = ((x - self.x_min) / self.map_res_m).astype(np.int32)
        iz = ((z - self.z_min) / self.map_res_m).astype(np.int32)
        # Flip Z so forward is "up" in the image.
        row = self.grid_h - 1 - iz
        col = ix

        self.free_counts *= self.map_decay
        self.occ_counts *= self.map_decay

        if np.any(gmask):
            self.free_counts[row[gmask], col[gmask]] += 1.0
        if np.any(omask):
            self.occ_counts[row[omask], col[omask]] += 1.0

    def render(self):
        # Visualize: green = free, red = occupied, dark = unknown.
        free_vis = np.log1p(self.free_counts)
        occ_vis = np.log1p(self.occ_counts)
        fmax = free_vis.max()
        omax = occ_vis.max()
        if fmax > 0:
            free_vis = free_vis / fmax
        if omax > 0:
            occ_vis = occ_vis / omax
        map_vis = np.zeros((self.grid_h, self.grid_w, 3), dtype=np.uint8)
        # Red channel for occupied, green channel for free.
        map_vis[:, :, 1] = (free_vis * 255.0).astype(np.uint8)
        map_vis[:, :, 2] = (occ_vis * 255.0).astype(np.uint8)
        return map_vis
