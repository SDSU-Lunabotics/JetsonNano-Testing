"""
Mining automation subsystem for the ZED-based Lunabotics rover.

Provides MiningAutomation: a state machine that coordinates excavation
and deposit cycles using the live occupancy map.

Zone setup (on the map window):
  Use the Draw Excav Zone button, then click 4 corners  - define excavation zone
  Use the Draw Deposit Zone button, then click 4 corners - define deposit zone
  Use the Berm Left/Right buttons                      - stamp the official berm box
  Use the Pick Dig Start button, then click inside excavation - choose first strip
  Press 'r'                                                - start automated run
  Press 't'                                                - abort run at any time

Each cycle:
  1. Navigate to next dig strip waypoint    (A* guided, normal speed)
  2. Slow forward creep through strip       (DIGGING — simulates bucket fill)
  3. Short reverse backup                   (BACKUP)
  4. Navigate to pre-deposit approach point (A* guided, normal speed)
  5. Reverse into deposit zone              (DEPOSITING — treadmill on back)
  6. Repeat for all strips, then DONE
"""

import os
import json
import math
import enum

import numpy as np

try:
    import cv2
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------

class MiningState(enum.Enum):
    IDLE             = "IDLE"
    DRAW_EXCAV       = "DRAW_EXCAV"
    DRAW_DEPOSIT     = "DRAW_DEPOSIT"
    PICK_DIG_START   = "PICK_DIG_START"
    PLAN_SWEEP       = "PLAN_SWEEP"
    NAVIGATE_DIG     = "NAVIGATE_DIG"
    DIGGING          = "DIGGING"
    BACKUP           = "BACKUP"
    NAVIGATE_DEPOSIT = "NAVIGATE_DEPOSIT"
    DEPOSITING       = "DEPOSITING"
    DONE             = "DONE"
    ABORTED          = "ABORTED"


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class MiningAutomation:
    """
    Self-contained state machine for excavation + deposit automation.

    Parameters
    ----------
    cfg : dict
        Configuration keys (all optional, with defaults):
          dig_duration          float  5.0   Seconds of forward creep per strip
          dig_speed             float  0.20  Motor value during dig creep
          backup_duration       float  2.0   Seconds to reverse after dig
          backup_speed          float  0.35  Motor value during backup
          deposit_duration      float  5.0   Seconds reversing into deposit zone
          deposit_backup_speed  float  0.35  Motor value during deposit reverse
          deposit_approach_dist float  1.0   Fallback metres outside deposit center
          deposit_boundary_inset_m float 0.05 Rear edge inset into deposit zone
          continuous_runs       bool   True   Restart sweep after the last strip
          strip_pitch_m         float  0.0   Row spacing (0 = rover_size_m * 0.8)
          goal_tol_m            float  0.4   Goal-reached distance (m)
          rover_size_m          float  0.305 Rover footprint (m, square)
          zones_path            str    mining_zones.json zone persistence file
    occ_map : OccupancyMap
        Used for coordinate conversion and obstacle checking.
    """

    def __init__(self, cfg, occ_map):
        self.cfg = cfg
        self.state = MiningState.IDLE

        # Zone corners stored as (row, col) grid indices
        self.excav_corners_rc = []    # set after 4 clicks in DRAW_EXCAV
        self.deposit_corners_rc = []  # set after 4 clicks in DRAW_DEPOSIT
        self._click_buffer = []       # accumulates corners while drawing
        self.deposit_zone_preset_side = None

        # Dig sweep
        self.dig_points_rc = []       # boustrophedon waypoints inside excav zone
        self.dig_index = 0
        self.visited = set()          # set of completed dig_index values
        self.preferred_start_rc = None # optional user-picked dig start

        # Deposit approach waypoint (recomputed when None)
        self._deposit_approach_rc = None

        # Phase timer
        self.phase_start = 0.0

        # Zones file
        self.zones_path = cfg.get("zones_path", os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "mining_zones.json"
        ))

        # Try to restore zones from last run
        self.load_zones(occ_map)

    # -----------------------------------------------------------------------
    # Public API — called from zed_ground_wall.py
    # -----------------------------------------------------------------------

    def consume_click(self, row, col, occ_map):
        """
        Called when the user left-clicks on the map.
        Returns True when the click is consumed (suppresses normal goal assignment).
        """
        if self.state == MiningState.PICK_DIG_START:
            if not self.excav_corners_rc:
                print("[Mining] Set the excavation zone before picking a dig start.")
                self.state = MiningState.IDLE
                return True
            if not self._point_in_polygon_rc(
                row, col, self.excav_corners_rc, (occ_map.grid_h, occ_map.grid_w)
            ):
                print("[Mining] Dig start must be inside the excavation zone.")
                return True
            self.preferred_start_rc = (row, col)
            self.state = MiningState.IDLE
            self.dig_points_rc = []
            print(f"[Mining] Dig start selected at r={row} c={col}.")
            if self.excav_corners_rc and self.deposit_corners_rc:
                self.save_zones(occ_map)
            return True

        if self.state not in (MiningState.DRAW_EXCAV, MiningState.DRAW_DEPOSIT):
            return False

        self._click_buffer.append((row, col))
        n = len(self._click_buffer)
        zone_name = "Excav" if self.state == MiningState.DRAW_EXCAV else "Deposit"
        print(f"[Mining] {zone_name} corner {n}/4  r={row} c={col}")

        if n < 4:
            return True

        # Fourth corner — finalise zone
        corners = list(self._click_buffer)
        self._click_buffer = []

        if self.state == MiningState.DRAW_EXCAV:
            self.excav_corners_rc = corners
            self.preferred_start_rc = None
            print("[Mining] Excavation zone defined.")
        else:
            self.deposit_corners_rc = corners
            self.deposit_zone_preset_side = None
            self._deposit_approach_rc = None   # invalidate cached approach point
            print("[Mining] Deposit zone defined.")

        self.state = MiningState.IDLE

        if self.excav_corners_rc and self.deposit_corners_rc:
            self.save_zones(occ_map)
            print("[Mining] Both zones set. Press 'r' to start the run.")
        else:
            missing = "excavation" if not self.excav_corners_rc else "deposit"
            print(f"[Mining] Use the Draw {missing.title()} Zone button "
                  f"and click 4 corners to set the {missing} zone.")
        return True

    def start_draw_excavation(self):
        """Begin collecting four map clicks for the excavation zone."""
        self._start_draw(MiningState.DRAW_EXCAV, "excavation")

    def start_draw_deposit(self):
        """Begin collecting four map clicks for the deposit zone."""
        self._start_draw(MiningState.DRAW_DEPOSIT, "deposit")

    def set_deposit_zone_preset(self, side, occ_map):
        """Stamp the official berm scoring box using the configured arena side."""
        side_name = str(side or "").strip().lower()
        if side_name not in ("left", "right"):
            print(f"[Mining] Invalid berm preset side: {side}")
            return False
        if self._zones_edit_blocked():
            print(f"[Mining] Cannot redefine zones while running "
                  f"(state={self.state.value}). Press 't' to abort first.")
            return False

        center_x = self._cfg_float(
            f"berm_{side_name}_center_x_m",
            -6.80 if side_name == "left" else 6.80,
        )
        center_z = self._cfg_float("berm_center_z_m", 3.57)
        width_x = max(0.10, self._cfg_float("berm_width_m", 1.50))
        depth_z = max(0.10, self._cfg_float("berm_depth_m", 0.90))

        half_w = 0.5 * width_x
        half_d = 0.5 * depth_z
        corners_world = [
            (center_x - half_w, center_z - half_d),
            (center_x + half_w, center_z - half_d),
            (center_x + half_w, center_z + half_d),
            (center_x - half_w, center_z + half_d),
        ]

        corners_rc = []
        for world_x, world_z in corners_world:
            rc = occ_map.world_to_grid(float(world_x), float(world_z))
            if rc is None:
                print(
                    "[Mining] Berm preset is outside the current map bounds. "
                    "This preset assumes the map frame is aligned to the UCF field "
                    "origin at the divider/ingress corner."
                )
                return False
            corners_rc.append(rc)

        self.deposit_corners_rc = corners_rc
        self.deposit_zone_preset_side = side_name
        self._deposit_approach_rc = None
        self.state = MiningState.IDLE
        self._click_buffer = []
        self.save_zones(occ_map)
        print(
            f"[Mining] Deposit zone set from {side_name} arena berm preset "
            f"(center x={center_x:+.2f}m, z={center_z:.2f}m, size {width_x:.2f}m x {depth_z:.2f}m)."
        )
        print(
            "[Mining] Preset assumes the occupancy map is field-aligned from starting-zone "
            "localization; use landmarks for drift correction, not for absolute field placement."
        )
        return True

    def start_pick_dig_start(self):
        """Begin collecting one map click for the preferred dig start."""
        self._start_pick_dig_start()

    def start_run(self):
        """Start the automated excavation/deposit run."""
        self._start_run()

    def abort(self):
        """Abort the automated run or cancel an in-progress zone draw."""
        self._abort()

    def handle_key(self, key):
        """
        Called for mining keyboard actions. Returns True if consumed.

        Zone drawing is intentionally button-only so the driving keys are not
        overloaded by the mining setup controls.
        """
        if not HAS_CV2:
            return False
        if key == ord("r"):
            self._start_run()
            return True
        if key == ord("t"):
            self._abort()
            return True
        return False

    def tick(self, cam_rc, occ_map, now):
        """
        Advance the state machine by one frame.

        Parameters
        ----------
        cam_rc : (row, col) or None   Current rover grid cell.
        occ_map : OccupancyMap
        now : float                   Current timestamp (time.time()).

        Returns
        -------
        goal_rc_override : (row, col) or None
            If not None, override goal_cell to this value before path planning.
        drive_override : (fwd, turn) or None
            If not None, send this command directly and skip A* steering.
        status_str : str
            Short label for display.
        """
        s = self.state

        # --- Drawing modes ---
        if s in (MiningState.DRAW_EXCAV, MiningState.DRAW_DEPOSIT):
            zname = "EXCAV" if s == MiningState.DRAW_EXCAV else "DEPOSIT"
            n = len(self._click_buffer)
            return None, None, f"DRAW_{zname} ({n}/4)"
        if s == MiningState.PICK_DIG_START:
            return None, None, "PICK_DIG_START"

        # --- Idle / terminal ---
        if s == MiningState.IDLE:
            return None, None, "IDLE"
        if s == MiningState.DONE:
            return None, (0.0, 0.0), "DONE"
        if s == MiningState.ABORTED:
            return None, (0.0, 0.0), "ABORTED"

        # --- Plan sweep ---
        if s == MiningState.PLAN_SWEEP:
            self._generate_dig_points(occ_map)
            if not self.dig_points_rc:
                print("[Mining] No valid dig points; aborting.")
                self.state = MiningState.ABORTED
                return None, (0.0, 0.0), "ABORTED"
            self.dig_index = 0
            self.visited = set()
            self.state = MiningState.NAVIGATE_DIG
            total = len(self.dig_points_rc)
            print(f"[Mining] {total} dig waypoint(s) planned. Starting navigation.")
            return (self.dig_points_rc[0], None,
                    f"NAV_DIG 1/{total}")

        # --- Navigate to dig point ---
        if s == MiningState.NAVIGATE_DIG:
            if self.dig_index >= len(self.dig_points_rc):
                self.state = MiningState.DONE
                return None, (0.0, 0.0), "DONE"
            target_rc = self.dig_points_rc[self.dig_index]
            total = len(self.dig_points_rc)
            if cam_rc is not None:
                dist_m = self._dist_rc(cam_rc, target_rc, occ_map.map_res_m)
                if dist_m <= float(self.cfg.get("goal_tol_m", 0.4)):
                    print(f"[Mining] At dig point {self.dig_index + 1}/{total}. Digging.")
                    self.state = MiningState.DIGGING
                    self.phase_start = now
            return (target_rc, None, f"NAV_DIG {self.dig_index + 1}/{total}")

        # --- Digging: slow forward creep ---
        if s == MiningState.DIGGING:
            dig_speed = float(self.cfg.get("dig_speed", 0.20))
            dig_dur   = float(self.cfg.get("dig_duration", 5.0))
            elapsed   = now - self.phase_start
            if elapsed >= dig_dur:
                self.visited.add(self.dig_index)
                print(f"[Mining] Dig complete ({self.dig_index + 1}). Backing up.")
                self.state = MiningState.BACKUP
                self.phase_start = now
            pct = min(1.0, elapsed / max(0.01, dig_dur)) * 100
            return (None, (dig_speed, 0.0), f"DIGGING {pct:.0f}%")

        # --- Backup after dig ---
        if s == MiningState.BACKUP:
            bk_speed = float(self.cfg.get("backup_speed", 0.35))
            bk_dur   = float(self.cfg.get("backup_duration", 2.0))
            elapsed  = now - self.phase_start
            if elapsed >= bk_dur:
                print("[Mining] Backup done. Backing toward deposit zone.")
                self.state = MiningState.NAVIGATE_DEPOSIT
                self._deposit_approach_rc = None   # recompute fresh each deposit trip
            return (None, (-bk_speed, 0.0), f"BACKUP {elapsed:.1f}/{bk_dur:.1f}s")

        # --- Navigate to deposit approach waypoint ---
        if s == MiningState.NAVIGATE_DEPOSIT:
            if self._deposit_approach_rc is None:
                self._deposit_approach_rc = self._find_deposit_approach(occ_map)
                if self._deposit_approach_rc is None:
                    print("[Mining] Cannot compute deposit approach; aborting.")
                    self.state = MiningState.ABORTED
                    return None, (0.0, 0.0), "ABORTED"
            ap_rc = self._deposit_approach_rc
            if cam_rc is not None:
                dist_m = self._dist_rc(cam_rc, ap_rc, occ_map.map_res_m)
                if dist_m <= float(self.cfg.get("goal_tol_m", 0.4)):
                    print("[Mining] At deposit approach. Reversing into zone.")
                    self.state = MiningState.DEPOSITING
                    self.phase_start = now
            return (ap_rc, None, "NAV_DEPOSIT")

        # --- Depositing: reverse into zone (treadmill on back) ---
        if s == MiningState.DEPOSITING:
            dep_speed = float(self.cfg.get("deposit_backup_speed", 0.35))
            dep_dur   = float(self.cfg.get("deposit_duration", 5.0))
            elapsed   = now - self.phase_start
            if elapsed >= dep_dur:
                next_idx = self.dig_index + 1
                total = len(self.dig_points_rc)
                if next_idx < total:
                    self.dig_index = next_idx
                    print(f"[Mining] Deposit done. Navigating to dig point "
                          f"{next_idx + 1}/{total}.")
                    self.state = MiningState.NAVIGATE_DIG
                    self._deposit_approach_rc = None
                    return (self.dig_points_rc[self.dig_index], None,
                            f"NAV_DIG {self.dig_index + 1}/{total}")
                else:
                    continuous = self._cfg_bool("continuous_runs", True)
                    if continuous:
                        print("[Mining] All strips done. Restarting excavation/deposit cycle.")
                        self.state = MiningState.PLAN_SWEEP
                        self.dig_index = 0
                        self.visited = set()
                        self._deposit_approach_rc = None
                        return None, (0.0, 0.0), "CYCLE_NEXT"
                    print("[Mining] All strips done. DONE.")
                    self.state = MiningState.DONE
                    return None, (0.0, 0.0), "DONE"
            pct = min(1.0, elapsed / max(0.01, dep_dur)) * 100
            return (None, (-dep_speed, 0.0), f"DEPOSITING {pct:.0f}%")

        return None, None, self.state.value

    def render_overlay(self, map_vis, occ_map):
        """
        Draw zone boxes and dig waypoints onto map_vis.
        map_vis must be the raw grid frame (grid_h x grid_w) BEFORE apply_map_view
        and BEFORE zoom-resize. Coordinates are in raw grid pixel space.
        """
        if not HAS_CV2 or map_vis is None:
            return
        h, w = map_vis.shape[:2]

        def _clamp(r, c):
            return max(0, min(h - 1, r)), max(0, min(w - 1, c))

        def _draw_poly_outline(corners_rc, color):
            if len(corners_rc) < 2:
                return
            pts = np.array([[c, r] for r, c in corners_rc], dtype=np.int32)
            cv2.polylines(map_vis, [pts], True, color, 1, cv2.LINE_AA)

        # -- Excavation zone (orange outline) --
        if self.excav_corners_rc:
            _draw_poly_outline(self.excav_corners_rc, (0, 140, 255))
            cr, cc = self._poly_centroid(self.excav_corners_rc)
            cr, cc = _clamp(int(cr), int(cc))
            cv2.putText(map_vis, "EX", (max(0, cc - 6), max(8, cr + 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, (0, 140, 255), 1, cv2.LINE_AA)

        # -- Deposit zone (cyan/gold outline) --
        if self.deposit_corners_rc:
            _draw_poly_outline(self.deposit_corners_rc, (255, 220, 0))
            cr, cc = self._poly_centroid(self.deposit_corners_rc)
            cr, cc = _clamp(int(cr), int(cc))
            cv2.putText(map_vis, "DEP", (max(0, cc - 9), max(8, cr + 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, (255, 220, 0), 1, cv2.LINE_AA)

        # -- Deposit approach waypoint (magenta diamond) --
        if self._deposit_approach_rc is not None:
            apr, apc = self._deposit_approach_rc
            if 0 <= apr < h and 0 <= apc < w:
                sz = 3
                diamond = np.array(
                    [[apc, apr - sz], [apc + sz, apr],
                     [apc, apr + sz], [apc - sz, apr]], dtype=np.int32
                )
                cv2.polylines(map_vis, [diamond], True, (255, 0, 255), 1, cv2.LINE_AA)

        # -- User-selected preferred dig start (green crosshair) --
        if self.preferred_start_rc is not None:
            sr, sc = self.preferred_start_rc
            if 0 <= sr < h and 0 <= sc < w:
                cv2.drawMarker(
                    map_vis,
                    (int(sc), int(sr)),
                    (80, 255, 140),
                    cv2.MARKER_CROSS,
                    9,
                    1,
                    cv2.LINE_AA,
                )

        # -- Planned dig sweep path ("snake") --
        if len(self.dig_points_rc) >= 2:
            pts = []
            idxs = []
            for idx, (r, c) in enumerate(self.dig_points_rc):
                if 0 <= r < h and 0 <= c < w:
                    pts.append([int(c), int(r)])
                    idxs.append(idx)
            pts = np.array(pts, dtype=np.int32)
            if pts.shape[0] >= 2:
                for i in range(pts.shape[0] - 1):
                    p0 = pts[i]
                    p1 = pts[i + 1]
                    seg_done = idxs[i] in self.visited and idxs[i + 1] in self.visited
                    color = (150, 150, 150) if seg_done else (0, 190, 255)
                    cv2.line(
                        map_vis,
                        (int(p0[0]), int(p0[1])),
                        (int(p1[0]), int(p1[1])),
                        color,
                        1,
                        cv2.LINE_AA,
                    )
                    dx = int(p1[0] - p0[0])
                    dy = int(p1[1] - p0[1])
                    if dx * dx + dy * dy >= 25:
                        mid = ((int(p0[0]) + int(p1[0])) // 2, (int(p0[1]) + int(p1[1])) // 2)
                        ang = math.atan2(dy, dx)
                        wing = 3
                        a1 = ang + math.pi * 0.78
                        a2 = ang - math.pi * 0.78
                        q1 = (int(mid[0] + math.cos(a1) * wing), int(mid[1] + math.sin(a1) * wing))
                        q2 = (int(mid[0] + math.cos(a2) * wing), int(mid[1] + math.sin(a2) * wing))
                        cv2.line(map_vis, mid, q1, color, 1, cv2.LINE_AA)
                        cv2.line(map_vis, mid, q2, color, 1, cv2.LINE_AA)

        # -- Dig waypoints --
        active_states = (
            MiningState.NAVIGATE_DIG,
            MiningState.DIGGING,
            MiningState.BACKUP,
        )
        for i, (dr, dc) in enumerate(self.dig_points_rc):
            if not (0 <= dr < h and 0 <= dc < w):
                continue
            if i in self.visited:
                # White X for completed strips
                cv2.line(map_vis, (dc - 2, dr - 2), (dc + 2, dr + 2), (210, 210, 210), 1)
                cv2.line(map_vis, (dc + 2, dr - 2), (dc - 2, dr + 2), (210, 210, 210), 1)
            elif i == self.dig_index and self.state in active_states:
                # Yellow circle for active strip
                cv2.circle(map_vis, (dc, dr), 3, (0, 255, 255), 1)
            else:
                # Small grey dot for pending strips
                safe_r, safe_c = _clamp(dr, dc)
                map_vis[safe_r, safe_c] = (110, 110, 110)

        # -- Corner-click preview (in-progress drawing) --
        if self._click_buffer:
            col = (0, 140, 255) if self.state == MiningState.DRAW_EXCAV else (255, 220, 0)
            for cr, cc in self._click_buffer:
                if 0 <= cr < h and 0 <= cc < w:
                    cv2.circle(map_vis, (cc, cr), 2, col, -1)

    def render_status_banner(self, map_vis):
        """
        Draw a fixed automation task banner on the already positioned map view.
        Call this AFTER apply_map_view so it stays pinned at the top of the map.
        """
        if not HAS_CV2 or map_vis is None:
            return
        h, w = map_vis.shape[:2]
        if h <= 0 or w <= 0:
            return

        text = self._task_text()
        cv2.rectangle(map_vis, (0, 0), (w, 24), (0, 0, 0), -1)
        cv2.putText(
            map_vis,
            text,
            (6, 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    def _task_text(self):
        """Human-readable task text for the map banner."""
        s = self.state
        if s == MiningState.IDLE:
            if not self.excav_corners_rc:
                return "TASK: Set excavation zone"
            elif not self.deposit_corners_rc:
                return "TASK: Set deposit zone"
            else:
                return "TASK: Ready to start auto run"
        if s == MiningState.DRAW_EXCAV:
            return f"TASK: Draw excavation zone ({len(self._click_buffer)}/4)"
        if s == MiningState.DRAW_DEPOSIT:
            return f"TASK: Draw deposit zone ({len(self._click_buffer)}/4)"
        if s == MiningState.PICK_DIG_START:
            return "TASK: Pick dig start inside excavation zone"
        if s == MiningState.PLAN_SWEEP:
            return "TASK: Planning excavation path"
        if s == MiningState.NAVIGATE_DIG:
            total = len(self.dig_points_rc)
            if total:
                return f"TASK: Driving to dig point {self.dig_index + 1}/{total}"
            return "TASK: Driving to dig point"
        if s == MiningState.DIGGING:
            return "TASK: Digging"
        if s == MiningState.BACKUP:
            return "TASK: Backing away from dig strip"
        if s == MiningState.NAVIGATE_DEPOSIT:
            return "TASK: Backing to deposit edge"
        if s == MiningState.DEPOSITING:
            return "TASK: Depositing - backing into zone"
        if s == MiningState.DONE:
            return "TASK: Done"
        if s == MiningState.ABORTED:
            return "TASK: Aborted"
        return f"TASK: {s.value}"

    def save_zones(self, occ_map):
        """Persist both zone polygons as world-space (x, z) pairs to JSON."""
        def _rc_to_world(corners_rc):
            out = []
            for r, c in corners_rc:
                w = occ_map.grid_to_world(r, c)
                if w is not None:
                    out.append([float(w[0]), float(w[1])])
            return out

        try:
            data = {
                "excav":   _rc_to_world(self.excav_corners_rc),
                "deposit": _rc_to_world(self.deposit_corners_rc),
            }
            if self.deposit_zone_preset_side:
                data["deposit_zone_preset"] = str(self.deposit_zone_preset_side)
            if self.preferred_start_rc is not None:
                w_start = occ_map.grid_to_world(
                    self.preferred_start_rc[0],
                    self.preferred_start_rc[1],
                )
                if w_start is not None:
                    data["dig_start"] = [float(w_start[0]), float(w_start[1])]
            d = os.path.dirname(self.zones_path)
            if d:
                os.makedirs(d, exist_ok=True)
            with open(self.zones_path, "w") as f:
                json.dump(data, f, indent=2)
            print(f"[Mining] Zones saved → {self.zones_path}")
        except Exception as exc:
            print(f"[Mining] Zone save failed: {exc}")

    def load_zones(self, occ_map):
        """Restore zone polygons from JSON if the file exists."""
        if not os.path.exists(self.zones_path):
            return

        def _world_to_rc(pairs):
            out = []
            for xz in pairs:
                rc = occ_map.world_to_grid(float(xz[0]), float(xz[1]))
                if rc is not None:
                    out.append(rc)
            return out

        try:
            with open(self.zones_path) as f:
                data = json.load(f)
            loaded = False
            if "excav" in data and len(data["excav"]) == 4:
                rc = _world_to_rc(data["excav"])
                if len(rc) == 4:
                    self.excav_corners_rc = rc
                    loaded = True
            if "deposit" in data and len(data["deposit"]) == 4:
                rc = _world_to_rc(data["deposit"])
                if len(rc) == 4:
                    self.deposit_corners_rc = rc
                    loaded = True
            preset_side = str(data.get("deposit_zone_preset", "")).strip().lower()
            self.deposit_zone_preset_side = preset_side if preset_side in ("left", "right") else None
            if "dig_start" in data and len(data["dig_start"]) == 2:
                rc = occ_map.world_to_grid(
                    float(data["dig_start"][0]),
                    float(data["dig_start"][1]),
                )
                if rc is not None:
                    self.preferred_start_rc = rc
            if loaded:
                print(f"[Mining] Zones loaded ← {self.zones_path}")
        except Exception as exc:
            print(f"[Mining] Zone load failed: {exc}")

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _zones_edit_blocked(self):
        return self.state in (
            MiningState.PLAN_SWEEP,
            MiningState.NAVIGATE_DIG,
            MiningState.DIGGING,
            MiningState.BACKUP,
            MiningState.NAVIGATE_DEPOSIT,
            MiningState.DEPOSITING,
        )

    def _start_draw(self, draw_state, name):
        if self._zones_edit_blocked():
            print(f"[Mining] Cannot redefine zones while running "
                  f"(state={self.state.value}). Press 't' to abort first.")
            return
        self.state = draw_state
        self._click_buffer = []
        print(f"[Mining] Drawing {name} zone — click 4 corners on the map.")

    def _start_pick_dig_start(self):
        if self._zones_edit_blocked():
            print(f"[Mining] Cannot pick dig start while running "
                  f"(state={self.state.value}). Press 't' to abort first.")
            return
        if not self.excav_corners_rc:
            print("[Mining] Set the excavation zone before picking a dig start.")
            return
        self.state = MiningState.PICK_DIG_START
        self._click_buffer = []
        print("[Mining] Pick dig start — click inside the excavation zone.")

    def _start_run(self):
        if not self.excav_corners_rc:
            print("[Mining] Excavation zone not set. Use Draw Excav Zone, then click 4 corners.")
            return
        if not self.deposit_corners_rc:
            print("[Mining] Deposit zone not set. Use Draw Deposit Zone, then click 4 corners.")
            return
        running = (
            MiningState.PLAN_SWEEP,
            MiningState.NAVIGATE_DIG,
            MiningState.DIGGING,
            MiningState.BACKUP,
            MiningState.NAVIGATE_DEPOSIT,
            MiningState.DEPOSITING,
        )
        if self.state in running:
            print(f"[Mining] Already running (state={self.state.value}). "
                  "Press 't' to abort first.")
            return
        print("[Mining] Starting run — planning sweep...")
        self.state = MiningState.PLAN_SWEEP
        self.dig_index = 0
        self.visited = set()
        self._deposit_approach_rc = None

    def _abort(self):
        if self.state in (MiningState.IDLE, MiningState.DONE, MiningState.ABORTED,
                          MiningState.DRAW_EXCAV, MiningState.DRAW_DEPOSIT,
                          MiningState.PICK_DIG_START):
            self.state = MiningState.IDLE
            self._click_buffer = []
            return
        print(f"[Mining] Aborted from {self.state.value}.")
        self.state = MiningState.ABORTED

    @staticmethod
    def _dist_rc(a_rc, b_rc, res_m):
        """Distance in metres between two grid cells."""
        dr = a_rc[0] - b_rc[0]
        dc = a_rc[1] - b_rc[1]
        return math.hypot(dr, dc) * float(res_m)

    def _cfg_bool(self, key, default=False):
        """Parse bool-like config values from env/config dictionaries."""
        value = self.cfg.get(key, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        if isinstance(value, str):
            return value.strip().lower() not in ("0", "false", "no", "off", "")
        return bool(value)

    def _cfg_float(self, key, default):
        """Parse float-like config values from env/config dictionaries."""
        value = self.cfg.get(key, default)
        try:
            return float(value)
        except Exception:
            return float(default)

    @staticmethod
    def _poly_centroid(corners_rc):
        """Return (avg_row, avg_col) of a list of (row, col) corners."""
        rs = [r for r, c in corners_rc]
        cs = [c for r, c in corners_rc]
        return sum(rs) / len(rs), sum(cs) / len(cs)

    @staticmethod
    def _polygon_cells(corners_rc, grid_shape):
        """
        Rasterise a polygon defined by corners_rc into a list of (row, col) cells.
        Uses OpenCV's fillPoly on a temporary uint8 mask.
        Falls back to bounding-box approach if OpenCV is unavailable.
        """
        if len(corners_rc) < 3:
            return []
        h, w = grid_shape
        if HAS_CV2:
            mask = np.zeros((h, w), dtype=np.uint8)
            pts = np.array([[c, r] for r, c in corners_rc], dtype=np.int32)
            cv2.fillPoly(mask, [pts], 1)
            rows, cols = np.where(mask)
            return list(zip(rows.tolist(), cols.tolist()))
        # Fallback: bounding-box only (less accurate but works without cv2)
        min_r = max(0, min(r for r, c in corners_rc))
        max_r = min(h - 1, max(r for r, c in corners_rc))
        min_c = max(0, min(c for r, c in corners_rc))
        max_c = min(w - 1, max(c for r, c in corners_rc))
        cells = []
        for r in range(min_r, max_r + 1):
            for c in range(min_c, max_c + 1):
                cells.append((r, c))
        return cells

    @staticmethod
    def _point_in_polygon_rc(row, col, corners_rc, grid_shape):
        """Return True when a grid cell is inside the polygon."""
        if len(corners_rc) < 3:
            return False
        h, w = grid_shape
        if row < 0 or row >= h or col < 0 or col >= w:
            return False
        if HAS_CV2:
            pts = np.array([[c, r] for r, c in corners_rc], dtype=np.int32)
            return cv2.pointPolygonTest(pts, (float(col), float(row)), False) >= 0
        min_r = max(0, min(r for r, c in corners_rc))
        max_r = min(h - 1, max(r for r, c in corners_rc))
        min_c = max(0, min(c for r, c in corners_rc))
        max_c = min(w - 1, max(c for r, c in corners_rc))
        return min_r <= row <= max_r and min_c <= col <= max_c

    @staticmethod
    def _ray_polygon_boundary_distance(origin_x, origin_z, dir_x, dir_z, poly_world):
        """
        Return distance from origin to the first polygon edge hit by a ray.
        poly_world is a list of (x, z) vertices in polygon order.
        """
        if len(poly_world) < 3:
            return None

        def _cross(ax, az, bx, bz):
            return ax * bz - az * bx

        best_t = None
        eps = 1e-6
        for i, (x1, z1) in enumerate(poly_world):
            x2, z2 = poly_world[(i + 1) % len(poly_world)]
            edge_x = x2 - x1
            edge_z = z2 - z1
            denom = _cross(dir_x, dir_z, edge_x, edge_z)
            if abs(denom) < eps:
                continue

            rel_x = x1 - origin_x
            rel_z = z1 - origin_z
            t = _cross(rel_x, rel_z, edge_x, edge_z) / denom
            u = _cross(rel_x, rel_z, dir_x, dir_z) / denom
            if t >= -eps and -eps <= u <= 1.0 + eps:
                if best_t is None or t < best_t:
                    best_t = max(0.0, t)

        return best_t

    def _generate_dig_points(self, occ_map):
        """
        Quarry-mode dig planning: find the largest contiguous obstacle-free
        region inside the excavation zone, then sweep it with a serpentine
        (boustrophedon) pattern.

        Focusing on the single largest clear blob means:
          - Every waypoint is reachable without crossing obstacles
          - Small pockets near walls/rocks are ignored automatically
          - Maximum material per run with minimum crash risk

        Steps:
          1. Rasterise the excav polygon onto a mask
          2. Zero out any cells in the obstacle mask (with an extra erosion
             margin of one rover-width to keep the path clear of edges)
          3. Run connected-component labelling — pick the largest blob
          4. Generate boustrophedon waypoints across that blob only
        """
        self.dig_points_rc = []
        if not self.excav_corners_rc:
            return

        grid_shape = (occ_map.grid_h, occ_map.grid_w)

        # --- 1. Polygon mask ---
        cells = self._polygon_cells(self.excav_corners_rc, grid_shape)
        if not cells:
            print("[Mining] Excavation polygon produced no grid cells.")
            return

        poly_mask = np.zeros(grid_shape, dtype=np.uint8)
        for r, c in cells:
            poly_mask[r, c] = 1

        # --- 2. Remove obstacles + erode by half a rover-width for safety margin ---
        rover_size_m  = float(self.cfg.get("rover_size_m", 0.305))
        strip_pitch_m = float(self.cfg.get("strip_pitch_m", 0.0))
        if strip_pitch_m <= 0.0:
            strip_pitch_m = max(0.05, rover_size_m * 0.8)
        step_cells = max(1, int(math.ceil(strip_pitch_m / occ_map.map_res_m)))

        obs = occ_map.obstacle_mask()
        free_mask = poly_mask.copy()
        free_mask[obs] = 0  # zero out obstacle cells

        # Erode by rover radius so waypoints never sit right on an obstacle edge.
        margin_cells = max(1, int(math.ceil((rover_size_m * 0.5) / occ_map.map_res_m)))
        if HAS_CV2:
            kernel = np.ones((margin_cells * 2 + 1, margin_cells * 2 + 1), np.uint8)
            free_mask = cv2.erode(free_mask, kernel, iterations=1)

        if not np.any(free_mask):
            print("[Mining] No obstacle-free cells inside excavation zone after erosion.")
            self.dig_points_rc = []
            return

        # --- 3. Largest connected component (quarry selection) ---
        if HAS_CV2:
            num_labels, label_img = cv2.connectedComponents(free_mask, connectivity=8)
            if num_labels <= 1:
                print("[Mining] No clear region found inside excavation zone.")
                return
            # Label 0 is background — find largest non-background label.
            best_label = 1 + int(
                np.argmax([np.sum(label_img == lbl) for lbl in range(1, num_labels)])
            )
            quarry_mask = (label_img == best_label).astype(np.uint8)
            quarry_cells_count = int(np.sum(quarry_mask))
            total_free = int(np.sum(free_mask))
            pct = 100.0 * quarry_cells_count / max(1, total_free)
            print(f"[Mining] Quarry blob: {quarry_cells_count} cells "
                  f"({pct:.0f}% of free area, {num_labels - 1} region(s) found).")
        else:
            # No cv2 — fall back to using all free cells.
            quarry_mask = free_mask
            print("[Mining] cv2 unavailable — using all free cells (no blob selection).")

        rows_arr, cols_arr = np.where(quarry_mask)

        # --- 4. Boustrophedon sweep across the quarry blob ---
        unique_rows = sorted(set(int(v) for v in rows_arr))
        selected_rows = unique_rows[::step_cells]

        points = []
        for i, r in enumerate(selected_rows):
            row_cols = sorted(int(cols_arr[j]) for j in range(len(rows_arr))
                              if rows_arr[j] == r)
            if not row_cols:
                continue
            if i % 2 == 1:
                row_cols = row_cols[::-1]
            mid_c = row_cols[len(row_cols) // 2]
            points.append((r, mid_c))

        if points and self.preferred_start_rc is not None:
            sr, sc = self.preferred_start_rc
            start_idx = min(
                range(len(points)),
                key=lambda idx: (points[idx][0] - sr) ** 2 + (points[idx][1] - sc) ** 2,
            )
            if start_idx > 0:
                points = points[start_idx:] + points[:start_idx]
            start_r, start_c = points[0]
            print(f"[Mining] Sweep starts near picked dig start "
                  f"(r={start_r} c={start_c}).")

        self.dig_points_rc = points
        print(f"[Mining] {len(points)} dig waypoints "
              f"(step={step_cells} cells ≈ {strip_pitch_m:.2f} m/strip).")

    def _find_deposit_approach(self, occ_map):
        """
        Compute the deposit-edge waypoint on the excavation side.

        The rover center is placed so its rear edge is just inside the
        deposit polygon boundary. That makes the rover footprint touch the
        drawn deposit zone instead of stopping at a fixed centroid offset.

        Returns (row, col) or None on failure.
        """
        if not self.deposit_corners_rc or not self.excav_corners_rc:
            return None

        # Centroids in grid space
        dep_r, dep_c = self._poly_centroid(self.deposit_corners_rc)
        exc_r, exc_c = self._poly_centroid(self.excav_corners_rc)

        # Convert to world (x, z)
        dep_world = occ_map.grid_to_world(int(dep_r), int(dep_c))
        exc_world = occ_map.grid_to_world(int(exc_r), int(exc_c))
        if dep_world is None or exc_world is None:
            return None

        dep_x, dep_z = dep_world
        exc_x, exc_z = exc_world

        # Direction from deposit toward excavation (rover faces this way to
        # reach the approach point, so its back points toward deposit)
        dx = exc_x - dep_x
        dz = exc_z - dep_z
        length = math.hypot(dx, dz)
        if length < 1e-6:
            return None
        dx /= length
        dz /= length

        deposit_poly_world = []
        for r, c in self.deposit_corners_rc:
            w = occ_map.grid_to_world(r, c)
            if w is not None:
                deposit_poly_world.append((float(w[0]), float(w[1])))

        boundary_dist = self._ray_polygon_boundary_distance(
            dep_x, dep_z, dx, dz, deposit_poly_world
        )

        rover_size_m = max(0.01, float(self.cfg.get("rover_size_m", 0.305)))
        rover_half_m = rover_size_m * 0.5
        inset_m = max(0.0, float(self.cfg.get("deposit_boundary_inset_m", 0.05)))
        inset_m = min(inset_m, max(0.0, rover_half_m - 0.01))

        if boundary_dist is None:
            approach_dist = float(self.cfg.get("deposit_approach_dist", 1.0))
            ap_dist = max(0.0, approach_dist)
            print("[Mining] Deposit edge intersection failed; using fallback "
                  f"approach distance {ap_dist:.2f} m.")
        else:
            ap_dist = boundary_dist + max(0.0, rover_half_m - inset_m)

        # Approach point in world space. The rear bumper will sit at
        # boundary - inset_m along this ray when the rover reaches the point.
        ap_x = dep_x + dx * ap_dist
        ap_z = dep_z + dz * ap_dist

        # Convert to grid
        rc = occ_map.world_to_grid(ap_x, ap_z)
        if rc is None:
            # Clamp near the grid boundary and retry
            ap_x = max(occ_map.x_min + occ_map.map_res_m,
                       min(occ_map.x_max - occ_map.map_res_m, ap_x))
            ap_z = max(occ_map.z_min + occ_map.map_res_m,
                       min(occ_map.z_max - occ_map.map_res_m, ap_z))
            rc = occ_map.world_to_grid(ap_x, ap_z)
        return rc
