import bisect
import copy
import json
import os
import re
import time


def _write_json_atomic(path, payload):
    if not path:
        return
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    os.replace(tmp_path, path)


def _read_json(path, default):
    if not path or (not os.path.exists(path)):
        return copy.deepcopy(default)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return copy.deepcopy(default)


class DriveCalibrationManager:
    def __init__(self, path):
        self.path = path
        self.active = False
        self.target_cell = None
        self.start_xz = None
        self.started_at = 0.0
        self.flip_checked = False
        self.last_result = "Calibration idle."
        self.last_saved_flip = None
        self.last_saved_hard_drive_flip = None
        self.last_saved_steering_flip = None
        self.last_saved_display_heading_flip = None
        self.last_saved_camera_map_angle_deg = None
        self.last_saved_camera_deposit_angle_deg = None
        self.min_progress_m = 0.35
        self.goal_tol_m = 0.30
        self.timeout_sec = 10.0
        self._load()

    def _load(self):
        data = _read_json(
            self.path,
            {
                "version": 1,
                "drive_heading_flip": None,
                "hard_drive_flip": None,
                "steering_flip": None,
                "display_heading_flip": None,
                "camera_map_angle_deg": None,
                "camera_deposit_angle_deg": None,
                "updated_ms": 0,
                "last_result": self.last_result,
            },
        )
        value = data.get("drive_heading_flip")
        if isinstance(value, bool):
            self.last_saved_flip = bool(value)
        hard_flip = data.get("hard_drive_flip")
        if isinstance(hard_flip, bool):
            self.last_saved_hard_drive_flip = bool(hard_flip)
        steering_flip = data.get("steering_flip")
        if isinstance(steering_flip, bool):
            self.last_saved_steering_flip = bool(steering_flip)
        display_flip = data.get("display_heading_flip")
        if isinstance(display_flip, bool):
            self.last_saved_display_heading_flip = bool(display_flip)
        map_angle = data.get("camera_map_angle_deg")
        if isinstance(map_angle, (int, float)):
            self.last_saved_camera_map_angle_deg = float(map_angle)
        deposit_angle = data.get("camera_deposit_angle_deg")
        if isinstance(deposit_angle, (int, float)):
            self.last_saved_camera_deposit_angle_deg = float(deposit_angle)
        self.last_result = str(data.get("last_result", self.last_result))

    def _write_settings(self):
        _write_json_atomic(
            self.path,
            {
                "version": 1,
                "drive_heading_flip": self.last_saved_flip,
                "hard_drive_flip": self.last_saved_hard_drive_flip,
                "steering_flip": self.last_saved_steering_flip,
                "display_heading_flip": self.last_saved_display_heading_flip,
                "camera_map_angle_deg": self.last_saved_camera_map_angle_deg,
                "camera_deposit_angle_deg": self.last_saved_camera_deposit_angle_deg,
                "updated_ms": int(time.time() * 1000),
                "last_result": self.last_result,
            },
        )

    def save_runtime_settings(
        self,
        drive_heading_flip,
        hard_drive_flip,
        steering_flip,
        display_heading_flip,
        camera_map_angle_deg,
        camera_deposit_angle_deg,
        result_text,
    ):
        self.last_saved_flip = None if drive_heading_flip is None else bool(drive_heading_flip)
        self.last_saved_hard_drive_flip = None if hard_drive_flip is None else bool(hard_drive_flip)
        self.last_saved_steering_flip = None if steering_flip is None else bool(steering_flip)
        self.last_saved_display_heading_flip = (
            None if display_heading_flip is None else bool(display_heading_flip)
        )
        self.last_saved_camera_map_angle_deg = (
            None if camera_map_angle_deg is None else float(camera_map_angle_deg)
        )
        self.last_saved_camera_deposit_angle_deg = (
            None if camera_deposit_angle_deg is None else float(camera_deposit_angle_deg)
        )
        self.last_result = str(result_text)
        self._write_settings()

    def save_result(self, drive_heading_flip, result_text):
        self.save_runtime_settings(
            drive_heading_flip,
            self.last_saved_hard_drive_flip,
            self.last_saved_steering_flip,
            self.last_saved_display_heading_flip,
            self.last_saved_camera_map_angle_deg,
            self.last_saved_camera_deposit_angle_deg,
            result_text,
        )

    def set_active(self, enabled):
        self.active = bool(enabled)
        self.clear_target("Calibration canceled." if not self.active else "Calibration armed. Click a drive target on the map.")

    def clear_target(self, result_text=None):
        self.target_cell = None
        self.start_xz = None
        self.started_at = 0.0
        self.flip_checked = False
        if result_text:
            self.last_result = str(result_text)

    def set_target(self, row, col):
        self.target_cell = (int(row), int(col))
        self.start_xz = None
        self.started_at = 0.0
        self.flip_checked = False
        self.last_result = f"Calibration target set: r={int(row)} c={int(col)}"

    def update(self, rover_xz, goal_xz, drive_heading_flip):
        if (not self.active) or self.target_cell is None or rover_xz is None or goal_xz is None:
            return None
        now = time.time()
        rover_xz = (float(rover_xz[0]), float(rover_xz[1]))
        goal_xz = (float(goal_xz[0]), float(goal_xz[1]))
        if self.start_xz is None:
            self.start_xz = rover_xz
            self.started_at = now
            self.last_result = "Calibration running. Drive motion is being evaluated."
            return None

        disp_x = rover_xz[0] - self.start_xz[0]
        disp_z = rover_xz[1] - self.start_xz[1]
        target_x = goal_xz[0] - self.start_xz[0]
        target_z = goal_xz[1] - self.start_xz[1]
        disp_norm = (disp_x * disp_x + disp_z * disp_z) ** 0.5
        target_norm = (target_x * target_x + target_z * target_z) ** 0.5
        goal_dx = goal_xz[0] - rover_xz[0]
        goal_dz = goal_xz[1] - rover_xz[1]
        goal_dist = (goal_dx * goal_dx + goal_dz * goal_dz) ** 0.5

        if (not self.flip_checked) and disp_norm >= self.min_progress_m and target_norm >= 1e-6:
            dot = ((disp_x * target_x) + (disp_z * target_z)) / max(1e-6, disp_norm * target_norm)
            self.flip_checked = True
            if dot < -0.20:
                new_flip = not bool(drive_heading_flip)
                self.save_result(
                    new_flip,
                    f"Drive calibration toggled heading flip to {'ON' if new_flip else 'OFF'}. Click again to verify.",
                )
                self.clear_target(self.last_result)
                return {
                    "apply_drive_heading_flip": new_flip,
                    "clear_goal": True,
                    "message": self.last_result,
                }
            self.last_result = "Drive direction matches the clicked target."

        if goal_dist <= self.goal_tol_m:
            self.save_result(
                bool(drive_heading_flip),
                f"Drive calibration passed with heading flip {'ON' if drive_heading_flip else 'OFF'}.",
            )
            self.clear_target(self.last_result)
            return {
                "clear_goal": False,
                "message": self.last_result,
            }

        if self.started_at > 0.0 and (now - self.started_at) >= self.timeout_sec:
            self.last_result = "Drive calibration timed out. Click a nearer target and retry."
            self.clear_target(self.last_result)
            return {
                "clear_goal": False,
                "message": self.last_result,
            }
        return None

    def ui_state(self):
        return {
            "active": bool(self.active),
            "target_cell": list(self.target_cell) if self.target_cell is not None else None,
            "last_result": self.last_result,
            "saved_drive_heading_flip": self.last_saved_flip,
            "saved_hard_drive_flip": self.last_saved_hard_drive_flip,
            "saved_steering_flip": self.last_saved_steering_flip,
            "saved_display_heading_flip": self.last_saved_display_heading_flip,
            "saved_camera_map_angle_deg": self.last_saved_camera_map_angle_deg,
            "saved_camera_deposit_angle_deg": self.last_saved_camera_deposit_angle_deg,
        }


class DigProfileLibrary:
    VALID_STYLES = ("short", "long")
    VALID_PHASES = ("dig", "retract")

    def __init__(self, path):
        self.path = path
        self.profiles = []
        self.selected = {
            style: {phase: None for phase in self.VALID_PHASES}
            for style in self.VALID_STYLES
        }
        self.cursor = {
            style: {phase: None for phase in self.VALID_PHASES}
            for style in self.VALID_STYLES
        }
        self.active_style = "long"
        self.active_phase = "dig"
        self.recording = False
        self.recording_style = None
        self.recording_phase = None
        self.recording_name_base = None
        self.recording_started_at = 0.0
        self.recording_samples = []
        self.last_recorded_signature = None
        self.last_recorded_t = -1.0
        self.sample_period_sec = 0.05
        self._load()

    def _default_payload(self):
        return {
            "version": 1,
            "active_style": self.active_style,
            "active_phase": self.active_phase,
            "selected": copy.deepcopy(self.selected),
            "cursor": copy.deepcopy(self.cursor),
            "profiles": [],
        }

    def _load(self):
        data = _read_json(self.path, self._default_payload())
        profiles = data.get("profiles", [])
        self.profiles = []
        for profile in profiles:
            if not isinstance(profile, dict) or not isinstance(profile.get("name"), str):
                continue
            item = copy.deepcopy(profile)
            phase = str(item.get("phase", "dig")).strip().lower()
            item["phase"] = phase if phase in self.VALID_PHASES else "dig"
            style = str(item.get("style", "")).strip().lower()
            if style not in self.VALID_STYLES:
                continue
            self.profiles.append(item)
        for style in self.VALID_STYLES:
            selected_entry = data.get("selected", {}).get(style)
            cursor_entry = data.get("cursor", {}).get(style)
            if isinstance(selected_entry, dict):
                for phase in self.VALID_PHASES:
                    selected_name = selected_entry.get(phase)
                    self.selected[style][phase] = selected_name if self.get_profile(selected_name) else None
            else:
                self.selected[style]["dig"] = selected_entry if self.get_profile(selected_entry) else None
            if isinstance(cursor_entry, dict):
                for phase in self.VALID_PHASES:
                    cursor_name = cursor_entry.get(phase)
                    self.cursor[style][phase] = cursor_name if self.get_profile(cursor_name) else None
            else:
                self.cursor[style]["dig"] = cursor_entry if self.get_profile(cursor_entry) else None
        active_style = str(data.get("active_style", self.active_style))
        if active_style in self.VALID_STYLES:
            self.active_style = active_style
        active_phase = str(data.get("active_phase", self.active_phase))
        if active_phase in self.VALID_PHASES:
            self.active_phase = active_phase
        self._normalize_cursor_state()

    def _save(self):
        _write_json_atomic(
            self.path,
            {
                "version": 1,
                "active_style": self.active_style,
                "active_phase": self.active_phase,
                "selected": copy.deepcopy(self.selected),
                "cursor": copy.deepcopy(self.cursor),
                "profiles": copy.deepcopy(self.profiles),
                "updated_ms": int(time.time() * 1000),
            },
        )

    def _normalize_style(self, style):
        style = str(style or self.active_style).strip().lower()
        return style if style in self.VALID_STYLES else self.active_style

    def _normalize_phase(self, phase):
        phase = str(phase or self.active_phase).strip().lower()
        return phase if phase in self.VALID_PHASES else self.active_phase

    def _normalize_cursor_state(self):
        for style in self.VALID_STYLES:
            for phase in self.VALID_PHASES:
                names = [profile["name"] for profile in self.list_profiles(style, phase)]
                if self.selected[style][phase] not in names:
                    self.selected[style][phase] = names[0] if names else None
                if self.cursor[style][phase] not in names:
                    self.cursor[style][phase] = self.selected[style][phase] or (names[0] if names else None)

    def _slugify_name(self, name):
        raw = str(name or "").strip().lower()
        if not raw:
            return ""
        raw = raw.replace(" ", "_")
        raw = re.sub(r"[^a-z0-9_\-]+", "", raw)
        raw = re.sub(r"_+", "_", raw).strip("_-")
        return raw

    def _make_unique_name(self, base_name):
        existing = {str(profile.get("name", "")) for profile in self.profiles}
        if base_name not in existing:
            return base_name
        idx = 2
        while True:
            candidate = f"{base_name}_{idx}"
            if candidate not in existing:
                return candidate
            idx += 1

    def list_profiles(self, style=None, phase=None):
        style = self._normalize_style(style)
        phase = self._normalize_phase(phase)
        items = [
            profile
            for profile in self.profiles
            if str(profile.get("style", "")).lower() == style
            and str(profile.get("phase", "dig")).lower() == phase
        ]
        return sorted(items, key=lambda item: int(item.get("created_ms", 0)), reverse=True)

    def get_profile(self, name):
        if not name:
            return None
        for profile in self.profiles:
            if profile.get("name") == name:
                return profile
        return None

    def get_selected_profile(self, style=None, phase=None):
        style = self._normalize_style(style)
        phase = self._normalize_phase(phase)
        return self.get_profile(self.selected.get(style, {}).get(phase))

    def get_cursor_profile(self, style=None, phase=None):
        style = self._normalize_style(style)
        phase = self._normalize_phase(phase)
        return self.get_profile(self.cursor.get(style, {}).get(phase))

    def cycle_active_style(self):
        self.active_style = "short" if self.active_style == "long" else "long"
        self._normalize_cursor_state()
        self._save()
        return self.active_style

    def cycle_active_phase(self):
        self.active_phase = "retract" if self.active_phase == "dig" else "dig"
        self._normalize_cursor_state()
        self._save()
        return self.active_phase

    def cycle_cursor(self, step, style=None, phase=None):
        style = self._normalize_style(style)
        phase = self._normalize_phase(phase)
        items = self.list_profiles(style, phase)
        if not items:
            self.cursor[style][phase] = None
            self._save()
            return None
        names = [profile["name"] for profile in items]
        current = self.cursor.get(style, {}).get(phase)
        try:
            idx = names.index(current)
        except ValueError:
            idx = 0
        idx = (idx + int(step)) % len(names)
        self.cursor[style][phase] = names[idx]
        self._save()
        return self.get_cursor_profile(style, phase)

    def select_cursor(self, style=None, phase=None):
        style = self._normalize_style(style)
        phase = self._normalize_phase(phase)
        cursor_profile = self.get_cursor_profile(style, phase)
        if cursor_profile is None:
            return None
        self.selected[style][phase] = cursor_profile["name"]
        self._save()
        return cursor_profile

    def delete_cursor(self, style=None, phase=None):
        style = self._normalize_style(style)
        phase = self._normalize_phase(phase)
        profile = self.get_cursor_profile(style, phase)
        if profile is None:
            return None
        name = profile["name"]
        self.profiles = [item for item in self.profiles if item.get("name") != name]
        if self.selected.get(style, {}).get(phase) == name:
            self.selected[style][phase] = None
        if self.cursor.get(style, {}).get(phase) == name:
            self.cursor[style][phase] = None
        self._normalize_cursor_state()
        self._save()
        return profile

    def begin_recording(self, style, phase, name_base=None):
        style = self._normalize_style(style)
        phase = self._normalize_phase(phase)
        if self.recording:
            return False
        self.recording = True
        self.recording_style = style
        self.recording_phase = phase
        self.recording_name_base = self._slugify_name(name_base)
        self.recording_started_at = time.time()
        self.recording_samples = []
        self.last_recorded_signature = None
        self.last_recorded_t = -1.0
        return True

    def capture_sample(self, now, fwd, turn, digger_on, lower_on, left_extend_on, right_extend_on):
        if not self.recording:
            return
        elapsed = max(0.0, float(now) - float(self.recording_started_at))
        sample = {
            "t": round(elapsed, 3),
            "fwd": round(float(fwd), 4),
            "turn": round(float(turn), 4),
            "digger_on": bool(digger_on),
            "lower_on": bool(lower_on),
            "left_extend_on": bool(left_extend_on),
            "right_extend_on": bool(right_extend_on),
        }
        signature = (
            sample["fwd"],
            sample["turn"],
            sample["digger_on"],
            sample["lower_on"],
            sample["left_extend_on"],
            sample["right_extend_on"],
        )
        if (
            (not self.recording_samples)
            or signature != self.last_recorded_signature
            or (elapsed - self.last_recorded_t) >= self.sample_period_sec
        ):
            self.recording_samples.append(sample)
            self.last_recorded_signature = signature
            self.last_recorded_t = elapsed

    def stop_recording(self, save=True):
        if not self.recording:
            return None
        style = self.recording_style
        phase = self.recording_phase
        samples = list(self.recording_samples)
        self.recording = False
        self.recording_style = None
        self.recording_phase = None
        name_base = self.recording_name_base
        self.recording_name_base = None
        self.recording_started_at = 0.0
        self.recording_samples = []
        self.last_recorded_signature = None
        self.last_recorded_t = -1.0
        if (not save) or len(samples) < 2:
            return None
        created_ms = int(time.time() * 1000)
        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(created_ms / 1000.0))
        if name_base:
            name = self._make_unique_name(f"{style}_{name_base}_{phase}")
        else:
            name = f"{style}_{phase}_{stamp}"
        profile = {
            "name": name,
            "style": style,
            "phase": phase,
            "created_ms": created_ms,
            "duration_sec": float(samples[-1]["t"]),
            "samples": samples,
        }
        self.profiles = [item for item in self.profiles if item.get("name") != name]
        self.profiles.append(profile)
        self.selected[style][phase] = name
        self.cursor[style][phase] = name
        self._normalize_cursor_state()
        self._save()
        return profile

    def selected_duration_sec(self, style=None, phase=None):
        profile = self.get_selected_profile(style, phase)
        if profile is None:
            return None
        try:
            return max(0.05, float(profile.get("duration_sec", 0.0)))
        except Exception:
            return None

    def playback_sample(self, elapsed_sec, style=None, phase=None):
        profile = self.get_selected_profile(style, phase)
        if profile is None:
            return None
        samples = profile.get("samples") or []
        if not samples:
            return None
        times = [float(sample.get("t", 0.0)) for sample in samples]
        idx = bisect.bisect_right(times, float(elapsed_sec)) - 1
        idx = max(0, min(idx, len(samples) - 1))
        sample = samples[idx]
        return {
            "profile_name": profile["name"],
            "style": profile["style"],
            "phase": profile.get("phase", "dig"),
            "fwd": float(sample.get("fwd", 0.0)),
            "turn": float(sample.get("turn", 0.0)),
            "digger_on": bool(sample.get("digger_on", False)),
            "lower_on": bool(sample.get("lower_on", False)),
            "left_extend_on": bool(sample.get("left_extend_on", False)),
            "right_extend_on": bool(sample.get("right_extend_on", False)),
            "duration_sec": float(profile.get("duration_sec", 0.0)),
        }

    def ui_state(self):
        profiles = []
        for style in self.VALID_STYLES:
            for phase in self.VALID_PHASES:
                for profile in self.list_profiles(style, phase):
                    profiles.append(
                        {
                            "name": profile["name"],
                            "style": style,
                            "phase": phase,
                            "duration_sec": float(profile.get("duration_sec", 0.0)),
                            "selected": profile["name"] == self.selected.get(style, {}).get(phase),
                            "cursor": profile["name"] == self.cursor.get(style, {}).get(phase),
                        }
                    )
        return {
            "active_style": self.active_style,
            "active_phase": self.active_phase,
            "recording": bool(self.recording),
            "recording_style": self.recording_style,
            "recording_phase": self.recording_phase,
            "selected": copy.deepcopy(self.selected),
            "cursor": copy.deepcopy(self.cursor),
            "profiles": profiles,
        }
