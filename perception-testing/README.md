# Perception Testing Lab

This folder is an isolated testing workspace for live perception tuning.
It is separate from `ZEDAuto` so you can test obstacle logic and AI detection
without changing your autonomous driving pipeline.

## What it does
- Live geometry-based obstacle detection from ZED point cloud + floor plane.
- Runtime controls (trackbars) for thresholds and filtering.
- Obstacle bounding boxes from connected components.
- Object boxes from detector source:
  - ZED SDK built-in detector (default, offline, no API key/Wi-Fi)
  - Optional YOLO (`ultralytics`) if a custom model is provided.
- Annotation mode: click/drag a box, then press class hotkey (`1..9`).
- Semantic top-down map that updates from labeled boxes using ZED depth projection.
- Single combined `Perception Map` window for map outputs (terrain + semantic overlays).

## Run
From repo root:

```bash
./perception-testing/RunPerceptionLab.sh
```

If executable bit is missing:

```bash
bash perception-testing/RunPerceptionLab.sh
```

## Simple Terrain View (Ground + Wall)
If you want a lightweight terrain-only view (similar to ZEDAuto ground/wall map), run:

```bash
./perception-testing/RunTerrainSimple.sh
```

This mode:
- skips AI/annotation workflow
- shows a camera overlay (`green=ground`, `red=obstacle/wall`, `blue=hole`)
- shows a top-down occupancy map (`Terrain Map (XZ)`)
- does **not** run people/object detectors (use `RunPerceptionLab.sh` for that)

Keys:
- `q` or `Esc`: quit
- `r`: refresh floor plane fit
- `c`: clear occupancy map

If you want people detection and terrain map together, use:

```bash
./perception-testing/RunPerceptionLab.sh
```

## Live Controls (Trackbars)
Set `SHOW_CONTROLS=1` in `perception-testing/perception_lab.env` to open this window.
- `Obstacle cm`: height above floor treated as obstacle.
- `Hole cm`: depth below floor treated as hole.
- `MaxAbove cm`: ignore obstacle points above this floor-relative height.
- `MaxFwd dm`: ignore points farther than this forward distance.
- `Stride`: point-cloud sampling stride (higher = faster, less detail).
- `MinBox x100`: minimum area for geometric obstacle boxes.
- `AI Conf %`, `AI IoU %`: AI thresholds if AI model is enabled.
- `ShowGeom`, `ShowBoxes`, `ShowAI`: overlay toggles.

## Keyboard
- `q` or `Esc`: quit
- `p`: pause/resume processing
- `r`: force floor-plane refresh
- `s`: save snapshot
- `l`: toggle annotation mode on/off
- `0`: clear pending box
- `1..8`: assign class from `CLASSES` order
- `9`: type a class name in terminal prompt, then label with that class
- In `Semantic Map (XZ)`: left-drag paints selected class, right-click erases

## Optional AI Detection
Default detector mode is `zed` (built-in ZED SDK object detection, local/offline).

To use your own custom model labels like `rock`, `wall`, etc., switch to YOLO mode:

1. Train/export a YOLO model.
2. Set paths in `perception-testing/perception_lab.env`:
   - `AI_MODEL_PATH=/abs/path/to/model.pt`
   - `AI_LABELS_PATH=/abs/path/to/labels.txt` (optional)
   - `DETECTOR_MODE="yolo"` (or `"both"`)
3. Run again with `RunPerceptionLab.sh`.

Notes:
- If `ultralytics` is missing, the script will continue in geometry-only mode.
- Class names (`rock`, `wall`, etc.) depend entirely on your training labels.

## Annotation Workflow
1. Run `./perception-testing/RunPerceptionLab.sh`.
2. In `Perception Lab` window:
- Drag a box manually around an object, or click inside an AI/geom box to select it.
3. Press class key `1..8` (from `CLASSES` order in `perception_lab.env`), or press `9` to type a class name.
4. The tool saves:
- Image: `perception-testing/dataset/images/*.jpg`
- YOLO label: `perception-testing/dataset/labels/*.txt`
- Metadata CSV: `perception-testing/dataset/annotations.csv`
5. If `SEMANTIC_MAP=1`, labeled box points are projected into `Semantic Map (XZ)` immediately.
6. Use class `ground` to teach/fill areas that depth marks incorrectly as obstacles.
7. For empty/missing depth regions, paint directly on `Semantic Map (XZ)` with the selected class.
8. Use `unknown_anomaly` for odd detections that should be reviewed later.

This means your exact use case is supported:
- If depth is weak but camera/AI sees a rock, you can label it and it gets marked on the semantic map for testing.
- If depth falsely paints ground as red, label `ground` boxes and/or paint ground directly on semantic map.
