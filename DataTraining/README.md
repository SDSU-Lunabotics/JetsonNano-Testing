# DataTraining

Offline dataset and model training workspace for Lunabotics perception.

This folder lets you:
- import videos/images from test runs
- label poles, rocks, and other objects
- train a YOLO model on laptop/desktop
- export model files and deploy to Jetson

## 1) What To Install

### Required on all OS
- Python 3.10+ (3.10 to 3.12 recommended)
- `pip`
- Git (optional, but recommended)

### Optional but useful
- FFmpeg (if some videos do not decode in OpenCV)
- NVIDIA GPU + CUDA for faster training

### OS notes
- macOS:
  - install Python from python.org or Homebrew
  - if using Apple Silicon, CPU training works; Metal acceleration depends on PyTorch build
- Windows:
  - install Python from python.org and enable "Add Python to PATH"
  - if PowerShell blocks virtual env activation, run:
    - `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`
- Linux:
  - install `python3`, `python3-venv`, `python3-pip`
  - install OpenCV GUI dependencies if needed (`libgl1`, `libgtk2.0-0` on Ubuntu-like systems)

## 2) Setup Virtual Environment

### macOS/Linux
```bash
cd /path/to/JetsonNano-Testing
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r DataTraining/requirements.txt
```

### Windows (PowerShell)
```powershell
cd C:\path\to\JetsonNano-Testing
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r DataTraining\requirements.txt
```

## 3) Folder Layout

- `data/raw_media/`: drop source videos and images here
- `data/raw_images/`: extracted/selected images for labeling
- `data/raw_labels/`: YOLO label txt files
- `data/dataset/`: final train/val/test dataset
- `classes.txt`: object class list (edit this for your project)

Default classes:
- `pole`
- `rock`
- `berm_marker`
- `hole_rgb`
- `unknown_obstacle`

## 4) Workflow

### Step A: Put media into raw input
Copy videos/images into:
- `DataTraining/data/raw_media/`

### Step B: Extract frames and copy images
```bash
python DataTraining/tools/extract_frames.py \
  --input-dir DataTraining/data/raw_media \
  --output-dir DataTraining/data/raw_images \
  --video-every-n-frames 10
```

### Step C1: Label image-by-image
```bash
python DataTraining/tools/annotate_images.py \
  --images-dir DataTraining/data/raw_images \
  --labels-dir DataTraining/data/raw_labels \
  --classes DataTraining/classes.txt
```

Keys in annotator:
- `0-9`: choose active class id
- `a`: draw bounding box using active class
- `d`: auto-detect boxes on current image (if `--auto-model` is provided)
- `r`: remove last box
- `u`: undo last add batch (useful after auto-detect)
- `c`: clear boxes for current image
- `s`: save current labels
- `n` or `Enter` or right arrow: next image (autosaves)
- `p` or left arrow: previous image (autosaves)
- `q`: quit (autosaves)

The tool now opens two windows:
- `Annotator`: image + boxes + status
- `Annotator Help`: key cheatsheet + class list

Auto-detect example:
```bash
python DataTraining/tools/annotate_images.py \
  --images-dir DataTraining/data/raw_images \
  --labels-dir DataTraining/data/raw_labels \
  --classes DataTraining/classes.txt \
  --auto-model /path/to/your_model.pt
```

### Step C2: Label while video is playing
```bash
python DataTraining/tools/review_video_and_label.py \
  --video DataTraining/data/raw_media/your_video.mp4 \
  --classes DataTraining/classes.txt
```

Video review keys:
- `space`: pause/resume
- `a`: annotate current frame and save to raw images/labels
- `n`: step one frame (when paused)
- `q`: quit

### Step D: Build train/val/test split + dataset yaml
```bash
python DataTraining/tools/split_dataset.py \
  --images-dir DataTraining/data/raw_images \
  --labels-dir DataTraining/data/raw_labels \
  --output-dir DataTraining/data/dataset \
  --ratios 0.8,0.15,0.05
```

### Step E: Train
```bash
python DataTraining/tools/train_yolo.py \
  --data DataTraining/data/dataset/dataset.yaml \
  --model yolov8n.pt \
  --epochs 120 \
  --imgsz 640
```

### Step F: Export for Jetson
```bash
python DataTraining/tools/export_for_jetson.py \
  --model DataTraining/runs/pole_rock_train/weights/best.pt \
  --format onnx \
  --imgsz 640
```

For TensorRT engine export (`--format engine`), run on the Jetson itself.

## 5) Copy Model To Jetson

From laptop/desktop:
```bash
scp DataTraining/runs/pole_rock_train/weights/best.pt \
  username@JETSON_IP:~/models/
```

Example for ONNX:
```bash
scp DataTraining/runs/pole_rock_train/weights/best.onnx \
  username@JETSON_IP:~/models/
```

## 6) Recommended Training Strategy For Your Arena

- Train `pole` and `rock` with bounding boxes.
- Keep `hole` detection primarily depth-based in ZEDAuto (more reliable than RGB-only).
- Collect data from multiple lighting conditions and viewpoints.
- Start with small model (`yolov8n`) for Jetson stability, then scale up only if needed.

## 7) Common Issues

- "No module named cv2": install requirements in the active virtual env.
- Video won't open: try MP4/H264, install FFmpeg, or transcode video.
- Labels look wrong: verify class IDs match `classes.txt` order.
- Jetson slow: use smaller model, lower `imgsz`, and reduce inference FPS.
