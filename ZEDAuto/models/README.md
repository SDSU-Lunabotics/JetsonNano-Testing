# ZEDAuto / models

Drop trained YOLO model weights here for use by the autonomy stack.

## Workflow — from training to deployment

1. **Classmates train** using `DataTraining/tools/train_yolo.py` (on a machine with a GPU).
   Each training run outputs:
   ```
   DataTraining/runs/<run_name>/weights/best.pt   ← best validation mAP
   DataTraining/runs/<run_name>/weights/last.pt   ← final epoch
   ```

2. **Compare runs** — open `DataTraining/runs/<run_name>/results.csv` or check
   `val/mAP50` in the training logs. Pick whichever `best.pt` has the highest mAP50.

3. **Copy the winner here** and give it a descriptive name:
   ```
   cp DataTraining/runs/pole_rock_train_v3/weights/best.pt \
      ZEDAuto/models/rock_detector_v3.pt
   ```

4. **Enable it** — edit `ZEDAuto/zed_ground_wall.env`:
   ```
   ROCK_MODEL="./ZEDAuto/models/rock_detector_v3.pt"
   ```
   Then restart `RunAuto.sh`. Detected rocks are stamped onto the occupancy map
   as red obstacle cells and A* automatically routes around them.

## Naming convention

```
rock_detector_v1.pt     first trained version
rock_detector_v2.pt     retrained with more data
rock_obstacle_v1.pt     model trained on both rock + obstacle classes
```

## Notes

- The model file is NOT committed to git (listed in .gitignore below) because
  `.pt` files are large binary blobs — share them via Google Drive, Slack, or
  copy them onto the Jetson directly over SSH.
- The training dataset and labels ARE committed (`DataTraining/data/`).
- Only the model file goes here — keep configs and training scripts in DataTraining.
