# ByteYOLO‑MOT  
**Multi‑Object Tracking with YOLOv8 + ByteTrack**

---

<div align="center">
  <img src="docs/demo/demo.gif" alt="Tracking demo" width="800"/>
</div>

> **ByteYOLO‑MOT** is a fully‑reproducible pipeline that fine‑tunes a COCO‑pre‑trained YOLOv8 detector on the MOT‑17 + MOT‑20 pedestrian datasets and couples it with ByteTrack for online multi‑object tracking.  
> It ships with ready‑to‑use model weights, CLI tools for batch/real‑time inference, and a Docker image for one‑command deployment.

---

## Table of Contents
1. [Project Structure](#project-structure)  
2. [Quick start](#quick-start)  
3. [Dataset preparation](#dataset-preparation)  
4. [Training pipeline](#training-pipeline)  
5. [Inference pipelines](#inference-pipelines)  
6. [Augmentations & image‑size experiments](#augmentations--image-size-experiments)  
7. [NumPy 2.0 compatibility patch](#numpy-20-compatibility-patch)  
8. [Docker support](#docker-support)  
9. [Reproducing the demo GIF](#reproducing-the-demo-gif)  
10. [License](#license)  

---

## Project Structure
~~~text
byteyolo-mot/
├── data/                    # (ignored) local MOT17/MOT20 downloads
├── docs/
│   └── demo/demo.gif        # 10‑second illustrative GIF (≈ 15 MB)
├── models/
│   └── weights/
│       ├── best.pt          # Fine‑tuned YOLOv8m (≈ 100 MB, LFS or Release)
│       └── README.md        # SHA256 checksum & training metadata
├── notebooks/
│   └── fawry_experiment.ipynb
├── src/
│   ├── cli.py
│   ├── utils/
│   │   ├── numpy_patch.py   # NumPy‑2.0 dtype fix
│   │   └── io.py            # Download helpers
│   └── pipelines/
│       ├── training/
│       │   ├── prepare_data.py
│       │   └── train.py
│       └── inference/
│           ├── predict.py
│           ├── track_video.py
│           └── submit.py
├── tests/
│   ├── test_numpy_patch.py
│   └── test_inference.py
├── Dockerfile
├── environment.yml
├── requirements.txt
├── setup.py
└── LICENSE
~~~

> **Tip 🔥** Have a beefier GPU? Pass a larger `--imgsz` (e.g. `1536` or `1920`) to `byteyolo train`; larger inputs help detect far‑away pedestrians.

---

## Quick start
~~~bash
# 1 · Clone & install
git clone https://github.com/your‑org/byteyolo‑mot.git
cd byteyolo‑mot
pip install -r requirements.txt
pip install -e .

# 2 · Fetch model weights & demo GIF
byteyolo fetch-assets     # downloads best.pt & demo.gif

# 3 · Run a real‑time demo on your webcam (device 0)
byteyolo track-video --source 0 --out demo.mp4
~~~

| Command                   | Description                                                 |
|---------------------------|-------------------------------------------------------------|
| `byteyolo train`          | Fine‑tune YOLOv8 on MOT17 + MOT20 (≈ 10 h on one P100)      |
| `byteyolo predict`        | Detection‑only on images / folders                          |
| `byteyolo track-video`    | Overlay ByteTrack IDs on a video / webcam stream            |
| `byteyolo submit`         | Produce `submission.csv` for MOT‑style leaderboards         |
| `byteyolo prepare`        | Convert MOT annotations to YOLO format                      |
| `byteyolo fetch-assets`   | Download model weights & demo GIF                           |

Each sub‑command supports `‑‑help`.

---

## Dataset preparation
~~~bash
byteyolo prepare \
  --root /path/to/MOT      \   # contains mot-20/tracking & mot-17/MOT17
  --output data/yolo_data
~~~
The script:

1. Ingests **MOT‑20** competition sequences `02 03 05` and **MOT‑17** (`train` + `val`).  
2. Converts every ground‑truth rectangle to YOLO format **once**.  
3. Writes
   ~~~text
   data/yolo_data/
   ├── images/{train,val}/...
   ├── labels/{train,val}/...
   └── dataset.yaml
   ~~~

---

## Training pipeline
~~~bash
byteyolo train \
  --data   data/yolo_data/dataset.yaml \
  --epochs 10 \
  --imgsz  1184 \
  --batch  10
~~~

| Hyper‑param   | Value | Rationale                                       |
|---------------|-------|-------------------------------------------------|
| `epochs`      | **10**| Early‑stops after ≈ 10 h on Kaggle P100         |
| `imgsz`       | 1184  | Best mAP for far‑view pedestrians               |
| `batch`       | 10    | Fits in 16 GB VRAM                              |
| `optimizer`   | SGD   | More stable than AdamW for detection            |
| `lr0 / lrf`   | 0.005 / 0.015 | Backbone already warm – higher LR OK |

### Augmentations
| Augmentation            | Value | Purpose                         |
|-------------------------|-------|---------------------------------|
| Horizontal flip         | 0.5   | Orientation diversity           |
| HSV h/s/v jitter        | 0.015 / 0.4 / 0.2 | Colour & brightness |
| Translation             | 0.05  | Minor shifts                    |
| Scale                   | 0.2   | Zoom variance                   |
| Random erasing (CutOut) | 0.1   | Occlusion robustness            |
| RandAugment, MixUp      | *off* | Preserve object geometry for MOT|

---

## Inference pipelines

### 1 · Detection‑only
~~~bash
byteyolo predict \
  --source imgs/ \
  --out dets/ \
  --conf 0.3 --iou 0.7
~~~

### 2 · Video tracking
~~~bash
byteyolo track-video \
  --source input.mp4 \
  --out tracked.mp4 \
  --conf 0.3 --iou 0.7
~~~

### 3 · MOT submission CSV
~~~bash
byteyolo submit \
  --seq /path/to/test/01 \
  --out submission.csv \
  --conf 0.3 --iou 0.7
~~~

---

## Augmentations & image‑size experiments

| Image size | mAP<sub>50</sub> | mAP<sub>50‑95</sub> | HOTA | Note                          |
|------------|------------------|---------------------|------|-------------------------------|
| 640        | 0.580            | 0.330              | 0.55 | Fast baseline                 |
| 832        | 0.610            | 0.360              | 0.60 |                               |
| 960        | 0.630            | 0.380              | 0.63 |                               |
| **1184**   | **0.655**        | **0.393**          | **0.65** | Best—resolves small objects |

> Bigger than 1184 (e.g. 1536, 1920) may improve recall if you have ≥16 GB VRAM—diminishing returns beyond ~2 k px.

---

## NumPy 2.0 compatibility patch
ByteTrack still uses *bare* dtype names (`float32`, `int32`, …).  
NumPy ≥ 2.0 removed those aliases, raising:

