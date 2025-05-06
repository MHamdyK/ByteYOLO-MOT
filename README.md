# ByteYOLO-MOT  
**Multi-Object Tracking with YOLOv8 + ByteTrack**

---

<div align="center">
  <img src="docs/demo/demo.gif" alt="Tracking demo" width="800"/>
</div>

> **ByteYOLO-MOT** is a fully-reproducible pipeline that fine-tunes a COCO-pre-trained YOLOv8 detector on the MOT-17 + MOT-20 pedestrian datasets and couples it with ByteTrack for online multi-object tracking. It ships with ready-to-use model weights, CLI tools for batch/real-time inference, and a Docker image for one-command deployment.

---

## Table of Contents
1. [Project Structure](#project-structure)  
2. [Quick start](#quick-start)  
3. [Dataset preparation](#dataset-preparation)  
4. [Training pipeline](#training-pipeline)  
5. [Inference pipelines](#inference-pipelines)  
6. [Augmentations & image-size experiments](#augmentations--image-size-experiments)  
7. [NumPy 2.0 compatibility patch](#numpy-20-compatibility-patch)  
8. [Docker support](#docker-support)  
9. [Reproducing the demo GIF](#reproducing-the-demo-gif)  
10. [License](#license)  

---

## Project Structure
```
text
byteyolo-mot/
├── data/                    # (ignored) for your local MOT17/MOT20 downloads
├── docs/
│   └── demo/demo.gif        # 10-second illustrative GIF (≈ 15 MB)
├── models/
│   └── weights/
│       ├── best.pt          # Fine-tuned YOLOv8m (≈ 100 MB, via Git LFS or Release)
│       └── README.md        # SHA256 checksum & training metadata
├── notebooks/
│   └── fawry_experiment.ipynb
├── src/
│   ├── cli.py
│   ├── utils/
│   │   ├── numpy_patch.py   # NumPy-2.0 dtype deprecation fix
│   │   └── io.py            # Download helper for weights & demo
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
├── README.md  ← this file
└── LICENSE
```