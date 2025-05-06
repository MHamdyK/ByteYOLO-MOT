# src/pipelines/inference/submit.py
import click
import cv2
import json
import pandas as pd
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from yolox.tracker.byte_tracker import BYTETracker, STrack
from dataclasses import dataclass

@dataclass(frozen=True)
class BYTETrackerArgs:
    track_thresh: float = 0.35
    track_buffer: int = 45
    match_thresh: float = 0.7
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = True

@click.command()
@click.argument('seq')
@click.argument('out')
@click.option('--conf', default=0.3)
@click.option('--iou', default=0.7)
def main(seq, out, conf, iou):
    """Generate a MOT-style submission.csv"""
    weight = 'models/weights/best.pt'
    model = YOLO(weight)
    seqinfo = {}
    for line in open(Path(seq)/'seqinfo.ini'):
        if '=' in line:
            k,v = line.strip().split('=')
            seqinfo[k.strip()] = v.strip()
    num_frames = int(seqinfo['seqLength'])
    tracker = BYTETracker(BYTETrackerArgs())
    submission = []

    for idx in range(1, num_frames+1):
        img_path = Path(seq)/'img1'/f\"{idx:06d}.jpg\"
        frame = cv2.imread(str(img_path))
        objs = []
        if frame is not None:
            res = model.predict(str(img_path), conf=conf, iou=iou)[0]
            xyxy = res.boxes.xyxy.cpu().numpy() if len(res.boxes) else np.empty((0,4))
            dets = np.hstack((xyxy, np.ones((len(xyxy),1)))) if len(xyxy) else None
            tracks = tracker.update(dets, frame.shape[:2], frame.shape[:2]) if dets is not None else []
            for trk in tracks:
                x1,y1,x2,y2 = trk.tlbr
                objs.append({
                    'tracked_id': int(trk.track_id),
                    'x': float(x1), 'y': float(y1),
                    'w': float(x2-x1), 'h': float(y2-y1),
                    'confidence': 1.0
                })
        submission.append({
            'ID': idx-1,
            'Frame': idx,
            'Objects': objs,
            'Objective': 'tracking'
        })

    df = pd.DataFrame(submission)
    df.to_csv(out, index=False)
    click.echo(f\"✅ Submission saved to {out}\")
