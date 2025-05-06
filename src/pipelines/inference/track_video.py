# src/pipelines/inference/track_video.py
import click
import cv2
import numpy as np
from ultralytics import YOLO
from yolox.tracker.byte_tracker import BYTETracker, STrack
from dataclasses import dataclass

@dataclass(frozen=True)
class BYTETrackerArgs:
    track_thresh: float = 0.35
    track_buffer: int = 30
    match_thresh: float = 0.7
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = True

@click.command()
@click.argument('source')
@click.argument('out')
@click.option('--conf', default=0.3)
@click.option('--iou', default=0.7)
@click.option('--track-thresh', default=0.35)
@click.option('--track-buffer', default=30)
@click.option('--match-thresh', default=0.7)
def main(source, out, conf, iou, track_thresh, track_buffer, match_thresh):
    """Overlay ByteTrack IDs on a video or webcam."""
    model = YOLO('models/weights/best.pt')
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    args = BYTETrackerArgs(track_thresh, track_buffer, match_thresh)
    tracker = BYTETracker(args)
    colors = {}

    def get_color(tid):
        if tid not in colors:
            colors[tid] = tuple(int(x) for x in np.random.randint(0, 255, 3))
        return colors[tid]

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(frame, conf=conf, iou=iou)[0]
        dets = None
        if len(results.boxes):
            boxes = results.boxes.xyxy.cpu().numpy()
            scores = results.boxes.conf.cpu().numpy().reshape(-1,1)
            dets = np.hstack((boxes, scores))
        tracks = tracker.update(dets, frame.shape[:2], frame.shape[:2]) if dets is not None else []
        for trk in tracks:
            x1,y1,x2,y2 = map(int, trk.tlbr)
            tid = trk.track_id
            color = get_color(tid)
            cv2.rectangle(frame, (x1,y1),(x2,y2), color, 2)
            cv2.putText(frame, f\"ID:{tid}\", (x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        writer.write(frame)

    cap.release()
    writer.release()
    click.echo(f\"✅ Tracked video saved to {out}\")
