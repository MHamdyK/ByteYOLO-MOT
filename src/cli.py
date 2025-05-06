# src/cli.py
import click
from src.utils.numpy_patch import apply_patch
from src.utils.io import fetch_assets
from src.pipelines.training.prepare_data import main as prepare_main
from src.pipelines.training.train import main as train_main
from src.pipelines.inference.predict import main as predict_main
from src.pipelines.inference.track_video import main as track_video_main
from src.pipelines.inference.submit import main as submit_main

@click.group()
def main():
    """ByteYOLO-MOT: train & inference CLI."""
    apply_patch()

@main.command()
@click.option('--root', required=True, help="MOT dataset root")
@click.option('--output', default="data/yolo_data", help="YOLO data output dir")
def prepare(root, output):
    """Prepare YOLO dataset from MOT annotations."""
    prepare_main(root, output)

@main.command()
@click.option('--data', default="data/yolo_data/dataset.yaml")
@click.option('--epochs', default=10)
@click.option('--imgsz', default=1184)
@click.option('--batch', default=10)
def train(data, epochs, imgsz, batch):
    """Fine-tune YOLOv8 on prepared data."""
    train_main(data, epochs, imgsz, batch)

@main.command()
def fetch_assets_cmd():
    """Download model weights & demo GIF."""
    fetch_assets()

@main.command()
@click.option('--source', required=True)
@click.option('--out', default='dets')
@click.option('--conf', default=0.3)
@click.option('--iou', default=0.7)
def predict(source, out, conf, iou):
    """Run detection only (no tracking)."""
    predict_main(source, out, conf, iou)

@main.command(name='track-video')
@click.option('--source', required=True)
@click.option('--out', default='tracked.mp4')
@click.option('--conf', default=0.3)
@click.option('--iou', default=0.7)
@click.option('--track-thresh', default=0.35)
@click.option('--track-buffer', default=30)
@click.option('--match-thresh', default=0.7)
def track_video(source, out, conf, iou, track_thresh, track_buffer, match_thresh):
    """Overlay ByteTrack IDs on video/webcam."""
    track_video_main(source, out, conf, iou, track_thresh, track_buffer, match_thresh)

@main.command(name='submit')
@click.option('--seq', required=True)
@click.option('--out', default='submission.csv')
@click.option('--conf', default=0.3)
@click.option('--iou', default=0.7)
def submit(seq, out, conf, iou):
    """Generate MOT submission CSV."""
    submit_main(seq, out, conf, iou)

if __name__ == '__main__':
    main()
