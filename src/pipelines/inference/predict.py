# src/pipelines/inference/predict.py
import click
from ultralytics import YOLO

@click.command()
@click.argument('source')
@click.argument('out')
@click.option('--conf', default=0.3)
@click.option('--iou', default=0.7)
def main(source, out, conf, iou):
    """Detect only (no tracking)."""
    model = YOLO('models/weights/best.pt')
    model.predict(source, conf=conf, iou=iou, save=True, save_dir=out)
    click.echo(f"✅ Detection results in {out}")
