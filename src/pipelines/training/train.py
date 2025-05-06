# src/pipelines/training/train.py
import torch
import click
from ultralytics import YOLO

@click.command()
@click.argument('data')
@click.option('--epochs', default=10, help="Number of epochs")
@click.option('--imgsz', default=1184, help="Image size")
@click.option('--batch', default=10, help="Batch size")
@click.option('--device', default=None, help="cuda or cpu")
def main(data, epochs, imgsz, batch, device):
    """Fine-tune YOLOv8 on the MOT pedestrian dataset."""
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.cuda.manual_seed(42)
    torch.manual_seed(42)

    model = YOLO('yolov8m.pt')
    model.train(
        data=data,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        optimizer='SGD',
        lr0=0.005,
        lrf=0.015,
        momentum=0.937,
        weight_decay=0.0005,
        cos_lr=True,
        warmup_epochs=2,
        augment=True,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.2,
        translate=0.05,
        scale=0.2,
        erasing=0.1,
        mixup=0.0,
        copy_paste=0.0,
    )
    click.echo("✅ Training completed.")
