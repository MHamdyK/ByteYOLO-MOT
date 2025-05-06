# src/pipelines/training/prepare_data.py
import os
import cv2
import click
import numpy as np
from pathlib import Path
from atomicwrites import atomic_write
from collections import defaultdict

def safe_bbox_conversion(x, y, w, h, img_w, img_h):
    x_c = max(0.0, min(1.0, (x + w/2) / img_w))
    y_c = max(0.0, min(1.0, (y + h/2) / img_h))
    w_n = max(0.001, min(1.0, w / img_w))
    h_n = max(0.001, min(1.0, h / img_h))
    return round(x_c,6), round(y_c,6), round(w_n,6), round(h_n,6)

def read_mot_gt(gt_file, kind):
    anns=[]
    for line in open(gt_file):
        parts=line.strip().split(',')
        f, tid, x, y, w, h, conf, cls, vis = parts[:9]
        f, tid, x, y, w, h, conf, cls, vis = map(float, (f, tid, x, y, w, h, conf, cls, vis))
        if kind=='competition' and cls==1 and vis>=0.06:
            anns.append((int(f), x,y,w,h))
        if kind=='mot17' and tid>0 and vis>0.1:
            anns.append((int(f), x,y,w,h))
    return anns

@click.command()
@click.argument('root', type=click.Path(exists=True))
@click.argument('output', type=click.Path())
def main(root, output):
    """
    root: path to /MOT-20/train and MOT17/train
    output: where to write YOLO-format images+labels
    """
    COMP_SEQS = ["02","03","05"]
    MOT17_TRAIN = ["MOT17-02-SDP","MOT17-04-SDP","MOT17-05-SDP","MOT17-09-SDP","MOT17-10-SDP"]
    MOT17_VAL   = ["MOT17-11-SDP","MOT17-13-SDP"]
    os.makedirs(output+"/images/train", exist_ok=True)
    os.makedirs(output+"/images/val",   exist_ok=True)
    os.makedirs(output+"/labels/train", exist_ok=True)
    os.makedirs(output+"/labels/val",   exist_ok=True)

    def process(kind, seqs, split):
        base = Path(root)/("train" if kind=="competition" else "MOT17/train")
        for seq in seqs:
            gt = base/seq/"gt"/"gt.txt"
            anns = read_mot_gt(gt, kind)
            # group by frame
            groups=defaultdict(list)
            for f,x,y,w,h in anns:
                groups[f].append((x,y,w,h))
            for f, boxes in groups.items():
                img_path = base/seq/"img1"/f"{f:06d}.jpg"
                if not img_path.exists(): continue
                img=cv2.imread(str(img_path))
                hgt, wdt = img.shape[:2]
                dest_img = Path(output)/"images"/split/f"{kind}_{seq}_{f:06d}.jpg"
                dest_lbl = Path(output)/"labels"/split/f"{kind}_{seq}_{f:06d}.txt"
                from atomicwrites import atomic_write
                with atomic_write(dest_img, 'wb', overwrite=True) as fout:
                    fout.write(img_path.read_bytes())
                lines=[]
                for x,y,w,h in boxes:
                    xc,yc,wn,hn = safe_bbox_conversion(x,y,w,h,wdt,hgt)
                    lines.append(f"0 {xc} {yc} {wn} {hn}")
                with atomic_write(dest_lbl, 'w', overwrite=True) as fout:
                    fout.write("\n".join(lines))

    process("competition", COMP_SEQS, "train")
    process("mot17", MOT17_TRAIN, "train")
    process("mot17", MOT17_VAL,   "val")

    yaml = f"train: {output}/images/train\nval: {output}/images/val\nnc: 1\nnames: ['pedestrian']\n"
    Path(output)/"dataset.yaml".write_text(yaml)
    click.echo("✅ Dataset prepared.")
