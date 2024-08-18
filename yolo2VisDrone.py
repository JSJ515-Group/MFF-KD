from PIL import Image
from tqdm import tqdm
from pathlib import Path
import os

def yolo2visdrone(dir):
    def convert_box(size, box):
        return int((box[0] - box[2] / 2) * size[0]), int((box[1] - box[3] / 2) * size[1]), int(box[2] *size[0]), int(box[3] * size[1])
    (dir / 'post_labels').mkdir(parents=True, exist_ok=True)
    pbar = tqdm((dir / 'labels').glob('*.txt'), desc=f'Converting {dir}')
    for f in pbar:
        img_size = Image.open((dir / f.name).with_suffix('.jpg')).size
        lines = []
        with open(f, 'r') as file:
            for row in [x.split(' ') for x in file.read().strip().splitlines()]:
                cls = int(row[0]) + 1
                box = convert_box(img_size, tuple(map(float, row[1:5])))
                conf = float(row[5])
                occ = -1
                jieduan = -1
                lines.append(f"{','.join(f'{x}' for x in box)},{conf:.4f},{cls},{occ},{jieduan}\n")
                with open(str(f).replace(os.sep +'labels' + os.sep, os.sep + 'post_labels' + os.sep), 'w') as fl:
                    fl.writelines(lines)
        if lines == []:
            with open(str(f).replace(os.sep + 'labels' + os.sep, os.sep + 'post_labels' + os.sep), 'w') as fl:
                fl.writelines(lines)

dir = Path('E:/SCAM_yolo/ultralytics-main/runs/detect/exp/')  # 预测的labels存放目录
yolo2visdrone(dir)
