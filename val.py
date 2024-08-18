import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('E:/gold_yolo/train/canshu/0.5GIoU+0.5NWD/weights/best.pt')
    model.val(data='E:/SCAM_yolo/ultralytics-main/ultralytics/cfg/datasets/VisDrone.yaml',
                split='val',
                save_json=True,# if you need to cal coco metrice
                project='runs/val',
                name='exp',
                )