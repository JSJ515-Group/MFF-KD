import warnings
warnings.filterwarnings('ignore')
# from ultralytics import YOLO
from ultralytics import YOLO
if __name__ == '__main__':
    model = YOLO('E:/SCAM_yolo/ultralytics-main/yolov8-C2f-EMSC.yaml')
    model.load('E:/SCAM_yolo/ultralytics-main/yolov8s.pt') # loading pretrain weights
    model.train(data='E:/SCAM_yolo/ultralytics-main/ultralytics/cfg/datasets/VisDrone.yaml',
                cache=False,
                imgsz=640,
                epochs=100,
                batch=1,
                close_mosaic=10,
                workers=4,
                device='0',
                optimizer='SGD', # using SGD
                # resume='', # last.pt path
                # amp=False # close amp
                # fraction=0.2,
                project='runs/train',
                name='exp',
                )