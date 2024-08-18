import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('E:/SCAM_yolo/runs的/DIOR-teacher-l模型/weights/best.pt') # select your model.pt path
    model.predict(source='E:/DIOR_dataset_yolo/images/val/',
                  imgsz=800,
                  project='runs/detect',
                  name='exp13',
                  save=True,
                #   visualize=True # visualize model features maps
                )