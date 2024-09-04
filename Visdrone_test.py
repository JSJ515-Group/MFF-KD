from ultralytics import YOLO
import multiprocessing

# Your validation function
def validate_model():
    # Load a model
    model = YOLO('E:/gold_yolo/train/xiaorong/yolov8+LKSPPF+C2f_DCNV2/weights/best.pt')  # load an official model
    # Validate the model
    metrics = model.val(data='E:/gold_yolo/ultralytics-main/ultralytics/cfg/datasets/VisDrone.yaml', iou=0.5, conf=0.001, half=False, device=0, save_json=True)
    print(metrics.box.map)  # map50-95
    print(metrics.box.map50)  # map50
    print(metrics.box.map75)  # map75
    print(metrics.box.maps)  # 包含每个类别的map50-95列表

if __name__ == '__main__':
    multiprocessing.freeze_support()
    validate_model()
