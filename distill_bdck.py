import warnings
warnings.filterwarnings('ignore')
import argparse, yaml, copy
from ultralytics.models.yolo.detect.distill import DetectionDistiller

if __name__ == '__main__':
    param_dict = {
        # origin
        'model': '/mnt/sdb/A515/limengyuan/yolov8/ultralytics-main/yolov8-C2f-EMSCs.yaml',
        'data': '/mnt/sdb/A515/limengyuan/yolov8/ultralytics-main/ultralytics/cfg/datasets/VisDrone.yaml',
        'imgsz': 800,
        'epochs': 200,
        'batch': 1,
        'workers': 8,
        'cache': True,
        'optimizer': 'SGD',
        'device': '0',
        'close_mosaic': 20,
        'project': 'runs/distill',
        'name': 'yolov8n-BCKD-CWD-exp4',

        # distill teacher
        'prune_model': False,
        'teacher_weights': '/mnt/sdb/A515/limengyuan/yolov8/ultralytics-main/runs/train/exp66/weights/best.pt',
        'teacher_cfg': '/mnt/sdb/A515/limengyuan/yolov8/ultralytics-main/runs/train/exp66/ultralytics-main/yolov8-C2f-EMSCl.yaml',
        'kd_loss_type': 'all',
        'kd_loss_decay': 'constant',

        'logical_loss_type': 'BCKD',
        'logical_loss_ratio': 1.0,

        'teacher_kd_layers': '30, 25, 20, 23',
        'student_kd_layers': '30, 25, 20, 23',
        'feature_loss_type': 'cwd',
        'feature_loss_ratio': 0.85
    }
    # param_dict = {
    #     # origin
    #     'model': 'E:/SCAM_yolo/ultralytics-main/yolov8-C2f-EMSCs.yaml',
    #     'data': 'E:/SCAM_yolo/ultralytics-main/ultralytics/cfg/datasets/VisDrone.yaml',
    #     'imgsz': 800,
    #     'epochs': 200,
    #     'batch': 1,
    #     'workers': 8,
    #     'cache': True,
    #     'optimizer': 'SGD',
    #     'device': '0',
    #     'close_mosaic': 20,
    #     'project': 'runs/distill',
    #     'name': 'yolov8n-cwd-exp3',
    #
    #     # distill
    #     'prune_model': False,
    #     'teacher_weights': 'E:/SCAM_yolo/yolov8s_EMSCs/weights/best.pt',
    #     'teacher_cfg': 'E:/SCAM_yolo/ultralytics-main/yolov8-C2f-EMSCl.yaml',
    #     'kd_loss_type': 'feature',
    #     'kd_loss_decay': 'constant',
    #
    #     'logical_loss_type': 'l2',
    #     'logical_loss_ratio': 1.0,
    #
    #     'teacher_kd_layers': '30,25,20,23',
    #     'student_kd_layers': '30,25,20,23',
    #     'feature_loss_type': 'cwd',
    #     'feature_loss_ratio': 1.0
    # }
    
    model = DetectionDistiller(overrides=param_dict)
    model.distill()