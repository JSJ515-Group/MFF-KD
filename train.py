
import argparse, sys, os, warnings
import torch.nn as nn
warnings.filterwarnings('ignore')
from pathlib import Path
from ultralytics import YOLO
import torch
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

def str2bool(str):
    return True if str.lower() == 'true' else False

def transformer_opt(opt):
    opt = vars(opt)
    if opt['unamp']:
        opt['amp'] = False
    else:
        opt['amp'] = True
    del opt['yaml']
    del opt['weight']
    del opt['info']
    del opt['unamp']
    return opt

def parse_opt():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--yaml', type=str, default='E:/SCAM_yolo/ultralytics-main/yolov8-C2f-EMSC.yaml', help='model.yaml path')
    parser.add_argument('--weight', type=str, default='E:/SCAM_yolo/ultralytics-main/yolov8s.pt', help='pretrained model path')
    parser.add_argument('--cfg', type=str, default='E:/gold_yolo/ultralytics-main/ultralytics/cfg/default.yaml', help='hyperparameters path')
    parser.add_argument('--data', type=str, default='E:/SCAM_yolo/ultralytics-main/ultralytics/cfg/datasets/VisDrone.yaml', help='data yaml path')
    
    parser.add_argument('--epochs', type=int, default=150, help='number of epochs to train for')
    parser.add_argument('--patience', type=int, default=100, help='EarlyStopping patience (epochs without improvement)')
    parser.add_argument('--unamp', action='store_true', help='Unuse Automatic Mixed Precision (AMP) training')
    parser.add_argument('--batch', type=int, default=1, help='number of images per batch (-1 for AutoBatch)')
    parser.add_argument('--imgsz', type=int, default=640, help='size of input images as integer')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='image --cache ram/disk')
    parser.add_argument('--device', type=str, default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--project', type=str, default=ROOT / 'runs/train', help='save to project/name')
    parser.add_argument('--name', type=str, default='exp', help='save to project/name')
    parser.add_argument('--resume', type=str, default='', help='resume training from last checkpoint')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'Adamax', 'NAdam', 'RAdam', 'AdamW', 'RMSProp', 'auto'], default='SGD', help='optimizer (auto -> ultralytics/yolo/engine/trainer.py in build_optimizer funciton.)')
    parser.add_argument('--close_mosaic', type=int, default=0, help='(int) disable mosaic augmentation for final epochs')
    parser.add_argument('--info', action="store_true", help='model info verbose')
    
    parser.add_argument('--save', type=str2bool, default='True', help='save train checkpoints and predict results')
    parser.add_argument('--save-period', type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--seed', type=int, default=1, help='Global training seed')
    parser.add_argument('--deterministic', action="store_true", default=True, help='whether to enable deterministic mode')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--cos-lr', action='store_true', help='cosine LR scheduler')
    parser.add_argument('--fraction', type=float, default=1.0, help='dataset fraction to train on (default is 1.0, all images in train set)')
    parser.add_argument('--profile', action='store_true', help='profile ONNX and TensorRT speeds during training for loggers')

    # region Segmentation
    # 没事了 试个东西
    parser.add_argument('--overlap_mask', type=str2bool, default='True', help='masks should overlap during training (segment train only)')
    parser.add_argument('--mask_ratio', type=int, default=4, help='mask downsample ratio (segment train only)')
    # endregion



    # Classification
    parser.add_argument('--dropout', type=float, default=0.0, help='use dropout regularization (classify train only)')

    return parser.parse_known_args()[0]


class YOLOV8(YOLO):
    '''
    yaml:model.yaml path
    weigth:pretrained model path
    '''

    def __init__(self, yaml='E:/gold_yolo/ultralytics-main/yolov8-goldyolo.yaml', weight='E:/SCAM_yolo/ultralytics-main/yolov8s.pt', task=None) -> None:
        super().__init__(yaml, task)
        if weight:
            self.load(weight)


if __name__ == '__main__':
    opt = parse_opt()
    model = YOLOV8(yaml=opt.yaml, weight=opt.weight)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)
    if opt.info:
        model.info(detailed=True, verbose=True)
        model.profile(opt.imgsz)

        print('before fuse...')
        model.info(detailed=False, verbose=True)
        print('after fuse...')
        model.fuse()
    else:
        model.train(**transformer_opt(opt))