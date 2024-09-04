from ultralytics import YOLO
import torch
import onnxruntime
# 加载模型
model = YOLO('E:/SCAM_yolo/ultralytics-main/yolov8s.pt')  # 加载官方模型（示例）
model = YOLO('E:/SCAM_yolo/distiil/distill/VisDrone-BCKD-ours-51.1/weights/best.pt')  # 加载自定义训练模型（示例）

# 设置模型为评估模式
model.eval()

# 示例输入张量
dummy_input = torch.randn(1, 3, 800, 800)

# 导出模型为ONNX格式
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    opset_version=11,  # 确保使用更高的opset版本以支持动态填充大小
    input_names=['input'],
    output_names=['output']
)