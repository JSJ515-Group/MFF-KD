# # Ultralytics YOLO 🚀, AGPL-3.0 许可
# import cv2
# from ultralytics import YOLO
#
# # 初始化YOLOv8模型，设置自己训练好的模型路径
# mdl = 'E:/gold_yolo/train/canshu/0.5GIoU+0.5NWD/weights/best.pt'
# model = YOLO(mdl)
#
# # 打开视频文件
# video_path = 'E:/gold_yolo/lv_0_20240429160510.mp4'  # 替换为你的视频文件路径
# cap = cv2.VideoCapture(video_path)
#
# # 逐帧处理视频
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     # 对每一帧进行预测，设置置信度阈值为0.8
#     results = model(frame, conf=0.8)
#
#     # 在帧上绘制预测结果
#     for result in results:
#         for box in result.boxes:
#             xyxy = box.xyxy.squeeze().tolist()  # 获取矩形框的坐标
#             x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 绘制矩形框
#
#             c = int(box.cls)  # 获取分类标签
#             conf = float(box.conf)  # 获取置信度
#             id = None if box.id is None else int(box.id.item())  # 获取ID（如果存在）
#             name = result.names[c]  # 获取分类名称
#             label = f'{name}' + (f' id:{id}' if id is not None else '')  # 创建标签
#
#             # 在矩形框上方绘制标签和置信度
#             cv2.putText(frame, f"{label}: {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#
#     # 显示带有预测结果的帧
#     cv2.imshow("Predictions", frame)
#
#     # 按下'q'键退出循环
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break
#
# # 释放视频捕捉设备并关闭所有OpenCV窗口
# cap.release()
# cv2.destroyAllWindows()
# Ultralytics YOLO 🚀, AGPL-3.0 许可
import cv2
from ultralytics import YOLO

# 初始化YOLOv8模型，设置自己训练好的模型路径
mdl = 'E:/gold_yolo/train/canshu/0.5GIoU+0.5NWD/weights/best.pt'
model = YOLO(mdl)

# 打开视频文件
video_path = 'E:/gold_yolo/20240520_171027.mp4'  # 替换为你的视频文件路径
cap = cv2.VideoCapture(video_path)

# 获取视频的宽度、高度和帧率
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# 设置保存视频的格式和文件名，并调整分辨率
output_path = 'E:/gold_yolo/video21.mp4'  # 替换为你想保存的视频文件路径
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用MP4编码器
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))  # 保持原始分辨率

# 逐帧处理视频
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 对每一帧进行预测，设置置信度阈值为0.8
    results = model(frame, conf=0.5)

    # 在帧上绘制预测结果
    for result in results:
        for box in result.boxes:
            xyxy = box.xyxy.squeeze().tolist()  # 获取矩形框的坐标
            x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 绘制矩形框

            c = int(box.cls)  # 获取分类标签
            conf = float(box.conf)  # 获取置信度
            id = None if box.id is None else int(box.id.item())  # 获取ID（如果存在）
            name = result.names[c]  # 获取分类名称
            label = f'{name}' + (f' id:{id}' if id is not None else '')  # 创建标签

            # 在矩形框上方绘制标签和置信度
            cv2.putText(frame, f"{label}: {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 将处理后的帧写入输出视频文件
    out.write(frame)

    # 显示带有预测结果的帧
    cv2.imshow("Predictions", frame)

    # 按下'q'键退出循环
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 释放视频捕捉设备和视频写入对象，并关闭所有OpenCV窗口
cap.release()
out.release()
cv2.destroyAllWindows()

