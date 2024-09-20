# import cv2
# from ultralytics import YOLO
# import os
#
# def detect_and_save_images(model, image_folder, output_folder):
#     # 检查输出文件夹是否存在，不存在则创建
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
#
#     # 遍历图片文件夹中的每张图片
#     for filename in os.listdir(image_folder):
#         # 检查文件是否为图片文件
#         if filename.endswith(('.jpg', '.jpeg', '.png')):
#             # 构建图片文件的完整路径
#             image_path = os.path.join(image_folder, filename)
#             # 读取图片
#             img = cv2.imread(image_path)
#             # 对图像进行推断
#             res = model(img)
#             # 绘制检测结果
#             ann = res[0].plot(labels=False)
#             # 保存检测结果图片到输出文件夹
#             output_path = os.path.join(output_folder, filename)
#             cv2.imwrite(output_path, ann)
#
# if __name__ == "__main__":
#     # 加载YOLO模型
#     model = YOLO("E:/SCAM_yolo/distiil/distill/DIOR-bckd-ours/weights/best.pt")
#     # 原始图片文件夹路径
#     image_folder = "E:/DIOR_dataset_yolo/images/val/"
#     # 检测结果输出文件夹路径
#     output_folder = "E:/SCAM_yolo/test/DIOR/"
#
#     # 对整个val图片集合进行目标检测并保存结果
#     detect_and_save_images(model, image_folder, output_folder)
# import cv2
# import os
#
# # 定义类别对应的颜色（BGR格式）
# class_colors = {
#     0: (0, 0, 255),     # 类别0使用红色
#     1: (0, 255, 0),     # 类别1使用绿色
#     2: (255, 0, 0),     # 类别2使用蓝色
#     3: (0, 255, 255),   # 类别3使用黄色
#     4: (255, 0, 255),   # 类别4使用紫色
#     5: (255, 255, 0),   # 类别5使用青色
#     6: (0, 128, 255),   # 类别6使用橙色
#     7: (255, 128, 0),   # 类别7使用淡蓝色
#     8: (128, 0, 255),   # 类别8使用粉色
#     9: (128, 255, 0),   # 类别9使用浅绿色
#     10: (255, 192, 203), # 类别10使用浅粉色
#     11: (128, 128, 0),  # 类别11使用橄榄色
#     12: (128, 0, 128),  # 类别12使用紫红色
#     13: (0, 128, 128),  # 类别13使用深青色
#     14: (0, 0, 128),    # 类别14使用深红色
#     15: (128, 128, 128),# 类别15使用灰色
#     16: (0, 255, 128),  # 类别16使用春绿色
#     17: (255, 165, 0),  # 类别17使用橙色
#     18: (75, 0, 130),   # 类别18使用靛色
#     19: (255, 20, 147)  # 类别19使用深粉色
# }
#
# def draw_boxes_from_labels(image_path, label_path, output_path):
#     # 读取图片
#     img = cv2.imread(image_path)
#     h, w, _ = img.shape
#
#     # 读取标签文件
#     with open(label_path, 'r') as file:
#         labels = file.readlines()
#
#     for label in labels:
#         # 解析每行标签
#         label = label.strip().split()
#         class_id = int(label[0])
#         x_center = float(label[1])
#         y_center = float(label[2])
#         box_width = float(label[3])
#         box_height = float(label[4])
#
#         # 将比例转换为实际像素值
#         x_center *= w
#         y_center *= h
#         box_width *= w
#         box_height *= h
#
#         # 计算边界框的左上角和右下角坐标
#         x1 = int(x_center - box_width / 2)
#         y1 = int(y_center - box_height / 2)
#         x2 = int(x_center + box_width / 2)
#         y2 = int(y_center + box_height / 2)
#
#         # 获取当前类别的颜色
#         color = class_colors.get(class_id, (255, 255, 255))  # 默认白色
#
#         # 绘制对应颜色的矩形框
#         cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
#
#     # 保存带有检测框的图片
#     cv2.imwrite(output_path, img)
#
# def detect_and_save_images(image_folder, label_folder, output_folder):
#     # 检查输出文件夹是否存在，不存在则创建
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
#
#     # 遍历图片文件夹中的每张图片
#     for filename in os.listdir(image_folder):
#         # 检查文件是否为图片文件
#         if filename.endswith(('.jpg', '.jpeg', '.png')):
#             # 构建图片文件的完整路径
#             image_path = os.path.join(image_folder, filename)
#             # 构建对应标签文件的路径
#             label_path = os.path.join(label_folder, os.path.splitext(filename)[0] + '.txt')
#             # 构建输出图片的路径
#             output_path = os.path.join(output_folder, filename)
#             # 在图片上绘制检测框并保存
#             draw_boxes_from_labels(image_path, label_path, output_path)
#
# if __name__ == "__main__":
#     # 原始图片文件夹路径
#     image_folder = "E:/DIOR_dataset_yolo/images/val/"
#     # 标签文件夹路径
#     label_folder = "E:/DIOR_dataset_yolo/labels/val/"
#     # 检测结果输出文件夹路径
#     output_folder = "E:/SCAM_yolo/test/DIOR/"
#     # 类别名称（如果需要绘制类别名称，可以在这里提供）
#     class_names = None
#
#     # 对整个val图片集合进行目标检测并保存结果
#     detect_and_save_images(image_folder, label_folder, output_folder)
# import cv2
# from ultralytics import YOLO
# import os
#
# def draw_bboxes(img, results, font_scale=0.5, line_thickness=1):
#     for result in results:
#         boxes = result.boxes.xyxy.cpu().numpy()  # 预测框的坐标
#         confs = result.boxes.conf.cpu().numpy()  # 置信度
#         classes = result.boxes.cls.cpu().numpy()  # 类别索引
#         names = result.names  # 类别名称
#
#         for box, conf, cls in zip(boxes, confs, classes):
#             x1, y1, x2, y2 = map(int, box)
#             label = f"{names[int(cls)]} {conf:.2f}"
#             # 绘制框线
#             cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), line_thickness)
#             # 绘制标签
#             (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
#             cv2.rectangle(img, (x1, y1 - h - 10), (x1 + w, y1), (0, 255, 0), -1)
#             cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1)
#
# def detect_and_save_images(model, image_folder, output_folder):
#     # 检查输出文件夹是否存在，不存在则创建
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
#
#     # 遍历图片文件夹中的每张图片
#     for filename in os.listdir(image_folder):
#         # 检查文件是否为图片文件
#         if filename.endswith(('.jpg', '.jpeg', '.png')):
#             # 构建图片文件的完整路径
#             image_path = os.path.join(image_folder, filename)
#             # 读取图片
#             img = cv2.imread(image_path)
#             # 对图像进行推断
#             res = model(img)
#             # 绘制检测结果，带有标签和调整后的字体和框线
#             draw_bboxes(img, res, font_scale=0.5, line_thickness=1)
#             # 保存检测结果图片到输出文件夹
#             output_path = os.path.join(output_folder, filename)
#             cv2.imwrite(output_path, img)
#
# if __name__ == "__main__":
#     # 加载YOLO模型
#     model = YOLO("E:/SCAM_yolo/distiil/distill/DIOR-bckd-ours/weights/best.pt")
#     # 原始图片文件夹路径
#     image_folder = "E:/DIOR_dataset_yolo/images/val/"
#     # 检测结果输出文件夹路径
#     output_folder = "E:/SCAM_yolo/test/DIOR"
#
#     # 对整个val图片集合进行目标检测并保存结果
#     detect_and_save_images(model, image_folder, output_folder)

# import os
# import cv2
# from ultralytics import YOLO
# # import torch
# # import numpy as np
# # from yolov5 import YOLOv5
#
# # 定义类别对应的颜色（BGR格式）
# class_colors = {
#     0: (0, 0, 255),     # 类别0使用红色
#     1: (0, 255, 0),     # 类别1使用绿色
#     2: (255, 0, 0),     # 类别2使用蓝色
#     3: (0, 255, 255),   # 类别3使用黄色
#     4: (255, 0, 255),   # 类别4使用紫色
#     5: (255, 255, 0),   # 类别5使用青色
#     6: (0, 128, 255),   # 类别6使用橙色
#     7: (255, 128, 0),   # 类别7使用淡蓝色
#     8: (128, 0, 255),   # 类别8使用粉色
#     9: (128, 255, 0),   # 类别9使用浅绿色
#     10: (255, 192, 203), # 类别10使用浅粉色
#     11: (128, 128, 0),  # 类别11使用橄榄色
#     12: (128, 0, 128),  # 类别12使用紫红色
#     13: (0, 128, 128),  # 类别13使用深青色
#     14: (0, 0, 128),    # 类别14使用深红色
#     15: (128, 128, 128),# 类别15使用灰色
#     16: (0, 255, 128),  # 类别16使用春绿色
#     17: (255, 165, 0),  # 类别17使用橙色
#     18: (75, 0, 130),   # 类别18使用靛色
#     19: (255, 20, 147)  # 类别19使用深粉色
# }
#
# def draw_boxes(image, detections, class_colors):
#     for det in detections:
#         x1, y1, x2, y2, conf, class_id = map(int, det)
#         color = class_colors.get(class_id, (255, 255, 255))  # 默认白色
#         cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
#     return image
#
# def detect_and_save_images(model, image_folder, output_folder, class_colors):
#     # 检查输出文件夹是否存在，不存在则创建
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
#
#     # 遍历图片文件夹中的每张图片
#     for filename in os.listdir(image_folder):
#         # 检查文件是否为图片文件
#         if filename.endswith(('.jpg', '.jpeg', '.png')):
#             # 构建图片文件的完整路径
#             image_path = os.path.join(image_folder, filename)
#             # 读取图片
#             img = cv2.imread(image_path)
#             results = model(img)
#             detections = results.xyxy[0].numpy()  # 获取检测结果
#
#             # 在图片上绘制检测框
#             img = draw_boxes(img, detections, class_colors)
#             # 构建输出图片的路径
#             output_path = os.path.join(output_folder, filename)
#             # 保存带有检测框的图片
#             cv2.imwrite(output_path, img)
#
# if __name__ == "__main__":
#     # 加载YOLOv5模型
#     model = YOLO("E:/SCAM_yolo/distiil/distill/DIOR-bckd-ours/weights/best.pt")
#
#     # 原始图片文件夹路径
#     image_folder = "E:/DIOR_dataset_yolo/images/val/"
#     # 检测结果输出文件夹路径
#     output_folder = "E:/SCAM_yolo/test/DIOR/"
#     # 类别名称（如果需要绘制类别名称，可以在这里提供）
#     class_names = None
#
#     # 对整个val图片集合进行目标检测并保存结果
#     detect_and_save_images(model, image_folder, output_folder, class_colors)
import os
import cv2
from ultralytics import YOLO

# # 定义类别对应的颜色（BGR格式）
# class_colors = {
#     0: (0, 0, 255),     # 类别0使用红色
#     1: (0, 255, 0),     # 类别1使用绿色
#     2: (255, 0, 0),     # 类别2使用蓝色
#     3: (0, 255, 255),   # 类别3使用黄色
#     4: (255, 0, 255),   # 类别4使用紫色
#     5: (255, 255, 0),   # 类别5使用青色
#     6: (0, 128, 255),   # 类别6使用橙色
#     7: (255, 128, 0),   # 类别7使用淡蓝色
#     8: (128, 0, 255),   # 类别8使用粉色
#     9: (128, 255, 0),   # 类别9使用浅绿色
#     10: (255, 192, 203),# 类别10使用浅粉色
#     11: (128, 128, 0),  # 类别11使用橄榄色
#     12: (128, 0, 128),  # 类别12使用紫红色
#     13: (0, 128, 128),  # 类别13使用深青色
#     14: (0, 0, 128),    # 类别14使用深红色
#     15: (128, 128, 128),# 类别15使用灰色
#     16: (0, 255, 128),  # 类别16使用春绿色
#     17: (255, 165, 0),  # 类别17使用橙色
#     18: (75, 0, 130),   # 类别18使用靛色
#     19: (255, 20, 147)  # 类别19使用深粉色
# }
#
# def draw_bboxes(img, results, line_thickness=6):
#     for result in results:
#         boxes = result.boxes.xyxy.cpu().numpy()  # 预测框的坐标
#         classes = result.boxes.cls.cpu().numpy()  # 类别索引
#
#         for box, cls in zip(boxes, classes):
#             x1, y1, x2, y2 = map(int, box)
#             class_id = int(cls)
#             color = class_colors.get(class_id, (255, 255, 255))  # 默认白色
#             # 绘制框线
#             cv2.rectangle(img, (x1, y1), (x2, y2), color, line_thickness)
#
# def detect_and_save_images(model, image_folder, output_folder):
#     # 检查输出文件夹是否存在，不存在则创建
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
#
#     # 遍历图片文件夹中的每张图片
#     for filename in os.listdir(image_folder):
#         # 检查文件是否为图片文件
#         if filename.endswith(('.jpg', '.jpeg', '.png')):
#             # 构建图片文件的完整路径
#             image_path = os.path.join(image_folder, filename)
#             # 读取图片
#             img = cv2.imread(image_path)
#             # 对图像进行推断
#             res = model(img)
#             # 绘制检测结果，不带标签
#             draw_bboxes(img, res, line_thickness=3)
#             # 保存检测结果图片到输出文件夹
#             output_path = os.path.join(output_folder, filename)
#             cv2.imwrite(output_path, img)
#
# if __name__ == "__main__":
#     # 加载YOLO模型
#     model = YOLO("E:/SCAM_yolo/distiil/distill/DIOR-bckd-ours/weights/best.pt")
#     # 原始图片文件夹路径
#     image_folder = "E:/DIOR_dataset_yolo/images/val/"
#     # 检测结果输出文件夹路径
#     output_folder = "E:/SCAM_yolo/test/DIOR/"
#
#     # 对整个val图片集合进行目标检测并保存结果
#     detect_and_save_images(model, image_folder, output_folder)
# import cv2
# import os
#
# # 定义类别对应的颜色（BGR格式）
# class_colors = {
#     0: (0, 0, 255),     # 类别0使用红色
#     1: (0, 255, 0),     # 类别1使用绿色
#     2: (255, 0, 0),     # 类别2使用蓝色
#     3: (0, 255, 255),   # 类别3使用黄色
#     4: (255, 0, 255),   # 类别4使用紫色
#     5: (255, 255, 0),   # 类别5使用青色
#     6: (0, 128, 255),   # 类别6使用橙色
#     7: (255, 128, 0),   # 类别7使用淡蓝色
#     8: (128, 0, 255),   # 类别8使用粉色
#     9: (128, 255, 0)    # 类别9使用浅绿色
# }
#
# def draw_boxes_from_labels(image_path, label_path, output_path):
#     # 读取图片
#     img = cv2.imread(image_path)
#     h, w, _ = img.shape
#
#     # 读取标签文件
#     with open(label_path, 'r') as file:
#         labels = file.readlines()
#
#     for label in labels:
#         # 解析每行标签
#         label = label.strip().split()
#         class_id = int(label[0])
#         x_center = float(label[1])
#         y_center = float(label[2])
#         box_width = float(label[3])
#         box_height = float(label[4])
#
#         # 将比例转换为实际像素值
#         x_center *= w
#         y_center *= h
#         box_width *= w
#         box_height *= h
#
#         # 计算边界框的左上角和右下角坐标
#         x1 = int(x_center - box_width / 2)
#         y1 = int(y_center - box_height / 2)
#         x2 = int(x_center + box_width / 2)
#         y2 = int(y_center + box_height / 2)
#
#         # 获取当前类别的颜色
#         color = class_colors.get(class_id, (255, 255, 255))  # 默认白色
#
#         # 绘制对应颜色的矩形框
#         cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
#
#     # 保存带有检测框的图片
#     cv2.imwrite(output_path, img)
#
# def detect_and_save_images(image_folder, label_folder, output_folder):
#     # 检查输出文件夹是否存在，不存在则创建
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
#
#     # 遍历图片文件夹中的每张图片
#     for filename in os.listdir(image_folder):
#         # 检查文件是否为图片文件
#         if filename.endswith(('.jpg', '.jpeg', '.png')):
#             # 构建图片文件的完整路径
#             image_path = os.path.join(image_folder, filename)
#             # 构建对应标签文件的路径
#             label_path = os.path.join(label_folder, os.path.splitext(filename)[0] + '.txt')
#             # 构建输出图片的路径
#             output_path = os.path.join(output_folder, filename)
#             # 在图片上绘制检测框并保存
#             draw_boxes_from_labels(image_path, label_path, output_path)
#
# if __name__ == "__main__":
#     # 原始图片文件夹路径
#     image_folder = "E:/xinVisDrone/images/val/"
#     # 标签文件夹路径
#     label_folder = "E:/xinVisDrone/labels/val/"
#     # 检测结果输出文件夹路径
#     output_folder = "E:/SCAM_yolo/test/ourskuang/"
#
#     # 对整个val图片集合进行目标检测并保存结果
#     detect_and_save_images(image_folder, label_folder, output_folder)
#
# import cv2
# import os
# from ultralytics import YOLO
#
# # 定义十个颜色，每个类别一个
# COLORS = [
#     (255, 0, 0),    # Red
#     (0, 255, 0),    # Green
#     (0, 0, 255),    # Blue
#     (0, 255, 255),  # Cyan
#     (255, 0, 255),  # Magenta
#     (255, 255, 0),  # Yellow
#     (0, 128, 255),    # Maroon
#     (255, 128, 0),    # Dark Green
#     (128, 0, 255),    # Navy
#     (128, 255, 0)   # Olive
# ]
#
# def get_color_for_class(class_id):
#     # 确保 class_id 在颜色列表的范围内
#     return COLORS[class_id % len(COLORS)]
#
# def detect_and_save_images(model, image_folder, output_folder):
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
#
#     for filename in os.listdir(image_folder):
#         if filename.endswith(('.jpg', '.jpeg', '.png')):
#             image_path = os.path.join(image_folder, filename)
#             img = cv2.imread(image_path)
#             if img is None:
#                 print(f"Failed to read {image_path}")
#                 continue
#
#             results = model(img)
#
#             if len(results) > 0:
#                 annotated_img = img.copy()
#                 for detection in results[0].boxes.data:
#                     x1, y1, x2, y2, score, class_id = map(int, detection)
#                     color = get_color_for_class(class_id)
#                     cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
#
#                 output_path = os.path.join(output_folder, filename)
#                 cv2.imwrite(output_path, annotated_img)
#                 print(f"Saved annotated image to {output_path}")
#             else:
#                 print(f"No detections in {image_path}")
#
# if __name__ == "__main__":
#     model_path = "E:/gold_yolo/train/canshu/0.5GIoU+0.5NWD/weights/best.pt"
#     if not os.path.exists(model_path):
#         raise FileNotFoundError(f"Model file not found: {model_path}")
#
#     model = YOLO(model_path)
#     image_folder = "E:/xinVisDrone/images/val/"
#     if not os.path.exists(image_folder):
#         raise FileNotFoundError(f"Image folder not found: {image_folder}")
#
#     output_folder = "E:/gold_yolo/jiance/800/"
#     detect_and_save_images(model, image_folder, output_folder)


# import cv2
# import os
# import numpy as np
# from ultralytics import YOLO
#
# # 定义十个颜色，每个类别一个
# COLORS = [
#     (255, 0, 0),    # Red
#     (0, 255, 0),    # Green
#     (0, 0, 255),    # Blue
#     (0, 255, 255),  # Cyan
#     (255, 0, 255),  # Magenta
#     (255, 255, 0),  # Yellow
#     (0, 128, 255),  # Maroon
#     (255, 128, 0),  # Dark Green
#     (128, 0, 255),  # Navy
#     (128, 255, 0)   # Olive
# ]
#
# # 类别名字
# CLASS_NAMES = {
#     0: 'pedestrian',
#     1: 'people',
#     2: 'bicycle',
#     3: 'car',
#     4: 'van',
#     5: 'truck',
#     6: 'tricycle',
#     7: 'awning-tricycle',
#     8: 'bus',
#     9: 'motor'
# }
#
# # 示例类别平均精度数据
# average_precision = {
#     0: 0.85,
#     1: 0.75,
#     2: 0.90,
#     3: 0.60,
#     4: 0.88,
#     5: 0.92,
#     6: 0.80,
#     7: 0.70,
#     8: 0.65,
#     9: 0.78
# }
#
# def get_color_for_class(class_id):
#     return COLORS[class_id % len(COLORS)]
#
# def add_legend(img, detected_classes):
#     # 创建图例的高度和初始宽度
#     legend_height = 50
#     legend_width = img.shape[1]
#
#     # 创建一个白色的图例图像
#     legend = np.ones((legend_height, legend_width, 3), dtype=np.uint8) * 255
#
#     # 字体设置
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     font_scale = 0.6
#     font_thickness = 1
#     rectangle_size = 20  # 矩形大小
#     rectangle_thickness = 2  # 矩形边框厚度
#     x_offset = 10
#     y_offset = 30  # 文字的y坐标
#
#     row = 0
#     for i, class_id in enumerate(detected_classes):
#         color = get_color_for_class(class_id)
#         precision = average_precision.get(class_id, 0.0)
#         class_name = CLASS_NAMES.get(class_id, f'Class {class_id}')
#         text = f"{class_name}: {precision:.2f}"
#
#         text_width = cv2.getTextSize(text, font, font_scale, font_thickness)[0][0]
#         if x_offset + text_width + rectangle_size + 30 > legend_width:
#             # 换行
#             row += 1
#             x_offset = 10
#             y_offset += legend_height + 10  # 换行后调整y坐标
#
#         x_pos = x_offset
#         y_pos = y_offset + row * (legend_height + 10)  # 加上间隔
#
#         # 画颜色矩形边框
#         cv2.rectangle(legend, (x_pos, y_pos - 15), (x_pos + rectangle_size, y_pos + 5), color, rectangle_thickness)
#
#         # 绘制文本
#         cv2.putText(legend, text, (x_pos + 30, y_pos), font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)
#
#         x_offset += text_width + rectangle_size + 50  # 调整下一个标签的x坐标
#
#     # 将图例图像与原始图像上下合并
#     combined_img = np.vstack((img, legend))
#     return combined_img
#
#
# def detect_and_save_images(model, image_folder, output_folder):
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
#
#     for filename in os.listdir(image_folder):
#         if filename.endswith(('.jpg', '.jpeg', '.png')):
#             image_path = os.path.join(image_folder, filename)
#             img = cv2.imread(image_path)
#             if img is None:
#                 print(f"Failed to read {image_path}")
#                 continue
#
#             results = model(img)
#
#             if len(results) > 0:
#                 annotated_img = img.copy()
#                 detected_classes = set()
#                 for detection in results[0].boxes.data:
#                     x1, y1, x2, y2, score, class_id = map(int, detection)
#                     color = get_color_for_class(class_id)
#                     cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
#                     detected_classes.add(class_id)
#
#                 # 添加图例
#                 img_with_legend = add_legend(annotated_img, detected_classes)
#
#                 output_path = os.path.join(output_folder, filename)
#                 cv2.imwrite(output_path, img_with_legend)
#                 print(f"Saved annotated image to {output_path}")
#             else:
#                 print(f"No detections in {image_path}")
# import cv2
# import os
# import numpy as np
# from ultralytics import YOLO
#
# # 定义十个颜色，每个类别一个
# COLORS = [
#     (255, 0, 0),    # Red
#     (0, 255, 0),    # Green
#     (0, 0, 255),    # Blue
#     (0, 255, 255),  # Cyan
#     (255, 0, 255),  # Magenta
#     (255, 255, 0),  # Yellow
#     (0, 128, 255),  # Maroon
#     (255, 128, 0),  # Dark Green
#     (128, 0, 255),  # Navy
#     (128, 255, 0)   # Olive
# ]
#
# # 类别名字
# CLASS_NAMES = {
#     0: 'pedestrian',
#     1: 'people',
#     2: 'bicycle',
#     3: 'car',
#     4: 'van',
#     5: 'truck',
#     6: 'tricycle',
#     7: 'awning-tricycle',
#     8: 'bus',
#     9: 'motor'
# }
#
# def get_color_for_class(class_id):
#     return COLORS[class_id % len(COLORS)]
#
# def add_legend(img, results):
#     # 创建图例的高度和初始宽度
#     legend_height = 50
#     legend_width = img.shape[1]
#
#     # 创建一个白色的图例图像
#     legend = np.ones((legend_height, legend_width, 3), dtype=np.uint8) * 255
#
#     # 字体设置
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     font_scale = 0.6
#     font_thickness = 1
#     rectangle_size = 20  # 矩形大小
#     rectangle_thickness = 2  # 矩形边框厚度
#     x_offset = 10
#     y_offset = 30  # 文字的y坐标
#
#     row = 0
#     for det in results:
#         class_id = int(det[5])
#         color = get_color_for_class(class_id)
#         confidence = float(det[4])
#         class_name = CLASS_NAMES.get(class_id, f'Class {class_id}')
#         text = f"{class_name}: {confidence:.2f}"
#
#         text_width = cv2.getTextSize(text, font, font_scale, font_thickness)[0][0]
#         if x_offset + text_width + rectangle_size + 30 > legend_width:
#             # 换行
#             row += 1
#             x_offset = 10
#             y_offset += legend_height + 10  # 换行后调整y坐标
#
#         x_pos = x_offset
#         y_pos = y_offset + row * (legend_height + 10)  # 加上间隔
#
#         # 画颜色矩形边框
#         cv2.rectangle(legend, (x_pos, y_pos - 15), (x_pos + rectangle_size, y_pos + 5), color, rectangle_thickness)
#
#         # 绘制文本
#         cv2.putText(legend, text, (x_pos + 30, y_pos), font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)
#
#         x_offset += text_width + rectangle_size + 50  # 调整下一个标签的x坐标
#
#     # 将图例图像与原始图像上下合并
#     combined_img = np.vstack((img, legend))
#     return combined_img
#
#
# def detect_and_save_images(model, image_folder, output_folder):
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
#
#     for filename in os.listdir(image_folder):
#         if filename.endswith(('.jpg', '.jpeg', '.png')):
#             image_path = os.path.join(image_folder, filename)
#             img = cv2.imread(image_path)
#             if img is None:
#                 print(f"Failed to read {image_path}")
#                 continue
#
#             results = model(img)
#
#             if len(results.xyxy) > 0:
#                 annotated_img = img.copy()
#
#                 # 添加图例
#                 img_with_legend = add_legend(annotated_img, results)
#
#                 output_path = os.path.join(output_folder, filename)
#                 cv2.imwrite(output_path, img_with_legend)
#                 print(f"Saved annotated image to {output_path}")
#             else:
#                 print(f"No detections in {image_path}")
#
# if __name__ == "__main__":
#     model_path = "E:/gold_yolo/train/xiaorong/yolov8_s/weights/best.pt"
#     if not os.path.exists(model_path):
#         raise FileNotFoundError(f"Model file not found: {model_path}")
#
#     model = YOLO(model_path)
#     image_folder = "E:/xinVisDrone/images/val/"
#     if not os.path.exists(image_folder):
#         raise FileNotFoundError(f"Image folder not found: {image_folder}")
#
#     output_folder = "E:/gold_yolo/jiance/v8sutuli/"
#     detect_and_save_images(model, image_folder, output_folder)
# import cv2
# import os
# from ultralytics import YOLO
#
# # 定义十个颜色，每个类别一个
# COLORS = [
#     (255, 0, 0),    # Red
#     (0, 255, 0),    # Green
#     (0, 0, 255),    # Blue
#     (0, 255, 255),  # Cyan
#     (255, 0, 255),  # Magenta
#     (255, 255, 0),  # Yellow
#     (0, 128, 255),  # Maroon
#     (255, 128, 0),  # Dark Green
#     (128, 0, 255),  # Navy
#     (128, 255, 0)   # Olive
# ]
#
# def get_color_for_class(class_id):
#     # 确保 class_id 在颜色列表的范围内
#     return COLORS[class_id % len(COLORS)]
#
# def detect_and_save_images(model, image_folder, output_folder):
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
#
#     for filename in os.listdir(image_folder):
#         if filename.endswith(('.jpg', '.jpeg', '.png')):
#             image_path = os.path.join(image_folder, filename)
#             img = cv2.imread(image_path)
#             if img is None:
#                 print(f"Failed to read {image_path}")
#                 continue
#
#             results = model(img)[0]  # 获取检测结果
#
#             if results.boxes is not None:
#                 annotated_img = img.copy()
#                 for box in results.boxes:
#                     x1, y1, x2, y2 = map(int, box.xyxy[0])
#                     confidence = box.conf[0]
#                     class_id = int(box.cls[0])
#                     color = get_color_for_class(class_id)
#
#                     cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
#
#                     # 添加类别标签及其置信度
#                     label = f"{model.names[class_id]}: {confidence:.2f}"
#                     cv2.putText(annotated_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
#
#                 output_path = os.path.join(output_folder, filename)
#                 cv2.imwrite(output_path, annotated_img)
#                 print(f"Saved annotated image to {output_path}")
#             else:
#                 print(f"No detections in {image_path}")
#
# if __name__ == "__main__":
#     model_path = "E:/gold_yolo/train/canshu/0.5GIoU+0.5NWD/weights/best.pt"
#     if not os.path.exists(model_path):
#         raise FileNotFoundError(f"Model file not found: {model_path}")
#
#     model = YOLO(model_path)
#     image_folder = "E:/xinVisDrone/images/val/"
#     if not os.path.exists(image_folder):
#         raise FileNotFoundError(f"Image folder not found: {image_folder}")
#
#     output_folder = "E:/gold_yolo/jiance/ceshi/"
#     detect_and_save_images(model, image_folder, output_folder)
# import cv2
# import os
# import numpy as np
# from ultralytics import YOLO
#
# # 定义十个颜色，每个类别一个
# COLORS = [
#     (255, 0, 0),    # Red
#     (0, 255, 0),    # Green
#     (0, 0, 255),    # Blue
#     (0, 255, 255),  # Cyan
#     (255, 0, 255),  # Magenta
#     (255, 255, 0),  # Yellow
#     (0, 128, 255),  # Maroon
#     (255, 128, 0),  # Dark Green
#     (128, 0, 255),  # Navy
#     (128, 255, 0)   # Olive
# ]
#
# def get_color_for_class(class_id):
#     return COLORS[class_id % len(COLORS)]
#
# def add_legend(img, class_confidences, class_names):
#     legend_height = 50  # 图例高度
#     legend = np.ones((legend_height, img.shape[1], 3), dtype=np.uint8) * 255  # 创建一个白色的图例图像
#
#     # 字体设置
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     font_scale = 0.6
#     font_thickness = 1
#     x_offset = 10
#     y_offset = 30  # 文字的y坐标
#
#     for class_id, confidence in class_confidences.items():
#         color = get_color_for_class(class_id)
#         class_name = class_names[class_id]
#         text = f"{class_name}: {confidence:.2f}"
#
#         # 画颜色矩形边框
#         cv2.rectangle(legend, (x_offset, y_offset - 15), (x_offset + 20, y_offset + 5), color, -1)
#
#         # 绘制文本
#         cv2.putText(legend, text, (x_offset + 30, y_offset), font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)
#
#         x_offset += 200  # 调整下一个标签的x坐标
#
#     # 将图例图像与原始图像上下合并
#     combined_img = np.vstack((img, legend))
#     return combined_img
#
# def detect_and_save_images(model, image_folder, output_folder):
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
#
#     for filename in os.listdir(image_folder):
#         if filename.endswith(('.jpg', '.jpeg', '.png')):
#             image_path = os.path.join(image_folder, filename)
#             img = cv2.imread(image_path)
#             if img is None:
#                 print(f"Failed to read {image_path}")
#                 continue
#
#             results = model(img)[0]  # 获取检测结果
#
#             class_confidences = {}
#             annotated_img = img.copy()
#
#             if results.boxes is not None:
#                 for box in results.boxes:
#                     x1, y1, x2, y2 = map(int, box.xyxy[0])
#                     confidence = box.conf[0]
#                     class_id = int(box.cls[0])
#                     color = get_color_for_class(class_id)
#
#                     cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
#
#                     # 更新最高置信度
#                     if class_id not in class_confidences or class_confidences[class_id] < confidence:
#                         class_confidences[class_id] = confidence
#
#             # 添加图例
#             img_with_legend = add_legend(annotated_img, class_confidences, model.names)
#
#             output_path = os.path.join(output_folder, filename)
#             cv2.imwrite(output_path, img_with_legend)
#             print(f"Saved annotated image to {output_path}")
#
# if __name__ == "__main__":
#     model_path = "E:/gold_yolo/train/canshu/0.5GIoU+0.5NWD/weights/best.pt"
#     if not os.path.exists(model_path):
#         raise FileNotFoundError(f"Model file not found: {model_path}")
#
#     model = YOLO(model_path)
#     image_folder = "E:/xinVisDrone/images/val/"
#     if not os.path.exists(image_folder):
#         raise FileNotFoundError(f"Image folder not found: {image_folder}")
#
#     output_folder = "E:/gold_yolo/jiance/ceshituli/"
#     detect_and_save_images(model, image_folder, output_folder)

# import cv2
# import os
# import numpy as np
# from ultralytics import YOLO
# from PIL import Image, ImageDraw, ImageFont
#
# # 定义十个颜色，每个类别一个
# COLORS = [
#     (255, 0, 0),    # Red
#     (0, 255, 0),    # Green
#     (0, 0, 255),    # Blue
#     (0, 255, 255),  # Cyan
#     (255, 0, 255),  # Magenta
#     (255, 255, 0),  # Yellow
#     (0, 128, 255),  # Maroon
#     (255, 128, 0),  # Dark Green
#     (128, 0, 255),  # Navy
#     (128, 255, 0)   # Olive
# ]
#
# rectangle_thickness = 2  # 矩形边框厚度
#
# def get_color_for_class(class_id):
#     return COLORS[class_id % len(COLORS)]
#
# def add_legend(img, class_confidences, class_names):
#     num_classes = len(class_confidences)
#     max_per_row = 5  # 每行最多显示7个标签
#     rows = (num_classes + max_per_row - 1) // max_per_row
#
#     legend_height = 70 * rows  # 每行高度为70像素
#     legend = np.ones((legend_height, img.shape[1], 3), dtype=np.uint8) * 255  # 创建一个白色的图例图像
#
#     pil_img = Image.fromarray(legend)
#     draw = ImageDraw.Draw(pil_img)
#     font_size = 30  # 初始字体大小
#
#     # 选择字体
#     try:
#         font = ImageFont.truetype("arial.ttf", font_size)  # 使用Arial字体
#     except IOError:
#         font = ImageFont.load_default()  # 如果找不到字体文件，使用默认字体
#
#     x_offset = 10
#     y_offset = 10  # 文字的初始y坐标
#     row_offset = 0
#
#     count = 0
#     for class_id, confidence in class_confidences.items():
#         if count > 0 and count % max_per_row == 0:
#             x_offset = 10
#             row_offset += 1
#             y_offset = 10 + row_offset * 70
#
#         color = get_color_for_class(class_id)
#         class_name = class_names[class_id]
#         text = f"{class_name}: {confidence:.2f}"
#
#         text_width, text_height = draw.textsize(text, font=font)
#
#         draw.rectangle([x_offset, y_offset, x_offset + 30, y_offset + 30], fill=color, outline=color)
#         draw.text((x_offset + 40, y_offset), text, fill="black", font=font)
#
#         x_offset += text_width + 70  # 动态调整下一个标签的x坐标
#         count += 1
#
#     combined_img = np.vstack((img, np.array(pil_img)))
#     return combined_img
#
# def detect_and_save_images(model, image_folder, output_folder):
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
#
#     for filename in os.listdir(image_folder):
#         if filename.endswith(('.jpg', '.jpeg', '.png')):
#             image_path = os.path.join(image_folder, filename)
#             img = cv2.imread(image_path)
#             if img is None:
#                 print(f"Failed to read {image_path}")
#                 continue
#
#             results = model(img)[0]
#
#             class_confidences = {}
#             annotated_img = img.copy()
#
#             if results.boxes is not None:
#                 for box in results.boxes:
#                     x1, y1, x2, y2 = map(int, box.xyxy[0])
#                     confidence = float(box.conf[0])
#                     class_id = int(box.cls[0])
#                     color = get_color_for_class(class_id)
#
#                     # 只描绘边框，不填充矩形
#                     cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, rectangle_thickness)
#
#                     if class_id not in class_confidences or class_confidences[class_id] < confidence:
#                         class_confidences[class_id] = confidence
#
#             img_with_legend = add_legend(annotated_img, class_confidences, model.names)
#
#             output_path = os.path.join(output_folder, filename)
#             cv2.imwrite(output_path, img_with_legend)
#             print(f"Saved annotated image to {output_path}")
#
# if __name__ == "__main__":
#     model_path = "E:/gold_yolo/train/xiaorong/yolov8_s/weights/best.pt"
#     if not os.path.exists(model_path):
#         raise FileNotFoundError(f"Model file not found: {model_path}")
#
#     model = YOLO(model_path)
#     image_folder = "E:/xinVisDrone/images/val/"
#     if not os.path.exists(image_folder):
#         raise FileNotFoundError(f"Image folder not found: {image_folder}")
#
#     output_folder = "E:/gold_yolo/jiance/v8s/"
#     detect_and_save_images(model, image_folder, output_folder)

#
# import cv2
# import os
# import numpy as np
# from ultralytics import YOLO
# from PIL import Image, ImageDraw, ImageFont
#
# # 定义十个颜色，每个类别一个
# COLORS = [
#     (255, 0, 0),    # Red
#     (0, 255, 0),    # Green
#     (0, 0, 255),    # Blue
#     (0, 255, 255),  # Cyan
#     (255, 0, 255),  # Magenta
#     (255, 255, 0),  # Yellow
#     (0, 128, 255),  # Maroon
#     (255, 128, 0),  # Dark Green
#     (128, 0, 255),  # Navy
#     (128, 255, 0)   # Olive
# ]
#
# rectangle_thickness = 2  # 矩形边框厚度
#
# def get_color_for_class(class_id):
#     return COLORS[class_id % len(COLORS)]
#
# def add_legend(img, class_confidences, class_names):
#     num_classes = len(class_confidences)
#     max_per_row = 4  # 每行最多显示7个标签
#     rows = (num_classes + max_per_row - 1) // max_per_row
#
#     legend_height = 70 * rows  # 每行高度为70像素
#     legend = np.ones((legend_height, img.shape[1], 3), dtype=np.uint8) * 255  # 创建一个白色的图例图像
#
#     pil_img = Image.fromarray(legend)
#     draw = ImageDraw.Draw(pil_img)
#     font_size = 30  # 初始字体大小
#
#     # 选择字体
#     try:
#         font = ImageFont.truetype("arial.ttf", font_size)  # 使用Arial字体
#     except IOError:
#         font = ImageFont.load_default()  # 如果找不到字体文件，使用默认字体
#
#     x_offset = 10
#     y_offset = 10  # 文字的初始y坐标
#     row_offset = 0
#
#     count = 0
#     for class_id, confidence in class_confidences.items():
#         if count > 0 and count % max_per_row == 0:
#             x_offset = 10
#             row_offset += 1
#             y_offset = 10 + row_offset * 70
#
#         color = get_color_for_class(class_id)
#         class_name = class_names[class_id]
#         text = f"{class_name}: {confidence:.2f}"
#
#         text_width, text_height = draw.textsize(text, font=font)
#
#         draw.rectangle([x_offset, y_offset, x_offset + 30, y_offset + 30], fill=color, outline=color)
#         draw.text((x_offset + 40, y_offset), text, fill="black", font=font)
#
#         x_offset += text_width + 70  # 动态调整下一个标签的x坐标
#         count += 1
#
#     combined_img = np.vstack((img, np.array(pil_img)))
#     return combined_img
#
# def detect_and_save_single_image(model, image_path, output_path):
#     img = cv2.imread(image_path)
#     if img is None:
#         print(f"Failed to read {image_path}")
#         return
#
#     results = model(img)[0]
#
#     class_confidences = {}
#     annotated_img = img.copy()
#
#     if results.boxes is not None:
#         for box in results.boxes:
#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             confidence = float(box.conf[0])
#             class_id = int(box.cls[0])
#             color = get_color_for_class(class_id)
#
#             # 只描绘边框，不填充矩形
#             cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, rectangle_thickness)
#
#             if class_id not in class_confidences or class_confidences[class_id] < confidence:
#                 class_confidences[class_id] = confidence
#
#     img_with_legend = add_legend(annotated_img, class_confidences, model.names)
#
#     cv2.imwrite(output_path, img_with_legend)
#     print(f"Saved annotated image to {output_path}")
#
# if __name__ == "__main__":
#     model_path = "E:/SCAM_yolo/runs的/VisDrone-l模型-800px-exp66/weights/best.pt"
#     if not os.path.exists(model_path):
#         raise FileNotFoundError(f"Model file not found: {model_path}")
#
#     model = YOLO(model_path)
#     image_path = "E:/xinVisDrone/images/val/0000116_00351_d_0000083.jpg"
#     output_path = "E:/gold_yolo/output_image4.jpg"
#
#     detect_and_save_single_image(model, image_path, output_path)

#  linux下
# import cv2
# import os
# import numpy as np
# from ultralytics import YOLO
# from PIL import Image, ImageDraw, ImageFont
#
# # 定义十个颜色，每个类别一个
# COLORS = [
#     (255, 0, 0),    # Red
#     (0, 255, 0),    # Green
#     (0, 0, 255),    # Blue
#     (0, 255, 255),  # Cyan
#     (255, 0, 255),  # Magenta
#     (255, 255, 0),  # Yellow
#     (0, 128, 255),  # Maroon
#     (255, 128, 0),  # Dark Green
#     (128, 0, 255),  # Navy
#     (128, 255, 0)   # Olive
# ]
#
# rectangle_thickness = 2  # 矩形边框厚度
#
# def get_color_for_class(class_id):
#     return COLORS[class_id % len(COLORS)]
#
# def add_legend(img, class_confidences, class_names):
#     num_classes = len(class_confidences)
#     max_per_row = 4  # 每行最多显示7个标签
#     rows = (num_classes + max_per_row - 1) // max_per_row
#
#     legend_height = 70 * rows  # 每行高度为70像素
#     legend = np.ones((legend_height, img.shape[1], 3), dtype=np.uint8) * 255  # 创建一个白色的图例图像
#
#     pil_img = Image.fromarray(legend)
#     draw = ImageDraw.Draw(pil_img)
#     font_size = 30  # 初始字体大小
#
#     # 选择字体
#     try:
#         font = ImageFont.truetype("arial.ttf", font_size)  # 使用Arial字体
#     except IOError:
#         font = ImageFont.load_default()  # 如果找不到字体文件，使用默认字体
#
#     x_offset = 10
#     y_offset = 10  # 文字的初始y坐标
#     row_offset = 0
#
#     count = 0
#     for class_id, confidence in class_confidences.items():
#         if count > 0 and count % max_per_row == 0:
#             x_offset = 10
#             row_offset += 1
#             y_offset = 10 + row_offset * 70
#
#         color = get_color_for_class(class_id)
#         class_name = class_names[class_id]
#         text = f"{class_name}: {confidence:.2f}"
#
#         bbox = draw.textbbox((0, 0), text, font=font)
#         text_width = bbox[2] - bbox[0]
#         text_height = bbox[3] - bbox[1]
#
#         draw.rectangle([x_offset, y_offset, x_offset + 30, y_offset + 30], fill=color, outline=color)
#         draw.text((x_offset + 40, y_offset), text, fill="black", font=font)
#
#         x_offset += text_width + 70  # 动态调整下一个标签的x坐标
#         count += 1
#
#     combined_img = np.vstack((img, np.array(pil_img)))
#     return combined_img
#
# def detect_and_save_single_image(model, image_path, output_path):
#     img = cv2.imread(image_path)
#     if img is None:
#         print(f"Failed to read {image_path}")
#         return
#
#     results = model(img)[0]
#
#     class_confidences = {}
#     annotated_img = img.copy()
#
#     if results.boxes is not None:
#         for box in results.boxes:
#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             confidence = float(box.conf[0])
#             class_id = int(box.cls[0])
#             color = get_color_for_class(class_id)
#
#             # 只描绘边框，不填充矩形
#             cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, rectangle_thickness)
#
#             if class_id not in class_confidences or class_confidences[class_id] < confidence:
#                 class_confidences[class_id] = confidence
#
#     img_with_legend = add_legend(annotated_img, class_confidences, model.names)
#
#     cv2.imwrite(output_path, img_with_legend)
#     print(f"Saved annotated image to {output_path}")
#
# if __name__ == "__main__":
#     model_path = "E:/SCAM_yolo/runs的/VisDrone-l模型-800px-exp66/weights/best.pt"
#     if not os.path.exists(model_path):
#         raise FileNotFoundError(f"Model file not found: {model_path}")
#
#     model = YOLO(model_path)
#     image_path = "E:/xinVisDrone/images/val/0000116_00351_d_0000083.jpg"
#     output_path = "E:/gold_yolo/output_image4.jpg"
#
#     detect_and_save_single_image(model, image_path, output_path)
import cv2
import os

# 定义十个颜色，每个类别一个
# COLORS = [
#     (255, 56, 56),    # Red
#     (255, 157, 151),  # Orange
#     (255, 112, 31),  # Yellow
#     (255, 178, 29),    # Green
#     (207, 210, 49),  # Cyan
#     (72, 249, 10),    # Blue
#     (146, 204, 23),  # Magenta
#     (61, 219, 134),  # Dark Green
#     (26, 147, 52), # Navy
#     (0, 212, 187)  # Olive
# ]
COLORS = [
    (255, 56, 56),    # airplane (Red)
    (255, 157, 151),  # airport (Orange)
    (255, 112, 31),   # baseballfield (Yellow)
    (255, 178, 29),   # basketballcourt (Green)
    (207, 210, 49),   # bridge (Cyan)
    (72, 249, 10),    # chimney (Blue)
    (146, 204, 23),   # dam (Magenta)
    (61, 219, 134),   # Expressway-Service-area (Dark Green)
    (26, 147, 52),    # Expressway-toll-station (Navy)
    (0, 212, 187),    # golffield (Olive)
    (168, 153, 44),   # groundtrackfield
    (255, 194, 0),    # harbor
    (147, 69, 52),    # overpass
    (255, 115, 100),  # ship
    (236, 24, 0),     # stadium
    (255, 56, 132),   # storagetank
    (133, 0, 82),     # tenniscourt
    (255, 56, 203),   # trainstation
    (200, 149, 255),  # vehicle
    (199, 55, 255)    # windmill
]
def get_color_for_class(class_id):
    # 确保 class_id 在颜色列表的范围内
    color_index = class_id if class_id >= 0 and class_id < len(COLORS) else 0
    return COLORS[color_index][::-1]  # 反转颜色通道顺序

def detect_and_save_images(image_folder, label_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历图片文件夹中的每张图片
    for filename in os.listdir(image_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(image_folder, filename)
            img = cv2.imread(image_path)
            if img is None:
                print(f"Failed to read {image_path}")
                continue

            # 获取图像尺寸
            height, width = img.shape[:2]

            # 获取对应的标签文件路径
            label_filename = os.path.splitext(filename)[0] + '.txt'
            label_path = os.path.join(label_folder, label_filename)

            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    lines = f.readlines()

                for line in lines:
                    parts = line.strip().split()
                    class_id = int(parts[0])
                    x_center, y_center, box_width, box_height = map(float, parts[1:])

                    # 转换为绝对像素坐标
                    x1 = int((x_center - box_width / 2) * width)
                    y1 = int((y_center - box_height / 2) * height)
                    x2 = int((x_center + box_width / 2) * width)
                    y2 = int((y_center + box_height / 2) * height)

                    color = get_color_for_class(class_id)
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 4)

                output_path = os.path.join(output_folder, filename)
                cv2.imwrite(output_path, img)
                print(f"Saved annotated image to {output_path}")
            else:
                print(f"No label file for {filename}")

if __name__ == "__main__":
    # 原始图片文件夹路径
    image_folder = "E:/DIOR_dataset_yolo/images/val/"
    if not os.path.exists(image_folder):
        raise FileNotFoundError(f"Image folder not found: {image_folder}")

    # 标签文件夹路径
    label_folder = "E:/DIOR_dataset_yolo/labels/val/"
    if not os.path.exists(label_folder):
        raise FileNotFoundError(f"Label folder not found: {label_folder}")

    # 检测结果输出文件夹路径
    output_folder = "E:/SCAM_yolo/ground/"

    # 对整个val图片集合进行目标检测并保存结果
    detect_and_save_images(image_folder, label_folder, output_folder)









