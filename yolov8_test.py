import torch
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

def load_image(image_path):
    # 加載圖片
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def show_image(img, detections):
    # 在圖片上繪製檢測結果
    for detection in detections:
        x1, y1, x2, y2 = detection[:4].astype(int)
        conf = detection[4]
        cls = int(detection[5])
        label = f'{cls} {conf:.2f}'
        color = (0, 255, 0)  # 綠色框
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    plt.imshow(img)
    plt.axis('off')
    plt.show()

def inference_yolov8(model_path, image_path):
    # 加載訓練好的模型
    model = YOLO(model_path)

    # 加載圖片
    img = load_image(image_path)

    # 執行推論
    results = model(img)

    # 解析結果
    detections = results[0].boxes.data.cpu().numpy()
    show_image(img, detections)

if __name__ == "__main__":
    model_path = 'runs/train/yolov8n_custom/weights/best.pt'  # 訓練好的模型權重文件路徑
    image_path = 'path_to_your_image.jpg'  # 要推論的圖片路徑
    inference_yolov8(model_path, image_path)
