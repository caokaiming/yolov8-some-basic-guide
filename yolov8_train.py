from ultralytics import YOLO

def train_yolov8():
    # 訓練配置
    config = {
        'imgsz': 640,  # 訓練圖片大小
        'batch': 16,  # 批次大小
        'epochs': 50,  # 訓練輪數
        'data': 'dataset.yaml',  # 數據集配置文件
        'cfg': 'yolov8n_custom.yaml',  # 模型配置文件
        'weights': 'yolov8n.pt',  # 預訓練模型權重文件
        'name': 'yolov8n_custom'  # 訓練過程的名稱
    }

    # 創建 YOLO 模型實例
    model = YOLO(config['weights'])

    # 開始訓練
    model.train(data=config['data'],
                imgsz=config['imgsz'],
                batch=config['batch'],
                epochs=config['epochs'],
                name=config['name'])

if __name__ == "__main__":
    train_yolov8()