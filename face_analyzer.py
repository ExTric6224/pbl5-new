import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model

class FaceAnalyzer:
    def __init__(self, model_path='./models/emotion_final4.keras'):
        self.img_size = 32  # Kích thước ảnh đầu vào cho model
        self.emotion_labels = ['Surprise', 'Fear', 'Disgust', 'Happiness', 'Sadness', 'Anger', 'Neutral']
        self.model = self._load_model(model_path)
        if self.model is None:
            raise ValueError(f"❌ Không thể tải model từ {model_path}")

    def _load_model(self, path):
        try:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Không tìm thấy model tại: {path}")
            model = load_model(path)
            print(f"✅ Đã tải model từ {path}")
            return model
        except Exception as e:
            print(f"❌ Lỗi khi tải model: {e}")
            return None

    def analyzeFace(self, face_crop):
        """
        Dự đoán cảm xúc từ một vùng ảnh mặt RGB.
        Args:
            face_rgb (numpy.ndarray): ảnh mặt RGB (3 kênh)

        Returns:
            dict: {emotion: probability} hoặc None nếu lỗi
        """
        try:
            # Resize về 32x32 và chuẩn hóa
            face_crop_resized = cv2.resize(face_crop, (32, 32))
            face_crop_normalized = face_crop_resized / 255.0
            face_crop_reshaped = np.expand_dims(face_crop_normalized, axis=0)

            # Dự đoán cảm xúc - sửa lỗi truy cập mảng
            predictions = self.model.predict(face_crop_reshaped,verbose=0)
            # Lấy mảng đầu tiên từ batch predictions (batch size = 1)
            prediction_values = predictions[0]
            
            result = {
                self.emotion_labels[i]: float(prediction_values[i])
                for i in range(len(self.emotion_labels))
            }
            return result
        except Exception as e:
            print(f"❌ Lỗi khi phân tích khuôn mặt: {e}")
            return None

