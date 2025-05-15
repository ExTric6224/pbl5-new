import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# Tải mô hình đã huấn luyện
model = load_model('models/emotion_final4.keras')

# Định nghĩa ánh xạ nhãn (dựa trên thứ tự thư mục 1-7)
label_map = {
    0: 'Surprise',
    1: 'Fear',
    2: 'Disgust',
    3: 'Happiness',
    4: 'Sadness',
    5: 'Anger',
    6: 'Neutral'
}

# Khởi tạo MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

# Khởi tạo webcam
cap = cv2.VideoCapture(0)  # 0 là webcam mặc định

if not cap.isOpened():
    print("Không thể mở webcam!")
    exit()

while True:
    # Đọc khung hình từ webcam
    ret, frame = cap.read()
    if not ret:
        print("Không thể nhận khung hình!")
        break

    # Chuyển đổi khung hình sang RGB (MediaPipe yêu cầu RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Phát hiện khuôn mặt
    results = face_detection.process(frame_rgb)

    # Nếu phát hiện được khuôn mặt
    if results.detections:
        for detection in results.detections:
            # Lấy tọa độ hình chữ nhật bao quanh khuôn mặt
            bboxC = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x = int(bboxC.xmin * w)
            y = int(bboxC.ymin * h)
            width = int(bboxC.width * w)
            height = int(bboxC.height * h)

            # Đảm bảo tọa độ không vượt ra ngoài khung hình
            x = max(0, x)
            y = max(0, y)
            width = min(width, w - x)
            height = min(height, h - y)

            # Cắt vùng khuôn mặt
            face_crop = frame_rgb[y:y + height, x:x + width]
            if face_crop.size == 0:  # Kiểm tra nếu vùng cắt rỗng
                continue

            # Resize về 32x32 và chuẩn hóa
            face_crop_resized = cv2.resize(face_crop, (32, 32))
            face_crop_normalized = face_crop_resized / 255.0
            face_crop_reshaped = np.expand_dims(face_crop_normalized, axis=0)            # Dự đoán cảm xúc
            prediction = model.predict(face_crop_reshaped)
            predicted_class = np.argmax(prediction, axis=1)[0]
            confidence_score = prediction[0][predicted_class] * 100  # Tính phần trăm
            predicted_label = label_map[predicted_class]
            
            # Tạo nhãn với phần trăm
            emotion_text = f"{predicted_label}: {confidence_score:.1f}%"

            # Vẽ hình vuông quanh khuôn mặt
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

            # Đặt nhãn cảm xúc với phần trăm trên hình vuông
            label_position = (x, y - 10)
            cv2.putText(frame, emotion_text, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Hiển thị khung hình
    cv2.imshow('Emotion Detection', frame)

    # Thoát khi nhấn 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
face_detection.close()