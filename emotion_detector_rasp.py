# emotion_detector_server_pi.py
import cv2
import numpy as np
# import argparse # Bỏ nếu không dùng args cho server
import os
import time
import threading
from datetime import datetime
import mediapipe as mp
import socket
import pickle
import struct
import tempfile


# --- Nhập các lớp Analyzer ---
from face_analyzer import FaceAnalyzer # Giữ nguyên, đảm bảo file tồn tại và đúng
from database.emotion_history_item import EmotionHistoryItem # Giữ nguyên
from voice_analyzer import VoiceAnalyzer # Giữ nguyên

# db
from database.db_utils import load_emotion_history_from_db, save_all_emotions_to_db # Giữ nguyên

# --- Thư viện âm thanh (vẫn giữ) ---
try:
    # import sounddevice as sd # Không cần thiết cho server nếu chỉ nhận dữ liệu
    SOUND_DEVICE_AVAILABLE = True # VoiceAnalyzer có thể vẫn cần thư viện này
except ImportError:
    SOUND_DEVICE_AVAILABLE = False
except Exception:
    SOUND_DEVICE_AVAILABLE = False

# --- Hàm tiện ích cho socket ---
def send_data(sock, payload, is_json=False):
    """Gửi dữ liệu (pickle hoặc JSON) với tiền tố độ dài."""
    if is_json:
        serialized_payload = json.dumps(payload).encode('utf-8')
    else:
        serialized_payload = pickle.dumps(payload)
    
    msg_len = struct.pack(">L", len(serialized_payload))
    try:
        sock.sendall(msg_len + serialized_payload)
    except socket.error as e:
        print(f"Lỗi khi gửi dữ liệu: {e}")
        raise

def recv_data(sock, is_json=False):
    """Nhận dữ liệu (pickle hoặc JSON) dựa trên tiền tố độ dài."""
    try:
        raw_msg_len = sock.recv(4)
        if not raw_msg_len:
            return None 
        msg_len = struct.unpack(">L", raw_msg_len)[0]

        data = b''
        while len(data) < msg_len:
            packet = sock.recv(msg_len - len(data))
            if not packet:
                return None
            data += packet
        
        if is_json:
            return json.loads(data.decode('utf-8'))
        else:
            return pickle.loads(data)
    except (struct.error, pickle.UnpicklingError, json.JSONDecodeError) as e:
        print(f"Lỗi khi nhận hoặc giải mã dữ liệu: {e}")
        return None
    except socket.error as e:
        print(f"Lỗi socket khi nhận dữ liệu: {e}")
        return None


class EmotionDetector:
    def __init__(self, face_analyzer, voice_analyzer, enable_analysis_face=True, enable_analysis_voice=True):
        if not isinstance(face_analyzer, FaceAnalyzer):
            raise TypeError("'face_analyzer' phải là một instance của FaceAnalyzer.")
        self.face_analyzer = face_analyzer
        self.voice_analyzer = voice_analyzer

        # Dữ liệu chia sẻ - được cập nhật bởi các luồng server
        self.latest_frame_from_pi = None # Khung hình gốc hoặc đã xử lý từ Pi để hiển thị
        self.last_face_emotion = "N/A"
        self.last_face_emotion_probabilities = None
        
        self.last_voice_emotion = "N/A"
        self.last_voice_probabilities = None
        
        self.frame_lock = threading.Lock() # Bảo vệ self.latest_frame_from_pi
        self.emotion_lock = threading.Lock() # Bảo vệ các biến emotion và history

        self.stop_event = threading.Event()
        self.face_server_thread = None
        self.voice_server_thread = None
        
        self.enable_analysis_face = enable_analysis_face
        self.enable_analysis_voice = enable_analysis_voice
        self.emotion_history = load_emotion_history_from_db("./database/emotion_log.db")
        
        # Khởi tạo MediaPipe Face Detection (vẫn cần cho face_analyzer)
        self.mp_face_detection = mp.solutions.face_detection
        # self.mp_drawing = mp.solutions.drawing_utils # Không vẽ trực tiếp ở đây nữa
        self.face_detection_model = self.mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        print("EmotionDetector (Server Mode) initialized.")

    def _handle_face_client(self, client_socket):
        print("Raspberry Pi (Face Client) connected.")
        try:
            while not self.stop_event.is_set():
                message = recv_data(client_socket) # Mong đợi {'data': image_bytes}
                if message is None:
                    print("Face client disconnected or sent invalid data.")
                    break
                
                image_bytes = message.get('data')
                if not isinstance(image_bytes, bytes):
                    print("Dữ liệu hình ảnh không hợp lệ từ face client.")
                    send_data(client_socket, {"status": "error", "message": "Invalid image data"})
                    continue

                nparr = np.frombuffer(image_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if frame is None:
                    print("Không thể giải mã hình ảnh từ face client.")
                    send_data(client_socket, {"status": "error", "message": "Cannot decode image"})
                    continue
                
                processed_frame_for_ui = frame.copy() # Khung hình để vẽ và hiển thị (nếu có UI)
                current_face_emotion = "N/A"
                current_face_probs = None
                face_location_str = "N/A"

                if self.enable_analysis_face:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = self.face_detection_model.process(frame_rgb)
                    
                    if results.detections:
                        detection = results.detections[0] # Chỉ xử lý khuôn mặt đầu tiên
                        bboxC = detection.location_data.relative_bounding_box
                        h_img, w_img, _ = frame.shape
                        x = int(bboxC.xmin * w_img)
                        y = int(bboxC.ymin * h_img)
                        width = int(bboxC.width * w_img)
                        height = int(bboxC.height * h_img)

                        x = max(0, x)
                        y = max(0, y)
                        width = min(width, w_img - x)
                        height = min(height, h_img - y)

                        face_crop = frame_rgb[y:y + height, x:x + width]
                        
                        if face_crop.size > 0:
                            predicted = self.face_analyzer.analyzeFace(face_crop)
                            if predicted:
                                current_face_emotion, prob_val = max(predicted.items(), key=lambda item: item[1])
                                current_face_probs = predicted # Lưu dict đầy đủ
                                face_location_str = f"{x}x{y}"
                                # Vẽ lên processed_frame_for_ui
                                cv2.rectangle(processed_frame_for_ui, (x, y), (x + width, y + height), (0, 255, 0), 2)
                                cv2.putText(processed_frame_for_ui, f"{current_face_emotion}: {prob_val:.2f}", (x, y - 10), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        else:
                            current_face_emotion = "N/A (empty crop)"
                
                # Cập nhật dữ liệu chia sẻ
                with self.frame_lock:
                    self.latest_frame_from_pi = processed_frame_for_ui
                
                with self.emotion_lock:
                    self.last_face_emotion = current_face_emotion
                    self.last_face_emotion_probabilities = current_face_probs
                    if current_face_emotion not in ["N/A", "N/A (empty crop)"]:
                        emotion_item = EmotionHistoryItem(
                            timestamp=datetime.now(),
                            face_location=face_location_str,
                            duration=None, 
                            result=current_face_emotion,
                            source="RaspberryPi (Face)",
                            emotion_distribution=current_face_probs
                        )
                        self.emotion_history.append(emotion_item)
                
                # Phản hồi cho Pi (ví dụ)
                send_data(client_socket, {"status": "success", "emotion": current_face_emotion, "probs": current_face_probs})

        except (socket.error, ConnectionResetError, BrokenPipeError) as e:
            print(f"Face client socket error: {e}")
        except Exception as e:
            print(f"Lỗi trong _handle_face_client: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("Face client disconnected.")
            client_socket.close()

    def _face_data_server_loop(self, host, port):
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            server_socket.bind((host, port))
            server_socket.listen(1) # Chỉ mong đợi 1 kết nối từ Pi cho face
            server_socket.settimeout(1.0) # Timeout để kiểm tra stop_event
            print(f"Face Data Server đang lắng nghe trên {host}:{port}")

            while not self.stop_event.is_set():
                try:
                    client_socket, addr = server_socket.accept()
                    # Mỗi client kết nối sẽ được xử lý tuần tự ở đây
                    # Nếu muốn xử lý nhiều client Pi cùng lúc cho face (ít khả năng), cần tạo luồng mới
                    self._handle_face_client(client_socket) # Xử lý client này cho đến khi nó ngắt kết nối
                except socket.timeout:
                    continue # Kiểm tra lại stop_event
                except Exception as e:
                    if not self.stop_event.is_set(): # Chỉ in lỗi nếu không phải do đang dừng
                         print(f"Lỗi trong face server loop (accept): {e}")
                    time.sleep(0.1)


        finally:
            print("Face Data Server đang đóng...")
            server_socket.close()
            print("Face Data Server đã đóng.")

    def _handle_voice_client(self, client_socket):
        print("Raspberry Pi (Voice Client) connected.")
        try:
            while not self.stop_event.is_set():
                message = recv_data(client_socket) # Mong đợi {'data': audio_bytes}
                if message is None:
                    print("Voice client disconnected or sent invalid data.")
                    break
                
                audio_bytes = message.get('data')
                if not isinstance(audio_bytes, bytes):
                    print("Dữ liệu âm thanh không hợp lệ từ voice client.")
                    send_data(client_socket, {"status": "error", "message": "Invalid audio data"})
                    continue
                
                current_voice_emotion = "N/A"
                current_voice_probs = None

                if self.enable_analysis_voice:
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio_file:
                            tmp_audio_file.write(audio_bytes)
                            temp_file_path = tmp_audio_file.name
                        
                        probabilities = self.voice_analyzer.predict_emotion(temp_file_path)
                        os.remove(temp_file_path)

                        if "error" not in probabilities and probabilities:
                            current_voice_emotion = max(probabilities, key=probabilities.get)
                            current_voice_probs = probabilities
                        elif probabilities: # Có thể trả về dict rỗng nếu không có lỗi nhưng không có kết quả
                             print(f"Voice analysis returned: {probabilities}")
                        else: # probabilities is None or empty
                            print("Voice analysis returned no probabilities or an error flag was not set properly.")


                    except Exception as e:
                        print(f"Lỗi khi phân tích âm thanh: {e}")
                        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
                            try: os.remove(temp_file_path)
                            except: pass
                        current_voice_emotion = "Error"

                with self.emotion_lock:
                    self.last_voice_emotion = current_voice_emotion
                    self.last_voice_probabilities = current_voice_probs
                    if current_voice_emotion not in ["N/A", "Error"] and current_voice_probs:
                        emotion_item = EmotionHistoryItem(
                            timestamp=datetime.now(),
                            face_location=None,
                            duration=len(audio_bytes), # Hoặc Pi gửi duration
                            result=current_voice_emotion,
                            source="RaspberryPi (Voice)",
                            emotion_distribution=current_voice_probs
                        )
                        self.emotion_history.append(emotion_item)
                
                send_data(client_socket, {"status": "success", "emotion": current_voice_emotion, "probs": current_voice_probs})
        
        except (socket.error, ConnectionResetError, BrokenPipeError) as e:
            print(f"Voice client socket error: {e}")
        except Exception as e:
            print(f"Lỗi trong _handle_voice_client: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("Voice client disconnected.")
            client_socket.close()

    def _voice_data_server_loop(self, host, port):
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            server_socket.bind((host, port))
            server_socket.listen(1) # Chỉ mong đợi 1 kết nối từ Pi cho voice
            server_socket.settimeout(1.0)
            print(f"Voice Data Server đang lắng nghe trên {host}:{port}")

            while not self.stop_event.is_set():
                try:
                    client_socket, addr = server_socket.accept()
                    self._handle_voice_client(client_socket)
                except socket.timeout:
                    continue
                except Exception as e:
                    if not self.stop_event.is_set():
                        print(f"Lỗi trong voice server loop (accept): {e}")
                    time.sleep(0.1)


        finally:
            print("Voice Data Server đang đóng...")
            server_socket.close()
            print("Voice Data Server đã đóng.")

    def get_latest_data(self):
        # Trả về dữ liệu mới nhất được cập nhật bởi các luồng server
        with self.frame_lock:
            frame = self.latest_frame_from_pi.copy() if self.latest_frame_from_pi is not None else None
        with self.emotion_lock:
            face_emo = self.last_face_emotion
            # face_emo_prob_dominant = max(self.last_face_emotion_probabilities.values()) if self.last_face_emotion_probabilities else 0.0
            # Lấy prob của emotion chính
            if self.last_face_emotion_probabilities and self.last_face_emotion in self.last_face_emotion_probabilities:
                face_emo_prob_dominant = self.last_face_emotion_probabilities[self.last_face_emotion]
            else:
                face_emo_prob_dominant = 0.0

            voice_emo = self.last_voice_emotion
            voice_emo_probs_full = self.last_voice_probabilities # Gửi cả dict probs cho UI
        
        # Đảm bảo face_emo_prob_dominant là float
        if not isinstance(face_emo_prob_dominant, (float, int)):
            try:
                face_emo_prob_dominant = float(face_emo_prob_dominant)
            except (ValueError, TypeError):
                face_emo_prob_dominant = 0.0


        return frame, face_emo, face_emo_prob_dominant, voice_emo, voice_emo_probs_full


    def start(self, face_server_host, face_server_port, voice_server_host, voice_server_port):
        if self.face_server_thread or self.voice_server_thread:
            print("Servers đã chạy rồi.")
            return

        self.stop_event.clear()
        print("Bắt đầu các luồng server xử lý dữ liệu từ Pi...")
        
        self.face_server_thread = threading.Thread(
            target=self._face_data_server_loop, 
            args=(face_server_host, face_server_port), 
            daemon=True
        )
        self.voice_server_thread = threading.Thread(
            target=self._voice_data_server_loop, 
            args=(voice_server_host, voice_server_port), 
            daemon=True
        )
        
        self.face_server_thread.start()
        self.voice_server_thread.start()

    def stop(self):
        if self.stop_event.is_set():
            print("Yêu cầu dừng đã được gửi.")
            return
        
        print("Bắt đầu quá trình dừng EmotionDetector (Server Mode)...")
        self.stop_event.set()

        # Không cần join trực tiếp ở đây nếu là daemon, nhưng cần đảm bảo socket được đóng
        # Các vòng lặp server sẽ tự thoát khi kiểm tra stop_event sau timeout của accept()
        # Chờ một chút để các luồng có thời gian nhận biết stop_event và đóng socket
        time.sleep(1.5) 

        # Kiểm tra xem luồng còn sống không (chủ yếu để gỡ lỗi)
        if self.face_server_thread and self.face_server_thread.is_alive():
            print(f"Cảnh báo: Luồng Face Server ({self.face_server_thread.name}) có thể chưa dừng hẳn.")
        if self.voice_server_thread and self.voice_server_thread.is_alive():
            print(f"Cảnh báo: Luồng Voice Server ({self.voice_server_thread.name}) có thể chưa dừng hẳn.")
            
        print("Các luồng server đã được yêu cầu dừng.")

    def cleanup(self):
        print("EmotionDetector: Thực hiện cleanup cuối cùng (nếu có).")
        # Việc lưu DB đã được chuyển vào finally của main
        pass


# --- Điểm khởi chạy chính ---
if __name__ == "__main__":
    # --- Import UIController ở đây (Nếu vẫn dùng UI) ---
    from ui_controller import EmotionGUI # Bỏ comment nếu dùng GUI
    from ttkbootstrap import Style      # Bỏ comment nếu dùng GUI

    # Cấu hình cho các server socket
    FACE_SERVER_HOST = '0.0.0.0'  # Nghe trên tất cả các interface
    FACE_SERVER_PORT = 65432     # Port cho server hình ảnh
    VOICE_SERVER_HOST = '0.0.0.0' # Nghe trên tất cả các interface
    VOICE_SERVER_PORT = 65433    # Port cho server âm thanh (KHÁC PORT VỚI FACE)    main_detector = None
    ui_controller = None # Bỏ comment nếu dùng GUI
    root = None          # Bỏ comment nếu dùng GUI

    try:
        print("Khởi tạo FaceAnalyzer...")
        face_analyzer_inst = FaceAnalyzer()
        print("Khởi tạo VoiceAnalyzer...")
        voi_analyzer_inst = VoiceAnalyzer(
            # model_path="path/to/your/voice_model.h5", # Cấu hình nếu cần
            # scaler_path="path/to/your/voice_scaler.pkl"
        )
        
        main_detector = EmotionDetector(
            face_analyzer=face_analyzer_inst,
            voice_analyzer=voi_analyzer_inst
        )

        # Bắt đầu các luồng server lắng nghe dữ liệu từ Pi
        main_detector.start(FACE_SERVER_HOST, FACE_SERVER_PORT, VOICE_SERVER_HOST, VOICE_SERVER_PORT)
        print(f"Face server Chạy trên {FACE_SERVER_HOST}:{FACE_SERVER_PORT}")
        print(f"Voice server Chạy trên {VOICE_SERVER_HOST}:{VOICE_SERVER_PORT}")
        print("Các luồng server đã bắt đầu. Chờ dữ liệu từ Raspberry Pi...")        print("Nhấn Ctrl+C để thoát.")

        # --- Phần GUI (Nếu muốn giữ lại để hiển thị) ---
        # 3. Khởi tạo giao diện EmotionGUI
        style = Style("superhero") 
        root = style.master        gui = EmotionGUI(root) # Đảm bảo EmotionGUI được định nghĩa hoặc import
        gui.detector = main_detector # Gán detector cho GUI để có thể sử dụng các chức năng
        
        # 5. Vòng lặp hiển thị và xử lý input (trong luồng chính)
        def update_gui():
            if main_detector: # Kiểm tra nếu main_detector đã được khởi tạo
                frame, face_emo, face_emo_pro, voice_emo, voice_emo_pros = main_detector.get_latest_data()
                gui.update_video_frame(
                    frame, # Đây sẽ là frame nhận từ Pi, đã được vẽ bounding box
                    face_emotion=face_emo,
                    emotion_probability=face_emo_pro, # Đây là prob của emotion chính
                    voice_emotion=voice_emo,
                    voice_probabilities=voice_emo_pros # Đây là dict đầy đủ các voice probs
                )
            if root: # Kiểm tra nếu root (cửa sổ tkinter) còn tồn tại
                root.after(50, update_gui) # Tăng thời gian chờ lên một chút
            else:
                print("Cửa sổ GUI đã đóng, dừng update_gui.")
        if root: # Chỉ chạy nếu có GUI
            update_gui()
            root.mainloop()
        else: # Nếu không có GUI, chạy vòng lặp chờ Ctrl+C
            while True:
                time.sleep(1) # Giữ luồng chính sống để bắt Ctrl+C và cho các luồng daemon chạy

    except (ValueError, TypeError, RuntimeError, IOError) as e:
        print(f"Lỗi nghiêm trọng khi khởi tạo hoặc chạy: {e}")
        import traceback
        traceback.print_exc()
    except KeyboardInterrupt:
        print("\nPhát hiện Ctrl+C! Đang dừng chương trình...")
    except Exception as e:
        print(f"\nĐã xảy ra lỗi không mong muốn: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Bắt đầu quá trình dọn dẹp cuối cùng...")
        if main_detector:
            print("Yêu cầu EmotionDetector dừng các luồng server...")
            main_detector.stop() 
            
            # Lưu lịch sử cảm xúc
            save_all_emotions_to_db("./database/emotion_log.db", main_detector.emotion_history)
            print(f"✅ Đã lưu {len(main_detector.emotion_history)} mục lịch sử cảm xúc vào cơ sở dữ liệu.")
              main_detector.cleanup()

        # Đóng cửa sổ giao diện (Nếu dùng GUI)
        if root:
            print("Đóng cửa sổ giao diện...")
            root.destroy() # Đóng cửa sổ tkinter

        print("Chương trình đã kết thúc hoàn toàn.")