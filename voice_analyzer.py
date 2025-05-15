import numpy as np
import librosa
import sounddevice as sd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LeakyReLU, Input # Cho custom_objects và định nghĩa model
import joblib
import json
import os
import tempfile
import pickle

class VoiceAnalyzer:
    def __init__(self, params=None):
        """
        Khởi tạo đối tượng VoiceAnalyzer với các thông số và mô hình.
        
        Args:
            params (dict, optional): Từ điển chứa các thông số cấu hình. 
                                    Nếu không cung cấp, sử dụng thông số mặc định.
        """
        # --- 1. KHỞI TẠO THÔNG SỐ ---
        # Các thông số chính (audio và mô hình)
        self.params = params or {
            "SR": 16000,              # Tần số lấy mẫu
            "N_FFT": 400,             # Độ dài FFT
            "DEFAULT_OFFSET": 0.0,    # Offset mặc định khi load file
            "HOP_LENGTH": 160,        # Độ dịch khung
            "N_MELS": 40,             # Số băng tần mel
            "MAX_FRAMES": 334,        # Số khung tối đa
            "MAX_PAD_LEN_SEQ": 334,   # Độ dài tối đa sau padding
            "num_expected_coeffs": 60 # Số hệ số đặc trưng
        }
        
        # --- 2. TẠO CÁC BIẾN THÀNH VIÊN TỪ PARAMS ---
        # Audio params
        self.sr = self.params["SR"]
        self.n_fft = self.params["N_FFT"] 
        self.hop_length = self.params["HOP_LENGTH"]
        self.n_mels = self.params["N_MELS"]
        self.max_frames = self.params["MAX_FRAMES"]
        self.DEFAULT_OFFSET = self.params["DEFAULT_OFFSET"]
        self.duration_sec = self.max_frames * self.hop_length / self.sr
        
        # Các tham số trích xuất đặc trưng
        self.INITIAL_LOAD_DURATION = None     # Không giới hạn thời gian
        self.TARGET_SR = self.sr              # Tần số mẫu đích
        self.FRAME_LENGTH_MS_SEQ = 25         # Độ dài khung (ms)
        self.HOP_LENGTH_MS_SEQ = 10           # Độ dịch khung (ms) 
        self.N_MFCC_PARAM_SEQ = 20            # Số hệ số MFCC
        self.USE_DELTA_MFCC = True            # Sử dụng đạo hàm bậc 1
        self.USE_DELTA2_MFCC = True           # Sử dụng đạo hàm bậc 2
        self.MAX_PAD_LEN_SEQ = self.params["MAX_PAD_LEN_SEQ"]  # Sử dụng thông số có sẵn

        # --- 3. ĐƯỜNG DẪN FILE ---
        self.MODEL_ASSETS_DIR = "model_assets"
        self.MODEL_DIR ="models"
        self.MODEL_CHECKPOINT_PATH = os.path.join(self.MODEL_DIR, "best_crnn_model.keras")
        self.LABEL_ENCODER_PATH = os.path.join(self.MODEL_ASSETS_DIR, "label_encoder_final.pkl")
        self.SCALERS_LIST_PATH = os.path.join(self.MODEL_ASSETS_DIR, "scalers_list.pkl")

        # --- 4. BIẾN CHO MÔ HÌNH ---
        self.model_loaded = None
        self.le_loaded = None
        self.scalers_list_loaded = None
        self.class_names_loaded = None
        self.num_classes_loaded = 0
        self.input_shape_loaded = (
            self.params["MAX_PAD_LEN_SEQ"],
            self.params["num_expected_coeffs"]
        )

        # --- 5. TẢI MÔ HÌNH VÀ DỮ LIỆU ---
        self._load_model_assets()
    
    def _load_model_assets(self):
        """
        Tải các tài nguyên mô hình: LabelEncoder, Scalers và mô hình dự đoán.
        Xử lý các lỗi nếu có và in thông báo tương ứng.
        """
        try:
            # Kiểm tra thư mục assets
            if not os.path.exists(self.MODEL_ASSETS_DIR):
                raise FileNotFoundError(f"Thư mục assets '{self.MODEL_ASSETS_DIR}' không tồn tại.")
            
            # Tải tuần tự các thành phần
            self._load_label_encoder()
            self._load_scalers()
            self._load_model()
            
        except Exception as e:
            print(f"⛔ LỖI KHỞI TẠO: {e}")
    
    def _load_label_encoder(self):
        """Tải và khởi tạo LabelEncoder."""
        if not os.path.exists(self.LABEL_ENCODER_PATH):
            raise FileNotFoundError(f"Không tìm thấy file LabelEncoder: {self.LABEL_ENCODER_PATH}")
            
        with open(self.LABEL_ENCODER_PATH, 'rb') as f:
            self.le_loaded = pickle.load(f)
        
        self.class_names_loaded = list(self.le_loaded.classes_)
        self.num_classes_loaded = len(self.class_names_loaded)
        print(f"✅ Đã tải LabelEncoder: {self.class_names_loaded}")
    
    def _load_scalers(self):
        """Tải và kiểm tra scalers."""
        if not os.path.exists(self.SCALERS_LIST_PATH):
            raise FileNotFoundError(f"Không tìm thấy file scaler: {self.SCALERS_LIST_PATH}")
            
        with open(self.SCALERS_LIST_PATH, 'rb') as f:
            self.scalers_list_loaded = pickle.load(f)
        
        print(f"✅ Đã tải {len(self.scalers_list_loaded)} scalers.")
        
        # Kiểm tra tính nhất quán
        if len(self.scalers_list_loaded) != self.input_shape_loaded[1]:
            raise ValueError(f"Số scaler không khớp: {len(self.scalers_list_loaded)} vs {self.input_shape_loaded[1]}")
    
    def _load_model(self):
        """Tải mô hình Keras và cấu hình."""
        if not os.path.exists(self.MODEL_CHECKPOINT_PATH):
            raise FileNotFoundError(f"Không tìm thấy mô hình: {self.MODEL_CHECKPOINT_PATH}")
            
        custom_objects = {'LeakyReLU': LeakyReLU}
        self.model_loaded = load_model(
            self.MODEL_CHECKPOINT_PATH,
            custom_objects=custom_objects,
            compile=False
        )
        print("✅ Đã tải mô hình thành công.")
        self.model_loaded.summary()


    def record_audio(self,duration=3.34, sample_rate=16000, filename="tmp.wav"):
        """
        Ghi âm giọng nói từ micro, lưu ra file WAV, và trả về:
            - Đường dẫn tới file WAV đã ghi.
            - Dữ liệu audio dạng numpy array (mono, float32).
        
        Parameters:
            duration (int or float): Thời gian ghi âm (giây).
            sample_rate (int): Tần số mẫu (Hz).
            filename (str): Tên file WAV (nếu muốn tự đặt). Nếu không, tạo file tạm.

        Returns:
            (wav_path: str, audio: np.ndarray)
        """
        print(f"🎙️ Đang ghi âm trong {duration:.2f} giây...")
        recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
        sd.wait()
        print("✅ Ghi âm xong.")

        audio = recording.flatten()

        # Xác định nơi lưu file WAV
        if filename is None:
            tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            filename = tmp_file.name

        # Lưu file WAV
        from scipy.io.wavfile import write
        write(filename, sample_rate, recording)
        print(f"📁 File WAV đã lưu tại: {filename}")

        return filename, audio
    #HÀM TRÍCH XUẤT ĐẶC TRƯNG
    def extract_features_for_gradio(self, data, sr,
                                n_fft_val, hop_length_val,
                                n_mfcc_val=None,
                                use_delta=None, use_delta2=None):
        # Đặt giá trị mặc định nếu chưa truyền vào
        if n_mfcc_val is None:
            n_mfcc_val = self.N_MFCC_PARAM_SEQ
        if use_delta is None:
            use_delta = self.USE_DELTA_MFCC
        if use_delta2 is None:
            use_delta2 = self.USE_DELTA2_MFCC
        # (Copy nội dung hàm extract_features_sequential từ Bước 4.1 vào đây)
        features_list = []
        mfccs = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=n_mfcc_val,
                                    n_fft=n_fft_val, hop_length=hop_length_val)
        features_list.append(mfccs)
        if use_delta:
            delta_mfccs = librosa.feature.delta(mfccs)
            features_list.append(delta_mfccs)
        if use_delta2:
            delta2_mfccs = librosa.feature.delta(mfccs, order=2)
            features_list.append(delta2_mfccs)
        combined_features = np.vstack(features_list)
        return combined_features.T

    # --- 6. HÀM TIỀN XỬ LÝ CHO MỘT FILE ÂM THANH ĐẦU VÀO ---
    def preprocess_single_audio_file(self,audio_filepath):
        if audio_filepath is None:
            return None, "Lỗi: Không có file âm thanh nào được cung cấp."

        try:
            # Tải file âm thanh (audio_filepath sẽ là đường dẫn tạm thời của file upload)
            data_orig, sr_orig = librosa.load(audio_filepath, sr=None, offset=self.DEFAULT_OFFSET, duration=self.INITIAL_LOAD_DURATION)
            
            if len(data_orig) == 0: return None, "Lỗi: File âm thanh rỗng hoặc không đọc được."

            # Resample
            if sr_orig != self.TARGET_SR:
                data_resampled = librosa.resample(y=data_orig, orig_sr=sr_orig, target_sr=self.TARGET_SR)
            else:
                data_resampled = data_orig
            current_sr = self.TARGET_SR
            if len(data_resampled) == 0: return None, "Lỗi: Dữ liệu rỗng sau khi resample."

            # Tính toán frame và hop samples
            frame_samples = int(current_sr * self.FRAME_LENGTH_MS_SEQ / 1000)
            hop_samples = int(current_sr * self.HOP_LENGTH_MS_SEQ / 1000)
            
            if len(data_resampled) < frame_samples:
                return None, f"Lỗi: File âm thanh quá ngắn ({len(data_resampled)} mẫu) so với độ dài khung ({frame_samples} mẫu)."

            # Trích xuất đặc trưng tuần tự
            feature_matrix_raw = self.extract_features_for_gradio(
                data_resampled, current_sr,
                n_fft_val=frame_samples, hop_length_val=hop_samples
            )
            if feature_matrix_raw is None or feature_matrix_raw.size == 0 :
                return None, "Lỗi: Không trích xuất được đặc trưng."


            # Padding / Truncating
            if feature_matrix_raw.shape[0] > self.MAX_PAD_LEN_SEQ:
                feature_matrix_padded = feature_matrix_raw[:self.MAX_PAD_LEN_SEQ, :]
            else:
                pad_width = self.MAX_PAD_LEN_SEQ - feature_matrix_raw.shape[0]
                feature_matrix_padded = np.pad(feature_matrix_raw, pad_width=((0, pad_width), (0, 0)), mode='constant', constant_values=(0,))
            
            if feature_matrix_padded.shape[0] != self.MAX_PAD_LEN_SEQ:
                return None, "Lỗi: Kích thước không đúng sau padding."

            # Scaling (sử dụng scalers_list_loaded)
            if self.scalers_list_loaded is None or len(self.scalers_list_loaded) != feature_matrix_padded.shape[1]:
                return None, "Lỗi: Scalers không được tải hoặc không khớp số lượng đặc trưng."
            
            feature_matrix_scaled = np.copy(feature_matrix_padded).astype(np.float32) # Đảm bảo float32
            for i in range(feature_matrix_padded.shape[1]): # Lặp qua từng cột feature
                feature_column_flat = feature_matrix_padded[:, i].reshape(-1, 1)
                feature_matrix_scaled[:, i] = self.scalers_list_loaded[i].transform(feature_column_flat).flatten()
                
            # Reshape cho đầu vào mô hình: (1, timesteps, features)
            final_features = np.expand_dims(feature_matrix_scaled, axis=0)
            return final_features, None # Không có lỗi

        except Exception as e:
            return None, f"Lỗi trong quá trình tiền xử lý file: {str(e)}"

    def predict_emotion(self,audio_file_path):
        if self.model_loaded is None or self.le_loaded is None:
            return {"Lỗi": "Mô hình hoặc LabelEncoder chưa được tải."}

        # audio_file_path là một đối tượng file tạm thời từ Gradio File/Audio input
        # Nó có thuộc tính .name chứa đường dẫn thực sự đến file tạm đó
        actual_path = audio_file_path if isinstance(audio_file_path, str) else audio_file_path.name
        
        print(f"Đang xử lý file: {actual_path}")
        
        features_scaled, error_message = self.preprocess_single_audio_file(actual_path)

        if error_message:
            print(error_message)
            # Trả về lỗi dưới dạng dictionary để Gradio hiển thị label
            return {error_message.split(':')[0]: 1.0} if ':' in error_message else {"Lỗi xử lý": 1.0}


        if features_scaled is not None and features_scaled.ndim == 3 and features_scaled.shape[1:] == self.input_shape_loaded:
            print(f"Đặc trưng đã xử lý có shape: {features_scaled.shape}")
            probabilities = self.model_loaded.predict(features_scaled)[0] # Lấy kết quả cho mẫu đầu tiên (và duy nhất)
            
            # Chuyển xác suất thành dictionary {label: probability}
            results = {label: float(prob) for label, prob in zip(self.le_loaded.classes_, probabilities)}
            
            # In ra console để debug
            print("Xác suất dự đoán:")
            for label, prob in results.items():
                print(f"  {label}: {prob:.4f}")
            
            return results
        else:
            print("Lỗi: Đặc trưng cuối cùng không hợp lệ hoặc shape không đúng.")
            return {"Lỗi đặc trưng": 1.0}

