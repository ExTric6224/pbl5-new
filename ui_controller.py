import cv2
from PIL import Image, ImageTk, ImageDraw, ImageFont
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from tkinter import Toplevel

class EmotionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Emotion Sense")

        # --- FRAME PHÂN BỐ ---
        self.frame_video = ttk.LabelFrame(root, text="🎥 Webcam", padding=10, bootstyle=PRIMARY)
        self.frame_video.grid(row=0, column=0, padx=10, pady=10)

        self.frame_audio = ttk.LabelFrame(root, text="🔊 Audio Analyzer", padding=10, bootstyle=INFO)
        self.frame_audio.grid(row=0, column=1, padx=10, pady=10)

        self.frame_controls = ttk.Frame(root, padding=10)
        self.frame_controls.grid(row=1, column=0, columnspan=2)

        # --- FRAME VIDEO ---
        self.video_label = ttk.Label(self.frame_video, text="Waiting for webcam...")
        self.video_label.pack()        # --- FRAME AUDIO ---
        self.audio_bars = {}
        emotions = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
        color_map = {
            "angry": DANGER,
            "disgust": WARNING,
            "fear": INFO,
            "happy": SUCCESS,
            "neutral": SECONDARY,
            "sad": PRIMARY,
            "surprise": SUCCESS
        }
        
        # Mapping between voice emotion keys and UI emotion keys
        self.voice_to_ui_emotion_map = {
            "angry": "angry",
            "disgust": "disgust",
            "fear": "fear", 
            "happy": "happy",
            "neutral": "neutral",
            "sad": "sad",
            "surprise": "surprise"
        }

        for emo in emotions:
            label = ttk.Label(self.frame_audio, text=emo.capitalize(), width=15, anchor='w')
            label.pack(pady=2)
            progress = ttk.Progressbar(self.frame_audio, length=200, maximum=100, bootstyle=color_map.get(emo, "secondary"))
            progress.pack(pady=2)
            self.audio_bars[emo] = progress

               # --- BUTTONS --- (SỬA)
        toggle_frame = ttk.Frame(self.frame_controls)  # Nhóm 2 toggle vào 1 dòng
        toggle_frame.pack(side='left', padx=10)

        self.image_toggle = ttk.Checkbutton(
            toggle_frame, 
            text="📸 Image Analysis", 
            bootstyle="info-outline-toolbutton",  # Nút dạng toggle hiện đại
            command=self.toggle_image_analysis
        )
        self.image_toggle.pack(side='left', padx=5)
        self.image_toggle.state(['selected'])  # Mặc định bật

        self.audio_toggle = ttk.Checkbutton(
            toggle_frame, 
            text="🎙️ Audio Analysis", 
            bootstyle="success-outline-toolbutton", 
            command=self.toggle_audio_analysis
        )
        self.audio_toggle.pack(side='left', padx=5)
        self.audio_toggle.state(['selected'])  # Mặc định bật

        # Nút xem lịch sử giữ nguyên
        ttk.Button(self.frame_controls, text="📜 Watch History", bootstyle=SECONDARY, command=self.watch_history).pack(side='left', padx=10)


        self.display_placeholder_image()

    def display_placeholder_image(self):
        try:
            img_path = "c:/Documents/PBL5/data/placeholder.png"
            img = Image.open(img_path).resize((320, 320))
            photo = ImageTk.PhotoImage(img)
            self.video_label.config(image=photo)
            self.video_label.image = photo
        except FileNotFoundError:
            self.video_label.config(text="Placeholder image not found.")

    def update_video_frame(self, frame, face_emotion="N/A", emotion_probability=0.0,voice_emotion="N/A", voice_probabilities=None):
        if frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)

            # Tạo nền mờ + text
            draw = ImageDraw.Draw(img)
            try:
                font = ImageFont.truetype("arial.ttf", 24)
            except:
                font = ImageFont.load_default()
            draw.rectangle((5, 5, 300, 45), fill=(0, 0, 0, 127))
            draw.text((10, 10), f"Emotion: {face_emotion} ({emotion_probability * 100:.2f}%)", font=font, fill=(255, 255, 0))

            img = img.resize((320, 320))
            photo = ImageTk.PhotoImage(img)
            self.video_label.config(image=photo)
            self.video_label.image = photo
        else:
            self.video_label.config(text="Không có khung hình để hiển thị.")                # --- CẬP NHẬT GIỌNG NÓI ---
        if voice_probabilities:
            # Update for the new format: angry, disgust, fear, happy, neutral, sad, surprise
            for emo, prob in voice_probabilities.items():
                if emo in self.audio_bars:
                    self.audio_bars[emo]['value'] = prob * 100  # vì maximum=100
        else:
            # Reset thanh tiến trình nếu voice analysis bị tắt
            for bar in self.audio_bars.values():
                bar['value'] = 0
                    
    def watch_history(self):
        if not hasattr(self, 'detector') or not self.detector:
            print("Emotion detector is not initialized.")
            return
        self.detector.can_send_to_UI = False  # k cho phép gửi dữ liệu đến UI
        # Toplevel mới
        win = Toplevel(self.root)
        win.title("Emotion History")
        win.geometry("800x350")
        win.protocol("WM_DELETE_WINDOW", lambda: self._on_close_history(win))

        # Layout chia làm 2 cột: left (Treeview), right (detail)
        frame_main = ttk.Frame(win)
        frame_main.pack(fill='both', expand=True, padx=10, pady=10)

        frame_table = ttk.Frame(frame_main)
        frame_table.pack(side='left', fill='both', expand=True)

        frame_detail = ttk.Frame(frame_main, padding=10, relief='solid')
        frame_detail.pack(side='right', fill='y')

        # Lưu reference để cập nhật chi tiết sau
        self.detail_widgets = {}
        self.tree = ttk.Treeview(frame_table, columns=("time", "result", "source", "detail"), show="headings", height=10)
        # Set heading + column...
        
        self.tree.heading("time", text="Time")
        self.tree.heading("result", text="Result")
        self.tree.heading("source", text="Source")

        self.tree.column("time", width=150)
        self.tree.column("result", width=100)
        self.tree.column("source", width=100)

        self.tree.pack(fill='both', expand=True)
        ttk.Button(frame_table, text="🧹 Delete All", bootstyle="danger-outline", command=self.delete_all_history).pack(pady=(10, 0))

        # Thêm dữ liệu từ history
        for i, item in enumerate(self.detector.emotion_history):
            self.tree.insert("", "end", iid=i, values=(
                item.timestamp.strftime("%H:%M:%S %d/%m/%Y"),
                item.result,
                item.source
            ))

        ttk.Label(frame_detail, text="Detail", font=('Helvetica', 12, 'bold')).pack()

        self.detail_widgets['emotion'] = ttk.Label(frame_detail, text="Emotion: ")
        self.detail_widgets['emotion'].pack(anchor='w')

        self.detail_widgets['location'] = ttk.Label(frame_detail, text="Face Location: ")
        self.detail_widgets['location'].pack(anchor='w')

        self.detail_widgets['time'] = ttk.Label(frame_detail, text="Time: ")
        self.detail_widgets['time'].pack(anchor='w')

        self.detail_widgets['source'] = ttk.Label(frame_detail, text="Source: ")
        self.detail_widgets['source'].pack(anchor='w')

        ttk.Label(frame_detail, text="Emotion Distribution:", font=('Helvetica', 10, 'bold')).pack(pady=(10, 0))

        # Frame chứa các dòng cảm xúc
        self.detail_widgets['emo_dist_frame'] = ttk.Frame(frame_detail)
        self.detail_widgets['emo_dist_frame'].pack(fill='both', anchor='w')
        # Nút xóa lịch sử
        ttk.Button(frame_detail, text="🗑️ Delete Selected", bootstyle=DANGER, command=self.delete_selected_history).pack(pady=10)

        self.tree.bind("<<TreeviewSelect>>", self.on_select)

    def toggle_image_analysis(self):
        if self.image_toggle.instate(['selected']):
            print("Image Analysis enabled")
            self.detector.enable_analysis_face = True
        else:
            print("Image Analysis disabled")
            self.detector.enable_analysis_face = False


    def toggle_audio_analysis(self):
        if self.audio_toggle.instate(['selected']):
            print("Audio Analysis enabled")
            self.detector.enable_analysis_voice = True
        else:
            print("Audio Analysis disabled")
            self.detector.enable_analysis_voice = False

    def on_select(self, event):
        selected = self.tree.selection()
        if not selected:
            return

        item_id = int(selected[0])
        history_item = self.detector.emotion_history[item_id]

        self.detail_widgets['emotion'].config(text=f"Emotion: {history_item.result}")

        # Phân biệt hiển thị tùy theo nguồn
        if history_item.source == "Webcam":
            self.detail_widgets['location'].config(text=f"Face Location: {history_item.face_location}")
        elif history_item.source == "Microphone":
            dur_str = f"{history_item.duration} ms" if history_item.duration is not None else "Unknown"
            self.detail_widgets['location'].config(text=f"Duration: {dur_str}")
        else:
            self.detail_widgets['location'].config(text="Location: N/A")

        self.detail_widgets['time'].config(text=f"Time: {history_item.timestamp.strftime('%H:%M:%S %d/%m/%Y')}")
        self.detail_widgets['source'].config(text=f"Source: {history_item.source}")

        # Reset emotion distribution
        for widget in self.detail_widgets['emo_dist_frame'].winfo_children():
            widget.destroy()

        for emo, val in history_item.emotion_distribution.items():
            label = ttk.Label(self.detail_widgets['emo_dist_frame'], text=f"{emo}: {val*100:.1f}%")
            label.pack(anchor='w')
    def delete_selected_history(self):
        selected = self.tree.selection()
        if not selected:
            print("Không có mục nào được chọn để xóa.")
            return

        item_id = int(selected[0])
        del self.detector.emotion_history[item_id]  # Xóa trong danh sách

        self.tree.delete(selected[0])  # Xóa khỏi bảng Treeview

        # Cập nhật lại Treeview (vì index đã thay đổi)
        self.tree.delete(*self.tree.get_children())
        for i, item in enumerate(self.detector.emotion_history):
            self.tree.insert("", "end", iid=i, values=(
                item.timestamp.strftime("%H:%M:%S %d/%m/%Y"),
                item.result,
                item.source
            ))

        print(f"🗑️ Đã xóa bản ghi #{item_id}")

    def _on_close_history(self, window):
        self.detector.can_send_to_UI = True
        print("✅ Đã bật lại gửi dữ liệu cho UI sau khi đóng lịch sử.")
        window.destroy()
    def delete_all_history(self):
        if not self.detector.emotion_history:
            print("Lịch sử cảm xúc trống, không có gì để xóa.")
            return

        self.detector.emotion_history.clear()
        self.tree.delete(*self.tree.get_children())
        print("🧹 Đã xóa toàn bộ lịch sử cảm xúc.")

