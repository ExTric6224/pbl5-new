from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class EmotionHistoryItem:
    timestamp: datetime
    face_location: str|None       # ví dụ "500x399"
    duration: int|None           # thời gian cảm xúc xuất hiện (ms)
    result: str              # ví dụ "Happy"
    source: str = "Webcam"
    emotion_distribution: dict = field(default_factory=dict)

