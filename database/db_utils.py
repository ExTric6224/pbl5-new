import sqlite3
import json
from datetime import datetime
from database.emotion_history_item import EmotionHistoryItem

def save_emotion_to_db(db_path, item):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO emotion_history (timestamp, source, result, face_location, duration, emotion_distribution)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        item.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        item.source,
        item.result,
        item.face_location if item.face_location else None,
        item.duration if item.duration else None,
        json.dumps(item.emotion_distribution)  # serialize dict
    ))

    conn.commit()
    conn.close()
    
def load_emotion_history_from_db(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT timestamp, source, result, face_location, duration, emotion_distribution FROM emotion_history")
    rows = cursor.fetchall()

    history_items = []
    for row in rows:
        timestamp = datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S")
        source = row[1]
        result = row[2]
        face_location = row[3]
        duration = row[4]
        emotion_distribution = json.loads(row[5]) if row[5] else {}

        item = EmotionHistoryItem(
            timestamp=timestamp,
            face_location=face_location,
            duration=duration,
            result=result,
            source=source,
            emotion_distribution=emotion_distribution
        )
        history_items.append(item)

    conn.close()
    return history_items

def save_all_emotions_to_db(db_path, history_items):
    import sqlite3, json
    from datetime import datetime

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    for item in history_items:
        cursor.execute("""
            INSERT INTO emotion_history (timestamp, source, result, face_location, duration, emotion_distribution)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            item.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            item.source,
            item.result,
            item.face_location,
            item.duration,
            json.dumps(item.emotion_distribution)
        ))

    conn.commit()
    conn.close()

# import sqlite3

# conn = sqlite3.connect("database/emotion_log.db")
# cursor = conn.cursor()

# cursor.execute("""
# CREATE TABLE IF NOT EXISTS emotion_history (
#     id INTEGER PRIMARY KEY AUTOINCREMENT,
#     timestamp TEXT,
#     source TEXT,
#     result TEXT,
#     face_location TEXT,
#     duration INTEGER,
#     emotion_distribution TEXT
# )
# """)

# conn.commit()
# conn.close()

