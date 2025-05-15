#xóa dữ liệu trong bảng emotion_history
import sqlite3


def clear_emotion_history(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("DELETE FROM emotion_history")

    conn.commit()
    conn.close()

if __name__ == "__main__":
    db_path = "database/emotion_log.db"  # Path to your database file
    clear_emotion_history(db_path)
    print("Emotion history cleared.")