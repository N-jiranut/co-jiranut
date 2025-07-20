import cv2
import mediapipe as mp
import numpy as np
import os

mp_hands = mp.solutions.hands
# ใช้ static_image_mode=False สำหรับการประมวลผลวิดีโอ
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

def extract_landmarks_from_video(video_path, max_frames=30):
    cap = cv2.VideoCapture(video_path)
    frames_landmarks = []
    frame_count = 0

    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        landmarks_frame = []
        if results.multi_hand_landmarks:
            # เลือกเฉพาะมือแรกที่ตรวจพบ (ถ้ามีหลายมือ)
            for hand_landmarks in results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    # เก็บเฉพาะ x, y, z coordinates
                    landmarks_frame.extend([landmark.x, landmark.y, landmark.z])
                break # หยุดหลังจากประมวลผลมือแรก
        else:
            # ถ้าไม่พบมือในเฟรมนี้ ให้เติมด้วย 0 หรือค่า NaN เพื่อรักษามิติ
            landmarks_frame = [0.0] * (21 * 3) # 21 จุด x 3 พิกัด (x,y,z)

        frames_landmarks.append(landmarks_frame)
        frame_count += 1

    cap.release()

    # ทำให้จำนวนเฟรมเท่ากัน (padding/trimming)
    if len(frames_landmarks) < max_frames:
        # Zero-padding
        while len(frames_landmarks) < max_frames:
            frames_landmarks.append([0.0] * (21 * 3))
    elif len(frames_landmarks) > max_frames:
        # Trimming (เอาแค่ max_frames แรก)
        frames_landmarks = frames_landmarks[:max_frames]

    return np.array(frames_landmarks) # จะมีมิติ (max_frames, 21*3)

# --- การรวบรวมข้อมูลสำหรับ Training ---
DATA_PATH = 'dataset'
ACTIONS = os.listdir(DATA_PATH) # ['Hello', 'Thank_You', 'Goodbye']
NO_SEQUENCES = 30 # จำนวนวิดีโอต่อท่าทาง
SEQUENCE_LENGTH = 30 # จำนวนเฟรมต่อวิดีโอ

features = [] # เก็บข้อมูล landmark sequences
labels = []   # เก็บ label (one-hot encoded)

for action_idx, action in enumerate(ACTIONS):
    action_path = os.path.join(DATA_PATH, action)
    video_files = [f for f in os.listdir(action_path) if f.endswith('.mp4')]

    for video_file in video_files[:NO_SEQUENCES]: # จำกัดจำนวนวิดีโอสำหรับตัวอย่าง
        video_path = os.path.join(action_path, video_file)
        sequence = extract_landmarks_from_video(video_path, max_frames=SEQUENCE_LENGTH)
        if sequence is not None and sequence.shape[0] == SEQUENCE_LENGTH:
            features.append(sequence)
            labels.append(action_idx) # ใช้ index เป็น label ก่อนแล้วค่อยแปลงเป็น one-hot

X = np.array(features) # (จำนวนตัวอย่างทั้งหมด, SEQUENCE_LENGTH, 21*3)
# แปลง labels เป็น One-Hot Encoding
from tensorflow.keras.utils import to_categorical
y = to_categorical(np.array(labels), num_classes=len(ACTIONS))

print(f"Shape of X (features): {X.shape}") # เช่น (90, 30, 63) -> 90 วิดีโอ, วิดีโอละ 30 เฟรม, เฟรมละ 63 ค่า
print(f"Shape of y (labels): {y.shape}")   # เช่น (90, 3) -> 90 วิดีโอ, 3 คลาส