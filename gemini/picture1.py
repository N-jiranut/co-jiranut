import cv2
import mediapipe as mp
import numpy as np
import os
import time

# --- 1. การตั้งค่า ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils # สำหรับวาดจุดบนมือ

DATA_PATH = os.path.join('data') # โฟลเดอร์หลักสำหรับเก็บข้อมูล
ACTIONS = np.array(['hello', 'thank_you', 'goodbye']) # ท่าทางที่คุณต้องการเก็บ
NO_SEQUENCES = 30 # จำนวนชุดข้อมูล (วิดีโอ/ท่าทาง) ที่จะเก็บต่อหนึ่งท่าทาง
SEQUENCE_LENGTH = 30 # จำนวนเฟรมที่จะเก็บต่อหนึ่งชุดข้อมูล (ความยาวของวิดีโอ/ท่าทาง)

# --- สร้างโฟลเดอร์เก็บข้อมูล ---
for action in ACTIONS:
    for sequence in range(NO_SEQUENCES):
        try:
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass # โฟลเดอร์อาจมีอยู่แล้ว

# --- 2. การเปิดกล้องและเริ่มเก็บข้อมูล ---
cap = cv2.VideoCapture(0) # 0 คือ ID ของกล้องเว็บแคมหลักของคุณ

if not cap.isOpened():
    print("Error: Could not open video stream. Check camera connection.")
    exit()

print("Press 's' to start recording a sequence.")
print("Press 'q' to quit.")

with hands: # ใช้ with statement เพื่อให้แน่ใจว่า MediaPipe resources ถูกปล่อย
    for action_idx, action in enumerate(ACTIONS):
        print(f"\n--- Collecting data for '{action}' action ---")
        # วนลูปเพื่อเก็บ NO_SEQUENCES สำหรับแต่ละ action
        for sequence in range(NO_SEQUENCES):
            sequence_frames = [] # เก็บ landmarks สำหรับ 1 sequence
            frames_collected_count = 0

            # --- รอการเริ่มบันทึก ---
            print(f"Starting collection for sequence {sequence+1}/{NO_SEQUENCES} of '{action}'. Press 's' to start...")
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # พลิกภาพเพื่อแสดงผลเหมือนกระจกเงา
                frame = cv2.flip(frame, 1)

                # ประมวลผลภาพด้วย MediaPipe (แค่แสดงผล ไม่ต้องบันทึกตอนนี้)
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_rgb.flags.writeable = False # เพื่อประสิทธิภาพ
                results = hands.process(img_rgb)
                img_rgb.flags.writeable = True

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                cv2.putText(frame, f'Ready to collect for {action} seq {sequence+1}', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                
                cv2.imshow('OpenCV Feed - Data Collection', frame)

                key = cv2.waitKey(10) & 0xFF
                if key == ord('s'):
                    print("Recording started!")
                    break
                elif key == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    exit()
            
            # --- เริ่มบันทึก sequence ---
            start_time = time.time()
            while frames_collected_count < SEQUENCE_LENGTH:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1) # Flip for mirror effect

                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_rgb.flags.writeable = False
                results = hands.process(img_rgb)
                img_rgb.flags.writeable = True

                current_frame_landmarks = []
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                        for landmark in hand_landmarks.landmark:
                            # เก็บ x, y, z (MediaPipe ให้ค่า normalized 0-1)
                            current_frame_landmarks.extend([landmark.x, landmark.y, landmark.z])
                        break # ใช้แค่มือแรกที่ตรวจพบ

                else:
                    # ถ้าไม่พบมือ ให้เติม 0 เพื่อรักษามิติของข้อมูล
                    current_frame_landmarks = [0.0] * (21 * 3) # 21 points * 3 coords (x,y,z)

                sequence_frames.append(current_frame_landmarks)
                frames_collected_count += 1

                # แสดงสถานะการบันทึก
                cv2.putText(frame, f'Collecting frames for {action} - {sequence+1}/{NO_SEQUENCES}', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, f'Frame: {frames_collected_count}/{SEQUENCE_LENGTH}', (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)

                cv2.imshow('OpenCV Feed - Data Collection', frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break # Allow quitting during recording

            # --- บันทึกข้อมูลที่เก็บได้ลงไฟล์ .npy ---
            if frames_collected_count == SEQUENCE_LENGTH:
                sequence_array = np.array(sequence_frames)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), f'{action}_{sequence}.npy')
                np.save(npy_path, sequence_array)
                print(f"Saved {npy_path} with shape {sequence_array.shape}")
            else:
                print(f"Skipped saving sequence {sequence+1} for {action} due to insufficient frames.")

            if cv2.waitKey(1) & 0xFF == ord('q'): # ให้สามารถกด q ออกได้หลังจากบันทึกเสร็จ
                break 
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print("Data collection finished.")