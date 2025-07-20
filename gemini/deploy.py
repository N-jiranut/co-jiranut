# --- ตัวอย่างการทำนายผลจากวิดีโอสด (Conceptual) ---
# ต้องมีการสร้าง buffer เพื่อเก็บ SEQUENCE_LENGTH ของเฟรม landmark
# เมื่อ buffer เต็ม ก็ป้อนเข้าโมเดล

# model = load_model('dynamic_sign_language_model.h5')
# class_names = ACTIONS # ['Hello', 'Thank_You', 'Goodbye']

# cap = cv2.VideoCapture(0) # เปิดกล้อง
# landmark_sequence_buffer = [] # เก็บ landmarks ของเฟรมล่าสุด

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # 1. ดึง Landmarks จากเฟรมปัจจุบัน
#     img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = hands.process(img_rgb)
#     
#     current_frame_landmarks = []
#     if results.multi_hand_landmarks:
#         for hand_landmarks in results.multi_hand_landmarks:
#             for landmark in hand_landmarks.landmark:
#                 current_frame_landmarks.extend([landmark.x, landmark.y, landmark.z])
#             break
#     else:
#         current_frame_landmarks = [0.0] * (21 * 3)

#     # 2. เพิ่ม Landmarks เข้า Buffer
#     landmark_sequence_buffer.append(current_frame_landmarks)
#     if len(landmark_sequence_buffer) > SEQUENCE_LENGTH:
#         landmark_sequence_buffer.pop(0) # ลบเฟรมเก่าที่สุดออก

#     # 3. ถ้า Buffer เต็ม ทำการทำนายผล
#     if len(landmark_sequence_buffer) == SEQUENCE_LENGTH:
#         input_sequence = np.expand_dims(np.array(landmark_sequence_buffer), axis=0) # เพิ่ม batch dim
#         predictions = model.predict(input_sequence)[0]
#         predicted_class_index = np.argmax(predictions)
#         confidence = predictions[predicted_class_index]
#         predicted_action = class_names[predicted_class_index]

#         # แสดงผลบนหน้าจอ
#         cv2.putText(frame, f'Sign: {predicted_action} ({confidence:.2f})', (50, 50), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
#     cv2.imshow('Sign Language Translator', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()