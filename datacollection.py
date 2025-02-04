import mediapipe as mp 
import numpy as np 
import cv2 
 
video_capture = cv2.VideoCapture(0)

data_name = input("Enter the name of the data: ")

mp_holistic = mp.solutions.holistic
mp_hands = mp.solutions.hands
holistic_processor = mp_holistic.Holistic()
drawing_utils = mp.solutions.drawing_utils

data_list = []
frame_count = 0

while True:
    landmarks = []

    ret, frame = video_capture.read()

    frame = cv2.flip(frame, 1)

    results = holistic_processor.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.face_landmarks:
        for landmark in results.face_landmarks.landmark:
            landmarks.append(landmark.x - results.face_landmarks.landmark[1].x)
            landmarks.append(landmark.y - results.face_landmarks.landmark[1].y)

        if results.left_hand_landmarks:
            for landmark in results.left_hand_landmarks.landmark:
                landmarks.append(landmark.x - results.left_hand_landmarks.landmark[8].x)
                landmarks.append(landmark.y - results.left_hand_landmarks.landmark[8].y)
        else:
            landmarks.extend([0.0] * 42)

        if results.right_hand_landmarks:
            for landmark in results.right_hand_landmarks.landmark:
                landmarks.append(landmark.x - results.right_hand_landmarks.landmark[8].x)
                landmarks.append(landmark.y - results.right_hand_landmarks.landmark[8].y)
        else:
            landmarks.extend([0.0] * 42)

        data_list.append(landmarks)
        frame_count += 1

    drawing_utils.draw_landmarks(frame, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)
    drawing_utils.draw_landmarks(frame, results.left_hand_landmarks, mp_hands.HAND_CONNECTIONS)
    drawing_utils.draw_landmarks(frame, results.right_hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.putText(frame, str(frame_count), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Data Collection", frame)

    if cv2.waitKey(1) == 27 or frame_count > 99:
        cv2.destroyAllWindows()
        video_capture.release()
        break

np.save(f"{data_name}.npy", np.array(data_list))
print(np.array(data_list).shape)