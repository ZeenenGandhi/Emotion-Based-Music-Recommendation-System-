import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2 
import numpy as np 
import mediapipe as mp 
from keras.models import load_model
import webbrowser

emotion_model = load_model("model.h5")
emotion_labels = np.load("labels.npy")
mp_holistic = mp.solutions.holistic
mp_hands = mp.solutions.hands
holistic_processor = mp_holistic.Holistic()
drawing_utils = mp.solutions.drawing_utils

st.header("Emotion Based Music Recommender")

if "run" not in st.session_state:
    st.session_state["run"] = "true"

try:
    detected_emotion = np.load("emotion.npy")[0]
except:
    detected_emotion = ""

if not detected_emotion:
    st.session_state["run"] = "true"
else:
    st.session_state["run"] = "false"

class EmotionProcessor:
    def recv(self, frame):
        frame_array = frame.to_ndarray(format="bgr24")

        frame_array = cv2.flip(frame_array, 1)

        results = holistic_processor.process(cv2.cvtColor(frame_array, cv2.COLOR_BGR2RGB))

        landmarks_list = []

        if results.face_landmarks:
            for landmark in results.face_landmarks.landmark:
                landmarks_list.append(landmark.x - results.face_landmarks.landmark[1].x)
                landmarks_list.append(landmark.y - results.face_landmarks.landmark[1].y)

            if results.left_hand_landmarks:
                for landmark in results.left_hand_landmarks.landmark:
                    landmarks_list.append(landmark.x - results.left_hand_landmarks.landmark[8].x)
                    landmarks_list.append(landmark.y - results.left_hand_landmarks.landmark[8].y)
            else:
                landmarks_list.extend([0.0] * 42)

            if results.right_hand_landmarks:
                for landmark in results.right_hand_landmarks.landmark:
                    landmarks_list.append(landmark.x - results.right_hand_landmarks.landmark[8].x)
                    landmarks_list.append(landmark.y - results.right_hand_landmarks.landmark[8].y)
            else:
                landmarks_list.extend([0.0] * 42)

            landmarks_array = np.array(landmarks_list).reshape(1, -1)

            prediction = emotion_labels[np.argmax(emotion_model.predict(landmarks_array))]

            print(prediction)
            cv2.putText(frame_array, prediction, (50, 50), cv2.FONT_ITALIC, 1, (255, 0, 0), 2)

            np.save("emotion.npy", np.array([prediction]))

        drawing_utils.draw_landmarks(frame_array, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                                     landmark_drawing_spec=drawing_utils.DrawingSpec(color=(0, 0, 255), thickness=-1, circle_radius=1),
                                     connection_drawing_spec=drawing_utils.DrawingSpec(thickness=1))
        drawing_utils.draw_landmarks(frame_array, results.left_hand_landmarks, mp_hands.HAND_CONNECTIONS)
        drawing_utils.draw_landmarks(frame_array, results.right_hand_landmarks, mp_hands.HAND_CONNECTIONS)

    
        return av.VideoFrame.from_ndarray(frame_array, format="bgr24")

if st.session_state["run"] != "false":
    webrtc_streamer(key="key", desired_playing_state=True,
                    video_processor_factory=EmotionProcessor)

recommend_button = st.button("Recommend me songs")

if recommend_button:
    if not detected_emotion:
        st.warning("Please let me capture your emotion first")
        st.session_state["run"] = "true"
    else:
        webbrowser.open(f"https://www.youtube.com/results?search_query={detected_emotion}+song")
        np.save("emotion.npy", np.array([""]))
        st.session_state["run"] = "false"
