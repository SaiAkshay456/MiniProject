import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2 
import numpy as np 
import mediapipe as mp 
from keras.models import load_model
import webbrowser
import os

# Paths for model and labels
model_path = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)), "model.h5")
labels_path = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)), "labels.npy")

# Folder path for saving emotion.npy

# Load model and labels
model  = load_model(model_path)
label = np.load(labels_path)

# Mediapipe setup
holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils

# Streamlit setup
st.header("Emotion Based Music Recommender")
if "run" not in st.session_state:
    st.session_state["run"] = "true"

try:
    emotion = np.load("emotion.npy")[0]
except:
    emotion = ""

if not emotion:
    st.session_state["run"] = "true"
else:
    st.session_state["run"] = "false"
    

class EmotionProcessor:
    def recv(self, frame):
        frm = frame.to_ndarray(format="bgr24")

        ##############################
        frm = cv2.flip(frm, 1)

        res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

        lst = []

        if res.face_landmarks:
            for i in res.face_landmarks.landmark:
                lst.append(i.x - res.face_landmarks.landmark[1].x)
                lst.append(i.y - res.face_landmarks.landmark[1].y)

            if res.left_hand_landmarks:
                for i in res.left_hand_landmarks.landmark:
                    lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
            else:
                for i in range(42):
                    lst.append(0.0)

            if res.right_hand_landmarks:
                for i in res.right_hand_landmarks.landmark:
                    lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
            else:
                for i in range(42):
                    lst.append(0.0)

            lst = np.array(lst).reshape(1, -1)

            pred = label[np.argmax(model.predict(lst))]

            print(pred)
            cv2.putText(frm, pred, (50, 50), cv2.FONT_ITALIC, 1, (255, 0, 0), 2)

            # Save the emotion prediction to the defined file path
            np.save("emotion.npy", np.array([pred]))

        drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_TESSELATION,
                               landmark_drawing_spec=drawing.DrawingSpec(color=(0, 0, 255), thickness=-1, circle_radius=1),
                               connection_drawing_spec=drawing.DrawingSpec(thickness=1))
        drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
        drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)

        ##############################

        return av.VideoFrame.from_ndarray(frm, format="bgr24")

# Inputs for language and singer
lang = st.text_input("Language")
singer = st.text_input("singer")

# Start the webrtc streamer if the inputs are provided
if lang and singer and st.session_state["run"] != "false":
    webrtc_streamer(key="key", desired_playing_state=True, video_processor_factory=EmotionProcessor)

# Button to recommend songs
btn = st.button("Recommend me songs")

if btn:
    if not emotion:
        st.warning("Please let me capture your emotion first.")
        st.session_state["run"] = "true"
    else:
        # Construct and open the YouTube query
        query = f"{lang} {emotion} song {singer}"
        st.info(f"Searching for: {query}")
        webbrowser.open(f"https://www.youtube.com/results?search_query={query}")
        
        # Reset emotion and session state
        np.save("emotion.npy", np.array([]))  # Clear saved emotion
        emotion = ""  # Reset local emotion variable
        st.session_state["run"] = "true"  # Enable fresh capture
        st.success("Emotion reset. Ready for a new detection.")
