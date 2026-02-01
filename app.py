import streamlit as st
import cv2

st.set_page_config(page_title="Happy Emotion Detection", layout="centered")
st.title("😊 Happy Emotion Detection")

# Session state
if "camera_on" not in st.session_state:
    st.session_state.camera_on = False

# Buttons
col1, col2 = st.columns(2)
with col1:
    if st.button("▶ Start Camera"):
        st.session_state.camera_on = True
with col2:
    if st.button("⏹ Stop Camera"):
        st.session_state.camera_on = False

frame_placeholder = st.empty()

# Load Haar Cascades
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
smile_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_smile.xml"
)

if st.session_state.camera_on:
    cap = cv2.VideoCapture(0)

    while st.session_state.camera_on:
        ret, frame = cap.read()
        if not ret:
            st.error("Camera not accessible")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]

            smiles = smile_cascade.detectMultiScale(
                roi_gray,
                scaleFactor=1.7,
                minNeighbors=20
            )

            emotion = "Happy 😊" if len(smiles) > 0 else "Not Happy 😐"

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(
                frame,
                emotion,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame, channels="RGB")

    cap.release()
    frame_placeholder.empty()
else:
    st.info("Camera OFF. Click ▶ Start Camera")

