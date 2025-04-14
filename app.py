import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os
import datetime
import pandas as pd
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

# --- Page Configuration ---
st.set_page_config(
    page_title="Facial Emotion App",
    page_icon="assets/favicon.png",
    layout="wide"
)

# --- Emotion Labels ---
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# --- Cache Model & Face Detector ---
@st.cache_resource
def load_fer_model():
    return load_model('fer_model.h5')

@st.cache_resource
def load_face_detector():
    return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# --- Save Feedback to CSV ---
def save_review(name, feedback):
    os.makedirs("reviews", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    review_data = {
        "Name": name,
        "Feedback": feedback,
        "Timestamp": timestamp
    }

    csv_file = "reviews/reviews.csv"

    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        df = pd.concat([df, pd.DataFrame([review_data])], ignore_index=True)
    else:
        df = pd.DataFrame([review_data])

    df.to_csv(csv_file, index=False)

# --- Streamlit-WebRTC Emotion Detector Class ---
class EmotionDetector(VideoTransformerBase):
    def __init__(self, model, face_cascade):
        self.model = model
        self.face_cascade = face_cascade

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (48, 48))
            face = face.astype("float") / 255.0
            face = img_to_array(face)
            face = np.expand_dims(face, axis=0)

            preds = self.model.predict(face, verbose=0)[0]
            label = emotion_labels[np.argmax(preds)]
            confidence = np.max(preds)

            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(img, f'{label} ({confidence:.2f})', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        return img

# --- Introduction Page ---
def introduction_page():
    st.title(" Welcome to Facial Emotion Recognition App")

    st.markdown("""
This is an AI-powered web application that uses your **webcam** to detect and recognize **facial emotions** in real time using deep learning.

The project builds upon previous research in the field of **affective computing** and **facial expression analysis**. It leverages a custom-trained deep learning model developed using a **combined dataset** from **FER-2013** and **CK+ (Extended Cohn-Kanade)** — two well-known benchmarks for facial emotion recognition.

These datasets provide a diverse range of emotional expressions, helping the model to generalize well across different facial structures, lighting conditions, and expressions. The model is capable of identifying core human emotions such as:

- 😠 Angry
- 😖 Disgust
- 😨 Fear
- 😄 Happy
- 😢 Sad
- 😲 Surprise
- 😐 Neutral

---

###  Technologies Used
- **Streamlit** for the user interface
- **OpenCV** for webcam and image processing
- **TensorFlow** (Keras) for emotion recognition model
- **Pandas** for data storage and review management

---

### ===l Privacy Notice
- All emotion detection happens **locally** in your browser
- No data is shared unless you manually submit it

---

### ✨ Future Improvements
- Live emotion tracking graph
- More advanced emotion categories
- Better accuracy on diverse lighting and angles
- Integration with cloud-based storage (optional)

""")

    if st.button("🚀 Start Emotion Detection"):
        st.session_state.page = "detect"

# --- Emotion Detection Page ---
def emotion_detection_page(model, face_cascade):
    st.header("🎥 Real-Time Emotion Detection")

    webrtc_streamer(
        key="emotion-detect",
        video_processor_factory=lambda: EmotionDetector(model, face_cascade),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )

    if st.button("➡️ Continue to Reviews & Contact"):
        st.session_state.page = "review"

# --- Review and Contact Page ---
def review_page():
    st.header("Leave a Review & Contact")

    st.markdown("""
We’d love to hear your thoughts about the app! Your feedback helps improve this project and inspire future features.

---
 **Contact Developer:** [Click to Email](mailto:gokunak@gmail.com)  
 **Source Code:** [GitHub - Lightyear/fer-webapp](https://github.com/lightyear/fer-webapp)
---
""")

    name = st.text_input("Your Name")
    feedback = st.text_area("Your Feedback")
    if st.button("Submit Review"):
        save_review(name, feedback)
        st.success("Thank you for your feedback! ")
    else:
        st.warning("Please fill out all fields before submitting.")

    csv_file = "reviews/reviews.csv"
    if os.path.exists(csv_file):
        st.subheader( "Past Reviews")
        df = pd.read_csv(csv_file)
        st.dataframe(df)

# --- Main Controller ---
def main():
    if "page" not in st.session_state:
        st.session_state.page = "intro"

    model = load_fer_model()
    face_cascade = load_face_detector()

    if st.session_state.page == "intro":
        introduction_page()
    elif st.session_state.page == "detect":
        emotion_detection_page(model, face_cascade)
    elif st.session_state.page == "review":
        review_page()

if __name__ == '__main__':
    main()
