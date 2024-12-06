import streamlit as st
import cv2
import mediapipe as mp
from FaceDetectionModule import FaceDetector  # Your FaceDetector class
import tempfile

def main():
    st.title("Real-Time Face Detection")

    # Upload video file
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])
    if uploaded_file is not None:
        # Save the uploaded video to a temp file
        with tempfile.NamedTemporaryFile(delete=False) as temp_video:
            temp_video.write(uploaded_file.read())
            video_path = temp_video.name

        cap = cv2.VideoCapture(video_path)
        detector = FaceDetector(0.66)

        stframe = st.empty()

        while cap.isOpened():
            success, img = cap.read()
            if not success:
                st.write("End of video.")
                break

            img, bboxs = detector.findFaces(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            stframe.image(img, channels="RGB", use_column_width=True)

        cap.release()

if __name__ == "__main__":
    main()
