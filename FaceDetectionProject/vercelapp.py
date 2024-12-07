import streamlit as st
import numpy as np

# Attempt to import with error handling
try:
    import cv2
except ImportError as e:
    st.error(f"OpenCV import error: {e}")
    st.stop()

try:
    import mediapipe as mp
except ImportError as e:
    st.error(f"MediaPipe import error: {e}")
    st.stop()

from FaceDetectionModule import FaceDetector
import tempfile
import os

def main():
    st.title("Real-Time Face Detection")
    
    # Upload video file
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])
    
    if uploaded_file is not None:
        try:
            # Use a more reliable temp file creation method for cloud environments
            temp_dir = tempfile.mkdtemp()
            temp_video_path = os.path.join(temp_dir, uploaded_file.name)
            
            with open(temp_video_path, 'wb') as temp_video:
                temp_video.write(uploaded_file.read())
            
            # Open video capture
            cap = cv2.VideoCapture(temp_video_path)
            
            if not cap.isOpened():
                st.error("Error: Could not open video file.")
                return
            
            detector = FaceDetector(0.66)
            
            stframe = st.empty()
            
            frame_count = 0
            while cap.isOpened():
                success, img = cap.read()
                if not success:
                    st.write("End of video.")
                    break
                
                # Process only every 3rd frame to reduce computational load
                frame_count += 1
                if frame_count % 3 != 0:
                    continue
                
                try:
                    img, bboxs = detector.findFaces(img)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    stframe.image(img, channels="RGB", use_column_width=True)
                except Exception as e:
                    st.error(f"Error processing frame: {e}")
                    break
            
            cap.release()
            
            # Clean up temporary files
            os.unlink(temp_video_path)
            os.rmdir(temp_dir)
        
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()