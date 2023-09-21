import cv2
import streamlit as st
import yt_dlp
import requests
from ultralytics import YOLO

# Global variables
CHUNK_SIZE = 512 * 1024  # Size of video chunk to process at once (512 KB)
VIDEO_FORMATS = ['mp4']

def main():
    # Streamlit UI
    st.title("YouTube Video Object Detection with YOLOv8")

    # Input for YouTube link
    link = st.text_input("Enter YouTube Video Link:")

    if link:
        st.write("Fetching video...")

        # Configuration options for yt_dlp
        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]',  # Choose MP4 format
            'quiet': True
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            video_info = ydl.extract_info(link, download=False)
            video_url = video_info['url']

        st.write("Processing video...")

        # Load yolov8 model
        model = YOLO('yolov8n.pt')

        # Fetch video and process
        with requests.get(video_url, stream=True) as response:
            buffer = BytesIO()
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                buffer.write(chunk)
                process_buffer(buffer, model)
                buffer = BytesIO()

        st.write("Processing completed!")

def process_buffer(buffer, model):
    buffer.seek(0)
    cap = cv2.VideoCapture(buffer)

    ret, frame = cap.read()
    while ret:
        results = model.track(frame, persist=True)
        frame_ = results[0].plot()

        # Convert frame to RGB and display directly in Streamlit
        frame_rgb = cv2.cvtColor(frame_, cv2.COLOR_BGR2RGB)
        st.image(frame_rgb, channels='RGB', use_column_width=True)

        ret, frame = cap.read()

    cap.release()

if __name__ == "__main__":
    main()
