import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
from pytube import YouTube
from ultralytics import YOLO

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = YOLO('yolov8n.pt')

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        results = self.model.track(img, persist=True)
        frame_ = results[0].plot()

        return av.VideoFrame.from_ndarray(frame_, format="bgr24")

def main():
    st.title("YouTube Video Object Detection with YOLOv8")
    link = st.text_input("Enter YouTube Video Link:")

    if link:
        # Note: You can still download the video if needed, but for real-time streaming, we won't
        # yt = YouTube(link)
        # stream = yt.streams.filter(file_extension="mp4").first()
        # video_path = stream.download()
        
        # Instead of processing the video using OpenCV's VideoCapture, we'll use streamlit-webrtc's streaming
        webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)

if __name__ == "__main__":
    main()
