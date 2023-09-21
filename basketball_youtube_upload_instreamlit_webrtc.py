import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import av
from pytube import YouTube
from ultralytics import YOLO
import imageio
import numpy as np

rtc_config = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = YOLO('yolov8n.pt')
        self.frames = []

    def load_video(self, link):
        yt = YouTube(link)
        stream = yt.streams.filter(file_extension="mp4").first()
        video_path = stream.download()
        
        vid = imageio.get_reader(video_path, 'ffmpeg')
        for image in vid.iter_data():
            self.frames.append(image)
        return self.frames

    def transform(self, frame):
        if not self.frames:
            return frame

        img = self.frames.pop(0)

        results = self.model.track(img, persist=True)
        frame_ = results[0].plot()
        
        return av.VideoFrame.from_ndarray(frame_, format="bgr24")

def main():
    st.title("YouTube Video Object Detection with YOLOv8")
    
    link = st.text_input("Enter YouTube Video Link:")
    if link:
        transformer = VideoTransformer()
        frames = transformer.load_video(link)

        # Now, we can use streamlit-webrtc for streaming these frames as if they're from webcam
        webrtc_streamer(key="example", video_transformer_factory=VideoTransformer, rtc_configuration=rtc_config)

if __name__ == "__main__":
    main()
