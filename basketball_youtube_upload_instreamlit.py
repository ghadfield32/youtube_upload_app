import cv2
import streamlit as st
from pytube import YouTube
from ultralytics import YOLO


def main():
    # Streamlit UI
    st.title("YouTube Video Object Detection with YOLOv8")

    # Input for YouTube link
    link = st.text_input("Enter YouTube Video Link:")

    if link:
        st.write("Downloading video...")
        yt = YouTube(link)
        stream = yt.streams.filter(file_extension="mp4").first()
        video_path = stream.download()
        st.write("Download completed!")

        # Load yolov8 model
        model = YOLO('yolov8n.pt')

        # Processing the video
        st.write("Processing video...")
        cap = cv2.VideoCapture(video_path)

        # Get the FPS of the original video
        fps = cap.get(cv2.CAP_PROP_FPS)
        delay = int(1000 / fps)  # delay in milliseconds

        # Define codec and create VideoWriter object to save processed video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('processed_video.mp4', fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

        # Create an empty slot for the video frame preview
        frame_placeholder = st.empty()

        ret = True
        while ret:
            ret, frame = cap.read()
            if ret:
                results = model.track(frame, persist=True)
                frame_ = results[0].plot()
                out.write(frame_)  # Save the processed frame to the new video file

                # Convert frame to RGB and display directly in Streamlit
                frame_rgb = cv2.cvtColor(frame_, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb, channels='RGB', use_column_width=True)

                # Wait for the specified delay
                cv2.waitKey(delay)

        cap.release()
        out.release()
        st.write("Processing completed!")

        # Display the final processed video within the Streamlit app
        video_file = open('processed_video.mp4', 'rb')
        video_bytes = video_file.read()
        st.video(video_bytes)


if __name__ == "__main__":
    main()
