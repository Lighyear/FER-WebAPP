def emotion_detection_page(model, face_cascade):
    st.header(" Live Webcam Emotion Detection")

    webrtc_streamer(
        key="emotion-detect",
        video_processor_factory=lambda: EmotionDetector(model, face_cascade),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )

    st.markdown(" This works in-browser and doesn't save your video.")
