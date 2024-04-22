import cv2
import streamlit as st
import numpy as np
import wikipedia
from gtts import gTTS
import os

# Function to perform object detection
def detect_objects(image):
    classNames = []
    classFiles = 'coco.names'
    with open(classFiles, 'rt') as f:
        classNames = f.read().rstrip('\n').split('\n')
    configfile = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    weight = 'frozen_inference_graph.pb'
    net = cv2.dnn_DetectionModel(weight, configfile)
    net.setInputSize(320, 320)
    net.setInputScale(1.0/127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)
    classIds, confs, bbox = net.detect(image, confThreshold=0.5)
    if isinstance(classIds, tuple):
        classIds = np.array(classIds)
    if isinstance(confs, tuple):
        confs = np.array(confs)
    if isinstance(bbox, tuple):
        bbox = np.array(bbox)
    for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
        if confidence < 75:
            cv2.rectangle(image, box, color=(0, 255, 0), thickness=2)
            cv2.putText(image, classNames[classId-1], (box[0]+10, box[1]+30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(image, str(round(confidence*100)), (box[0]+150, box[1]+30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            try:
                summary = wikipedia.summary(classNames[classId-1], sentences=3)
                st.write(f"**{classNames[classId-1]}**\n{summary}")
                
                # Convert text to speech using gTTS
                tts = gTTS(text=f"Detected {classNames[classId-1]}. {summary}", lang='en')
                audio_file = f"{classNames[classId-1]}.mp3"
                tts.save(audio_file)
                
                # Display download link for the audio file
                st.audio(audio_file, format='audio/mp3')
                
                # Remove the audio file after displaying
                os.remove(audio_file)
            except wikipedia.exceptions.DisambiguationError as e:
                st.write(f"**{classNames[classId-1]}**\n{e.options[0]}")
                # Convert text to speech using gTTS
                tts = gTTS(text=f"Detected {classNames[classId-1]}. {e.options[0]}", lang='en')
                audio_file = f"{classNames[classId-1]}.mp3"
                tts.save(audio_file)
                # Display download link for the audio file
                st.audio(audio_file, format='audio/mp3')
                # Remove the audio file after displaying
                os.remove(audio_file)
            return image

def main():
    st.title('Object Detection')

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the uploaded image
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
        
        # Perform object detection
        detected_image = detect_objects(image)

        # Display the detected image
        st.image(detected_image, channels="RGB", use_column_width=True)

if __name__ == '__main__':
    main()
