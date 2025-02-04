# Emotion-Based-Music-Recommendation-System-
A real-time music recommendation system that detects facial expressions using MediaPipe and suggests music by opening a YouTube search query based on the detected emotion. Built with Streamlit, Streamlit-WebRTC, and OpenCV.

## Features
* Real-time emotion detection using MediaPipe
* Webcam-based face tracking with Streamlit-WebRTC
* Automatic YouTube search for music based on detected emotion
* Interactive web interface using Streamlit

## Technologies Used
* Python
* MediaPipe (for face and emotion detection)
* Streamlit & Streamlit-WebRTC (for web-based UI and real-time webcam input)
* OpenCV (for image processing)
* Web Browser Automation (to open YouTube search query)

## Usage

### Running the Application

To run the main application, use the following command:
```sh
streamlit run main_app.py
```

### Data Collection

To collect data for training the model, run:
```sh
python datacollection.py
```

### Model Training

To train the model with the collected data, run:
```sh
python modeltrain.py
```
This will save the trained model as `model.h5` and the labels as `labels.npy`.

## How It Works
* The webcam captures the user's facial expression in real-time.
* MediaPipe processes the face and detects emotions like happy, sad, angry, surprised, neutral.
* Based on the detected emotion, the app generates a YouTube search query for songs related to the emotion.
* The browser automatically opens a new tab with the relevant music suggestions.

## Future Enhancements
* Integrate Spotify API for direct music playback
* Add user customization options for music preferences

