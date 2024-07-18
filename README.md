# Emotion-Recognition-from-Audio-using-CNN-AI-Project-

Objective: Develop a system that can recognise emotions from voice recordings using Convolutional Neural Networks (CNNs) with Librosa for audio processing and PyTorch for model building.

Scope:
1. Data Collection & Preprocessing
• Collect and preprocess a suitable dataset for emotion recognition.
• Extract relevant features (e.g., Mel spectrograms, MFCCs) using Librosa.
2. Model Development
• Design and implement a CNN architecture for emotion recognition.
• Train and validate the CNN model on the preprocessed dataset.
3. System Development
• Integrate the trained CNN model into a system that can take new audio inputs and provide emotion predictions.
• Develop a user interface for interaction and display of results.
4. Evaluation & Optimization
• Evaluate model performance on a test set.
• Optimize the model for better performance.
5. Documentation & Presentation
• Document the entire process.
• Prepare a presentation to showcase the project.

THE DATASET 

Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)
Full dataset of speech and song, audio and video (24.8 GB). This portion of the RAVDESS contains 1440 files: 60 trials per actor x 24 actors = 1440. The RAVDESS contains 24 professional actors (12 female, 12 male), vocalizing two lexically-matched statements in a neutral North American accent. Speech emotions includes calm, happy, sad, angry, fearful, surprise, and disgust expressions. Each expression is produced at two levels of emotional intensity (normal, strong), with an additional neutral expression.

Filename identifiers
Modality (01 = full-AV, 02 = video-only, 03 = audio-only).
Vocal channel (01 = speech, 02 = song).
Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the ‘neutral’ emotion.
Statement (01 = “Kids are talking by the door”, 02 = “Dogs are sitting by the door”).
Repetition (01 = 1st repetition, 02 = 2nd repetition).
Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).

Filename example: 02-01-06-01-02-01-12.mp4
Video-only (02)
Speech (01)
Fearful (06)
Normal intensity (01)
Statement “dogs” (02)
1st Repetition (01)
12th Actor (12)
Female, as the actor ID number is even.
<img width="1014" alt="Screenshot 2024-07-17 at 10 35 05 PM" src="https://github.com/user-attachments/assets/fc56610e-0500-45a3-81d0-851db00dbf5c">


