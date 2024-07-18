import streamlit as st
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from pydub import AudioSegment
from io import BytesIO
import librosa

# Define the CNN model class with dynamic input size calculation
class EmotionCNN(nn.Module):
    def __init__(self):
        super(EmotionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        
        self.fc_input_size = self._get_fc_input_size((1, 1, 128, 200))
        self.fc1 = nn.Linear(self.fc_input_size, 128)
        self.fc2 = nn.Linear(128, 8)  # 8 emotions

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        x = x.view(x.size(0), -1)  # Flatten the tensor while keeping the batch size
        
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

    def _get_fc_input_size(self, shape):
        x = torch.rand(shape)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        return x.view(x.size(0), -1).size(1)

# Load the saved model
model = EmotionCNN()
model_load_path = "emotion_cnn_model.pth"
model.load_state_dict(torch.load(model_load_path))
model.eval()

# Define preprocessing function (same as in training)
def preprocess_audio(signal, sr=22050, n_mels=128, n_fft=2048, hop_length=512):
    mel_spectrogram = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    return mel_spectrogram_db

# Streamlit UI
st.title("Emotion Classification from Audio")
uploaded_file = st.file_uploader("Choose an audio file...", type=["wav"])

if uploaded_file is not None:
    # Load and preprocess the audio file using pydub
    audio = AudioSegment.from_file(BytesIO(uploaded_file.read()), format="wav")
    signal = np.array(audio.get_array_of_samples(), dtype=np.float32)
    sr = audio.frame_rate

    # Ensure signal is mono
    if audio.channels == 2:
        signal = signal.reshape((-1, 2)).mean(axis=1)

    mel_spectrogram = preprocess_audio(signal, sr)
    
    # Pad the spectrogram to the expected input size
    max_len = 200
    if mel_spectrogram.shape[1] < max_len:
        pad_width = max_len - mel_spectrogram.shape[1]
        mel_spectrogram = np.pad(mel_spectrogram, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mel_spectrogram = mel_spectrogram[:, :max_len]
    
    # Convert to tensor
    mel_spectrogram = torch.tensor(mel_spectrogram).unsqueeze(0).unsqueeze(0).float()

    # Make prediction
    with torch.no_grad():
        outputs = model(mel_spectrogram)
        _, predicted = torch.max(outputs, 1)
        emotion = predicted.item()

    # Map the prediction to the corresponding emotion label
    emotion_labels = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
    emotion_colors = {
        'neutral': '#808080',  # gray
        'calm': '#00FF00',  # green
        'happy': '#FFFF00',  # yellow
        'sad': '#0000FF',  # blue
        'angry': '#FF0000',  # red
        'fearful': '#800080',  # purple
        'disgust': '#FFA500',  # orange
        'surprised': '#FFC0CB'  # pink
    }
    predicted_emotion = emotion_labels[emotion]
    emotion_color = emotion_colors[predicted_emotion]

    st.markdown(f"<h1 style='color: {emotion_color};'>Predicted Emotion: {predicted_emotion}</h1>", unsafe_allow_html=True)
