import streamlit as st
from scipy.io import wavfile
from IPython.display import Audio
import librosa
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import pickle
import numpy as np


with open('/content/drive/MyDrive/speech_recognition/labels.pkl', 'rb') as f:
        labels = pickle.load(f)

best_model = load_model('/content/drive/MyDrive/speech_recognition/custom_model.h5')

def recognize_speech(x):
    samples, _ = librosa.load(x, sr=22050)
    padded_sequence = pad_sequences([samples],maxlen=22050,dtype='float32',
                                              padding='post')
    spectrum = librosa.feature.melspectrogram(y=padded_sequence, sr=22050, n_mels=64)
    logmel_spectrum_data = librosa.power_to_db(S=spectrum, ref=np.max)
    prediction = np.argmax(best_model.predict(logmel_spectrum_data,verbose=0))
    predicted_label = labels[prediction]
    return predicted_label
            
            
st.title("Speech Recognition System")
st.write("Upload an audio file and let me recognize the speech!")

uploaded_file = st.file_uploader("Upload Audio File", type=["wav"])

if uploaded_file is not None:
    st.write("File uploaded successfully.")
    st.audio(uploaded_file, format='audio/wav')
    if st.button("Recognize Speech"):
        prediction = recognize_speech(uploaded_file)
        st.write(f"Predicted Word: {prediction}")