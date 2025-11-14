import numpy as np
import librosa
import soundfile as sf

spec = np.load("/Users/miilee/OrcaCallGAN/Results/epoch10_sample_7.npy")

# Undo GAN tanh normalization [-1,1] → [0,1]
spec = (spec + 1) / 2

sr = 16000
n_fft = 1024
hop_length = 256

mel_inv = librosa.feature.inverse.mel_to_stft(
    spec,
    sr=sr,
    n_fft=n_fft,
    power=1.0
)

audio = librosa.griffinlim(mel_inv, hop_length=hop_length)

sf.write("generated_audio/spec_7.wav", audio, sr)

print("✅ Saved: generated_audio/spec_0.wav")
