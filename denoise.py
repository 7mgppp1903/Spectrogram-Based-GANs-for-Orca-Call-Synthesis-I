import librosa
import soundfile as sf
import noisereduce as nr

# Load your generated audio
y, sr = librosa.load("/Users/miilee/OrcaCallGAN/generated_audio/spec_6.wav", sr=16000)

# Estimate noise from the first 0.5 seconds
noise_sample = y[:int(0.5*sr)]

# Apply noise reduction (FIXED VERSION)
y_denoised = nr.reduce_noise(y=y, y_noise=noise_sample, sr=sr, prop_decrease=0.8)

# Save result
sf.write("generated_clean_denoised.wav", y_denoised, sr)

print("Noise Reduced Audio Saved: generated_clean_denoised.wav")
