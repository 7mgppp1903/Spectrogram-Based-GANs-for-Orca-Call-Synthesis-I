import librosa
import soundfile as sf
import scipy.signal as signal

y, sr = librosa.load("generated_clean_denoised.wav", sr=16000)

# High-pass filter to remove low-frequency rumble < 4kHz
b, a = signal.iirfilter(5, 4000/(sr/2), btype="highpass", ftype="butter")
y_filtered = signal.filtfilt(b, a, y)

sf.write("generated_clean_denoised_eq.wav", y_filtered, sr)
print("Final Enhanced Audio Saved: generated_clean_denoised_eq.wav")
