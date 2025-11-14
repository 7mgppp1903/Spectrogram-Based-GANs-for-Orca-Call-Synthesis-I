import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

input_dir = "/Users/miilee/PycharmProjects/OrcaCallGAN/data/test_audio_files"
output_dir = "/Users/miilee/PycharmProjects/OrcaCallGAN/data/SpectrogramTest"
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.endswith(".wav"):
        filepath = os.path.join(input_dir, filename)
        try:
            y, sr = librosa.load(filepath, sr=16000)


            y, _ = librosa.effects.trim(y)

            S = librosa.feature.melspectrogram(
                y=y,
                sr=sr,
                n_mels=128,
                n_fft=1024,
                hop_length=256
            )


            S_dB = librosa.power_to_db(S, ref=np.max)

            np.save(os.path.join(output_dir, filename.replace(".wav", ".npy")), S_dB)


            plt.figure(figsize=(4, 4))
            plt.axis('off')
            librosa.display.specshow(S_dB, sr=sr, hop_length=256, cmap='magma')
            plt.savefig(os.path.join(output_dir, filename.replace(".wav", ".png")), bbox_inches='tight', pad_inches=0)
            plt.close()

        except Exception as e:
            print(f"Error processing {filename}: {e}")

