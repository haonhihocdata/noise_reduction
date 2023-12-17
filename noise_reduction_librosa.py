from __future__ import print_function
import numpy as np
import time
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
import librosa.display

start_time = time.time()

# Tải file âm thanh
y, sr = librosa.load('./input/noise_reduction_input.wav', duration=200)

# Tính toán Spectrogram
S_full, phase = librosa.magphase(librosa.stft(y))

# Giảm tiếng ồn
S_filter = librosa.decompose.nn_filter(S_full,
                                       aggregate=np.median,
                                       metric='cosine',
                                       width=int(librosa.time_to_frames(2, sr=sr)))

S_filter = np.minimum(S_full, S_filter)

# Áp dụng Mask
width = int(librosa.time_to_frames(5, sr=sr)) 
margin_i, margin_v = 1, 10
power = 1

mask_i = librosa.util.softmask(S_filter,
                               margin_i * (S_full - S_filter),
                               power=power)

mask_v = librosa.util.softmask(S_full - S_filter,
                               margin_v * S_filter,
                               power=power)

# Tách foreground và background
S_foreground = mask_v * S_full
S_background = mask_i * S_full

# Nghịch đảo STFT và Tái tạo
y_foreground = librosa.istft(S_foreground * phase)

end_time = time.time()
processing_time = end_time - start_time

print(f"Thời gian xử lý: {processing_time} giây")
cleanliness_ratio = np.sum(S_foreground) / np.sum(S_full)

print(f"Tỉ lệ sạch so với file gốc: {cleanliness_ratio * 100}%")

sf.write('./output/librosa_output.wav', y_foreground, samplerate=sr)