from __future__ import print_function
import numpy as np
import time
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
from pydub import AudioSegment

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

def calculate_cleanliness_ratio(original_audio, denoised_audio):
    # Chuyển đổi AudioSegment thành mảng numpy
    original_array = np.array(original_audio.get_array_of_samples())
    denoised_array = np.array(denoised_audio.get_array_of_samples())

    # Đảm bảo rằng kích thước của hai mảng là giống nhau hoặc có thể broadcast cho nhau
    min_len = min(len(original_array), len(denoised_array))
    original_array = original_array[:min_len]
    denoised_array = denoised_array[:min_len]

    # Tính toán công suất tín hiệu và công suất nhiễu
    signal_power = np.sum(original_array ** 2)
    noise_power = np.sum((original_array - denoised_array) ** 2)

    # Tính toán SNR
    snr = 10 * np.log10(signal_power / noise_power)

    # Tính toán tỷ lệ sạch dưới dạng phần trăm
    cleanliness_ratio = (1 - 10**(-snr / 10)) * 100

    return cleanliness_ratio

# Đọc file âm thanh gốc và sau khi giảm ồn
original_audio = AudioSegment.from_file("./input/noise_reduction_input.wav")
denoised_audio = AudioSegment.from_file("output/librosa_output.wav")

# Tính toán tỷ lệ sạch
cleanliness_ratio = calculate_cleanliness_ratio(original_audio, denoised_audio)

sf.write('./output/librosa_output.wav', y_foreground, samplerate=sr)


print(f"Thời gian xử lý: {processing_time:.2f} giây")
print(f"Tỷ lệ sạch: {cleanliness_ratio:.2f}%")