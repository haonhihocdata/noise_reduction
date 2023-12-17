import streamlit as st
from pydub import AudioSegment

def main():
    st.markdown("# <center>ỨNG DỤNG GIẢM NHIỄU VÀ ỒN</center>", unsafe_allow_html=True)
    st.markdown("## <span style='color: #FF7F00;'><br>Thư viện Librosa</span>", unsafe_allow_html=True)

    # Tên file âm thanh
    input_audio_file = "./input/noise_reduction_input.wav"
    output_audio_file = "./output/librosa_output.wav"

    # Đọc và hiển thị file âm thanh
    st.subheader(f"Âm thanh gốc")
    play_audio(input_audio_file)
    st.subheader(f"Âm thanh đã giảm nhiễu, ồn")
    play_audio(output_audio_file)

def play_audio(audio_file):
    # Đọc file âm thanh
    audio = AudioSegment.from_file(audio_file)

    # Hiển thị thanh điều khiển âm thanh và phát âm thanh
    audio_bytes = audio.export(format="wav").read()
    st.audio(audio_bytes, format="audio/wav")

if __name__ == "__main__":
    main()
