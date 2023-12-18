import streamlit as st
from pydub import AudioSegment

def main():
    st.markdown("# <center>ỨNG DỤNG GIẢM NHIỄU VÀ ỒN<br></center>", unsafe_allow_html=True)
    

    # Tên file âm thanh
    input_file = "./input/noise_reduction_input.wav"
    librosa_output_file = "./output/librosa_output.wav"
    spleeter_output_file = "./output/spleeter_output.wav"

    # Đọc và hiển thị file âm thanh
    st.markdown(f"### <br>ÂM THANH GỐC", unsafe_allow_html=True)
    play_audio(input_file)

    st.markdown("## <span style='color: #FF7F00;'>LIBROSA</span>", unsafe_allow_html=True)

    st.subheader(f"Âm thanh đã giảm nhiễu, ồn")
    play_audio(librosa_output_file)

    st.markdown("## <span style='color: #FF7F00;'>SPLEETER </span>", unsafe_allow_html=True)
    st.subheader(f"Âm thanh đã giảm nhiễu, ồn")
    play_audio(spleeter_output_file)

def play_audio(audio_file):
    # Đọc file âm thanh
    audio = AudioSegment.from_file(audio_file)

    # Hiển thị thanh điều khiển âm thanh và phát âm thanh
    audio_bytes = audio.export(format="wav").read()
    st.audio(audio_bytes, format="audio/wav")

if __name__ == "__main__":
    main()
