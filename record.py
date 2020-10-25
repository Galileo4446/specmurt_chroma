import pyaudio
import wave
import numpy as np
import matplotlib.pyplot as plt
import datetime
from scipy.io import wavfile

import specmurt as sp

rec_time = 3            # 録音時間[s]
sampling_rate = 16000 # サンプリング周波数
audio_buffer_size = 2048       # オーディオバッファサイズ
lower = 3
upper = 6
dt_now = datetime.datetime.now().strftime('%Y_%m%d_%H%M%S')
file_path = 'sound_data/record/' + dt_now + '.wav' #音声を保存するファイル名

def main():
    record()
    print(file_path)
    result_cv = []
    result_sp = []
    fs, wav = wavfile.read(file_path)

    chroma_vector = sp.make_chroma_vector(wav)
    for chroma in chroma_vector:
        result_cv.append(sp.return_max(chroma))
    print('chroma vector')
    print(' '.join(result_cv))

    specmurt_chroma = make_specmurt_chroma_vector_for_record(chroma_vector)
    for sp_chroma in specmurt_chroma:
        result_sp.append(sp.return_max(sp_chroma.tolist()))
    print('specmurt chroma vector')
    print(' '.join(result_sp))
    return

def record():
    fmt = pyaudio.paInt16  # 音声のフォーマット
    ch = 1              # チャンネル1(モノラル)
    audio = pyaudio.PyAudio()
    index = 1 # 録音デバイスのインデックス番号（デフォルト1）

    stream = audio.open(format=fmt, channels=ch, rate=sampling_rate, input=True, input_device_index = index, frames_per_buffer=audio_buffer_size)
    print("recording start...")

    # 録音処理
    frames = []
    for i in range(0, int(sampling_rate / audio_buffer_size * rec_time)):
        data = stream.read(audio_buffer_size)
        frames.append(data)

    print("recording  end...")

    # 録音終了処理
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # 録音データをファイルに保存
    wav = wave.open(file_path, 'wb')
    wav.setnchannels(ch)
    wav.setsampwidth(audio.get_sample_size(fmt))
    wav.setframerate(sampling_rate)
    wav.writeframes(b''.join(frames))
    wav.close()
    print('saved as ' + dt_now + '.wav')
    return

def make_specmurt_chroma_vector_for_record(chroma_vector):
    index = 1
    specmurt_chroma_vectors = []
    chroma_of_harmonic_structure = sp.harmonic_structure()
    cv_harmonic = np.roll(chroma_of_harmonic_structure[index], -9)
    for chroma in chroma_vector:
        specmurt_chroma_vectors.append(sp.make_specmurt_chroma_vector(chroma, cv_harmonic))
    return specmurt_chroma_vectors

if __name__ == '__main__':
    main()
