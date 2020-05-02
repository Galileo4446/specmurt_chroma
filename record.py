import pyaudio
import wave
import numpy as np
import matplotlib.pyplot as plt
import datetime
from scipy.io import wavfile

import chroma_vector as cv

rec_time = 3            # 録音時間[s]
sampling_rate = 16000 # サンプリング周波数
audio_buffer_size = 1024       # オーディオバッファサイズ
lower = 1
upper = 6
dt_now = datetime.datetime.now().strftime('%Y_%m%d_%H%M%S')
file_path = 'sound_data/record/' + dt_now + '.wav' #音声を保存するファイル名

def main():
    record()
    print(file_path)
    result_cv = []
    result_spec = []
    fs, wav = wavfile.read(file_path)

    chroma_vector = cv.make_chroma_vector(wav, audio_buffer_size, lower, upper)
    for chroma in chroma_vector:
        result_cv.append(cv.return_max(chroma))
    print('chroma vector')
    print(' '.join(result_cv))

    # harmonic_structure = cv.harmonic_structure()
    # average_chroma_f = cv.make_chroma_vector(harmonic_structure, audio_buffer_size, lower, upper)
    # chroma_vectors = [chroma_vector]
    # cv.make_specmurt_chroma_vector(chroma_vectors, 1, np.roll(average_chroma_f[1], -9), 'record', 'harmonic_structure')

    print('specmurt chroma vector')
    # print(' '.join(result))

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

def make_specmurt_chroma_vector_for_record(vectors):
    specmurt_chroma_vectors = []
    for i, vector in enumerate(vectors):
        specmurt_chroma = np.absolute(np.fft.ifft(np.fft.fft(vectors[note_number][index]) / np.fft.fft(avg_chroma)))
        specmurt_chroma_vectors.append(specmurt_chroma)
    save_chroma_specmurt(specmurt_chroma_vectors, inst, type)
    return specmurt_chroma_vectors

if __name__ == '__main__':
    main()
