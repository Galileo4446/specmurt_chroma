import numpy as np
import math
import matplotlib.pyplot as plt

audio_buffer_size = 1024
lower = 3
upper = 6
sample_late = 16000.
label = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


# クロマベクトルを作成
def make_chroma_vector(wav):
    # w_ham = np.hamming(audio_buffer_size)
    chroma_vector = []
    # bin_nums = []
    for buf_num in range(len(wav) // audio_buffer_size):
        sample = []
        value = []
        # bin_num = []
        # b = wav[buf_num * audio_buffer_size: (buf_num + 1) * audio_buffer_size]
        for num, b in enumerate(wav[buf_num * audio_buffer_size: (buf_num + 1) * audio_buffer_size]):
            # sample.append(w_ham[num] * b / (2 ** 15))
            sample.append(b / (2 ** 15))
        # FFT点数のルートで割ってから計算すれば最大値が0dBになる？  / math.sqrt(audio_buffer_size)
        # print(note, buf_num, b, num,sample==[])
        fft_wave = np.fft.fft(sample) / math.sqrt(audio_buffer_size)
        # plt.plot(abs(fft_wave[0:10000]))
        # plt.show()
        for i in range(12):
            count = 0
            sum = 0
            for j in range(lower, upper + 1):
                min_bin = (440 * pow(2., ((i + 1.) - 10.) / 12. + j - 3. - (1. / 24.))) * audio_buffer_size / sample_late
                min_k =  math.ceil(min_bin)
                max_bin = (440 * pow(2., ((i + 1.) - 10.) / 12. + j - 3. + (1. / 24.))) * audio_buffer_size / sample_late
                max_k =  math.floor(max_bin)
                if min_k <= max_k:
                    for k in range(min_k, max_k + 1):
                        sum += (fft_wave[k].real ** 2) + (fft_wave[k].imag ** 2)
                        count += 1
                        # bin_num.append(k)
            value.append(sum / count)
            # bin_nums.append(bin_num)
        chroma_vector.append(value)
    return 10 * np.log10(chroma_vector)

# 12音に対するクロマベクトルを作成
def make_chroma_vector_list(wavs):
    chroma_vectors=[]
    for note in range(12):
        chroma_vector = make_chroma_vector(wavs[note])
        chroma_vectors.append(chroma_vector)
    return chroma_vectors

# 画像の保存
def save_fig_chroma(vector, title, path, label):
    plt.barh(range(12), vector, color='#2980b9', edgecolor='#2980b9', align='center', tick_label=label)
    plt.title(title)
    plt.xlabel('power')
    plt.ylabel('note')
    plt.savefig(path + '.pdf')
    # plt.show()
    plt.close()
    # print(title + ' has been saved.')
    return

# 入力音源をもとに共通のクロマベクトル構造を定義
def make_average_chroma_vector(chroma_vectors, index):
    average_chroma = 0
    for num in range(12):
        average_chroma += np.roll(chroma_vectors[num][index], -num)
    average_chroma = average_chroma / 12
    return average_chroma

def make_specmurt_chroma_vector(vector, avg_chroma):
    return np.real(np.fft.ifft(np.fft.fft(vector) / np.fft.fft(avg_chroma)))

def make_specmurt_chroma_vector_2(vector, avg_chroma):
    return np.real(np.fft.fft(np.fft.ifft(vector) / np.fft.ifft(avg_chroma)))


def make_debug_chroma_vector(vector, avg_chroma):
    return np.real(np.fft.ifft(np.fft.fft(vector) * np.fft.fft(avg_chroma)))

def make_debug_chroma_vector_2(vector, avg_chroma):
    return np.real(np.fft.fft(np.fft.ifft(vector) * np.fft.ifft(avg_chroma)))


def return_max(chroma):
    max_index = chroma.index(max(chroma))
    label[max_index] = label[max_index] + ' ' if len(label[max_index]) == 1 else label[max_index] + ''
    return label[max_index]

def harmonic_structure():
    wav = []
    # fs = 16000
    sec = 1
    f0 = 220
    a = 2 ** 15
    for n in np.arange(int(sample_late) * sec):
        s = 0
        for i in range(1, 7):
            s += a / i * np.sin(2.0 * np.pi * f0 * i * n / int(sample_late))
        wav.append(s)
    # plt.plot(wav[0:100])
    # plt.show()
    return make_chroma_vector(wav)
