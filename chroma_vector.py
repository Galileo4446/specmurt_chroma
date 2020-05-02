import numpy as np
import math
import matplotlib.pyplot as plt
label = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
label_num = range(12)


# 12音に対するクロマベクトルを作成
def make_chroma_vectors(wavs, audio_buffer_size, lower, upper, inst):
    chroma_vectors=[]
    for note in range(12):
        chroma_vector = make_chroma_vector(wavs[note], audio_buffer_size, lower, upper)
        chroma_vectors.append(chroma_vector)
    save_chroma_vector(chroma_vectors, 1, inst, 'chroma')
    return chroma_vectors

# クロマベクトルを作成
def make_chroma_vector(wav, audio_buffer_size, lower, upper):
    w_ham = np.hamming(audio_buffer_size)
    chroma_vector = []
    bin_nums = []
    for a in range(int(len(wav) / audio_buffer_size)):
        sample = []
        value = []
        bin_num = []
        for num, b in enumerate(wav[a * audio_buffer_size: (a + 1) * audio_buffer_size]):
            # sample.append(w_ham[num] * b / (2 ** 15))
            sample.append(b / (2 ** 15))
        # FFT点数のルートで割ってから計算すれば最大値が0dBになる？  / math.sqrt(audio_buffer_size)
        # print(note, a, b, num,sample==[])
        fft_wave = np.fft.fft(sample) / math.sqrt(audio_buffer_size)
        # plt.plot(abs(fft_wave[0:10000]))
        # plt.show()
        for i in range(12):
            count = 0
            sum = 0
            for j in range(lower, upper + 1):
                min_bin = (440. * pow(2., ((float(i) + 1.) - 10.) / 12. + float(j) - 3. - (1. / 24.))) * 1024. / 16000.
                min_k =  math.ceil(min_bin)
                max_bin = (440. * pow(2., ((float(i) + 1.) - 10.) / 12. + float(j) - 3. + (1. / 24.))) * 1024. / 16000.
                max_k =  math.floor(max_bin)
                if min_k <= max_k:
                    for k in range(min_k, max_k + 1):
                        sum += (fft_wave[k].real ** 2) + (fft_wave[k].imag ** 2)
                        count += 1
                        bin_num.append(k)
            value.append(sum / float(count))
            bin_nums.append(bin_num)
        chroma_vector.append(value)
    return chroma_vector

# specmurtを用いたクロマベクトルのグラフを保存
def save_chroma_specmurt(vectors, inst, type):
    using = '' if type == 'chroma' else 'using_' + type
    for note_number, note_name in enumerate(label):
        plt.barh(range(12), vectors[note_number], color='#2980b9', edgecolor='#2980b9', align='center', tick_label=label)
        plt.title('chroma vector of ' + note_name + ' ' + using)
        plt.xlabel('power')
        plt.ylabel('note')
        plt.savefig('image/' + inst + '/' + type + '_' + note_name + '.pdf')
        # plt.savefig('image/' + inst + '/chroma/chroma_' + note_name + '.png')
        # plt.show()
        plt.close()
    # print('Image of chroma vector has been saved. (type:' + type + ')')
    return

# クロマベクトルのグラフを保存
def save_chroma_vector(vectors, index, inst, type):
    using = '' if type == 'chroma' else 'using_' + type
    for note_number, note_name in enumerate(label):
        plt.barh(range(12), vectors[note_number][index], color='#2980b9', edgecolor='#2980b9', align='center', tick_label=label)
        plt.title('chroma vector of ' + note_name + ' ' + using)
        plt.xlabel('power')
        plt.ylabel('note')
        plt.savefig('image/' + inst + '/' + type + '_' + note_name + '.pdf')
        # plt.savefig('image/' + inst + '/chroma/chroma_' + note_name + '.png')
        # plt.show()
        plt.close()
    # print('Image of chroma vector has been saved. (instrument:' + inst + ')')
    return

# 入力音源をもとに共通のクロマベクトル構造を定義
def make_average_chroma_vector(chroma_vectors, inst):
    average_chroma = 0
    for note_number, name in enumerate(label):
        average_chroma += np.roll(chroma_vectors[note_number][1], -note_number)
    average_chroma = average_chroma / 12

    save_fig(average_chroma, inst)
    return average_chroma

# 共通クロマベクトル構造のグラフを保存
def save_fig(average_chroma, inst):
    plt.barh(range(12), average_chroma, color='#2980b9', edgecolor='#2980b9', align='center', tick_label=label_num)
    plt.title('average chroma vector from ' + inst)
    plt.xlabel('power')
    plt.ylabel('note')
    plt.savefig('image/' + inst + '_AVG_CHROMA.pdf')
    # plt.savefig('specmurt_chroma/AVERAGE_CHROMA.png')
    # plt.show()
    plt.close()
    # print('Image of the average chroma vector has been saved. (instrument:' + inst + ')')
    return

# 1/fの共通クロマベクトルを定義
def harmonic_structure():
    wav = []
    fs = 16000
    sec = 1
    f0 = 440
    a = 2 ** 15
    for n in np.arange(fs * sec):
        s = 0
        for i in range(1, 7):
            s += a / i * np.sin(2.0 * np.pi * f0 * i * n / fs)
        wav.append(s)
    # plt.plot(wav[0:100])
    # plt.show()
    return wav

# 共通クロマベクトル構造と従来のクロマベクトルからspecmurtクロマベクトルを定義
def make_specmurt_chroma_vectors(vectors, index, avg_chroma, inst, type):
    specmurt_chroma_vectors = []
    for note_number, name in enumerate(label):
        specmurt_chroma = np.absolute(np.fft.ifft(np.fft.fft(vectors[note_number][index]) / np.fft.fft(avg_chroma)))
        specmurt_chroma_vectors.append(specmurt_chroma)
    save_chroma_specmurt(specmurt_chroma_vectors, inst, type)
    return specmurt_chroma_vectors

def return_max(chroma):
    max_index = chroma.index(max(chroma))
    label[max_index] = label[max_index] + ' ' if len(label[max_index]) == 1 else label[max_index] + ''
    return label[max_index]
