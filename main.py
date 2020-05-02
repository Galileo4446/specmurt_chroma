import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

import chroma_vector as cv

instrument = 'trumpet'
# instrument = 'piano'

AUDIO_BUFFER_SIZE = 1024
LOWER_LIMIT = 3
UPPER_LIMIT = 6

notes = ['C', 'Cs', 'D', 'Ds', 'E', 'F', 'Fs', 'G', 'Gs', 'A', 'As', 'B']
wavs = []

print('\nThe sound of "' + instrument.upper() + '" is used in this process.\n')
for note in notes:
    fs, wav = wavfile.read('sound_data/' + instrument + '/' + instrument + '_' + note + '3' + '.wav')
    wavs.append(wav[:,0])

# クロマベクトルの作成と保存
chroma_vectors = cv.make_chroma_vectors(wavs, AUDIO_BUFFER_SIZE, LOWER_LIMIT, UPPER_LIMIT, instrument)

# 従来のクロマベクトルをもとにした共通クロマベクトル構造の定義
average_chroma = cv.make_average_chroma_vector(chroma_vectors, instrument)
# 逆畳み込み
cv.make_specmurt_chroma_vectors(chroma_vectors, 1, average_chroma, instrument, 'average_chroma')

# 1/fによる共通クロマベクトル構造の定義
harmonic_structure = cv.harmonic_structure()
average_chroma_f = cv.make_chroma_vector(harmonic_structure, AUDIO_BUFFER_SIZE, LOWER_LIMIT, UPPER_LIMIT)
cv.save_fig(np.roll(average_chroma_f[1], -9) ,'harmonic_structure')
# specmurtを適用したクロマベクトルの作成と保存
cv.make_specmurt_chroma_vectors(chroma_vectors, 1, np.roll(average_chroma_f[1], -9), instrument, 'harmonic_structure')

print('All process has been completed.')
