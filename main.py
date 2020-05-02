import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import chroma_vector as cv
import specmurt as sp

instrument = 'trumpet'
# instrument = 'piano'

AUDIO_BUFFER_SIZE = 1024
LOWER_LIMIT = 3
UPPER_LIMIT = 6
index = 1

label = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
notes = ['C', 'Cs', 'D', 'Ds', 'E', 'F', 'Fs', 'G', 'Gs', 'A', 'As', 'B']

print('\nThe sound of "' + instrument.upper() + '" is used in this process.\n')

# wavファイルの読み込み
wavs = []
for note in notes:
    fs, wav = wavfile.read('sound_data/' + instrument + '/' + instrument + '_' + note + '3' + '.wav')
    wavs.append(wav[:,0])

# クロマベクトルの作成
chroma_vectors = sp.make_chroma_vectors(wavs)

# 従来のクロマベクトルをもとにした共通クロマベクトル構造の定義
average_chroma = sp.make_average_chroma_vector(chroma_vectors, index)
title = 'average chroma vector of trumpet'
path = 'image/' + instrument + '/_AVG_CHROMA'
sp.save_fig_chroma(average_chroma, title, path, label)

# 1/fによる共通クロマベクトル構造の定義
chroma_of_harmonic_structure = sp.harmonic_structure()
title = 'average chroma vector of harmonic_structure'
path = 'image/harmonic_structure_CV'
cv_harmonic = np.roll(chroma_of_harmonic_structure[index], -9)
sp.save_fig_chroma(cv_harmonic, title, path, label)

path = 'image/' + instrument
using = ' using average chroma vector of '
for i, chroma in enumerate(chroma_vectors):
# クロマベクトルの画像保存
    title = 'chroma vector of ' + label[i]
    name = path + '/chroma_' + label[i]
    sp.save_fig_chroma(chroma[index], title, name, label)
# 逆畳み込みの画像保存（平均クロマ）
    title_avg = title + using + instrument
    name = path + '/average_chroma_' + label[i]
    sp.save_fig_chroma(sp.make_specmurt_chroma_vector(chroma[index], average_chroma), title_avg, name, label)
# 逆畳み込みの画像保存（1/f）
    title_hs = title + using + 'harmonic structure'
    name = path + '/harmonic_structure_' + label[i]
    sp.save_fig_chroma(sp.make_specmurt_chroma_vector(chroma[index], cv_harmonic), title_hs, name, label)

print('All process has been completed.')
