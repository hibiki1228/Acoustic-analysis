import sys
import scipy.io.wavfile
import numpy as np
import matplotlib.pyplot as plt


# 音声ファイル読み込み
args = sys.argv
wav_filename = args[1]
rate, data = scipy.io.wavfile.read(wav_filename)


# （振幅）の配列を作成
data = data / 32768

##### 周波数成分を表示 #####
# 縦軸：dataを高速Fourier変換する（時間領域から周波数領域に変換）
fft_data = np.abs(np.fft.fft(data))
#横軸：周波数の取得         # np.fft.fftfreq(データ点数, サンプリング周期)
freqlist = np.fft.fftfreq(data.shape[0], d=1.0/rate)
# データプロット
plt.plot(freqlist, fft_data)
plt.xlim(0, 8000) # 0～8000Hzまで表示
plt.show()
