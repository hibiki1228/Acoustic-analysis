import sys
import numpy as np
import librosa
import matplotlib.pyplot as plt
import scipy.io.wavfile
import librosa.display


#音声ファイル読み込み
args = sys.argv
wav_filename = args[1]
rate, data = scipy.io.wavfile.read(wav_filename)
#16bitの音声ファイルデータを-1から1に正規化
data = data /32768
#フレーム長
fft_size = 1024
#フレームシフト長
hop_length = int(fft_size / 4)


#短時間Fourier変換実行
amplitude = np.abs(librosa.core.stft(
    data,
    n_fft=fft_size,
    hop_length=hop_length)
)

#振幅をデシベル単位に変換
log_power = librosa.core.amplitude_to_db(amplitude)
print(log_power)
#グラフ表示
librosa.display.specshow(log_power,
                        sr=rate,
                        hop_length=hop_length,
                        x_axis='time',
                        y_axis='hz',
                        cmap='magma'
)
plt.colorbar(format='%+2.0f dB')
plt.show()
