import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf


# 音声の読み込み
path = 'C:/Users/hibiki/Desktop/Acoustic_analysis/sample_sound/sounds_a.wav'
master, fs = sf.read(path)

# 音声波形の中心部分（定常部）を切り出す
center = len(master) // 2   # 中心のサンプル番号
cuttime = 0.04  # 秒
x = master[int(center - cuttime / 2 * fs):int(center + cuttime / 2 * fs)]

# ハミング窓をかける
hamming = np.hamming(len(x))
x = x * hamming

#振動スペクトラムを求める
N = 2048    #FFTのサンプル数
spec = np.abs(np.fft.fft(x, N))[:N//2]
fscale = np.fft.fftfreq(N, d = 1.0 / fs)[:N//2]

plt.plot(fscale, spec)
plt.xlabel("frequency [Hz]")
plt.ylabel("amplitude spectrum")
plt.show()
