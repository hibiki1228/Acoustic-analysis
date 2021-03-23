import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf


# 音声の読み込み
path = 'C:/Users/hibiki/Desktop/Acoustic_analysis/sample_sound/sounds_a.wav'
master, fs = sf.read(path)

t = np.arange(0, len(master) / fs, 1/fs)

# 音声波形の中心部分（定常部）を切り出す
center = len(master) // 2   # 中心のサンプル番号
cuttime = 0.04  # 秒
x = master[int(center - cuttime / 2 * fs):int(center + cuttime / 2 *fs)]
time = t[int(center - cuttime / 2 * fs):int(center + cuttime / 2 * fs)]

plt.plot(time * 1000, x)
plt.xlabel("time [ms]")
plt.ylabel("amplitude")
plt.show()