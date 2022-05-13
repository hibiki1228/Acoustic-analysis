import numpy as np
from scipy import fftpack
import matplotlib.pyplot as plt

# フーリエ変換をする関数
def calc_fft(data, samplerate):
    # 信号のフーリエ変換
    spectrum = fftpack.fft(data)
    amp = np.sqrt((spectrum.real ** 2) + (spectrum.imag ** 2))       # 振幅成分
    # 振幅成分の正規化（辻褄合わせ）
    amp = amp / (len(data) / 2)
    phase = np.arctan2(spectrum.imag, spectrum.real)                 # 位相を計算
    # 位相をラジアンから度に変換
    phase = np.degrees(phase)
    freq = np.linspace(0, samplerate, len(data))                     # 周波数軸を作成
    return spectrum, amp, phase, freq


# 様々な時間長の時間波形を生成
samplerate = 25600
x1 = np.arange(0, 128000)/samplerate
data1 = np.sin(2 * np.pi * 1 * x1)

x2 = np.arange(0, 51200)/samplerate
data2 = np.sin(2 * np.pi * 1 * x2)

x3 = np.arange(0, 25600)/samplerate
data3 = np.sin(2 * np.pi * 1 * x3)

# フーリエ変換をそれぞれ実行
spectrum1, amp1, phase1, freq1 = calc_fft(data1, samplerate)
spectrum2, amp2, phase2, freq2 = calc_fft(data2, samplerate)
spectrum3, amp3, phase3, freq3 = calc_fft(data3, samplerate)

# 各信号で情報をプリント
print('df1=', np.round(freq1[1], 2), '[Hz]', 'T1=', len(
    data1) * (1/samplerate), '[s]', '1/T=', 1/(len(data1) * (1/samplerate)), '[Hz]')
print('df2=', np.round(freq2[1], 2), '[Hz]', 'T2=', len(
    data2) * (1/samplerate), '[s]', '1/T=', 1/(len(data2) * (1/samplerate)), '[Hz]')
print('df3=', np.round(freq3[1], 2), '[Hz]', 'T3=', len(
    data3) * (1/samplerate), '[s]', '1/T=', 1/(len(data3) * (1/samplerate)), '[Hz]')

# ここからグラフ描画-------------------------------------
# フォントの種類とサイズを設定する。
plt.rcParams['font.size'] = 14
plt.rcParams['font.family'] = 'Times New Roman'

# 目盛を内側にする。
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

# グラフの上下左右に目盛線を付ける。
fig = plt.figure()
ax1 = fig.add_subplot(211)
ax1.yaxis.set_ticks_position('both')
ax1.xaxis.set_ticks_position('both')
ax2 = fig.add_subplot(212)
ax2.yaxis.set_ticks_position('both')
ax2.xaxis.set_ticks_position('both')

# 軸のラベルを設定する。
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Amplitude')
ax2.set_xlabel('Frequency [Hz]')
ax2.set_ylabel('Amplitude')

# スケールの設定をする。
ax2.set_xticks(np.arange(0, 5, 0.5))
ax2.set_xlim(0, 3)
ax2.set_yticks(np.arange(0, 10, 0.2))
ax2.set_ylim(0, 1)

# データプロットの準備とともに、ラベルと線の太さ、凡例の設置を行う。
ax1.plot(x1, data1, label='data1', lw=1)
ax1.plot(x2, data2, label='data2', lw=1)
ax1.plot(x3, data3, label='data3', lw=1)

ax2.plot(freq1, amp1, label='data1', lw=1, marker='o')
ax2.plot(freq2, amp2, label='data2', lw=1, marker='o')
ax2.plot(freq3, amp3, label='data3', lw=1, marker='o')

# レイアウト設定
fig.tight_layout()
plt.legend()

# グラフを表示する。
plt.show()
plt.close()
# ---------------------------------------------------
