import numpy as np
import matplotlib.pyplot as plt


data = np.load("mmwave_adc.npy")

# 查看npy数据
print("=====View npy data========")
print(data.shape)
print(data)

# 画ADC信号
adc = data[0]  # 第一个通道
print("=====Draw ADC signal========")
plt.imshow(np.abs(adc), aspect="auto")
plt.title("ADC magnitude(Ch0)")
plt.xlabel("Sample Index")
plt.ylabel("Chirp Index")
plt.colorbar(label="Magnitude")
plt.tight_layout()
plt.show()

# 生成 Range Profile距离剖面图
print("===== Range Profile========")
range_fft = np.fft.fft(adc, axis=1)
range_profile = np.abs(range_fft[0])

threshold = np.max(range_profile) * 0.01
valid_indices = np.where(range_profile > threshold)[0]

if len(valid_indices) > 0:
    # 计算有效数据的边界，并留出10%的边距
    start_idx = max(0, valid_indices[0] - len(valid_indices) // 10)
    end_idx = min(len(range_profile), valid_indices[-1] + len(valid_indices) // 10)

    # 只绘制有信号的区域
    x_range = np.arange(start_idx, end_idx)
    plt.plot(x_range, range_profile[start_idx:end_idx], "b-", linewidth=1.5)
    plt.title(f"Range Profile (Display range: {start_idx}~{end_idx})")
    plt.xlabel("Range Bin")

else:
    # 如果没有有效数据，显示全部
    plt.plot(range_profile, "b-", linewidth=1.5)
    plt.title("Range Profile (Full range)")
    plt.xlabel("Range Bin")

plt.ylabel("Amplitude")
plt.grid(True, alpha=0.3)  # 添加网格线，设置透明度 0（完全透明）-1（完全不透明）
plt.tight_layout()
plt.show()

# 生成Range-Doppler Map距离多普勒图
print("===== Range-Doppler Map========")
range_fft = np.fft.fft(adc, axis=1)
doppler_fft = np.fft.fft(range_fft, axis=0)
rd_map = np.abs(doppler_fft)

plt.imshow(20 * np.log10(rd_map), aspect="auto", origin="lower")

plt.title("Range-Doppler Map")
plt.xlabel("Range Bin")
plt.ylabel("Doppler Bin")
plt.colorbar(label="Magnitude(dB)")
plt.tight_layout()
plt.show()
