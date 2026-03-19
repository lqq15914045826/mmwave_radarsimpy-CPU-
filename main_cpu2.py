import numpy as np
import radarsimpy as rsp
import matplotlib.pyplot as plt

from radarsimpy.processing import range_fft
from radarsimpy.radar import Radar
from radarsimpy.transmitter import Transmitter
from radarsimpy.receiver import Receiver


def create_radar():

    fc = 77e9
    bandwidth = 1e9
    chirp_time = 60e-6
    num_chirps = 128

    tx = Transmitter(
        f=[fc, fc + bandwidth],
        t=chirp_time,
        tx_power=10,
        prp=chirp_time,
        pulses=num_chirps,
    )
    rx = Receiver(fs=10e6, noise_figure=5, rf_gain=20, baseband_gain=30)

    radar = Radar(transmitter=tx, receiver=rx)
    return radar


def create_hand_scatterers(frame_id, total_frames, fps=20):
    """
    完整的手势散射点模型：包含手掌中心和五根手指
    """
    scatterers = []
    base_x = 0.5
    base_z = 0.2
    amplitude = 0.05

    # 手势运动轨迹 (假设 y 轴摆动)
    amplitude = 0.05
    angular_freq = 2 * np.pi / total_frames
    # 当前位置
    pos_y = amplitude * np.sin(angular_freq * frame_id)
    # 瞬时速度 v = dy/dt (假设帧率为 fps)
    # 导数: A * w * cos(w * t) * 帧率转换
    vel_y = amplitude * angular_freq * np.cos(angular_freq * frame_id) * fps

    palm = [base_x, pos_y, base_z]
    # 注意：这里把瞬时速度加进去了！
    scatterers.append({"location": palm, "speed": [0, vel_y, 0], "rcs": 1})

    # 五个手指
    for i in range(5):

        finger = [base_x + 0.02 * i, palm[1] + 0.02 * (i - 2), base_z + 0.03]

        scatterers.append({"location": finger, "speed": [0, 0, 0], "rcs": 0.5})

    return scatterers


def compute_range_doppler(adc):

    range_fft = np.fft.fft(adc, axis=1)
    doppler_fft = np.fft.fft(range_fft, axis=0)
    rd_map = np.abs(doppler_fft)
    return rd_map


def generate_dataset():
    radar = create_radar()
    total_frames = 100
    adc_dataset = []
    for frame_id in range(total_frames):
        print(f"Generating frame {frame_id + 1}/{total_frames}")
        targets = create_hand_scatterers(frame_id, total_frames)
        data = rsp.sim_radar(radar, targets)
        adc = data["baseband"][0]  # 形状应为 (num_chirps, num_samples)
        adc_dataset.append(adc)

    adc_dataset = np.array(adc_dataset)
    print("ADC Dataset shape:", adc_dataset.shape)
    np.save("mmwave_adc_dataset.npy", adc_dataset)
    print("Dataset saved")


def visualize_example():
    data = np.load("mmwave_adc_dataset.npy")
    rd_map = data[10]  # 取第10帧

    num_chirps, num_samples = adc.shape
    print("===== Processing & Visualization ========")
    # 1. Range FFT (加汉宁窗)
    window_range = np.hanning(num_samples)
    adc_windowed = adc * window_range
    range_fft = np.fft.fft(adc_windowed, axis=1)

    # 绘制距离像 (取第一个 Chirp)
    range_profile = np.abs(range_fft[0])
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(range_profile, "b-")
    plt.title("Range Profile")
    plt.xlabel("Range Bin")
    plt.ylabel("Magnitude")
    plt.grid(True, alpha=0.3)

    # 2. Doppler FFT (加汉宁窗 + 移频)
    window_doppler = np.hanning(num_chirps).reshape(-1, 1)
    range_fft_windowed = range_fft * window_doppler
    doppler_fft = np.fft.fft(range_fft_windowed, axis=0)

    # 关键：将 0 频移到中心
    doppler_fft_shifted = np.fft.fftshift(doppler_fft, axes=0)
    rd_map = np.abs(doppler_fft_shifted)

    plt.subplot(1, 2, 2)
    # 使用 20*log10 转换为 dB 显示
    plt.imshow(20 * np.log10(rd_map + 1e-6), aspect="auto", origin="lower", cmap="jet")
    plt.title("Range-Doppler Map")
    plt.xlabel("Range Bin")
    plt.ylabel("Doppler Bin")
    # 设置 Y 轴刻度，让中心显示为 0
    plt.yticks([0, num_chirps // 2, num_chirps - 1], ["-Max V", "0", "+Max V"])
    plt.colorbar(label="Magnitude (dB)")

    plt.tight_layout()
    plt.show()

    """
    print("===== Range Profile========")
    range_fft = np.fft.fft(rd_map, axis=1)
    range_profile = np.abs(range_fft[0])
    plt.plot(range_profile, "b-", linewidth=1.5)
    plt.title("Range Profile (Full range)")
    plt.xlabel("Range Bin")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)  # 添加网格线，设置透明度 0（完全透明）-1（完全不透明）
    plt.tight_layout()
    plt.show()

    print("===== Range-Doppler Map========")
    range_fft = np.fft.fft(rd_map, axis=1)
    doppler_fft = np.fft.fft(range_fft, axis=0)
    rd_map = np.abs(doppler_fft)

    plt.imshow(20 * np.log10(rd_map + 1e-6), aspect="auto", origin="lower")

    plt.title("Range-Doppler Map")
    plt.xlabel("Range Bin")
    plt.ylabel("Doppler Bin")
    plt.colorbar(label="Magnitude(dB)")
    plt.tight_layout()
    plt.show()
    """


if __name__ == "__main__":
    generate_dataset()
    visualize_example()
