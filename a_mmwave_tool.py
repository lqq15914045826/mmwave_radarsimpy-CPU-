import numpy as np
import radarsimpy as rsp
from datetime import datetime
import os
import csv

from radarsimpy.radar import Radar
from radarsimpy.transmitter import Transmitter
from radarsimpy.receiver import Receiver


def create_radar():

    fc = 77e9
    bandwidth = 3.5e9  # 3 GHz bandwidth for better range resolution
    chirp_time = 60e-6
    num_chirps = 128

    tx = Transmitter(
        f=[fc, fc + bandwidth],
        t=chirp_time,
        tx_power=20,
        prp=80e-6,
        pulses=num_chirps,
    )

    rx = Receiver(fs=10e6, noise_figure=2, rf_gain=20, baseband_gain=60)

    radar = Radar(transmitter=tx, receiver=rx)

    return radar


def create_targets(num_targets=2, frame_idx=0, gesture_type="push"):
    """
    gesture_type:
    - 'push': Hand extends towards the radar from a distance (0.8m) to (0.2m)
    - 'pull': Hand retracts from a distance (2m) to (5m)
    - 'swipe': Hand swings from left (-0.3m) to right (0.3m) from a distance of 0.5m.
    """
    targets = []
    target_params = []  # To store parameters for the CSV

    dt = 0.05  # 20 FPS
    t = frame_idx * dt

    jitter = np.random.normal(0, 0.005)  # 5mm jitter to add realism

    if gesture_type == "push":
        start_dist = 1.2
        v_val = -1.2
        center_x = max(0.2, start_dist + v_val * t)
        center_y = 0.0 + jitter
        v_main = [v_val, 0, 0]

    elif gesture_type == "pull":
        start_dist = 0.2
        v_val = 1.5
        center_x = min(1.8, start_dist + v_val * t)
        center_y = 0.0 + jitter
        v_main = [v_val, 0, 0]

    elif gesture_type == "swipe":
        center_x = 0.8 + jitter
        dist_variation = 0.1 * np.cos(np.pi * (t - 0.25))
        center_x += dist_variation

        center_y = -0.4 + 1.2 * t
        v_main = [-0.3, 1.2, 0]

    else:
        center_x, center_y = 0.6, 0.0
        v_main = [0, 0, 0]

    for i in range(num_targets):
        if i == 0:
            loc = [center_x, center_y, 0.2]
            vel = v_main
            rcs = 1.5
        else:
            micro_oscillation = 0.05 * np.sin(2 * np.pi * 8 * t)
            loc = [
                center_x + 0.08,  # 距离比手掌远 8cm
                center_y + 0.1,
                0.15 + micro_oscillation,
            ]
            vel = [v * 1.25 for v in v_main]  # 速度稍微快一点，增加多普勒扩展
            rcs = 0.6

        """
        # Range: 2m to 15m; Cross-range: -4m to 4m
        random_x = np.random.uniform(2.0, 15.0)
        random_y = np.random.uniform(-4.0, 4.0)
        random_speed = np.random.uniform(-5.0, 5.0)  # -5 to 5 m/s
        rand_rcs = np.random.uniform(0.1, 1.5)
        """
        target = {
            "location": loc,
            "speed": vel,
            "rcs": rcs,
        }

        targets.append(target)
        target_params.append(
            {
                "location": loc,
                "speed": vel,
                "rcs": rcs,
            }
        )
        # params_list.extend([loc[0], loc[1], v_vector[0], v_vector[1], targets[-1]["rcs"]])
    return targets, target_params


def generate_mmwave_sample(
    radar, dataset_dir, gesture_type, num_targets=2, sample_idx=0
):

    # radar = create_radar()
    targets, params = create_targets(
        num_targets, frame_idx=sample_idx, gesture_type=gesture_type
    )
    data = rsp.sim_radar(radar, targets)
    baseband = data["baseband"]
    baseband = baseband - np.mean(baseband, axis=1, keepdims=True)  # DC offset removal
    noise = (
        np.random.randn(*baseband.shape) + 1j * np.random.randn(*baseband.shape)
    ) * 1e-4
    baseband += noise

    flat_params = []
    for t in params:
        flat_params += t["location"]
        flat_params += t["speed"]
        flat_params.append(t["rcs"])

    # 控制小数位
    flat_params = [
        round(p, 4) if isinstance(p, (float, np.floating)) else p for p in flat_params
    ]

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    date_str = datetime.now().strftime("%Y-%m-%d")
    filename = f"{gesture_type}_mmwave_{timestamp}_{sample_idx:04d}.npy"
    filepath = os.path.join(dataset_dir, filename)
    np.save(filepath, baseband)

    # --- CSV Logging Logic ---
    log_filename = f"radar_log_{date_str}.csv"
    log_path = os.path.join(dataset_dir, log_filename)

    file_exists = os.path.isfile(log_path)

    # Write to CSV
    with open(log_path, mode="a", newline="") as f:  # append mode
        writer = csv.writer(f)
        # Create header if file is new
        if not file_exists:
            header = ["filename"]
            for i in range(num_targets):
                header += [
                    f"t{i}_x",
                    f"t{i}_y",
                    f"t{i}_z",
                    f"t{i}_vx",
                    f"t{i}_vy",
                    f"t{i}_vz",
                    f"t{i}_rcs",
                ]
            writer.writerow(header)

        # Write data row
        writer.writerow([filename] + flat_params)

    return baseband
