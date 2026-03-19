from fileinput import filename

import numpy as np
import radarsimpy as rsp
from datetime import datetime

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


def create_targets():
    """
    手势用散射点表示
    """
    targets = []

    for i in range(2):
        target = {"location": [0.5, 0.6 * i, 0.2], "speed": [0.2, 0, 0], "rcs": 1}
        targets.append(target)

    return targets


def main():

    print("Creating radar")
    radar = create_radar()

    print("Radar created successfully")

    print("Creating target")
    targets = create_targets()

    print("Running simulation")

    data = rsp.sim_radar(radar, targets)

    print("Simulation finished")
    baseband = data["baseband"]

    print("ADC data shape:", baseband.shape)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    filename = f"mmwave_adc_{timestamp}.npy"

    np.save(filename, baseband)

    print("Saved to", filename)

    """
    print("Channels:", radar.num_channels)
    print("Samples per pulse:", radar.samples_per_pulse)
    print("Timestamp shape:", radar.time_prop["timestamp"].shape)
    """


if __name__ == "__main__":
    main()
