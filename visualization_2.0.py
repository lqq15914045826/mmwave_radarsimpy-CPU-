import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime


def visualize_mmwave_dataset(dataset_dir, output_dir):
    """
    Batch process radar ADC data for quality inspection.
    - Generates a 3-panel figure for each sample (ADC, Range, Doppler).
    - Uses log-scaling/dB for better dynamic range visibility.
    - Saves results to disk to avoid manual UI interaction.
    """
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    all_files = []

    # Search for all .npy files in the target directory
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith(".npy"):
                all_files.append(os.path.join(root, file))

    total_files = len(all_files)
    print(f"Found {total_files} .npy files in {dataset_dir}")

    new_files_count = 0
    skipped_count = 0

    for i, file_path in enumerate(all_files):
        file_name = os.path.basename(file_path)
        parent_dir_name = os.path.basename(os.path.dirname(file_path))

        try:
            parts = file_name.split("_")
            if len(parts) >= 3:
                gesture_type = parts[0]
                date_str = parts[2]
            else:
                gesture_type = "unknown"
                date_str = "unknown"

            # filename = f"{gesture_type}_{date_str}"

            # output_dir/gesture_type/date_str/
            target_dir = os.path.join(output_dir, gesture_type, parent_dir_name)
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)

            save_path = os.path.join(target_dir, file_name.replace(".npy", ".png"))

            if os.path.exists(save_path):
                skipped_count += 1
                continue
            try:
                # Load data (assuming shape: [Channels, Chirps, Samples])
                data = np.load(file_path)
                # Use Channel 0 for visualization
                adc_data = data[0]

                # --- Signal Processing ---
                # 1. Range FFT (across samples)
                range_fft = np.fft.fft(adc_data, axis=1)

                # 2. Doppler FFT (across chirps)
                # Shift zero-frequency component to the center of the spectrum
                rd_map = np.fft.fftshift(np.fft.fft(range_fft, axis=0), axes=0)
                rd_db = 20 * np.log10(np.abs(rd_map) + 1e-6)  # Convert to dB

                # 3. Range Profile (Average across all chirps)
                avg_range_profile = np.mean(np.abs(range_fft), axis=0)

                # --- Plotting ---
                fig, axes = plt.subplots(1, 3, figsize=(20, 6))
                fig.suptitle(f"Radar Data Quality Report: {file_name}", fontsize=16)

                # Panel 1: ADC Time-Domain Magnitude
                im0 = axes[0].imshow(np.abs(adc_data), aspect="auto", cmap="viridis")
                axes[0].set_title("ADC Magnitude (Ch0)")
                axes[0].set_xlabel("Sample Index")
                axes[0].set_ylabel("Chirp Index")
                plt.colorbar(im0, ax=axes[0])

                # Panel 2: Range Profile (Log Scale)
                axes[1].plot(avg_range_profile)
                axes[1].set_title("Average Range Profile")
                axes[1].set_xlabel("Range Bin")
                axes[1].set_ylabel("Amplitude (Linear)")
                axes[1].set_yscale("log")  # Log scale is better for radar dynamic range
                axes[1].grid(True, which="both", ls="-", alpha=0.2)

                # Panel 3: Range-Doppler Map (dB Scale)
                im2 = axes[2].imshow(rd_db, aspect="auto", origin="lower", cmap="jet")
                axes[2].set_title("Range-Doppler Map (dB)")
                axes[2].set_xlabel("Range Bin")
                axes[2].set_ylabel("Doppler Bin (Centered)")
                plt.colorbar(im2, ax=axes[2], label="Magnitude (dB)")

                # Save and Close
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                plt.savefig(save_path, dpi=150)
                plt.close(fig)  # Critical: Free memory

                new_files_count += 1
                print(f"[{i+1}/{total_files}] Saved visualization to: {save_path}")

            except Exception as e:
                print(f"Failed to process {file_name}: {e}")
        except Exception as e:
            print(f"Error processing {file_name}: {e}")

    print("\nVisualization batch complete.")
    print(f"Already existed: {skipped_count}")
    print(f"Newly visualized: {new_files_count}")
    print(f"Total files checked: {total_files}")


if __name__ == "__main__":
    # Example usage:
    visualize_mmwave_dataset(dataset_dir="dataset", output_dir="vis_reports")
