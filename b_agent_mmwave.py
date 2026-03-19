import os
import re
from datetime import datetime
from a_mmwave_tool import create_radar, generate_mmwave_sample


class RadarAgent:

    def __init__(self):

        print("Radar Simulation Agent Ready")
        print(
            "Gesture type (push, pull, swipe), number of frames (samples), and targets can be specified."
        )
        print("Example prompts:")
        print("- generate 20 frames of pull gesture data with 2 targets")
        print('- input "exit" to quit')

    def parse_samples(self, text):

        match = re.search(r"(\d+).*?(?:samples?|frames?)", text)
        if match:
            return int(match.group(1))
        return 20 if any(g in text for g in ["push", "pull", "swipe"]) else 1

    def parse_targets(self, text):

        match = re.search(r"(\d+)\s*targets?", text)
        if match:
            return int(match.group(1))
        return 2

    def parse_gesture(self, text):
        # 识别手势类型
        if "push" in text:
            return "push"
        if "pull" in text:
            return "pull"
        if "swipe" in text:
            return "swipe"
        return "static"

    def run(self, prompt):

        prompt = prompt.lower()
        if (
            "mmwave" in prompt
            or "radar" in prompt
            or any(g in prompt for g in ["push", "pull", "swipe"])
        ):
            num_targets = self.parse_targets(prompt)
            num_frames = self.parse_samples(prompt)
            gesture_type = self.parse_gesture(prompt)

            print(
                f"Running radar simulation with {num_frames} frames and {num_targets} targets"
            )
            # establish directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            sequence_name = f"{gesture_type}_{timestamp}"

            dataset_dir = "dataset"
            os.makedirs(dataset_dir, exist_ok=True)
            save_dir = os.path.join(dataset_dir, sequence_name)
            os.makedirs(save_dir, exist_ok=True)

            print(f"\n--- Starting Gesture Generation ---")
            print(f"Type: {gesture_type.upper()}")
            print(f"Frames: {num_frames}")
            print(f"Targets per frame: {num_targets}")
            print(f"Saving to: {save_dir}\n")

            # Generate continuous frame data
            last_data = None
            radar = create_radar()
            for i in range(num_frames):
                print(f"Generating sample {i + 1}/{num_frames}")
                last_data = generate_mmwave_sample(
                    radar=radar,
                    dataset_dir=save_dir,
                    gesture_type=gesture_type,
                    num_targets=num_targets,
                    sample_idx=i,
                )

            print("Simulation finished")
            if last_data is not None:
                print("ADC shape:", last_data.shape)
            return last_data
        else:
            return "Please specify a gesture type (push, pull, or swipe) to simulate."
