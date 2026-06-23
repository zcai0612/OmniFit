import os
import json

lmk_indices_json_files = {
    "100": "data/lmk_indices/smplx_100_landmark.json",
    "300": "data/lmk_indices/smplx_300_landmark.json",
    "600": "data/lmk_indices/smplx_600_landmark.json",
    "900": "data/lmk_indices/smplx_900_landmark.json",
}

def get_smplx_landmarks_from_file(
    indices_json_path: str = "data/lmk_indices/smplx_1000_landmark.json"
):
    with open(indices_json_path, "r") as f:
        data = json.load(f)
    lmk_indices = data["smplx_landmark_indices"]
    return lmk_indices

def get_smplx_landmarks(
    num_lmks: int = 600,
):
    if str(num_lmks) not in lmk_indices_json_files:
        raise ValueError(f"Cannot find landmark indices file for {num_lmks} landmarks.")
    indices_json_path = lmk_indices_json_files[str(num_lmks)]
    return get_smplx_landmarks_from_file(indices_json_path)

