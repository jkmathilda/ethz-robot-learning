"""Augment multicube dataset via goal relabeling.

For each episode, we create all 6 permutations of (red, green, blue) cube
identities. The actions stay the same — only cube position labels and the
one-hot goal vector are permuted. This turns N episodes into 6N episodes
without any additional teleoperation.

Usage:
python scripts/augment_multicube.py --input datasets/processed/multi_cube/processed_ee_xyz.zarr --output datasets/processed/multi_cube/processed_ee_xyz_augmented.zarr
"""

from __future__ import annotations

import argparse
from itertools import permutations
from pathlib import Path

import numpy as np
import zarr


def main() -> None:
    parser = argparse.ArgumentParser(description="Augment multicube zarr via goal relabeling.")
    parser.add_argument("--input", type=Path, required=True, help="Input processed zarr.")
    parser.add_argument("--output", type=Path, required=True, help="Output augmented zarr.")
    args = parser.parse_args()

    # Load input zarr
    root = zarr.open_group(str(args.input), mode="r")
    data = root["data"]
    meta = root["meta"]

    ep_ends = np.asarray(meta["episode_ends"][:], dtype=np.int64)

    # Load all data arrays
    arrays = {}
    for key in data:
        arrays[key] = np.asarray(data[key][:])

    # The three cube position arrays and goal one-hot
    cube_keys = ["original_pos_cube_red", "original_pos_cube_green", "original_pos_cube_blue"]
    # Verify they exist
    for k in cube_keys:
        assert k in arrays, f"Missing key {k} in zarr — is this a multicube dataset?"
    assert "state_goal" in arrays, "Missing state_goal in zarr"

    # All 6 permutations of (red, green, blue) indices
    # Original order: [0=red, 1=green, 2=blue]
    perms = list(permutations([0, 1, 2]))
    print(f"Generating {len(perms)} permutations (including original)")
    print(f"Original: {len(ep_ends)} episodes, {arrays[cube_keys[0]].shape[0]} transitions")

    # Build augmented arrays
    aug_arrays = {k: [] for k in arrays}
    aug_ep_ends = []
    offset = 0

    # Episode ranges
    starts = np.concatenate([[0], ep_ends[:-1]])

    for perm in perms:
        # perm is a tuple like (0, 1, 2) or (1, 0, 2) etc.
        # perm[i] = which original cube goes into position i
        # e.g. perm=(1, 0, 2) means: red_slot gets green_data, green_slot gets red_data, blue stays

        for key in arrays:
            if key == cube_keys[0]:
                # Red slot gets data from cube_keys[perm[0]]
                aug_arrays[key].append(arrays[cube_keys[perm[0]]])
            elif key == cube_keys[1]:
                # Green slot gets data from cube_keys[perm[1]]
                aug_arrays[key].append(arrays[cube_keys[perm[1]]])
            elif key == cube_keys[2]:
                # Blue slot gets data from cube_keys[perm[2]]
                aug_arrays[key].append(arrays[cube_keys[perm[2]]])
            elif key == "state_goal":
                # perm[i] = which original cube is now in slot i
                # So original cube j is now in slot where perm[slot]=j
                # For one-hot goal: if target was cube j, after perm the
                # target cube sits in the slot found by indexing with perm.
                # goal[:, perm] reorders columns so the 1 moves to the
                # correct new slot.
                goal = arrays["state_goal"]
                new_goal = goal[:, list(perm)]
                aug_arrays[key].append(new_goal)
            else:
                # All other arrays (state_ee_xyz, actions, etc.) stay the same
                aug_arrays[key].append(arrays[key])

        # Shift episode ends
        n_transitions = arrays[cube_keys[0]].shape[0]
        aug_ep_ends.append(ep_ends + offset)
        offset += n_transitions

    # Concatenate
    for key in aug_arrays:
        aug_arrays[key] = np.concatenate(aug_arrays[key], axis=0)

    aug_ep_ends = np.concatenate(aug_ep_ends)

    n_new = aug_arrays[cube_keys[0]].shape[0]
    print(f"Augmented: {len(aug_ep_ends)} episodes, {n_new} transitions")

    # Write output zarr
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_root = zarr.open_group(str(args.output), mode="w", zarr_format=3)
    compressor = zarr.codecs.Blosc(cname="zstd", clevel=3, shuffle=2)
    compressors = (compressor,)

    out_data = out_root.require_group("data")
    out_meta = out_root.require_group("meta")

    for key, arr in aug_arrays.items():
        out_data.create_array(key, data=arr.astype(np.float32), compressors=compressors)

    out_meta.create_array("episode_ends", data=aug_ep_ends.astype(np.int64), compressors=compressors)

    # Copy metadata from original
    for attr_key in root.attrs:
        out_root.attrs[attr_key] = root.attrs[attr_key]
    out_root.attrs["num_episodes"] = int(len(aug_ep_ends))
    out_root.attrs["num_transitions"] = int(n_new)
    out_root.attrs["augmented"] = True
    out_root.attrs["augmentation"] = "goal_relabeling_6x"

    print(f"Written to {args.output}")


if __name__ == "__main__":
    main()
