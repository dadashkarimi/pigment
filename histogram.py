import numpy as np
import json
import cv2
import os

from skimage.io import imread


import numpy as np
import json
import cv2
import os

def load_wsi_rgb(path, max_dim=2048, prefer_openslide=True, auto_fix_rb=True):
    import pyvips

    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")

    # 1) Prefer OpenSlide loader for SVS (more consistent for WSI pyramids)
    img = None
    if prefer_openslide:
        try:
            img = pyvips.Image.openslideload(path)  # level=0 by default
        except Exception:
            img = None

    # 2) Fallback
    if img is None:
        img = pyvips.Image.new_from_file(path, access="sequential")

    # Downsample for preview/mapping
    w, h = img.width, img.height
    scale = max(w, h) / float(max_dim)
    if scale > 1:
        img = img.resize(1.0 / scale)

    # Ensure sRGB interpretation (important across files)
    try:
        img = img.colourspace("srgb")
    except Exception:
        pass

    # Drop alpha/extra bands safely
    if img.bands >= 3:
        img = img.extract_band(0, n=3)
    else:
        raise RuntimeError(f"Unexpected bands={img.bands} for {path}")

    img = img.cast("uchar")
    mem = img.write_to_memory()
    arr = np.frombuffer(mem, dtype=np.uint8).reshape(img.height, img.width, img.bands)

    # Optional heuristic: if R/B look swapped, fix it.
    # (Works surprisingly well on H&E-ish tissue: red channel tends to be higher than blue overall.)
    if auto_fix_rb:
        if arr[..., 0].mean() < arr[..., 2].mean():  # suspicious: "blue > red"
            arr = arr[..., ::-1]  # swap R<->B

    return arr


def get_histogram_mapping_smooth(source_channel, reference_channel, smoothing=0.1):
    # Use range=(0,256) so value 255 is included
    source_hist, _ = np.histogram(source_channel.ravel(), bins=256, range=(0, 256))
    reference_hist, _ = np.histogram(reference_channel.ravel(), bins=256, range=(0, 256))

    source_hist = source_hist.astype(np.float64) + 1e-7
    reference_hist = reference_hist.astype(np.float64) + 1e-7

    source_cdf = np.cumsum(source_hist)
    source_cdf /= source_cdf[-1]

    reference_cdf = np.cumsum(reference_hist)
    reference_cdf /= reference_cdf[-1]

    # Make reference_cdf strictly usable for interpolation by uniquing it
    ref_cdf_u, ref_idx = np.unique(reference_cdf, return_index=True)
    ref_vals = np.arange(256)[ref_idx]

    mapping_values = np.interp(source_cdf, ref_cdf_u, ref_vals)

    if smoothing > 0:
        for i in range(1, len(mapping_values)):
            mapping_values[i] = mapping_values[i - 1] + smoothing * (mapping_values[i] - mapping_values[i - 1])

    mapping_values = np.maximum.accumulate(mapping_values)
    mapping_values = np.clip(mapping_values, 0, 255).astype(np.uint8)
    return {i: int(mapping_values[i]) for i in range(256)}


def apply_histogram_mapping(image, mapping_dict):
    result = np.empty_like(image)
    for i, channel in enumerate(["R", "G", "B"]):
        lut = np.array([mapping_dict[channel][j] for j in range(256)], dtype=np.uint8)
        result[..., i] = lut[image[..., i]]
    return result


def save_preview(rgb_img, path):
    cv2.imwrite(path, cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))



def save_mapping_dict(mapping_dict, path):
    json_serializable_dict = {}
    for channel, mapping in mapping_dict.items():
        json_serializable_dict[channel] = {str(k): int(v) for k, v in mapping.items()}
    with open(path, 'w') as f:
        json.dump(json_serializable_dict, f, indent=2)


def create_histogram_mapping_dict_smooth(target_img, reference_img, smoothing=0.1):
    return {
        'R': get_histogram_mapping_smooth(target_img[..., 0], reference_img[..., 0], smoothing),
        'G': get_histogram_mapping_smooth(target_img[..., 1], reference_img[..., 1], smoothing),
        'B': get_histogram_mapping_smooth(target_img[..., 2], reference_img[..., 2], smoothing),
    }


def main():
    # Experiment 2
    ref_path = "data/2016-6-A1.svs"
    targets = [
        ("2016-5-A2", "data/2016-5-A2.svs"),
        ("2016-5-P2", "data/2016-5-P2.svs"),
        ("2017-1-A3", "data/2017-1-A3.svs"),
    ]

    out_dir = "exp2_mappings"
    os.makedirs(out_dir, exist_ok=True)

    smoothing_factor = 0.1

    print("[INFO] Loading reference:", ref_path)
    ref_img = load_wsi_rgb(ref_path)
    print("[INFO] Reference loaded:", ref_img.shape, ref_img.dtype)

    for name, target_path in targets:
        print(f"\n=== Processing target {name} ===")
        print("[INFO] Loading target:", target_path)
        tgt_img = load_wsi_rgb(target_path)
        print("[INFO] Target loaded:", tgt_img.shape, tgt_img.dtype)

        mapping_dict = create_histogram_mapping_dict_smooth(tgt_img, ref_img, smoothing_factor)

        mapping_save_path = os.path.join(
            out_dir,
            f"histogram_mapping_exp2_{name}_to_2016-6-A1_s{str(smoothing_factor).replace('.','p')}.json"
        )
        save_mapping_dict(mapping_dict, mapping_save_path)
        print(f"[OK] Saved mapping: {mapping_save_path}")

        normalized_img = apply_histogram_mapping(tgt_img, mapping_dict)
        preview_path = os.path.join(out_dir, f"normalized_{name}_to_2016-6-A1.jpg")
        cv2.imwrite(preview_path, cv2.cvtColor(normalized_img, cv2.COLOR_RGB2BGR))
        print(f"[OK] Saved preview: {preview_path}")

    print("\nDONE ✅ Send the 3 JSON mapping files in exp2_mappings/ to Pushkar.")


if __name__ == "__main__":
    main()
