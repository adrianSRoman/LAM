"""
infer_visualize.py — LAM inference with acoustic map visualization.

Runs the LAM model on an audio file (via the same config format as infer.py)
and saves one acoustic-map PNG per time frame (default: 10 ms) into an output
directory.  The frame duration matches the T_sti parameter of
get_visibility_matrix (default 10 ms).  Override via "T_sti_ms" in the config.

Usage:
    python infer_visualize.py -C <config.json> [-D <gpu_id>] [--per-band]

Modes
-----
Default (combined):  one PNG per time frame, all frequency bands collapsed to
                     a single RGB image via to_RGB().

Per-band (--per-band):  one PNG per frame × band, saved as
                        <clip>/bands/band<bb>/frame_<NN>_<ttt>ms_band<bb>.png
                        Each map shows a single band's intensity as a
                        greyscale acoustic map.

The config JSON should follow the same schema as the one used by infer.py:
    {
        "output_dir":  "<path to write PNG files>",
        "model_path":  "<path to .pth checkpoint>",
        "FS":          24000,
        "n_max":       3,
        "per_band":    false,        // optional: enable per-band mode from config
        "model": { "module": "...", "main": "...", "args": {} },
        "dataset": { "module": "...", "main": "...", "args": {} }
    }
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")          # non-interactive backend, safe for headless servers
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from util.utils import initialize_config, load_checkpoint
from dataset.gen_dataset.gen_dataset import get_visibility_matrix
from trainer.utils import draw_map, get_field, to_RGB

# ── CLI ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser("LAM: acoustic map visualisation")
parser.add_argument("-C", "--config", type=str, required=True,
                    help="Config JSON (same schema as infer.py).")
parser.add_argument("-D", "--device", default="0", type=str,
                    help="GPU index to use (default: 0). Use 'cpu' to force CPU.")
parser.add_argument("-B", "--per-band", action="store_true", default=False,
                    help="Save one map per frequency band instead of a combined RGB image.")
args = parser.parse_args()

# ── Config & paths ───────────────────────────────────────────────────────────

config = json.load(open(args.config))
output_dir = config["output_dir"]
os.makedirs(output_dir, exist_ok=True)

# CLI flag takes precedence; fall back to config key; default False
per_band: bool = args.per_band or bool(config.get("per_band", False))

model_checkpoint_path = config["model_path"]
assert os.path.exists(model_checkpoint_path), \
    f"Checkpoint not found: {model_checkpoint_path}"

# ── Device ───────────────────────────────────────────────────────────────────

if args.device.lower() == "cpu":
    device = torch.device("cpu")
else:
    device = (torch.device(f"cuda:{args.device}")
              if torch.cuda.is_available()
              else torch.device("cpu"))

# ── DataLoader ───────────────────────────────────────────────────────────────

dataloader = DataLoader(
    dataset=initialize_config(config["dataset"]),
    batch_size=1,
    num_workers=4,
)

# ── Model ────────────────────────────────────────────────────────────────────

model = initialize_config(config["model"])
model.load_state_dict(load_checkpoint(model_checkpoint_path, device))
model.to(device)
model.eval()

# ── Shared visualisation resources ───────────────────────────────────────────

R_field = get_field()                          # (3, N_px) Fibonacci grid
lon_ticks = np.linspace(-180, 180, 5)

# ── Inference & visualisation loop ────────────────────────────────────────────

with torch.no_grad():
    for audio, name in tqdm(dataloader):
        assert len(name) == 1, "Only batch_size=1 is supported."
        name = name[0]

        audio = audio.cpu().detach().numpy()[0].T   # (samples, channels)

        # For the 32-channel LAM model, down-select to 4 Eigenmike capsules for UpLAM
        if config["model"].get("main") == "UpLAM":
            audio = audio[:, [5, 9, 25, 21]]

        # Build visibility matrix  ->  S_in: (n_bands, time_frames, N_ch, N_ch)
        S_in, _ = get_visibility_matrix(audio, fs=config["FS"], nbands=10, apgd=False)
        print(f"{name}: S_in shape = {S_in.shape} (bands, frames, N_ch, N_ch)")
        S_in = torch.from_numpy(S_in).to(device).permute(1, 0, 2, 3)

        # forward pass  -> I_pred: (time_frames, n_bands, N_px)
        _, I_pred = model(S_in)
        # shape: (time_frames, n_bands, N_px)
        I_pred_np = I_pred.cpu().detach().numpy()

        n_frames, n_bands, N_px = I_pred_np.shape
        print(f"{name}: {n_frames} frames × {n_bands} bands × {N_px} pixels")

        # Create a per-clip sub-directory to keep PNG files organised
        clip_dir = os.path.join(output_dir, name)
        os.makedirs(clip_dir, exist_ok=True)

        # T_sti defaults to 10 ms in get_visibility_matrix
        T_sti_ms = config.get("T_sti_ms", 10)   # ms per frame (override via config if needed)

        for i in range(n_frames):
            frame_bands  = I_pred_np[i]       # (n_bands, N_px)
            timestamp_ms = i * T_sti_ms

            if per_band:
                # ── Per-band mode: one PNG per band ─────────────────────────
                for b in range(n_bands):
                    band_intensity = frame_bands[b].copy()   # (N_px,)
                    b_max = band_intensity.max()
                    if b_max > 0:
                        band_intensity /= b_max
                    # draw_map expects (3, N_px); tile as greyscale
                    band_rgb = np.tile(band_intensity[np.newaxis, :], (3, 1))

                    band_dir = os.path.join(clip_dir, "bands", f"band{b:02d}")
                    os.makedirs(band_dir, exist_ok=True)

                    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
                    draw_map(
                        band_rgb, R_field,
                        lon_ticks=lon_ticks,
                        catalog=None,
                        show_labels=True, show_axis=True,
                        fig=fig, ax=ax,
                        kmeans=False, gaussian_mixture=False,
                    )
                    ax.set_title(f"{name}  —  t = {timestamp_ms} ms  |  band {b}")
                    out_path = os.path.join(
                        band_dir,
                        f"frame_{i:04d}_{timestamp_ms:06d}ms_band{b:02d}.png"
                    )
                    fig.savefig(out_path, bbox_inches="tight", dpi=100)
                    plt.close(fig)

            else:
                # ── Combined RGB mode: collapse bands → single image ─────────
                frame_rgb = to_RGB(frame_bands)   # (3, N_px)
                max_val = frame_rgb.max()
                if max_val > 0:
                    frame_rgb /= max_val

                fig, ax = plt.subplots(1, 1, figsize=(10, 5))
                draw_map(
                    frame_rgb, R_field,
                    lon_ticks=lon_ticks,
                    catalog=None,
                    show_labels=True, show_axis=True,
                    fig=fig, ax=ax,
                    kmeans=False, gaussian_mixture=False,
                )
                ax.set_title(f"{name}  —  t = {timestamp_ms} ms")
                out_path = os.path.join(
                    clip_dir, f"frame_{i:04d}_{timestamp_ms:06d}ms.png"
                )
                fig.savefig(out_path, bbox_inches="tight", dpi=100)
                plt.close(fig)

        mode_str = f"per-band ({n_bands} bands/frame)" if per_band else "combined RGB"
        print(f"  → saved {n_frames} frames [{mode_str}] to {clip_dir}")
