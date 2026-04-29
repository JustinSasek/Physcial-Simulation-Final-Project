#!/usr/bin/env python3
import argparse
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import json
from copy import deepcopy


def render_iteration(cli, iter_json: Path, tmp_out: Path, device: str = "cpu"):
    cmd = [
        cli,
        "-m",
        "pygoo.app",
        "render-video",
        "--json",
        str(iter_json),
        "--output",
        str(tmp_out),
        "--device",
        device,
    ]
    try:
        subprocess.check_call(cmd)
        return True, None
    except subprocess.CalledProcessError as e:
        return False, str(e)
    except Exception as e:
        return False, str(e)


def extract_frames(video_path: Path, count: int = 5, zoom: float = 1.0):
    reader = imageio.get_reader(str(video_path))
    frames = list(reader.iter_data())
    reader.close()
    indices = np.linspace(0, len(frames) - 1, count, dtype=int)
    sampled = [frames[i] for i in indices]

    if zoom <= 1.0:
        return sampled

    zoomed = []
    for frame in sampled:
        h, w = frame.shape[:2]
        crop_h, crop_w = h / zoom, w / zoom
        top = int((h - crop_h) / 2)
        left = int((w - crop_w) / 2)
        cropped = frame[top : top + int(crop_h), left : left + int(crop_w)]
        resized = np.array(Image.fromarray(cropped).resize((w, h), Image.LANCZOS))
        zoomed.append(resized)
    return zoomed


def make_grid_image(target_frames, pred_frames, diff_frames, iter_idx, out_path: Path):
    # assume all frames same size
    h, w, _ = target_frames[0].shape
    cols = len(target_frames)
    rows = 3
    pad = 8
    label_h = 24
    canvas_w = cols * w + (cols + 1) * pad
    canvas_h = rows * (h + label_h) + (rows + 1) * pad
    canvas = Image.new("RGB", (canvas_w, canvas_h), (40, 40, 40))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    # place images
    for col in range(cols):
        x = pad + col * (w + pad)
        # column label
        draw.text((x, 2), f"t={col}", fill=(255, 255, 255), font=font)
        # target
        tgt = Image.fromarray(target_frames[col])
        canvas.paste(tgt, (x, pad + label_h))
        # pred
        pred = Image.fromarray(pred_frames[col])
        canvas.paste(pred, (x, pad + label_h + h + pad))
        # diff
        diff = Image.fromarray(diff_frames[col])
        canvas.paste(diff, (x, pad + label_h + (h + pad) * 2))
    # row labels
    draw.text((4, pad + label_h + h // 2), "Target", fill=(255, 255, 255), font=font)
    draw.text(
        (4, pad + label_h + h + pad + h // 2),
        "Predicted",
        fill=(255, 255, 255),
        font=font,
    )
    draw.text(
        (4, pad + label_h + (h + pad) * 2 + h // 2),
        "Diff",
        fill=(255, 255, 255),
        font=font,
    )
    # title
    draw.text(
        (canvas_w // 2 - 40, 2),
        f"Iteration {iter_idx}",
        fill=(255, 255, 255),
        font=font,
    )
    canvas.save(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv_path", required=True, help="Path to optimization.csv from training logs"
    )
    ap.add_argument("--output", required=True, help="Output timelapse video path")
    ap.add_argument(
        "--target-video", required=True, help="Ground-truth target video path"
    )
    ap.add_argument(
        "--target-json", required=True, help="Ground-truth target json path"
    )
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    ap.add_argument(
        "--zoom", type=float, default=1.0, help="Zoom factor for grid images"
    )
    args = ap.parse_args()

    # load json
    with open(args.target_json) as f:
        target_json = json.load(f)

    df = pd.read_csv(args.csv_path)
    to_set = set(df.columns) - {"iteration", "loss"}
    to_set = {c for c in to_set if not c.startswith("gt.")}

    print("Parameters to set from CSV:", to_set)

    logs_dir = Path(args.output).parent / "frames"

    for i, row in df.iloc[1:].iterrows():
        new_json = deepcopy(target_json)
        for c in to_set:
            val = row[c]
            # skip missing values
            if pd.isna(val):
                continue
            # convert to native python number where possible
            try:
                if isinstance(val, (int, float, np.floating, np.integer)):
                    v = float(val)
                else:
                    v = float(str(val))
            except Exception:
                v = val

            # handle edge constants
            if c.startswith("edge."):
                parts = c.split(".")
                if len(parts) >= 2:
                    idx = int(parts[1])
                    edge_k = new_json["state"].setdefault("edge_k", [])
                    if idx >= len(edge_k):
                        edge_k.extend([None] * (idx - len(edge_k) + 1))
                    edge_k[idx] = v

            # handle particle attributes like particle.0.mass
            elif c.startswith("particle."):
                parts = c.split(".")
                if len(parts) >= 3:
                    try:
                        idx = int(parts[1])
                    except Exception:
                        continue
                    attr = parts[2]
                    if attr == "mass":
                        mass_list = new_json["state"].setdefault("mass", [])
                        if idx >= len(mass_list):
                            mass_list.extend([None] * (idx - len(mass_list) + 1))
                        mass_list[idx] = v
                    else:
                        # place other particle attributes under state.particles
                        particles = new_json["state"].setdefault("particles", [])
                        if idx >= len(particles):
                            particles.extend([{}] * (idx - len(particles) + 1))
                        particles[idx][attr] = v

            # legacy mass.N columns
            elif c.startswith("mass."):
                idx = int(c.split(".")[1])
                mass_list = new_json["state"].setdefault("mass", [])
                if idx >= len(mass_list):
                    mass_list.extend([None] * (idx - len(mass_list) + 1))
                mass_list[idx] = v

            else:
                # config keys may contain dots; keep the original column name
                try:
                    new_json["config"][c] = v
                except Exception:
                    new_json["config"][c] = row[c]

        out_json_path = logs_dir / f"{i}.json"
        out_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json_path, "w") as f:
            json.dump(new_json, f, indent=2)

    if not logs_dir.exists():
        print("logs dir not found", logs_dir)
        sys.exit(2)
    jsons = [p for p in logs_dir.iterdir() if p.suffix == ".json"]
    jsons = sorted(jsons, key=lambda p: int(p.stem))
    if not jsons:
        print("No json files in logs dir")
        sys.exit(2)

    target_vid = Path(args.target_video)
    if not target_vid.exists():
        print("Target video not found", target_vid)
        sys.exit(2)

    tmp_dir = Path(tempfile.mkdtemp(prefix="timelapse_tmp_"))
    os.makedirs(tmp_dir, exist_ok=True)
    grid_images = []

    for idx, j in enumerate(jsons):
        try:
            tmp_video = tmp_dir / f"iter_{idx:04d}.avi"
            ok, err = render_iteration(sys.executable, j, tmp_video, device=args.device)
            if not ok:
                print(f"Render failed for {j}: {err}")
                continue
            pred_frames = extract_frames(tmp_video, count=5, zoom=args.zoom)
            target_frames = extract_frames(target_vid, count=5, zoom=args.zoom)
            if len(pred_frames) < 5 or len(target_frames) < 5:
                print(f"Insufficient frames for iteration {idx}, skipping")
                continue
            # ensure uint8
            pred_frames = [f.astype("uint8") for f in pred_frames]
            target_frames = [f.astype("uint8") for f in target_frames]
            diff_frames = [
                (
                    target_frames[i].astype(int) / 2
                    - pred_frames[i].astype(int) / 2
                    + 128
                )
                .clip(0, 255)
                .astype("uint8")
                for i in range(5)
            ]
            grid_out = tmp_dir / f"grid_{idx:04d}.png"
            make_grid_image(target_frames, pred_frames, diff_frames, idx, grid_out)
            grid_images.append(grid_out)
            print(f"Created grid image for iteration {idx}")
        except Exception as e:
            print(f"Error processing iteration {j}: {e}")
            continue

    # write timelapse video at 3 fps
    writer = imageio.get_writer(args.output, fps=3)
    for img in grid_images:
        arr = imageio.imread(str(img))
        writer.append_data(arr)
    writer.close()
    print(f"Timelapse written to {args.output}")
    # cleanup temp
    for f in tmp_dir.iterdir():
        f.unlink()
    tmp_dir.rmdir()


if __name__ == "__main__":
    main()
