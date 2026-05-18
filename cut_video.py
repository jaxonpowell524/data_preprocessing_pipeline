import os
import csv
import json
import argparse
import subprocess
import sys
from dataclasses import dataclass, asdict
from typing import List, Optional
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
import ffmpeg
from scipy.signal import savgol_filter

# Terminal line to run with a dir of videos:
# python cut_video.py --inputs ./{directory name}/*.MOV --model pose_landmarker_full.task --outdir results --save-landmarks-csv --cut-clips --review-clips
# python cut_video.py --inputs ./your_folder/*.MOV --model pose_landmarker_full.task --outdir results --save-landmarks-csv --cut-clips --review-clips


@dataclass
class Segment:
    start_sec: float
    end_sec: float
    peak_score: float


def convert_mov_to_mp4(input_video: str) -> str:
    input_path = Path(input_video).resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input video does not exist: {input_path}")

    if input_path.suffix.lower() != ".mov":
        return str(input_path)

    output_video = str(input_path.with_suffix(".mp4"))

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(input_path),
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "22",
        "-c:a", "aac",
        "-b:a", "192k",
        "-movflags", "+faststart",
        output_video,
    ]
    subprocess.run(cmd, check=True)

    if not Path(output_video).exists():
        raise RuntimeError(f"Conversion failed: {output_video}")

    return output_video


def robust_stats(x: np.ndarray):
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med))) + 1e-6
    return med, mad


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x.copy()
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(x, (pad_left, pad_right), mode="edge")
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(padded, kernel, mode="valid")


def median_filter_1d(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x.copy()
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = np.pad(x, (pad, pad), mode="edge")
    out = np.empty_like(x)
    for i in range(len(x)):
        out[i] = np.median(padded[i:i + window])
    return out


def torso_scale(points_xy: np.ndarray) -> float:
    ls = points_xy[11]
    rs = points_xy[12]
    lh = points_xy[23]
    rh = points_xy[24]

    shoulder_width = np.linalg.norm(ls - rs)
    hip_width = np.linalg.norm(lh - rh)
    return max(0.5 * (shoulder_width + hip_width), 1e-3)


def weighted_motion_score(curr_xyz: np.ndarray, prev_xyz: np.ndarray) -> float:
    if curr_xyz is None or prev_xyz is None:
        return 0.0

    scale = torso_scale(curr_xyz[:, :2])
    if scale < 1e-6:
        return 0.0

    weights = {
        15: 3.0,
        16: 3.0,
        13: 2.0,
        14: 2.0,
        11: 1.8,
        12: 1.8,
        23: 1.2,
        24: 1.2,
        0: 0.6,
    }

    total = 0.0
    denom = 0.0
    for idx, w in weights.items():
        d = np.linalg.norm(curr_xyz[idx, :2] - prev_xyz[idx, :2]) / scale
        total += w * d
        denom += w

    return total / max(denom, 1e-6)


def apply_savgol_to_landmarks(
    coords: np.ndarray,
    valid_mask: np.ndarray,
    window_length: int = 11,
    polyorder: int = 3
) -> np.ndarray:
    smoothed = coords.copy()
    T = coords.shape[0]

    if T < 3:
        return smoothed

    wl = window_length
    if wl > T:
        wl = T if T % 2 == 1 else T - 1
    if wl < 3:
        return smoothed
    if wl % 2 == 0:
        wl -= 1
    po = min(polyorder, wl - 1)

    for lm_idx in range(33):
        for axis in range(3):
            arr = coords[:, lm_idx, axis].astype(np.float64).copy()
            arr[~valid_mask] = np.nan

            if np.all(np.isnan(arr)):
                continue

            idx = np.arange(T)
            good = ~np.isnan(arr)
            arr = np.interp(idx, idx[good], arr[good])
            arr = savgol_filter(arr, window_length=wl, polyorder=po, axis=0, mode="interp")
            smoothed[:, lm_idx, axis] = arr

    return smoothed


def save_segments_json(segments: List[Segment], output_path: str):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump([asdict(s) for s in segments], f, indent=2)


def save_segments_csv(segments: List[Segment], output_path: str):
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["segment_id", "start_sec", "end_sec", "duration_sec", "peak_score"])
        for i, seg in enumerate(segments, start=1):
            writer.writerow([i, seg.start_sec, seg.end_sec, seg.end_sec - seg.start_sec, seg.peak_score])


def save_landmarks_csv(
    times: np.ndarray,
    raw_coords: np.ndarray,
    smooth_coords: np.ndarray,
    valid_mask: np.ndarray,
    output_path: str
):
    header = ["time_sec", "valid"]
    for i in range(33):
        header += [f"x_{i}", f"y_{i}", f"z_{i}"]
    for i in range(33):
        header += [f"sx_{i}", f"sy_{i}", f"sz_{i}"]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for t in range(len(times)):
            row = [float(times[t]), int(valid_mask[t])]
            for i in range(33):
                row += [
                    float(raw_coords[t, i, 0]),
                    float(raw_coords[t, i, 1]),
                    float(raw_coords[t, i, 2]),
                ]
            for i in range(33):
                row += [
                    float(smooth_coords[t, i, 0]),
                    float(smooth_coords[t, i, 1]),
                    float(smooth_coords[t, i, 2]),
                ]
            writer.writerow(row)


def play_clip_opencv(video_path: str, window_name: str = "Clip Review") -> str:
    """
    Play a video clip using OpenCV.

    Controls while video is playing:
      space = pause/resume
      r     = replay from beginning
      k     = keep
      d     = discard/delete
      q/esc = quit review, keep by default

    Returns:
      "keep", "discard", "replay", or "quit"
    """
    video_path = str(Path(video_path).resolve())

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open clip with OpenCV:\n  {video_path}")
        return "quit"

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30

    delay_ms = max(1, int(1000 / fps))
    paused = False

    print("\nOpenCV controls:")
    print("  space = pause/resume")
    print("  r     = replay")
    print("  k     = keep")
    print("  d     = discard/delete")
    print("  q/esc = quit review, keep by default")

    while True:
        if not paused:
            ok, frame = cap.read()

            if not ok:
                # End of clip. Pause on last state and wait for user choice.
                paused = True
                print("\nEnd of clip. Press r to replay, k to keep, d to discard, or q to quit.")
                continue

            frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            cv2.imshow(window_name, frame)

        key = cv2.waitKey(delay_ms if not paused else 0) & 0xFF

        if key == 255:
            # No key pressed.
            continue

        if key == ord(" "):
            paused = not paused

        elif key == ord("r"):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            paused = False

        elif key == ord("k"):
            cap.release()
            cv2.destroyWindow(window_name)
            return "keep"

        elif key == ord("d"):
            cap.release()
            cv2.destroyWindow(window_name)
            return "discard"

        elif key == ord("q") or key == 27:
            cap.release()
            cv2.destroyWindow(window_name)
            return "quit"


def review_clip_interactively(video_path: str) -> bool:
    """
    Plays a clip with OpenCV and asks whether to keep, replay, or discard it.

    Returns:
      True  = keep clip
      False = discard/delete clip
    """
    while True:
        print(f"\nReviewing clip:\n  {video_path}")

        choice = play_clip_opencv(video_path)

        if choice == "keep":
            print("Kept clip.")
            return True

        if choice == "replay":
            continue

        if choice == "discard":
            try:
                os.remove(video_path)
                print(f"Deleted clip:\n  {video_path}")
            except FileNotFoundError:
                print("Clip was already deleted.")
            return False

        if choice == "quit":
            print("Stopped interactive review. Keeping this clip by default.")
            return True

        print("Invalid choice. Keeping clip by default.")
        return True


def cut_segments_ffmpeg(
    input_video: str,
    segments: List[Segment],
    output_dir: str,
    reencode: bool = True,
    fps: Optional[float] = None,
    review_clips: bool = False
):
    os.makedirs(output_dir, exist_ok=True)
    video_stem = Path(input_video).stem

    kept_paths = []
    deleted_paths = []

    for i, seg in enumerate(segments, start=1):
        out_path = os.path.join(output_dir, f"{video_stem}_swing_{i:03d}.mp4")

        if reencode:
            stream = ffmpeg.input(input_video, ss=seg.start_sec, to=seg.end_sec)
            kwargs = {
                "vcodec": "libx264",
                "acodec": "aac",
                "movflags": "+faststart",
            }
            if fps is not None:
                kwargs["r"] = fps

            (
                stream
                .output(out_path, **kwargs)
                .overwrite_output()
                .run(quiet=True)
            )
        else:
            (
                ffmpeg
                .input(input_video, ss=seg.start_sec, to=seg.end_sec)
                .output(out_path, c="copy")
                .overwrite_output()
                .run(quiet=True)
            )

        print(f"\nCreated clip {i}/{len(segments)}:\n  {out_path}")

        if review_clips:
            kept = review_clip_interactively(out_path)
            if kept:
                kept_paths.append(out_path)
            else:
                deleted_paths.append(out_path)
        else:
            kept_paths.append(out_path)

    return kept_paths, deleted_paths


def detect_swings(
    input_video: str,
    model_path: str,
    frame_stride: int = 2,
    sg_window_length: int = 11,
    sg_polyorder: int = 3,
    score_smooth_window: int = 9,
    score_median_window: int = 5,
    start_z: float = 2.8,
    end_z: float = 1.2,
    min_swing_sec: float = 0.7,
    max_swing_sec: float = 4.0,
    min_gap_sec: float = 0.75,
    pre_pad_sec: float = 0.25,
    post_pad_sec: float = 0.35,
    presence_threshold: float = 0.5,
    save_landmarks_path: Optional[str] = None,
):
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        raise RuntimeError("Could not read video FPS.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps

    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    PoseLandmarker = mp.tasks.vision.PoseLandmarker

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    all_times = []
    all_valid = []
    all_coords = []

    frame_idx = 0

    with PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            if frame_idx % frame_stride != 0:
                frame_idx += 1
                continue

            timestamp_ms = int((frame_idx / fps) * 1000)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            coords = np.full((33, 3), np.nan, dtype=np.float32)
            valid = False

            if result.pose_landmarks and len(result.pose_landmarks) > 0:
                lms = result.pose_landmarks[0]
                visible = []
                for idx in [11, 12, 13, 14, 15, 16, 23, 24]:
                    lm = lms[idx]
                    vis = getattr(lm, "visibility", 1.0)
                    pres = getattr(lm, "presence", 1.0)
                    visible.append(min(vis, pres))

                if np.mean(visible) >= presence_threshold:
                    valid = True
                    for i, lm in enumerate(lms):
                        coords[i, 0] = lm.x
                        coords[i, 1] = lm.y
                        coords[i, 2] = lm.z

            all_times.append(frame_idx / fps)
            all_valid.append(valid)
            all_coords.append(coords)

            frame_idx += 1

    cap.release()

    if len(all_times) == 0:
        return [], fps

    times = np.asarray(all_times, dtype=np.float32)
    valid_mask = np.asarray(all_valid, dtype=bool)
    raw_coords = np.stack(all_coords, axis=0)

    smooth_coords = apply_savgol_to_landmarks(
        raw_coords,
        valid_mask=valid_mask,
        window_length=sg_window_length,
        polyorder=sg_polyorder,
    )

    if save_landmarks_path is not None:
        save_landmarks_csv(
            times=times,
            raw_coords=raw_coords,
            smooth_coords=smooth_coords,
            valid_mask=valid_mask,
            output_path=save_landmarks_path,
        )

    scores = []
    prev_xyz = None
    for t in range(len(times)):
        curr_xyz = smooth_coords[t] if valid_mask[t] or np.isfinite(smooth_coords[t]).any() else None
        score = weighted_motion_score(curr_xyz, prev_xyz)
        scores.append(score)
        prev_xyz = curr_xyz

    scores = np.asarray(scores, dtype=np.float32)
    scores = median_filter_1d(scores, score_median_window)
    scores = moving_average(scores, score_smooth_window)

    med, mad = robust_stats(scores)
    robust_z = (scores - med) / (1.4826 * mad + 1e-6)

    segments = []
    active = False
    seg_start_idx = None
    seg_peak = 0.0
    last_end_time = -1e9

    for i in range(len(scores)):
        t = float(times[i])
        z = float(robust_z[i])
        sc = float(scores[i])

        if not active:
            if z >= start_z and (t - last_end_time) >= min_gap_sec:
                active = True
                seg_start_idx = i
                seg_peak = sc
        else:
            seg_peak = max(seg_peak, sc)

            if z <= end_z:
                start_time = float(times[seg_start_idx])
                end_time = t

                start_time = max(0.0, start_time - pre_pad_sec)
                end_time = min(duration_sec, end_time + post_pad_sec)

                dur = end_time - start_time
                if min_swing_sec <= dur <= max_swing_sec:
                    segments.append(Segment(start_time, end_time, seg_peak))
                    last_end_time = end_time

                active = False
                seg_start_idx = None
                seg_peak = 0.0

    if active and seg_start_idx is not None:
        start_time = max(0.0, float(times[seg_start_idx]) - pre_pad_sec)
        end_time = min(duration_sec, float(times[-1]) + post_pad_sec)
        dur = end_time - start_time
        if min_swing_sec <= dur <= max_swing_sec:
            segments.append(Segment(start_time, end_time, seg_peak))

    merged = []
    for seg in segments:
        if not merged:
            merged.append(seg)
            continue

        prev = merged[-1]
        if seg.start_sec - prev.end_sec < min_gap_sec:
            merged[-1] = Segment(
                start_sec=prev.start_sec,
                end_sec=max(prev.end_sec, seg.end_sec),
                peak_score=max(prev.peak_score, seg.peak_score),
            )
        else:
            merged.append(seg)

    return merged, fps


def gather_input_videos(inputs: Optional[List[str]], input_list: Optional[str]) -> List[str]:
    video_paths = []

    if inputs:
        video_paths.extend(inputs)

    if input_list:
        with open(input_list, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    video_paths.append(line)

    # Remove duplicates while preserving order.
    deduped = []
    seen = set()
    for p in video_paths:
        resolved = str(Path(p).resolve())
        if resolved not in seen:
            seen.add(resolved)
            deduped.append(resolved)

    return deduped


def process_one_video(args, video_path: str):
    video_path = convert_mov_to_mp4(video_path)
    input_stem = Path(video_path).stem

    video_outdir = os.path.join(args.outdir, input_stem)
    os.makedirs(video_outdir, exist_ok=True)

    landmarks_csv_path = None
    if args.save_landmarks_csv:
        landmarks_csv_path = os.path.join(video_outdir, f"{input_stem}_landmarks.csv")

    segments, fps = detect_swings(
        input_video=video_path,
        model_path=args.model,
        frame_stride=args.frame_stride,
        sg_window_length=args.sg_window_length,
        sg_polyorder=args.sg_polyorder,
        score_smooth_window=args.score_smooth_window,
        score_median_window=args.score_median_window,
        start_z=args.start_z,
        end_z=args.end_z,
        min_swing_sec=args.min_swing_sec,
        max_swing_sec=args.max_swing_sec,
        min_gap_sec=args.min_gap_sec,
        pre_pad_sec=args.pre_pad,
        post_pad_sec=args.post_pad,
        save_landmarks_path=landmarks_csv_path,
    )

    json_path = os.path.join(video_outdir, f"{input_stem}_segments.json")
    csv_path = os.path.join(video_outdir, f"{input_stem}_segments.csv")

    save_segments_json(segments, json_path)
    save_segments_csv(segments, csv_path)

    print(f"\n[{input_stem}] Detected {len(segments)} swing(s):")
    for i, seg in enumerate(segments, start=1):
        print(
            f"  {i:03d}: "
            f"start={seg.start_sec:.3f}s, "
            f"end={seg.end_sec:.3f}s, "
            f"duration={seg.end_sec - seg.start_sec:.3f}s, "
            f"peak_score={seg.peak_score:.5f}"
        )

    print(f"Saved timestamps to:\n  {json_path}\n  {csv_path}")
    if landmarks_csv_path:
        print(f"Saved landmark CSV to:\n  {landmarks_csv_path}")

    if args.cut_clips and segments:
        clips_dir = os.path.join(video_outdir, "clips")
        kept_paths, deleted_paths = cut_segments_ffmpeg(
            input_video=video_path,
            segments=segments,
            output_dir=clips_dir,
            reencode=not args.copy_codec,
            fps=fps,
            review_clips=args.review_clips,
        )

        print(f"Saved clips to:\n  {clips_dir}")
        print(f"Kept {len(kept_paths)} clip(s), discarded {len(deleted_paths)} clip(s).")


def main():
    parser = argparse.ArgumentParser(description="Detect golf swings for one or many videos.")
    parser.add_argument("--inputs", nargs="*", help="One or more input video paths")
    parser.add_argument("--input-list", help="Text file with one input video path per line")
    parser.add_argument("--model", required=True, help="Path to pose_landmarker.task")
    parser.add_argument("--outdir", default="golf_swings_output", help="Root output directory")
    parser.add_argument("--frame-stride", type=int, default=2, help="Process every Nth frame")
    parser.add_argument("--cut-clips", action="store_true", help="Export one clip per swing")
    parser.add_argument("--copy-codec", action="store_true", help="Use stream copy instead of re-encoding")
    parser.add_argument("--save-landmarks-csv", action="store_true", help="Save raw + smoothed landmarks to CSV")
    parser.add_argument("--review-clips", action="store_true", help="After each clip is created, play it and choose whether to keep, replay, or delete it")

    parser.add_argument("--sg-window-length", type=int, default=11)
    parser.add_argument("--sg-polyorder", type=int, default=3)

    parser.add_argument("--score-smooth-window", type=int, default=9)
    parser.add_argument("--score-median-window", type=int, default=5)

    parser.add_argument(
        "--start-z",
        type=float,
        default=3.0,
        help="Motion activation threshold. Higher z means more motion is needed to start a detection."
    )
    parser.add_argument(
        "--end-z",
        type=float,
        default=2.0,
        help="Motion deactivation threshold. Higher z ends clips sooner; lower z ends clips later."
    )
    parser.add_argument("--pre-pad", type=float, default=2.0)
    parser.add_argument("--post-pad", type=float, default=1.0)
    parser.add_argument("--min-swing-sec", type=float, default=0.7)
    parser.add_argument("--max-swing-sec", type=float, default=4.0)
    parser.add_argument("--min-gap-sec", type=float, default=0.75)

    args = parser.parse_args()

    if not args.inputs and not args.input_list:
        parser.error("Provide either --inputs or --input-list")

    os.makedirs(args.outdir, exist_ok=True)

    all_videos = gather_input_videos(args.inputs, args.input_list)

    if not all_videos:
        raise RuntimeError("No input videos found.")

    print(f"Found {len(all_videos)} video(s) to process.")

    for idx, video_path in enumerate(all_videos, start=1):
        print(f"\n=== Processing {idx}/{len(all_videos)}: {video_path} ===")
        try:
            process_one_video(args, video_path)
        except Exception as e:
            print(f"FAILED on {video_path}: {e}")


if __name__ == "__main__":
    main()
