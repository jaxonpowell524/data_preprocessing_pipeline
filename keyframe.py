#!/usr/bin/env python3
"""
Video Player with Keyframe Clip Extraction
==========================================
Usage:
    python keyframe.py INPUT OUTPUT_FOLDER

    INPUT can be a single video file (.mp4 or .mov) or a folder of videos.

Controls:
    SPACE       Play / Pause
    LEFT        Rewind 5 seconds
    RIGHT       Fast-forward 5 seconds
    ,           Slow down (2x -> 1.5x -> 1.25x -> 1x)
    .           Speed up  (1x -> 1.25x -> 1.5x -> 2x)
    /           Go to start of video
    K           Set keyframe (clips 4 s before -> keyframe are saved when video ends)
    Q / ESC     Quit current video and open next

Output clips are saved to OUTPUT_FOLDER/
"""

import sys
import os
import subprocess
import argparse
import time

import cv2
import pygame
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_LOCAL_FFMPEG = os.path.join(_SCRIPT_DIR, "ffmpeg.exe")
FFMPEG = _LOCAL_FFMPEG if os.path.isfile(_LOCAL_FFMPEG) else "ffmpeg"

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
SPEEDS = [1.0, 1.25, 1.5, 2.0]
PRE_KEYFRAME_SECS = 4.0
FONT_SIZE = 22
BAR_HEIGHT = 80
KEYFRAME_FLASH_DURATION = 1.0
WIN_MIN_W, WIN_MIN_H = 640, 480


def _is_readable(path: str) -> bool:
    cap = cv2.VideoCapture(path)
    ok = cap.isOpened()
    cap.release()
    return ok


def convert_to_mp4(path: str) -> str:
    """Convert a .mov file to .mp4 using ffmpeg. Returns path to mp4."""
    base = os.path.splitext(path)[0]
    out = base + ".mp4"

    if os.path.exists(out) and _is_readable(out):
        print(f"[convert] {out} already exists, skipping conversion.")
        return out

    print(f"[convert] {path} -> {out} (remux)...")
    result = subprocess.run(
        [FFMPEG, "-y", "-i", path, "-c", "copy", out],
        capture_output=True, text=True
    )
    if result.returncode == 0 and _is_readable(out):
        return out

    print(f"[convert] remux unreadable, re-encoding with ultrafast...")
    result = subprocess.run(
        [FFMPEG, "-y", "-i", path, "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23", "-c:a", "aac", out],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"[error] ffmpeg conversion failed:\n{result.stderr}")
        sys.exit(1)
    return out


def extract_clip(src: str, keyframe_secs: float, clip_index: int, output_dir: str) -> str:
    """Use ffmpeg to cut a clip [keyframe-4s ... keyframe] from src."""
    start = max(0.0, keyframe_secs - PRE_KEYFRAME_SECS)
    duration = keyframe_secs - start
    if duration <= 0:
        print(f"[clip] Keyframe too close to start, skipping clip #{clip_index}.")
        return ""

    stem = os.path.splitext(os.path.basename(src))[0]
    out = os.path.join(output_dir, f"{stem}_clip{clip_index:02d}.mp4")
    print(f"[clip] Extracting {start:.2f}s - {keyframe_secs:.2f}s -> {out}")

    process = subprocess.Popen(
        [FFMPEG, "-y", "-ss", str(start), "-i", src, "-t", str(duration),
         "-c:v", "libx264", "-c:a", "aac", out],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    while process.poll() is None:
        pygame.event.pump()
        time.sleep(0.05)

    if process.returncode != 0:
        print(f"[error] clip extraction failed")
        return ""
    print(f"[clip] Saved: {out}")
    return out


# ──────────────────────────────────────────────
# HUD helpers
# ──────────────────────────────────────────────

def fmt_time(secs: float) -> str:
    if secs == float("inf"):
        return "--:--"
    secs = max(0.0, secs)
    m = int(secs) // 60
    s = int(secs) % 60
    return f"{m:02d}:{s:02d}"


def draw_hud(surface: pygame.Surface, font: pygame.font.Font,
             current: float, total: float, speed: float,
             paused: bool, keyframes: list, flash_msg: str,
             video_w: int, video_h: int, ended: bool = False):
    sw, sh = surface.get_size()

    bar_surf = pygame.Surface((sw, BAR_HEIGHT), pygame.SRCALPHA)
    bar_surf.fill((10, 10, 10, 180))
    surface.blit(bar_surf, (0, sh - BAR_HEIGHT))

    bar_x, bar_y = 12, sh - BAR_HEIGHT + 10
    bar_w, bar_h = sw - 24, 6
    pygame.draw.rect(surface, (60, 60, 60), (bar_x, bar_y, bar_w, bar_h), border_radius=3)
    if total > 0 and total != float("inf"):
        fill = int(bar_w * current / total)
        pygame.draw.rect(surface, (220, 80, 80), (bar_x, bar_y, fill, bar_h), border_radius=3)
        for kf in keyframes:
            kx = bar_x + int(bar_w * kf / total)
            pygame.draw.rect(surface, (255, 220, 0), (kx - 1, bar_y - 3, 3, bar_h + 6))

    y2 = sh - BAR_HEIGHT + 24
    state_str = "END OF VIDEO" if ended else ("PAUSED" if paused else "PLAYING")
    labels = [
        state_str,
        f"{fmt_time(current)} / {fmt_time(total)}",
        f"Speed: {speed:.2f}x",
        f"Keyframes: {len(keyframes)}",
    ]
    x = 14
    for label in labels:
        txt = font.render(label, True, (230, 230, 230))
        surface.blit(txt, (x, y2))
        x += txt.get_width() + 28

    if ended:
        hint = font.render("K = set keyframe here  |  Q/ESC = next video", True, (255, 220, 0))
    else:
        hint = font.render("SPC=play/pause  </>=5s  ,/.=speed  /=restart  K=keyframe  Q=next", True, (140, 140, 140))
    surface.blit(hint, (14, sh - BAR_HEIGHT + 52))

    if flash_msg:
        flash = pygame.font.Font(None, 54).render(flash_msg, True, (255, 220, 0))
        fx = sw // 2 - flash.get_width() // 2
        fy = sh // 2 - flash.get_height() // 2 - BAR_HEIGHT // 2
        bg = pygame.Surface((flash.get_width() + 20, flash.get_height() + 12), pygame.SRCALPHA)
        bg.fill((0, 0, 0, 160))
        surface.blit(bg, (fx - 10, fy - 6))
        surface.blit(flash, (fx, fy))


def show_status(message: str):
    """Render a centred message on the current pygame display and pump events."""
    screen = pygame.display.get_surface()
    if screen is None:
        return
    screen.fill((0, 0, 0))
    font = pygame.font.Font(None, 36)
    txt = font.render(message, True, (200, 200, 200))
    sw, sh = screen.get_size()
    screen.blit(txt, (sw // 2 - txt.get_width() // 2, sh // 2 - txt.get_height() // 2))
    pygame.display.flip()
    pygame.event.pump()


# ──────────────────────────────────────────────
# Player
# ──────────────────────────────────────────────

def play_video(mp4_path: str, video_number: int, total_videos: int) -> list:
    """Play one mp4 file. Returns list of keyframe timestamps (seconds)."""
    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        print(f"[error] Cannot open {mp4_path}")
        return []

    fps_native = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_secs = (total_frames / fps_native) if total_frames > 0 else float("inf")
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    pygame.display.set_caption(
        f"[{video_number}/{total_videos}] {os.path.basename(mp4_path)}"
    )
    win_w = max(vid_w // 3, WIN_MIN_W)
    win_h = max(vid_h // 3 + BAR_HEIGHT, WIN_MIN_H)
    screen = pygame.display.set_mode((win_w, win_h), pygame.RESIZABLE)

    clock = pygame.time.Clock()
    font = pygame.font.Font(None, FONT_SIZE)

    paused = False
    ended = False
    speed_idx = 0
    keyframes = []
    flash_msg = ""
    flash_until = 0.0
    last_frame = None

    playback_pos = 0.0
    wall_last = time.perf_counter()

    def seek_to(secs: float):
        nonlocal playback_pos
        playback_pos = max(0.0, min(secs, total_secs if total_secs != float("inf") else playback_pos))
        frame_no = int(playback_pos * fps_native)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)

    running = True

    while running:
        now = time.perf_counter()
        dt = min(now - wall_last, 1.0 / 30.0)  # cap to avoid jumps on window drag/resize
        wall_last = now

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.VIDEORESIZE:
                screen = pygame.display.set_mode(event.size, pygame.RESIZABLE)

            elif event.type == pygame.KEYDOWN:
                key = event.key

                if key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False

                elif key == pygame.K_SPACE:
                    if not ended:
                        paused = not paused
                        wall_last = time.perf_counter()

                elif key == pygame.K_LEFT:
                    if not ended:
                        seek_to(playback_pos - 5.0)

                elif key == pygame.K_RIGHT:
                    if not ended:
                        seek_to(playback_pos + 5.0)

                elif key == pygame.K_SLASH:
                    if not ended:
                        seek_to(0.0)

                elif key == pygame.K_PERIOD:
                    speed_idx = min(speed_idx + 1, len(SPEEDS) - 1)
                    wall_last = time.perf_counter()

                elif key == pygame.K_COMMA:
                    speed_idx = max(speed_idx - 1, 0)
                    wall_last = time.perf_counter()

                elif key == pygame.K_k:
                    keyframes.append(playback_pos)
                    flash_msg = f"KEYFRAME SET @ {fmt_time(playback_pos)}"
                    flash_until = time.perf_counter() + KEYFRAME_FLASH_DURATION
                    print(f"[keyframe] {fmt_time(playback_pos)} ({playback_pos:.3f}s)")
                    if ended:
                        running = False

        speed = SPEEDS[speed_idx]

        if not paused and not ended:
            playback_pos += dt * speed
            if playback_pos >= total_secs:
                ended = True
                paused = True

        if not ended and not paused:
            target_frame = int(playback_pos * fps_native)
            actual_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

            if target_frame < actual_frame - 1 or target_frame > actual_frame + int(fps_native * 2):
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

            ret, frame = cap.read()
            if ret:
                last_frame = frame
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, target_frame - 1))
                ret, frame = cap.read()
                if ret:
                    last_frame = frame
                else:
                    ended = True
                    paused = True

        sw, sh = screen.get_size()
        video_area_h = sh - BAR_HEIGHT

        if last_frame is not None:
            scale = min(sw / vid_w, video_area_h / vid_h)
            disp_w = int(vid_w * scale)
            disp_h = int(vid_h * scale)

            frame_rgb = cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB)
            frame_surf = pygame.surfarray.make_surface(
                np.transpose(frame_rgb, (1, 0, 2))
            )
            frame_surf = pygame.transform.smoothscale(frame_surf, (disp_w, disp_h))

            screen.fill((0, 0, 0))
            ox = (sw - disp_w) // 2
            oy = (video_area_h - disp_h) // 2
            screen.blit(frame_surf, (ox, oy))

        if time.perf_counter() > flash_until:
            flash_msg = ""

        draw_hud(
            screen, font,
            playback_pos, total_secs, speed,
            paused, keyframes, flash_msg,
            vid_w, vid_h, ended,
        )

        pygame.display.flip()
        clock.tick(60)

    cap.release()
    return keyframes


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Video player with keyframe clip export.")
    parser.add_argument("input", help="Video file (.mp4 or .mov) or folder of videos")
    parser.add_argument("output_folder", help="Folder where extracted clips will be saved")
    args = parser.parse_args()

    supported = (".mp4", ".mov")

    if os.path.isfile(args.input):
        ext = os.path.splitext(args.input)[1].lower()
        if ext not in supported:
            print(f"[error] Unsupported file type: {ext}")
            sys.exit(1)
        all_inputs = [args.input]
        input_dir = os.path.dirname(os.path.abspath(args.input))
    elif os.path.isdir(args.input):
        all_files = sorted(
            f for f in os.listdir(args.input)
            if os.path.splitext(f)[1].lower() in supported
        )
        if not all_files:
            print(f"[error] No .mp4 or .mov files found in {args.input}")
            sys.exit(1)
        all_inputs = [os.path.join(args.input, f) for f in all_files]
        input_dir = args.input
    else:
        print(f"[error] Not a file or directory: {args.input}")
        sys.exit(1)

    output_dir = args.output_folder
    os.makedirs(output_dir, exist_ok=True)
    print(f"[output] Clips will be saved to: {output_dir}")

    # Convert .mov files and deduplicate (skip .mp4 that share a stem with a .mov).
    mov_stems = {os.path.splitext(os.path.basename(p))[0].lower() for p in all_inputs if p.lower().endswith(".mov")}
    mp4_paths = []
    seen_paths = set()
    for path in all_inputs:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".mov":
            converted = convert_to_mp4(path)
            norm = os.path.normcase(os.path.abspath(converted))
            if norm not in seen_paths:
                seen_paths.add(norm)
                mp4_paths.append(converted)
        elif os.path.splitext(os.path.basename(path))[0].lower() not in mov_stems:
            norm = os.path.normcase(os.path.abspath(path))
            if norm not in seen_paths:
                seen_paths.add(norm)
                mp4_paths.append(path)

    pygame.init()
    pygame.display.set_caption("Video Player")

    total = len(mp4_paths)
    for i, mp4 in enumerate(mp4_paths, start=1):
        print(f"\n[player] Opening video {i}/{total}: {mp4}")
        keyframes = play_video(mp4, i, total)

        for clip_idx, kf in enumerate(keyframes, start=1):
            show_status(f"Saving clip {clip_idx} of {len(keyframes)}...")
            extract_clip(mp4, kf, clip_idx, output_dir)

        if i < total:
            show_status("Loading next video...")
            print(f"[player] Moving to next video...")

    pygame.quit()
    print("\n[done] All videos played.")


if __name__ == "__main__":
    main()
