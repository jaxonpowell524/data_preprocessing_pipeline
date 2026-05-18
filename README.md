# keyframe.py — Video Player with Keyframe Clip Extraction

A keyboard-driven video player that lets you mark keyframes while watching footage, then automatically extracts short clips around each marked moment.

Designed for reviewing sports footage (e.g. golf swings) and saving the best clips for further analysis.

---

## Features

- Play `.mp4` and `.mov` videos from a folder, one by one
- Mark keyframes with a single key press while watching
- Auto-extracts a 4-second clip ending at each marked keyframe
- Adjustable playback speed (1x, 1.25x, 1.5x, 2x)
- On-screen HUD showing playback position, speed, and keyframe markers
- Auto-converts `.mov` files to `.mp4` via ffmpeg before playback

---

## Requirements

- Python 3.8+
- [ffmpeg](https://ffmpeg.org/download.html) — either on your PATH or placed as `ffmpeg.exe` in the same folder as the script

Install Python dependencies:

```bash
pip install -r requirements.txt
```

**Dependencies:** `opencv-python`, `pygame`, `numpy`

---

## Usage

```bash
python keyframe.py /path/to/video/folder
```

The script will open every `.mp4` and `.mov` file in the folder sequentially. Extracted clips are saved to `<folder>/output_clips/`.

---

## Controls

| Key | Action |
|-----|--------|
| `SPACE` | Play / Pause |
| `LEFT` | Rewind 5 seconds |
| `RIGHT` | Fast-forward 5 seconds |
| `,` | Slow down playback |
| `.` | Speed up playback |
| `/` | Jump to start |
| `K` | Set keyframe at current position |
| `Q` / `ESC` | Skip to next video |

> **Tip:** At the end of a video, the player pauses. You can still press `K` to mark the final position as a keyframe before moving on.

---

## Output

Clips are named `<original_filename>_clip01.mp4`, `_clip02.mp4`, etc., and saved to:

```
<input_folder>/
└── output_clips/
    ├── swing01_clip01.mp4
    ├── swing01_clip02.mp4
    └── ...
```

Each clip covers the **4 seconds leading up to** the marked keyframe.

---

## Notes

- If a `.mov` file already has a matching `.mp4` in the same folder, conversion is skipped.
- If the keyframe is within 4 seconds of the video start, the clip is trimmed to whatever is available.
- The window is resizable; video is letterboxed to fit.
