r"""
make_sample_video.py - assemble generated clips + narration + captions into a finished vertical video.

Use it to turn clips from a free GPU notebook (Kaggle/Colab) into a 15-second sample reel. Needs only
what your venv already has (Pillow, imageio-ffmpeg; edge-tts only if you pass --text).

Examples (from C:\AiVideoGenerator, clips in sample\clips as clip_1.mp4, clip_2.mp4, clip_3.mp4):
    .\.venv\Scripts\python.exe make_sample_video.py --clips sample\clips --text "Green beans enter a rotating drum. They crackle, turn brown, and cool. Then they are ground fresh." --out sample\sample_reel.mp4
    .\.venv\Scripts\python.exe make_sample_video.py --clips sample\clips --audio sample\narration.mp3 --text "caption text here" --out sample\sample_reel.mp4

Rules kept from the project: the narration is never cut or sped up; the video is at least as long as the
audio (rounded up to a whole 24 fps frame); the clips are shortened or held to fit, not the audio.
"""
import argparse
import asyncio
import glob
import math
import os
import re
import subprocess
import sys
import tempfile

from PIL import Image, ImageDraw, ImageFont
import imageio_ffmpeg

FF = imageio_ffmpeg.get_ffmpeg_exe()
FPS = 24
FONTS = ("arialbd.ttf", "arial.ttf", "segoeuib.ttf", "DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf")


def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        print("ffmpeg failed:\n" + " ".join(cmd) + "\n" + p.stderr[-1500:])
        sys.exit(1)
    return p



def media_duration(path):
    p = subprocess.run(
        [FF, "-hide_banner", "-i", path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", p.stderr or "")
    if not m:
        raise RuntimeError(
            f"Cannot read duration of {path}.\n"
            f"FFmpeg output:\n{p.stderr[-1500:] if p.stderr else '(no output)'}"
        )
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))


def make_narration(text, voice, out_path):
    import edge_tts

    async def go():
        await edge_tts.Communicate(text, voice).save(out_path)
    asyncio.run(go())


def trim_edges_if_possible(path):
    """Use the project's silence trimmer when run from the project folder (optional)."""
    try:
        sys.path.insert(0, os.getcwd())
        from app.services.audio_trim import trim_edge_silence
        if trim_edge_silence(path):
            print("  trimmed silence at the ends of the narration")
    except Exception:
        pass


IMG_EXT = (".png", ".jpg", ".jpeg", ".webp")


def image_to_clip(path, W, H, seconds, out, crop_bottom=0.0, max_zoom=1.10):
    """Turn a still image into a clip with a slow, smooth push-in (fallback when no AI clip is available)."""
    src = Image.open(path).convert("RGB")
    if crop_bottom > 0:
        src = src.crop((0, 0, src.width, int(src.height * (1 - crop_bottom))))
    sw, sh = src.size
    s = max(W / sw, H / sh)                       # cover-fit scale
    n = max(1, round(seconds * FPS))
    frames_dir = out + "_frames"
    os.makedirs(frames_dir, exist_ok=True)
    for i in range(n):
        z = 1.0 + (max_zoom - 1.0) * i / max(1, n - 1)
        a = 1.0 / (s * z)
        c, d = (sw - W * a) / 2.0, (sh - H * a) / 2.0
        src.transform((W, H), Image.AFFINE, (a, 0.0, c, 0.0, a, d), resample=Image.BICUBIC).save(
            os.path.join(frames_dir, f"{i:05d}.jpg"), quality=95)
    run([FF, "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", os.path.join(frames_dir, "%05d.jpg"),
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", "-r", str(FPS), out])


def load_font(size):
    for name in FONTS:
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def split_caption_chunks(text, max_words=5):
    words = text.split()
    chunks, cur = [], []
    for w in words:
        cur.append(w)
        if len(cur) >= max_words or re.search(r"[.!?;,:]$", w) and len(cur) >= 2:
            chunks.append(" ".join(cur)); cur = []
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def caption_png(text, size, path):
    w, h = size
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    font = load_font(max(30, w // 15))
    lines, line = [], ""
    for word in text.split():
        test = (line + " " + word).strip()
        if d.textlength(test, font=font) <= w * 0.82:
            line = test
        else:
            lines.append(line); line = word
    lines.append(line)
    lh = int(font.size * 1.25)
    box_h = lh * len(lines) + 36
    box_w = int(max(d.textlength(l, font=font) for l in lines)) + 56
    y0 = int(h * 0.66)                           # leave the bottom ~20% free for app buttons
    d.rounded_rectangle([(w - box_w) // 2, y0, (w + box_w) // 2, y0 + box_h], radius=22, fill=(0, 0, 0, 165))
    for i, l in enumerate(lines):
        d.text((w // 2, y0 + 18 + i * lh + lh // 2), l, font=font, fill=(255, 255, 255, 255), anchor="mm")
    img.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True, help="folder with the video clips (sorted by name)")
    ap.add_argument("--text", default="", help="narration text (also used for captions)")
    ap.add_argument("--audio", default="", help="existing narration audio; if omitted it is made with edge-tts from --text")
    ap.add_argument("--voice", default="en-US-ChristopherNeural")
    ap.add_argument("--size", default="1080x1920", help="WIDTHxHEIGHT, default 9:16")
    ap.add_argument("--out", default="sample_reel.mp4")
    ap.add_argument("--no-captions", action="store_true")
    ap.add_argument("--crop-bottom", type=float, default=0.0,
                    help="cut this fraction off the bottom of every clip first (e.g. 0.06 hides a corner watermark)")
    ap.add_argument("--fill", choices=("slow", "hold"), default="slow",
                    help="clip shorter than its slot: slow = play slower (up to 2x), hold = freeze the last frame")
    a = ap.parse_args()

    W, H = [int(x) for x in a.size.lower().split("x")]
    clips = sorted(sum((glob.glob(os.path.join(a.clips, e)) for e in ("*.mp4", "*.mov", "*.webm", "*.mkv", "*.png", "*.jpg", "*.jpeg", "*.webp")), []))
    if not clips:
        sys.exit(f"no video clips or images found in {a.clips}")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        audio = a.audio
        if not audio:
            if not a.text:
                sys.exit("give --text (to make narration) or --audio")
            audio = os.path.join(tmp, "narration.mp3")
            print("Making narration with edge-tts ...")
            make_narration(a.text, a.voice, audio)
            trim_edges_if_possible(audio)
        audio_dur = media_duration(audio)
        total = math.ceil(audio_dur * FPS) / FPS          # video never shorter than the narration
        per = total / len(clips)
        print(f"Narration {audio_dur:.2f}s -> video {total:.3f}s, {len(clips)} clips of {per:.2f}s each")

        norm = []
        for i, c in enumerate(clips, 1):
            o = os.path.join(tmp, f"n{i}.mp4")
            if c.lower().endswith(IMG_EXT):
                print(f"  clip {i}: still image -> slow push-in (fallback style): {os.path.basename(c)}")
                image_to_clip(c, W, H, per, o, a.crop_bottom)
                norm.append(o)
                continue
            cdur = media_duration(c)
            pre = ""
            if a.fill == "slow" and 0.3 < cdur < per:
                factor = min(2.0, per / cdur)          # slow down a short clip instead of freezing it
                pre = f"setpts={factor:.4f}*PTS,"
                print(f"  clip {i}: {cdur:.2f}s is shorter than {per:.2f}s -> slowed x{factor:.2f}")
            cb = f"crop=iw:ih*{1 - a.crop_bottom:.4f}:0:0," if a.crop_bottom > 0 else ""
            vf = (f"{pre}{cb}scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},fps={FPS},setsar=1,"
                  f"tpad=stop_mode=clone:stop_duration={per:.3f},trim=duration={per:.3f},setpts=PTS-STARTPTS,format=yuv420p")
            run([FF, "-y", "-loglevel", "error", "-i", c, "-an", "-vf", vf, "-c:v", "libx264", "-preset", "veryfast",
                 "-crf", "18", o])
            norm.append(o)
        lst = os.path.join(tmp, "list.txt")
        with open(lst, "w") as f:
            f.writelines(f"file '{p.replace(chr(92), '/')}'\n" for p in norm)
        joined = os.path.join(tmp, "joined.mp4")
        run([FF, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", joined])

        inputs = ["-i", joined, "-i", audio]
        filt, last, n_png = [], "0:v", 0
        if a.text and not a.no_captions:
            chunks = split_caption_chunks(a.text)
            weights = [max(1, len(c)) for c in chunks]
            start = 0.10 if not a.audio else 0.0
            span = max(0.5, audio_dur - start - 0.1)
            t = start
            for i, (c, wgt) in enumerate(zip(chunks, weights)):
                dur = span * wgt / sum(weights)
                png = os.path.join(tmp, f"cap{i}.png")
                caption_png(c, (W, H), png)
                inputs += ["-i", png]
                n_png += 1
                out = f"v{i}"
                filt.append(f"[{last}][{i + 2}:v]overlay=0:0:enable='between(t,{t:.3f},{t + dur:.3f})'[{out}]")
                last = out
                t += dur
        cmd = [FF, "-y", "-loglevel", "error", *inputs]
        if filt:
            cmd += ["-filter_complex", ";".join(filt), "-map", f"[{last}]"]
        else:
            cmd += ["-map", "0:v"]
        cmd += ["-map", "1:a", "-af", "apad", "-t", f"{total:.3f}", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                "-pix_fmt", "yuv420p", "-r", str(FPS), "-c:a", "aac", "-b:a", "192k", a.out]
        run(cmd)

    print(f"\nDONE: {a.out}  (video {media_duration(a.out):.2f}s, narration {audio_dur:.2f}s)")


if __name__ == "__main__":
    main()
