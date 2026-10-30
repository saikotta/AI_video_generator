"""
Title-card clips for shots that stock footage cannot depict truthfully
(e.g. internal biochemical processes such as enzymes breaking down starch).

Rendering an honest card is better than substituting unrelated footage
(like water pouring into a kitchen glass).
"""
import os
import subprocess
import textwrap

from PIL import Image, ImageDraw, ImageFont

_FONT_CANDIDATES = ("arialbd.ttf", "arial.ttf", "segoeui.ttf", "DejaVuSans-Bold.ttf", "DejaVuSans.ttf",
                    "LiberationSans-Bold.ttf")


def _load_font(size: int):
    for name in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def render_card_image(heading: str, size=(1280, 720)) -> Image.Image:
    w, h = size
    img = Image.new("RGB", size)
    px = img.load()
    for y in range(h):  # soft vertical gradient
        t = y / max(1, h - 1)
        color = (int(14 + 12 * t), int(32 + 26 * t), int(52 + 34 * t))
        for x in range(w):
            px[x, y] = color
    draw = ImageDraw.Draw(img)

    label_font = _load_font(28)
    draw.text((w // 2, 150), "KEY IDEA", font=label_font, fill=(130, 190, 230), anchor="mm")
    draw.rectangle([w // 2 - 60, 178, w // 2 + 60, 182], fill=(80, 150, 200))

    text = " ".join((heading or "").split())[:140] or "Key idea"
    head_font = _load_font(52)
    lines = textwrap.wrap(text, width=30) or [text]
    line_h = 70
    y0 = h // 2 - (len(lines) * line_h) // 2 + 10
    for i, line in enumerate(lines[:4]):
        draw.text((w // 2, y0 + i * line_h), line, font=head_font, fill=(240, 246, 250), anchor="mm")
    return img


def render_info_card(heading: str, out_path: str, duration_seconds: float, fps: int = 24,
                     size=(1280, 720)) -> str:
    """Render a still title card to an H.264 MP4 of the given duration. Returns out_path."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    png_path = os.path.splitext(out_path)[0] + ".png"
    render_card_image(heading, size).save(png_path)
    cmd = [
        _ffmpeg_exe(), "-y", "-loglevel", "error",
        "-loop", "1", "-framerate", str(fps), "-i", png_path,
        "-t", f"{max(0.5, float(duration_seconds)):.3f}",
        "-c:v", "libx264", "-tune", "stillimage", "-pix_fmt", "yuv420p", "-r", str(fps), "-an",
        out_path,
    ]
    subprocess.run(cmd, check=True)
    return out_path
