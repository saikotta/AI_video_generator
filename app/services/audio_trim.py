"""
Trim the silence that the TTS engine puts at the start and end of every narration clip.

Each scene's narration is generated as a separate clip, and each clip carries roughly half a second of
leading and trailing silence. Played back to back that is a ~1.1s dead pause at every scene boundary,
which makes the narration sound like dictation. Only digital silence at the two ends is removed
(a short, natural margin is kept); spoken words and pauses inside the sentence are never touched, and the
measured duration returned to the pipeline is the trimmed length, so video still follows audio.

Turn off with the environment variable TTS_TRIM_EDGE_SILENCE=0.
"""
import logging
import os
import re
import subprocess

logger = logging.getLogger(__name__)

KEEP_HEAD_SECONDS = 0.10
KEEP_TAIL_SECONDS = 0.15
THRESHOLD_DB = -48.0

UNTRIMMED_PAUSE_PER_SCENE = 1.1   # measured on real runs (two clips back to back)
TRIMMED_PAUSE_PER_SCENE = 0.3


def edge_trim_enabled() -> bool:
    return os.getenv("TTS_TRIM_EDGE_SILENCE", "1").strip().lower() not in ("0", "false", "no", "off")


def pause_seconds_per_scene() -> float:
    """Expected silence added per scene; used by the script word budget."""
    return TRIMMED_PAUSE_PER_SCENE if edge_trim_enabled() else UNTRIMMED_PAUSE_PER_SCENE


def _ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def measure_duration_seconds(path: str) -> float:
    """Decode the whole file and return its duration (0.0 if it cannot be decoded)."""
    try:
        import av
        container = av.open(path)
        rate = container.streams.audio[0].codec_context.sample_rate or 24000
        total = sum(frame.samples for frame in container.decode(audio=0))
        container.close()
        return total / float(rate)
    except Exception:
        pass
    try:
        r = subprocess.run([_ffmpeg_exe(), "-hide_banner", "-i", path, "-f", "null", "-"],
                           capture_output=True, text=True)
        found = re.findall(r"time=(\d+):(\d+):(\d+\.\d+)", r.stderr)
        if found:
            h, m, s = found[-1]
            return int(h) * 3600 + int(m) * 60 + float(s)
    except Exception:
        pass
    return 0.0


def trim_edge_silence(path: str, keep_head: float = KEEP_HEAD_SECONDS, keep_tail: float = KEEP_TAIL_SECONDS,
                      threshold_db: float = THRESHOLD_DB) -> bool:
    """Trim leading/trailing silence of an audio file in place. Returns True only if the file was replaced."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return False
    original = measure_duration_seconds(path)
    if original <= 0.5:
        return False

    base, ext = os.path.splitext(path)
    tmp = base + ".trimtmp" + ext
    af = (
        f"silenceremove=start_periods=1:start_threshold={threshold_db}dB:start_silence={keep_head},"
        f"areverse,"
        f"silenceremove=start_periods=1:start_threshold={threshold_db}dB:start_silence={keep_tail},"
        f"areverse"
    )
    codec = ["-c:a", "libmp3lame", "-q:a", "2"] if ext.lower() == ".mp3" else []
    try:
        subprocess.run([_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", path, "-af", af, *codec, tmp],
                       check=True, capture_output=True)
        trimmed = measure_duration_seconds(tmp)
        # Safety: never keep a result that lost more than 40% of the clip (that would mean speech was cut)
        if trimmed < 0.5 or trimmed < 0.6 * original or trimmed > original + 0.05:
            logger.warning(f"Silence trim rejected for {path}: {original:.2f}s -> {trimmed:.2f}s")
            return False
        os.replace(tmp, path)
        return True
    except Exception as e:
        logger.warning(f"Silence trim failed for {path}: {e}")
        return False
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def trim_narration_edges(path: str, measured_duration: float) -> float:
    """Trim a freshly generated narration clip and return its (possibly new) measured duration."""
    if not edge_trim_enabled():
        return measured_duration
    try:
        if trim_edge_silence(path):
            new_duration = measure_duration_seconds(path)
            if new_duration > 0:
                logger.info(f"Trimmed narration edge silence: {measured_duration:.3f}s -> {new_duration:.3f}s ({path})")
                return round(new_duration, 3)
    except Exception as e:
        logger.warning(f"Narration trim skipped: {e}")
    return measured_duration
