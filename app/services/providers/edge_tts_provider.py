import os
import asyncio
import logging
from typing import Dict, Any, Optional
import edge_tts
import av

from app.config import settings
from app.services.providers.tts_base import TTSProvider
from app.services.providers.exceptions import (
    TTSError,
    TTSConnectionError,
    TTSValidationError,
    TTSAudioCorruptError
)

# NOTE: Step 5 exposes the measured audio duration for downstream consumption.
# It does NOT modify Step 3's existing scene.duration_seconds implementation or any locked Step 3 code —
# how this measured value gets stored or consumed is a Step 6/7 decision, out of scope here.

logger = logging.getLogger(__name__)

class EdgeTTSProvider(TTSProvider):
    """
    EdgeTTS implementation of TTSProvider using Microsoft Edge TTS service.
    
    Exposes generate_audio(text, output_path) -> float, where the returned float
    is the ACTUAL measured audio duration in seconds obtained by genuinely decoding
    the generated MP3 file via PyAV.
    """

    def __init__(self, voice: Optional[str] = None):
        self.voice = voice or os.getenv("TTS_VOICE", getattr(settings, "TTS_VOICE", "en-US-AriaNeural"))

    @staticmethod
    def decode_audio_metadata(audio_path: str) -> Dict[str, Any]:
        """
        Perform a full genuine audio stream decode using PyAV (av) to verify structural
        readability, playability, and extract exact sample-based duration and stream properties.
        """
        if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
            raise TTSAudioCorruptError(f"Audio file does not exist or is empty: {audio_path}")

        try:
            container = av.open(audio_path)
            if not container.streams.audio:
                container.close()
                raise TTSAudioCorruptError(f"No audio stream found in file: {audio_path}")

            stream = container.streams.audio[0]
            sample_rate = stream.codec_context.sample_rate or 24000
            channels = stream.codec_context.channels or 1
            codec_name = stream.codec_context.name or "mp3"
            format_name = container.format.name or "mp3"

            total_samples = 0
            for frame in container.decode(audio=0):
                total_samples += frame.samples

            container.close()

            if sample_rate <= 0:
                raise TTSAudioCorruptError("Invalid audio sample rate (<= 0).")

            duration_seconds = total_samples / float(sample_rate)

            if duration_seconds <= 0:
                raise TTSAudioCorruptError(f"Decoded audio has 0 or negative duration ({duration_seconds}s).")

            return {
                "duration": round(duration_seconds, 3),
                "format": format_name,
                "codec": codec_name,
                "sample_rate": sample_rate,
                "channels": channels,
                "total_samples": total_samples
            }

        except TTSAudioCorruptError:
            raise
        except Exception as e:
            raise TTSAudioCorruptError(f"PyAV audio decoding failed for file {audio_path}: {e}")

    async def generate_audio(self, text: str, output_path: str) -> float:
        """
        Generate narration audio from scene text using edge-tts and save to output_path.
        
        Validates non-empty text, handles network errors, decodes audio stream via PyAV,
        and returns actual measured duration in seconds.
        """
        # 1. Validation: Genuinely reject empty or whitespace-only scene text
        if not text or not text.strip():
            raise TTSValidationError("Scene text for TTS narration cannot be empty or whitespace-only.")

        # Ensure target directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        # 2. Call edge-tts service dependency
        try:
            communicate = edge_tts.Communicate(text.strip(), self.voice)
            await communicate.save(output_path)
        except (edge_tts.exceptions.EdgeTTSException, Exception) as e:
            logger.error(f"EdgeTTS service audio generation failed: {e}")
            raise TTSConnectionError(f"EdgeTTS network/service dependency error: {e}")

        # 3. Decode generated audio file via PyAV to confirm playability & measure actual duration
        metadata = self.decode_audio_metadata(output_path)
        actual_duration = metadata["duration"]

        logger.info(
            f"Generated TTS audio ({actual_duration}s) saved to {output_path} "
            f"[Format: {metadata['format']}, Codec: {metadata['codec']}, Sample Rate: {metadata['sample_rate']}Hz, Channels: {metadata['channels']}]"
        )

        return actual_duration
