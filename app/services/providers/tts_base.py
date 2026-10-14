from abc import ABC, abstractmethod

# NOTE: Step 5 exposes the measured audio duration for downstream consumption.
# It does NOT modify Step 3's existing scene.duration_seconds implementation or any locked Step 3 code —
# how this measured value gets stored or consumed is a Step 6/7 decision, out of scope here.

class TTSProvider(ABC):
    """
    Abstract Base Class interface for text-to-speech voiceover providers.
    
    In alignment with the project's asynchronous architecture, methods are defined as async.
    """

    @abstractmethod
    async def generate_audio(self, text: str, output_path: str) -> float:
        """
        Generate narration audio from text and save to output_path.
        
        Returns the ACTUAL generated audio duration in seconds, measured directly
        from the decoded audio file itself — not estimated or derived from word count.
        """
        pass
