class ScriptProviderError(Exception):
    """Base exception for script provider errors."""
    pass

class OllamaUnavailableError(ScriptProviderError):
    """Raised when the Ollama service is unreachable or requested model is missing."""
    pass

class ScriptGenerationTimeoutError(ScriptProviderError):
    """Raised when script generation times out."""
    pass

class InvalidScriptResponseError(ScriptProviderError):
    """Raised when the script provider returns an empty or invalid response."""
    pass

# Stock Media Exceptions
class StockMediaError(Exception):
    """Base exception for stock media provider errors."""
    pass

class PexelsAPIError(StockMediaError):
    """Raised when the Pexels API returns an unexpected error."""
    pass

class PexelsRateLimitError(PexelsAPIError):
    """Raised when Pexels API rate limit is exceeded after retries."""
    pass

class PexelsAuthenticationError(PexelsAPIError):
    """Raised when Pexels API authentication fails due to missing or invalid key."""
    pass

class NoSuitableMediaFoundError(StockMediaError):
    """Raised when no media item meeting selection criteria is found for keywords."""
    pass

class UncertainFootageRequiresApprovalError(StockMediaError):
    """
    Raised when candidate stock video footage is visually uncertain and requires explicit
    manual reviewer approval before it can proceed to final video assembly.
    Never silently sent to final assembly.
    """
    def __init__(self, message: str, pending_approvals: list = None):
        super().__init__(message)
        self.pending_approvals = pending_approvals or []

# TTS Provider Exceptions
class TTSError(Exception):
    """Base exception for text-to-speech provider errors."""
    pass

class TTSConnectionError(TTSError):
    """Raised when edge-tts service or network dependency is unavailable or fails."""
    pass

class TTSValidationError(TTSError):
    """Raised when scene text for TTS is invalid, empty, or whitespace-only."""
    pass

class TTSAudioCorruptError(TTSError):
    """Raised when generated audio file cannot be decoded or has 0 duration."""
    pass


