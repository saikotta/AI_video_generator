from app.services.providers.base import ScriptProvider
from app.services.providers.factory import get_script_provider
from app.services.providers.exceptions import (
    ScriptProviderError,
    OllamaUnavailableError,
    ScriptGenerationTimeoutError,
    InvalidScriptResponseError
)

__all__ = [
    "ScriptProvider",
    "OllamaScriptProvider",
    "GroqScriptProvider",
    "get_script_provider",
    "ScriptProviderError",
    "OllamaUnavailableError",
    "ScriptGenerationTimeoutError",
    "InvalidScriptResponseError"
]

def __getattr__(name: str):
    if name == "GroqScriptProvider":
        from app.services.providers.groq_script_provider import GroqScriptProvider
        return GroqScriptProvider
    if name == "OllamaScriptProvider":
        from app.services.providers.ollama_script_provider import OllamaScriptProvider
        return OllamaScriptProvider
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
