from typing import Optional
from app.config import settings
from app.services.providers.base import ScriptProvider

def get_script_provider(provider_name: Optional[str] = None) -> ScriptProvider:
    name = (provider_name or settings.SCRIPT_PROVIDER).strip().lower()
    if name == "groq":
        from app.services.providers.groq_script_provider import GroqScriptProvider
        return GroqScriptProvider()
    elif name == "ollama":
        from app.services.providers.ollama_script_provider import OllamaScriptProvider
        return OllamaScriptProvider()
    else:
        raise ValueError(
            f"Unsupported script provider '{name}'. Supported providers are: 'groq', 'ollama'."
        )
