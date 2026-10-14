from app.config import settings
from app.services.providers.base import ScriptProvider
from app.services.providers.ollama_script_provider import OllamaScriptProvider
from app.services.providers.groq_script_provider import GroqScriptProvider

def get_script_provider(provider_name: str = None) -> ScriptProvider:
    name = (provider_name or settings.SCRIPT_PROVIDER).lower()
    if name == "groq":
        return GroqScriptProvider()
    return OllamaScriptProvider()
