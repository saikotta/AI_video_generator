import hashlib
from typing import Optional
from app.config import settings

class ScriptCache:
    def __init__(self):
        self._cache = {}

    def _generate_key(self, topic: str, duration_seconds: int, wpm: int) -> str:
        raw = f"{topic.strip().lower()}:{duration_seconds}:{wpm}"
        return hashlib.sha256(raw.encode('utf-8')).hexdigest()

    def get_script(self, topic: str, duration_seconds: int, wpm: int) -> Optional[str]:
        if not settings.ENABLE_SCRIPT_CACHE:
            return None
        key = self._generate_key(topic, duration_seconds, wpm)
        return self._cache.get(key)

    def set_script(self, topic: str, duration_seconds: int, wpm: int, script: str) -> None:
        if not settings.ENABLE_SCRIPT_CACHE:
            return
        key = self._generate_key(topic, duration_seconds, wpm)
        self._cache[key] = script

    def clear(self) -> None:
        self._cache.clear()

script_cache = ScriptCache()
