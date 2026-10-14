import os
import asyncio
import logging
from typing import List, Dict, Any, Optional, Set, Tuple
import httpx

from app.config import settings
from app.services.providers.exceptions import (
    StockMediaError,
    PexelsAPIError,
    PexelsRateLimitError,
    PexelsAuthenticationError
)

logger = logging.getLogger(__name__)

class FreesoundAudioProvider:
    """
    Freesound.org API v2 provider for action-oriented sound effects matching specific visible scene actions.
    Enforces strict physical action verification, segment-level timestamp placement, and graceful fallback to NO SFX.
    """

    FREESOUND_SEARCH_URL = "https://freesound.org/apiv2/search/text/"

    # Action mappings: Keyword -> Queries, Action Terms required, Human-readable Action Description
    ACTION_MAPPINGS = {
        "coffee beans": {
            "queries": ["pouring coffee beans", "coffee beans falling", "coffee beans scoop"],
            "action_terms": ["pouring", "pour", "falling", "drop", "scoop", "rattling", "beans"],
            "action_name": "coffee beans pouring / dropping / rattling action"
        },
        "coffee origin": {
            "queries": ["harvesting coffee", "plucking coffee berries", "foliage rustle"],
            "action_terms": ["harvest", "picking", "plucking", "foliage", "leaf", "rustle"],
            "action_name": "coffee plant / harvesting / picking action"
        },
        "coffeehouse": {
            "queries": ["coffee shop chatter", "quiet cafe chatter"],
            "action_terms": ["chatter", "crowd", "walla", "voices", "cafe", "restaurant"],
            "action_name": "coffeehouse social gathering chatter"
        },
        "coffee culture": {
            "queries": ["coffee preparation", "pouring coffee into cup"],
            "action_terms": ["pour", "pouring", "cup", "preparation", "simmer", "brew"],
            "action_name": "coffee preparation action"
        },
        "european coffeehouse": {
            "queries": ["cafe chatter", "coffeehouse chatter"],
            "action_terms": ["chatter", "cafe", "walla", "restaurant"],
            "action_name": "European cafe chatter & clatter"
        },
        "coffee shop Europe": {
            "queries": ["espresso machine extraction", "coffee pouring into cup"],
            "action_terms": ["espresso", "extraction", "pouring", "cup"],
            "action_name": "espresso machine extraction action"
        },
        "colonial coffee": {
            "queries": ["pouring coffee into mug", "pouring hot water coffee"],
            "action_terms": ["pour", "pouring", "liquid", "mug", "pot", "simmer", "bubble"],
            "action_name": "coffee / hot liquid pouring into mug action"
        },
        "tea to coffee": {
            "queries": ["stirring coffee cup", "spoon in coffee cup", "pouring tea into cup"],
            "action_terms": ["stirring", "stir", "spoon", "clink", "pour", "pouring"],
            "action_name": "spoon stirring / pouring tea cup action"
        },
        "global coffee industry": {
            "queries": ["coffee roaster machine", "roasting coffee beans"],
            "action_terms": ["roaster", "roasting", "machine", "drum", "mill"],
            "action_name": "coffee roaster machine roasting action"
        },
        "modern coffee roaster": {
            "queries": ["coffee roaster machine", "espresso machine extraction"],
            "action_terms": ["roaster", "roasting", "machine", "extraction", "mill"],
            "action_name": "modern coffee roaster / extraction machine action"
        },
        "coffee grinder": {
            "queries": ["coffee grinder grinding", "coffee bean grinder"],
            "action_terms": ["grinder", "grind", "grinding"],
            "action_name": "coffee bean grinding action"
        },
        "coffee harvest": {
            "queries": ["harvesting coffee", "plucking coffee berries", "foliage rustle"],
            "action_terms": ["harvest", "picking", "plucking", "foliage", "leaf", "leaves", "rustle"],
            "action_name": "coffee harvesting action"
        },
        "coffee plant": {
            "queries": ["harvesting coffee", "foliage rustle", "leaves rustling"],
            "action_terms": ["harvest", "picking", "foliage", "leaf", "leaves", "rustle"],
            "action_name": "coffee plant / foliage rustle action"
        },
        "coffee picking": {
            "queries": ["harvesting coffee", "plucking coffee berries"],
            "action_terms": ["harvest", "picking", "plucking", "foliage", "leaf", "rustle"],
            "action_name": "coffee berry picking action"
        },
        "coffee berries": {
            "queries": ["harvesting coffee", "foliage rustle"],
            "action_terms": ["harvest", "picking", "foliage", "leaf", "rustle"],
            "action_name": "coffee berries harvesting action"
        },
        "coffee farming": {
            "queries": ["harvesting coffee", "foliage rustle"],
            "action_terms": ["harvest", "picking", "foliage", "rustle"],
            "action_name": "coffee farming / harvest action"
        },
        "foliage rustle": {
            "queries": ["foliage rustle", "leaves rustling"],
            "action_terms": ["foliage", "rustle", "leaf", "leaves"],
            "action_name": "foliage rustling action"
        },
        "coffee beans scoop": {
            "queries": ["coffee beans scoop", "pouring coffee beans"],
            "action_terms": ["scoop", "beans", "pouring", "rattling"],
            "action_name": "coffee beans scoop action"
        },
        "espresso machine": {
            "queries": ["espresso machine extraction", "espresso stream"],
            "action_terms": ["espresso", "extraction", "pump", "steam"],
            "action_name": "espresso machine extraction action"
        },
        "espresso extraction": {
            "queries": ["espresso machine extraction", "espresso stream"],
            "action_terms": ["espresso", "extraction", "pump", "steam"],
            "action_name": "espresso extraction action"
        },
        "coffee roasting": {
            "queries": ["coffee roaster machine", "roasting coffee beans"],
            "action_terms": ["roaster", "roasting", "machine", "drum"],
            "action_name": "coffee roasting action"
        },
        "coffee roaster": {
            "queries": ["coffee roaster machine", "roasting coffee beans"],
            "action_terms": ["roaster", "roasting", "machine", "drum"],
            "action_name": "coffee roaster machine action"
        },
        "coffee pouring": {
            "queries": ["pouring coffee into cup", "pouring liquid cup"],
            "action_terms": ["pour", "pouring", "stream", "cup"],
            "action_name": "coffee pouring action"
        },
        "coffeehouse chatter": {
            "queries": ["coffee shop chatter", "cafe chatter"],
            "action_terms": ["chatter", "crowd", "walla", "voices", "cafe", "restaurant"],
            "action_name": "coffeehouse chatter ambience"
        },
        "french cafe": {
            "queries": ["cafe chatter", "coffee shop chatter"],
            "action_terms": ["chatter", "cafe", "walla", "restaurant"],
            "action_name": "cafe chatter ambience"
        },
        "french coffee": {
            "queries": ["pouring coffee into cup", "coffee preparation"],
            "action_terms": ["pour", "pouring", "cup", "preparation"],
            "action_name": "coffee pouring / prep action"
        },
        "industrial coffee": {
            "queries": ["coffee roaster machine", "industry processing machine"],
            "action_terms": ["machine", "motor", "mill", "roaster", "industrial"],
            "action_name": "industrial processing machine action"
        },
        "coffee machine": {
            "queries": ["espresso machine extraction", "coffee roaster machine"],
            "action_terms": ["machine", "espresso", "extraction", "motor"],
            "action_name": "coffee machine operation action"
        },
        "specialty coffee": {
            "queries": ["espresso machine extraction", "coffee preparation"],
            "action_terms": ["espresso", "extraction", "pour", "grind", "steam"],
            "action_name": "specialty coffee preparation action"
        },
        "coffee barista": {
            "queries": ["espresso machine extraction", "coffee preparation"],
            "action_terms": ["espresso", "extraction", "pour", "grind", "steam"],
            "action_name": "barista coffee preparation action"
        }
    }

    # Hard rejection keywords (Music, Unrelated Ambiences, Speech, Mismatched Actions)
    EXCLUDED_TERMS = {
        "music", "song", "melody", "synth", "techno", "beat", "tune", "piano", "guitar",
        "orchestra", "dark", "sorrow", "scary", "horror", "dramatic", "loop", "instrumental",
        "bpm", "chords", "album", "soundtrack", "dystopian", "cinematic",
        "traffic", "city", "covid", "siren", "car", "highway", "storm", "thunder", "rain",
        "jungle", "ocean", "beach", "forest", "birds", "zoo", "dog", "cat", "construction",
        "train", "plane", "street", "town", "outdoor",
        "talkshow", "podcast", "interview", "speech", "narration", "dialogue", "voiceover",
        "sip", "sipping", "slurp", "cough", "snore", "chewing", "swallow"
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        max_retries: int = 3,
        retry_base_delay: float = 1.0
    ):
        self.api_key = (api_key or settings.FREESOUND_API_KEY or os.getenv("FREESOUND_API_KEY", "")).strip()
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay

        if not self.api_key:
            logger.warning("FREESOUND_API_KEY is missing. Freesound integration disabled.")

    def _is_hard_rejected(self, sound: Dict[str, Any]) -> Tuple[bool, str]:
        """Check if sound metadata contains any hard excluded terms."""
        name = sound.get("name", "").lower()
        tags = [t.lower() for t in sound.get("tags", [])]
        description = sound.get("description", "").lower()
        combined = f"{name} {' '.join(tags)} {description[:300]}"

        for term in self.EXCLUDED_TERMS:
            if term in combined:
                return True, f"Contains hard excluded term '{term}'"
        return False, ""

    async def _make_api_request(self, client: httpx.AsyncClient, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute GET request to Freesound API with token auth and retry logic."""
        request_params = dict(params)
        request_params["token"] = self.api_key

        for attempt in range(self.max_retries + 1):
            try:
                response = await client.get(self.FREESOUND_SEARCH_URL, params=request_params, timeout=15.0)

                if response.status_code in (401, 403):
                    raise PexelsAuthenticationError(f"Freesound API authentication failed (HTTP {response.status_code}).")

                if response.status_code == 429:
                    if attempt < self.max_retries:
                        await asyncio.sleep(self.retry_base_delay * (2 ** attempt))
                        continue
                    raise PexelsRateLimitError("Freesound API rate limit exceeded.")

                if response.status_code >= 500:
                    if attempt < self.max_retries:
                        await asyncio.sleep(self.retry_base_delay * (2 ** attempt))
                        continue
                    raise PexelsAPIError(f"Freesound API server error HTTP {response.status_code}")

                if response.status_code != 200:
                    raise PexelsAPIError(f"Freesound API error HTTP {response.status_code}")

                return response.json()

            except (httpx.TimeoutException, httpx.NetworkError) as e:
                if attempt < self.max_retries:
                    await asyncio.sleep(self.retry_base_delay * (2 ** attempt))
                    continue
                raise PexelsAPIError(f"Freesound API network request failed: {e}")

        raise PexelsAPIError("Freesound retry loop terminated unexpectedly.")

    async def search_segment_action_sfx(
        self,
        keyword: str,
        segment_duration: float,
        used_sound_ids: Optional[Set[int]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Search Freesound API for a specific physical action SFX matching the visual clip segment.
        If no candidate strongly matches the specific physical action, returns None (NO SUITABLE SFX).
        """
        if not self.api_key or not keyword:
            return None

        spec = self.ACTION_MAPPINGS.get(keyword.lower().strip())
        if not spec:
            logger.info(f"No specific action mapping defined for keyword '{keyword}'. Returning NO SFX.")
            return None

        queries = list(spec["queries"])
        action_terms = list(spec["action_terms"])
        action_name = spec["action_name"]
        active_used_ids = used_sound_ids if used_sound_ids is not None else set()

        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            for q in queries:
                params = {
                    "query": q,
                    "fields": "id,name,tags,license,previews,duration,username,description",
                    "filter": "duration:[2.0 TO 180.0]",
                    "page_size": 15
                }
                try:
                    data = await self._make_api_request(client, params)
                except Exception as e:
                    logger.warning(f"Freesound search error for query '{q}': {e}")
                    continue

                results = data.get("results", [])
                for sound in results:
                    sound_id = sound.get("id")
                    if not sound_id or sound_id in active_used_ids:
                        continue

                    rejected, _ = self._is_hard_rejected(sound)
                    if rejected:
                        continue

                    name = sound.get("name", "")
                    tags = [t.lower() for t in sound.get("tags", [])]
                    previews = sound.get("previews", {})
                    download_url = previews.get("preview-hq-mp3") or previews.get("preview-lq-mp3")
                    duration = float(sound.get("duration", 0.0))

                    if not download_url or duration < 2.0:
                        continue

                    combined = f"{name.lower()} {' '.join(tags)}"

                    # Verify at least one required action term is explicitly present
                    matched_action_terms = [term for term in action_terms if term in combined]
                    if not matched_action_terms:
                        continue

                    matched_terms_str = ", ".join(matched_action_terms)
                    justification = f"Specifically matches {action_name} ({matched_terms_str})"

                    # Verify source recording loudness before committing (must have RMS >= 400.0 to contribute audibly under 1.5x cap)
                    try:
                        prev_res = await client.get(download_url, follow_redirects=True)
                        if prev_res.status_code == 200:
                            import io, av as av_mod
                            buf = io.BytesIO(prev_res.content)
                            with av_mod.open(buf) as av_c:
                                frames = [f.to_ndarray() for f in av_c.decode(audio=0)]
                            if frames:
                                pcm_f = np.concatenate(frames, axis=1)
                                cand_rms = float(np.sqrt(np.mean(pcm_f ** 2))) * 32768.0
                                if cand_rms < 400.0:
                                    logger.info(f"Rejecting candidate {sound_id} ({name}): too quiet at source (RMS={cand_rms:.1f} < 400.0). Retrying next...")
                                    continue
                    except Exception as err:
                        logger.debug(f"Could not verify loudness for sound {sound_id}: {err}")

                    license_url = sound.get("license", "")
                    license_name = "CC0 Public Domain" if ("zero" in license_url or "publicdomain" in license_url) else "CC-BY Attribution"
                    username = sound.get("username", "Unknown")

                    active_used_ids.add(sound_id)
                    logger.info(f"Matched Action SFX ID {sound_id} ({name}) for keyword '{keyword}' -> {justification}")

                    return {
                        "freesound_id": sound_id,
                        "name": name,
                        "download_url": download_url,
                        "duration": duration,
                        "username": username,
                        "license": license_name,
                        "license_url": license_url,
                        "search_query": q,
                        "action_name": action_name,
                        "justification": justification,
                        "keyword": keyword
                    }

        logger.info(f"No high-confidence action SFX matched on Freesound for keyword '{keyword}'. Segment will have NO SFX.")
        return None

    async def download_audio(self, download_url: str, output_dir: str, filename_prefix: str) -> str:
        """Download sound effect MP3 preview into output_dir."""
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, f"{filename_prefix}.mp3")

        if os.path.exists(file_path) and os.path.getsize(file_path) > 3000:
            return file_path

        async with httpx.AsyncClient(timeout=45.0, follow_redirects=True) as client:
            for attempt in range(self.max_retries + 1):
                try:
                    res = await client.get(download_url)
                    if res.status_code == 200:
                        content = res.content
                        if len(content) < 3000:
                            raise StockMediaError(f"Downloaded audio file size too small ({len(content)} bytes).")

                        with open(file_path, "wb") as f:
                            f.write(content)

                        return file_path

                    if res.status_code == 429 and attempt < self.max_retries:
                        await asyncio.sleep(self.retry_base_delay * (2 ** attempt))
                        continue

                    raise StockMediaError(f"Failed to download audio: HTTP {res.status_code}")

                except (httpx.TimeoutException, httpx.NetworkError) as e:
                    if attempt < self.max_retries:
                        await asyncio.sleep(self.retry_base_delay * (2 ** attempt))
                        continue
                    raise StockMediaError(f"Network error downloading audio: {e}")

        raise StockMediaError(f"Download failed after retries for {download_url}")
