import os
import re
import asyncio
import logging
from typing import List, Dict, Any, Optional, Set, Tuple
import httpx
import numpy as np

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

    # Universal hard rejection keywords (Music, Speech, Bodily Noises, Memes, Unrelated Violence/Gore)
    EXCLUDED_TERMS = {
        "music", "song", "melody", "synth", "techno", "beat", "tune", "piano", "guitar",
        "orchestra", "dark", "sorrow", "scary", "horror", "dramatic", "loop", "instrumental",
        "bpm", "chords", "album", "soundtrack", "dystopian", "cinematic", "remix", "band",
        "talkshow", "podcast", "interview", "speech", "narration", "dialogue", "voiceover",
        "monologue", "singing", "acapella", "vocal", "rap", "hip hop", "dance", "electronic",
        "bass", "drum", "percussion", "lead", "riff",
        "cough", "snore", "burp", "fart", "slurp", "chewing", "swallow", "sneezing",
        "screamer", "jump scare", "earrape",
        "gore", "flesh", "blood", "body hit", "human being hit", "stab", "punch", "slap", "whip",
        "splatter", "screaming", "grunt", "gunshot", "explosion"
    }

    # Synthetic, musical, and ethereal atmospheric texture descriptors to reject for physical Foley
    SYNTHETIC_ATMOSPHERE_TERMS = {
        "celestial", "divine", "ethereal", "drone", "drones", "pad", "pads",
        "meditation", "dreamy", "psychedelic", "astral", "new age", "chillout",
        "binaural", "frequency", "soundscape", "soundscapes", "texture", "textures",
        "synthesizer", "ambient texture", "synthetic", "fantasy pad", "musical texture",
        "electronic texture", "ambient drone"
    }

    # Ambience detection keywords in queries
    AMBIENCE_QUERY_KEYWORDS = {
        "ambience", "ambient", "room tone", "atmosphere", "background", "walla"
    }

    # Stop words to filter out when constructing dynamic search terms
    STOP_WORDS = {
        "a", "an", "the", "in", "on", "at", "to", "into", "onto", "of", "for", "with",
        "by", "from", "and", "or", "while", "as", "is", "are", "beneath", "slowly",
        "across", "over", "under", "through", "downward", "upward"
    }

    # Visual-only keywords that pollute audio search queries
    VISUAL_FILTER_WORDS = {
        "timelapse", "macro", "cinematic", "4k", "hd", "1080p", "footage", "video",
        "clip", "close up", "close-up", "slow motion", "aerial", "shot", "framing",
        "photograph", "photo", "wide shot", "view"
    }

    # Non-acoustic descriptive adjectives that shouldn't lead search queries
    DESCRIPTIVE_MODIFIERS = {
        "soft", "gently", "gentle", "slow", "slowly", "fast", "quickly",
        "quiet", "quietly", "loud", "loudly", "tiny", "small", "big",
        "large", "deep", "shallow", "heavy", "light", "delicate",
        "subtle", "smooth", "rough", "sharp"
    }

    @classmethod
    def _clean_audio_keyword(cls, keyword: str) -> str:
        """Strip visual-only terminology and excess punctuation from audio search queries."""
        kw_clean = keyword.lower().strip()
        for v_term in cls.VISUAL_FILTER_WORDS:
            kw_clean = re.sub(r'\b' + re.escape(v_term) + r'\b', ' ', kw_clean)
        kw_clean = re.sub(r'[^\w\s]', ' ', kw_clean)
        kw_clean = " ".join(kw_clean.split())
        return kw_clean

    @classmethod
    def _detect_sound_category(cls, keyword: str) -> str:
        """Classify requested sound query into 'ambience' or 'foley'."""
        kw_lower = keyword.lower()
        if any(re.search(r'\b' + re.escape(w) + r'\b', kw_lower) for w in cls.AMBIENCE_QUERY_KEYWORDS):
            return "ambience"
        if any(w in kw_lower for w in ["rain on", "rainfall", "thunderstorm", "ocean waves", "forest atmosphere", "cafe chatter", "room tone"]):
            return "ambience"
        return "foley"

    @classmethod
    def _build_dynamic_spec(cls, keyword: str) -> Dict[str, Any]:
        """
        Dynamically construct candidate queries, physical subject terms, action/motion terms,
        sound category, and human-readable descriptions for arbitrary physical action and environmental sound requests.
        """
        cleaned = cls._clean_audio_keyword(keyword)
        sound_category = cls._detect_sound_category(cleaned)
        tokens = [w for w in cleaned.split() if w not in cls.STOP_WORDS and len(w) > 1]

        if not tokens:
            tokens = [cleaned] if cleaned else ["sound"]

        # Filter out pure descriptive adjectives from leading query positions
        salient_tokens = [w for w in tokens if w not in cls.DESCRIPTIVE_MODIFIERS]
        if not salient_tokens:
            salient_tokens = tokens

        queries: List[str] = []
        if len(salient_tokens) >= 3:
            queries.append(" ".join(salient_tokens[:3]))
            queries.append(" ".join(salient_tokens[:2]))
            queries.append(salient_tokens[0])
        elif len(salient_tokens) == 2:
            queries.append(" ".join(salient_tokens))
            queries.append(salient_tokens[0])
            queries.append(salient_tokens[1])
        else:
            queries.append(salient_tokens[0])

        unique_queries: List[str] = []
        for q in queries:
            if q and q not in unique_queries:
                unique_queries.append(q)

        # Build subjects and actions
        token_str = " ".join(tokens)
        subject_terms: Set[str] = set()
        action_terms: Set[str] = set()

        # 1. Grains, Seeds & Agriculture
        if any(w in token_str for w in ["grain", "seed", "wheat", "rice", "corn", "bean", "cereal"]):
            subject_terms.update(["grain", "grains", "seed", "seeds", "wheat", "rice", "corn", "bean", "beans", "cereal", "stone", "table", "surface", "floor", "hard"])
            action_terms.update(["scatter", "scattering", "drop", "dropping", "pour", "pouring", "spill", "spilling", "fall", "falling", "rattle", "rattling", "clatter", "clattering", "foley", "sfx", "impact", "hit"])

        # 2. Water & Liquids
        elif any(w in token_str for w in ["water", "liquid", "drop", "droplet", "drip", "splash", "pour", "stream", "flow", "glass", "cup", "mug"]):
            subject_terms.update(["water", "liquid", "glass", "cup", "mug", "pool", "puddle", "ice", "bottle", "sink", "surface"])
            action_terms.update(["pour", "pouring", "splash", "splashing", "drip", "dripping", "drop", "droplet", "droplets", "trickle", "trickling", "stream", "flow", "spill", "bubble", "foley", "sfx"])

        # 3. Footsteps & Walking
        elif any(w in token_str for w in ["footstep", "walk", "step", "run", "tread", "gravel", "dirt", "shoe", "boot"]):
            subject_terms.update(["gravel", "dirt", "path", "pavement", "ground", "floor", "wood", "grass", "shoes", "boots", "shoe", "boot"])
            action_terms.update(["footstep", "footsteps", "walk", "walking", "step", "steps", "tread", "treading", "pace", "pacing", "crunch", "crunching", "stride", "foley", "sfx"])

        # 4. Leaves & Foliage
        elif any(w in token_str for w in ["leaf", "leaves", "foliage", "rustle", "wind", "breeze", "branch", "tree", "plant"]):
            subject_terms.update(["leaf", "leaves", "foliage", "branch", "branches", "tree", "trees", "plant", "plants", "twig", "twigs", "wood", "sprout"])
            action_terms.update(["rustle", "rustling", "breeze", "wind", "swish", "swishing", "flutter", "fluttering", "crack", "snap", "foley", "sfx"])

        # 5. Soil & Earth
        elif any(w in token_str for w in ["soil", "dirt", "earth", "ground", "mud", "sand", "clay", "dig", "displace"]):
            subject_terms.update(["soil", "dirt", "earth", "ground", "mud", "sand", "clay", "dust"])
            action_terms.update(["dig", "digging", "displace", "displacing", "displacement", "shovel", "scrape", "scraping", "crumble", "crumbling", "shift", "shifting", "movement", "move", "foley", "sfx"])

        # 6. Doors & Latches
        elif any(w in token_str for w in ["door", "close", "shut", "open", "slam", "creak", "knock", "latch"]):
            subject_terms.update(["door", "gate", "window", "wood", "metal", "latch", "lock", "hinge"])
            action_terms.update(["door", "close", "closing", "shut", "slam", "creak", "knock", "latch", "lock", "hinge", "click", "foley", "sfx"])

        # 7. Vehicles & Engines
        elif any(w in token_str for w in ["car", "engine", "motor", "vehicle", "start", "drive", "rev"]):
            subject_terms.update(["car", "vehicle", "engine", "motor", "machine"])
            action_terms.update(["engine", "motor", "start", "starting", "ignition", "idle", "rev", "drive", "machine", "foley", "sfx"])

        # 8. Environmental Ambience (forest, rain, chatter, etc.)
        elif any(w in token_str for w in ["forest", "woods", "rain", "storm", "thunder", "wind", "ocean", "waves", "chatter", "cafe", "birds"]):
            subject_terms.update(["forest", "woods", "rain", "rainfall", "storm", "thunder", "wind", "ocean", "waves", "cafe", "birds", "nature", "wilderness"])
            action_terms.update(["ambience", "ambient", "atmosphere", "room tone", "field recording", "background", "weather", "wildlife", "chirp", "chirping"])

        else:
            # Generic fallback
            subject_terms.update([t for t in tokens if t not in cls.DESCRIPTIVE_MODIFIERS])
            action_terms.update(["foley", "sfx", "sound", "impact", "action", "movement"] + tokens)

        action_name = f"{cleaned} sound effect"
        return {
            "queries": unique_queries,
            "subject_terms": sorted(list(subject_terms)),
            "action_terms": sorted(list(action_terms)),
            "sound_category": sound_category,
            "action_name": action_name
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

    def _is_hard_rejected(
        self,
        sound: Dict[str, Any],
        query_terms: Optional[Set[str]] = None,
        sound_category: str = "foley"
    ) -> Tuple[bool, str]:
        """
        Check if sound metadata contains any hard excluded terms (music, speech, bodily noises),
        or category-inappropriate synthetic atmospheric textures.
        """
        if not isinstance(sound, dict):
            return True, "Sound metadata is not a valid dictionary"

        name = str(sound.get("name") or "").lower()
        raw_tags = sound.get("tags") or []
        if isinstance(raw_tags, str):
            raw_tags = raw_tags.split()
        elif not isinstance(raw_tags, (list, tuple, set)):
            raw_tags = []
        tags = [str(t).lower() for t in raw_tags if t]
        desc = str(sound.get("description") or "").lower()[:400]
        combined = f"{name} {' '.join(tags)} {desc}"

        protected = query_terms if query_terms else set()

        # 1. Universal hard rejection (music, song, speech, vocals, etc.)
        for term in self.EXCLUDED_TERMS:
            if term in combined and term not in protected:
                return True, f"Contains hard excluded term '{term}'"

        # 2. Category-specific rejection for Physical Action Foley
        if sound_category == "foley":
            for term in self.SYNTHETIC_ATMOSPHERE_TERMS:
                if term in combined and term not in protected:
                    return True, f"Contains synthetic/musical atmosphere term '{term}' for physical Foley request"
            if "ambient" in combined and "ambient" not in protected:
                return True, "Contains 'ambient' descriptor for physical Foley request"
            if "texture" in combined and "texture" not in protected:
                return True, "Contains 'texture' descriptor for physical Foley request"

        # 3. Category-specific rejection for Environmental Ambience
        elif sound_category == "ambience":
            for term in ["synth", "synthesizer", "celestial", "divine", "meditation", "dreamy", "psychedelic", "astral", "new age"]:
                if term in combined and term not in protected:
                    return True, f"Contains musical/synthetic term '{term}' for environmental ambience request"

        return False, ""

    def _check_topic_conflict(self, sound: Dict[str, Any], topic: Optional[str]) -> Tuple[bool, str]:
        """Check if candidate sound has an explicit domain conflict with the video topic."""
        if not topic or not isinstance(sound, dict):
            return False, ""

        topic_lower = topic.lower()
        s_name = str(sound.get("name") or "").lower()
        raw_tags = sound.get("tags") or []
        tags_str = " ".join(raw_tags) if isinstance(raw_tags, list) else str(raw_tags)
        s_desc = str(sound.get("description") or "").lower()[:300]
        cand_all_text = f"{s_name} {tags_str} {s_desc}".lower()

        if any(w in topic_lower for w in ("cook", "stir fry", "recipe", "kitchen", "food", "vegetable", "baking")):
            if any(w in cand_all_text for w in ("human being", "gore", "body hit", "punch", "weapon", "engine", "motor", "gun", "combat", "bathtub", "bathroom")):
                return True, f"Candidate sound conflicts with cooking topic: {cand_all_text[:60]}"
        elif any(w in topic_lower for w in ("seed", "sprout", "plant", "botany", "germination", "photosynthesis")):
            if any(w in cand_all_text for w in ("kitchen", "chef", "cook", "recipe", "punch", "gun", "motor", "engine", "combat")):
                return True, f"Candidate sound conflicts with botanical/plant topic: {cand_all_text[:60]}"
        elif any(w in topic_lower for w in ("bath", "bathing", "infant", "baby bath")):
            if any(w in cand_all_text for w in ("kitchen", "cooking", "knife", "engine", "motor", "gun", "combat")):
                return True, f"Candidate sound conflicts with bathing topic: {cand_all_text[:60]}"

        return False, ""

    def _score_candidate(
        self,
        sound: Dict[str, Any],
        spec: Dict[str, Any]
    ) -> Tuple[float, List[str]]:
        """
        Compute semantic relevance score based on physical subject, action/motion,
        acoustic event category, candidate title, candidate tags, and CC0 licensing.
        Enforces dual-match requirement: Foley candidates cannot qualify on generic nouns alone.
        """
        if not isinstance(sound, dict):
            return 0.0, []

        name = str(sound.get("name") or "")
        name_lower = name.lower()
        raw_tags = sound.get("tags") or []
        if isinstance(raw_tags, str):
            raw_tags = raw_tags.split()
        elif not isinstance(raw_tags, (list, tuple, set)):
            raw_tags = []
        tags_lower = [str(t).lower() for t in raw_tags if t]
        desc_lower = str(sound.get("description") or "").lower()[:400]
        combined_tags = " ".join(tags_lower)

        subject_terms = spec.get("subject_terms") or []
        action_terms = spec.get("action_terms") or []
        sound_category = spec.get("sound_category", "foley")

        if not subject_terms:
            subject_terms = [t for t in action_terms if t in {"beans", "coffee", "cup", "pot", "mug", "machine", "grinder", "roaster", "leaf", "leaves", "foliage", "cafe", "restaurant"}] or action_terms

        matched_subj_title = [s for s in subject_terms if re.search(r'\b' + re.escape(s) + r'\b', name_lower)]
        matched_subj_tags = [s for s in subject_terms if re.search(r'\b' + re.escape(s) + r'\b', combined_tags)]
        matched_subj_desc = [s for s in subject_terms if re.search(r'\b' + re.escape(s) + r'\b', desc_lower)]

        matched_act_title = [a for a in action_terms if re.search(r'\b' + re.escape(a) + r'\b', name_lower)]
        matched_act_tags = [a for a in action_terms if re.search(r'\b' + re.escape(a) + r'\b', combined_tags)]
        matched_act_desc = [a for a in action_terms if re.search(r'\b' + re.escape(a) + r'\b', desc_lower)]

        all_subjects = sorted(list(set(matched_subj_title + matched_subj_tags + matched_subj_desc)))
        all_actions = sorted(list(set(matched_act_title + matched_act_tags + matched_act_desc)))

        # Strict requirement for Physical Foley:
        # Must match at least one subject/material AND at least one action/motion/acoustic event.
        # A candidate matching solely on one generic noun (e.g. 'grain') without any action match is rejected!
        if sound_category == "foley":
            if not all_subjects or not all_actions:
                return 0.0, []

        # Requirement for Ambience: must match environmental subject and atmosphere descriptor
        elif sound_category == "ambience":
            if not all_subjects and not all_actions:
                return 0.0, []

        score = 0.0
        score += len(matched_subj_title) * 3.5
        score += len(matched_act_title) * 3.5
        score += len(matched_subj_tags) * 2.0
        score += len(matched_act_tags) * 2.0
        score += (len(matched_subj_desc) + len(matched_act_desc)) * 0.5

        # Co-occurrence bonus: both subject and action in title or tags
        has_subj_top = bool(matched_subj_title or matched_subj_tags)
        has_act_top = bool(matched_act_title or matched_act_tags)
        if has_subj_top and has_act_top:
            score += 4.0

        # Dedicated Foley / Field-recording credibility bonus
        if any(t in combined_tags or t in name_lower for t in ["foley", "field recording", "sfx", "sound effect"]):
            score += 2.0

        # CC0 public domain license bonus (favors commercial CC0 content)
        lic = (sound.get("license") or "").lower()
        if "zero" in lic or "publicdomain" in lic:
            score += 3.0

        reasons = [f"subjects: {all_subjects}", f"actions: {all_actions}"]
        return score, reasons

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
        used_sound_ids: Optional[Set[int]] = None,
        topic: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Search Freesound API for a specific physical action SFX matching the visual clip segment.
        Supports both curated action mappings and arbitrary dynamic sound queries.
        Uses category separation (Foley vs Ambience), multi-field semantic scoring, and conservative fallback.
        If no candidate strongly matches the specific physical action, returns None (NO SUITABLE SFX).
        """
        if not self.api_key or not keyword:
            return None

        clean_kw = keyword.lower().strip()
        spec = self.ACTION_MAPPINGS.get(clean_kw)

        if not spec:
            spec = self._build_dynamic_spec(keyword)
            logger.info(f"Using dynamic action SFX specification for '{keyword}': queries={spec['queries']}")
        else:
            spec = dict(spec)
            if "sound_category" not in spec:
                spec["sound_category"] = self._detect_sound_category(spec.get("action_name", clean_kw))
            if "subject_terms" not in spec:
                spec["subject_terms"] = [t for t in spec["action_terms"] if t in {"beans", "coffee", "cup", "pot", "mug", "machine", "grinder", "roaster", "leaf", "leaves", "foliage", "cafe", "restaurant"}] or spec["action_terms"]

        queries = list(spec["queries"])
        action_name = spec["action_name"]
        sound_category = spec.get("sound_category", "foley")
        active_used_ids = used_sound_ids if used_sound_ids is not None else set()
        query_token_set = set(clean_kw.split())

        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            for q in queries:
                params = {
                    "query": q,
                    "fields": "id,name,tags,license,previews,duration,username,description",
                    "filter": "duration:[1.0 TO 180.0]",
                    "page_size": 15
                }
                try:
                    data = await self._make_api_request(client, params)
                except Exception as e:
                    logger.warning(f"Freesound search error for query '{q}': {e}")
                    continue

                results = data.get("results") or []
                ranked_candidates = []

                for sound in results:
                    if not isinstance(sound, dict):
                        continue

                    sound_id = sound.get("id")
                    if not sound_id or sound_id in active_used_ids:
                        continue

                    rejected, reject_reason = self._is_hard_rejected(sound, query_token_set, sound_category)
                    if rejected:
                        logger.debug(f"Candidate {sound_id} rejected: {reject_reason}")
                        continue

                    # Topic-aware domain exclusion check
                    conflict, conflict_reason = self._check_topic_conflict(sound, topic)
                    if conflict:
                        logger.debug(f"Candidate {sound_id} rejected due to topic conflict: {conflict_reason}")
                        continue

                    try:
                        duration = float(sound.get("duration", 0.0) or 0.0)
                    except (ValueError, TypeError):
                        continue

                    previews = sound.get("previews")
                    if not isinstance(previews, dict):
                        continue

                    download_url = previews.get("preview-hq-mp3") or previews.get("preview-lq-mp3")
                    if not download_url or duration < 1.0:
                        continue

                    score, match_reasons = self._score_candidate(sound, spec)
                    if score < 4.0:
                        continue

                    ranked_candidates.append((score, sound, match_reasons, download_url, duration))

                # Sort candidates by score descending (highest semantic relevance first)
                ranked_candidates.sort(key=lambda x: x[0], reverse=True)

                for score, sound, match_reasons, download_url, duration in ranked_candidates:
                    sound_id = sound["id"]
                    name = sound.get("name") or "Sound"

                    # Verify source recording loudness before committing (must have RMS >= 150.0 to contribute audibly)
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
                                if cand_rms < 150.0:
                                    logger.info(f"Rejecting candidate {sound_id} ({name}): too quiet at source (RMS={cand_rms:.1f} < 150.0). Retrying next...")
                                    continue
                    except Exception as err:
                        logger.debug(f"Could not verify loudness for sound {sound_id}: {err}")

                    license_url = sound.get("license") or ""
                    license_url_lower = license_url.lower()
                    if "zero" in license_url_lower or "publicdomain" in license_url_lower:
                        license_name = "CC0 Public Domain"
                    elif "by-nc" in license_url_lower or "-nc" in license_url_lower:
                        license_name = "CC-BY-NC Non-Commercial"
                    else:
                        license_name = "CC-BY Attribution"

                    username = sound.get("username") or "Unknown"
                    justification = f"Specifically matches {action_name} (score={score:.1f}, {'; '.join(match_reasons)})"

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
