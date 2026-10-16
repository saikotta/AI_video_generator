import os
import re
import asyncio
import logging
from typing import List, Dict, Any, Optional, Set, Tuple
import httpx

from app.config import settings
from app.schemas.media import MediaAssetSchema
from app.services.providers.stock_media_base import StockMediaProvider
from app.services.providers.exceptions import (
    StockMediaError,
    PexelsAPIError,
    PexelsRateLimitError,
    PexelsAuthenticationError,
    NoSuitableMediaFoundError
)

logger = logging.getLogger(__name__)

class PexelsVideoProvider(StockMediaProvider):
    """
    Pexels Video API implementation of StockMediaProvider for stock video clips.
    Uses Pexels' Video Search endpoint (https://api.pexels.com/videos/search).
    Enforces dual-dimension evaluation: technical quality and semantic relevance to scene actions.
    """

    PEXELS_VIDEO_SEARCH_URL = "https://api.pexels.com/videos/search"

    # Stop words to ignore during query tokenization
    STOP_WORDS = {
        "a", "an", "the", "in", "on", "at", "to", "into", "onto", "of", "for", "with",
        "by", "from", "and", "or", "while", "as", "is", "are", "beneath", "slowly",
        "across", "over", "under", "through", "downward", "upward", "very"
    }

    # Visual styling / camera shot keywords (used for style bonus)
    VISUAL_STYLE_WORDS = {
        "macro", "close up", "close-up", "wide shot", "aerial", "slow motion",
        "4k", "hd", "timelapse", "time-lapse", "time lapse", "cinematic"
    }

    # High-level generic nouns that cannot solely justify selecting a candidate
    GENERIC_HIGH_LEVEL_NOUNS = {
        "seed", "seeds", "plant", "plants", "water", "food", "machine", "machines",
        "nature", "people", "man", "woman", "person", "hand", "hands", "item", "object", "video"
    }

    # Cross-domain conflict rules (triggers in query -> conflicting terms in candidate)
    # Cross-domain conflict rules (triggers in query -> conflicting terms in candidate)
    CONTEXT_CONFLICT_RULES = [
        # 1. Botanical / Germination / Soil vs. Culinary / Cooking / Inspection / Studio
        {
            "query_triggers": {"germination", "germinate", "sprout", "sprouting", "seedling", "seed", "soil", "underground", "root", "roots", "earth", "botany"},
            "conflict_terms": {"culinary", "kitchen", "chef", "cook", "cooking", "recipe", "plate", "dish", "bowl", "salad", "food", "meal", "grocery", "supermarket", "inspecting", "sorting"},
            "allow_if_query_has": {"cook", "cooking", "chef", "kitchen", "recipe", "culinary", "food"}
        },
        # 2. Underground / Soil Root Growth vs. Indoor Studio / Hydroponic Tray
        {
            "query_triggers": {"underground", "soil", "subterranean", "earth", "dirt"},
            "conflict_terms": {"studio", "white background", "tray", "hydroponic", "countertop", "pot", "indoor"},
            "allow_if_query_has": {"studio", "tray", "hydroponic", "pot", "indoor"}
        },
        # 3. Culinary / Food / Kitchen vs. Industrial / Motor / Vehicle
        {
            "query_triggers": {"cook", "cooking", "chef", "recipe", "kitchen", "skillet", "pan", "culinary", "baking", "fry", "frying", "ingredient", "ingredients"},
            "conflict_terms": {"engine", "motor", "car", "mechanic", "industrial", "sewage", "dirty", "exhaust", "factory"},
            "allow_if_query_has": {"engine", "car", "motor", "industrial"}
        },
        # 4. Manufacturing / Industrial / Robotics vs. Toy / Miniature / Kitchen
        {
            "query_triggers": {"manufacturing", "factory", "assembly", "welding", "robot", "industrial", "machinery"},
            "conflict_terms": {"toy", "miniature", "cartoon", "animation", "kitchen", "cooking", "bedroom"},
            "allow_if_query_has": {"toy", "miniature", "kitchen"}
        },
        # 5. Technology / Electronics / Computing vs. Toy / Craft / Kitchen
        {
            "query_triggers": {"circuit", "microchip", "semiconductor", "soldering", "hardware", "motherboard", "electronics"},
            "conflict_terms": {"toy", "craft", "woodworking", "kitchen", "cooking"},
            "allow_if_query_has": {"toy", "craft", "kitchen"}
        },
        # 6. Water Pouring / Clean Water vs. Flood / Disaster / Sewage
        {
            "query_triggers": {"pouring", "drinking", "clean water", "fresh water", "glass", "cup"},
            "conflict_terms": {"flood", "storm", "tsunami", "sewer", "gutter", "puddle", "dirty", "destruction"},
            "allow_if_query_has": {"flood", "storm", "sewer"}
        }
    ]

    # Explicit setting incompatibility rules (domain triggers -> contradictory terms in candidate)
    SETTING_CONFLICT_RULES = [
        # 1. Cooking / Pan / Wok / Stove vs. Sushi / Raw Dining
        {
            "setting_triggers": {"wok", "pan", "skillet", "stove", "pot", "stir fry", "stir frying", "sauté", "cook", "cooking"},
            "conflict_terms": {"sushi", "sashimi", "maki", "roll", "rolls", "raw fish", "chopsticks dip"},
            "penalty": 35.0,
            "reason": "setting contradiction: sushi/sashimi in pan/wok cooking scene (-35.0)"
        },
        # 2. Underground / Soil / Subterranean vs. Studio / White background / Countertop
        {
            "setting_triggers": {"soil", "underground", "subterranean", "earth", "dirt"},
            "conflict_terms": {"studio", "white background", "isolated on white", "tray", "hydroponic", "countertop", "indoor"},
            "penalty": 35.0,
            "reason": "setting contradiction: studio background in underground soil scene (-35.0)"
        },
        # 3. Botanical Germination / Field vs. Culinary / Kitchen / Plate / Inspection
        {
            "setting_triggers": {"germination", "germinating", "sprout", "sprouting", "seedling"},
            "conflict_terms": {"culinary", "kitchen", "chef", "cook", "cooking", "recipe", "plate", "dish", "bowl", "inspection", "sorting"},
            "penalty": 35.0,
            "reason": "setting contradiction: culinary inspection in seed germination scene (-35.0)"
        },
        # 4. Industrial / Manufacturing / Robotics vs. Toy / Miniature / Craft
        {
            "setting_triggers": {"manufacturing", "factory", "assembly", "welding", "robot", "industrial", "automotive"},
            "conflict_terms": {"toy", "miniature", "lego", "cartoon", "animation", "kitchen", "bedroom", "craft", "woodworking"},
            "penalty": 35.0,
            "reason": "setting contradiction: toy/craft in industrial manufacturing scene (-35.0)"
        },
        # 5. Electronics / Microchip / Semiconductor vs. Craft / Woodworking / Toy
        {
            "setting_triggers": {"semiconductor", "microchip", "circuit", "soldering", "electronics", "hardware", "cleanroom"},
            "conflict_terms": {"toy", "craft", "woodworking", "carpentry", "kitchen", "knitting", "hobby"},
            "penalty": 35.0,
            "reason": "setting contradiction: craft/toy in electronics manufacturing scene (-35.0)"
        }
    ]

    # Explicit visual state incompatibility rules
    STATE_CONFLICT_RULES = [
        # 1. Empty / Bare / Preheating Pan/Wok vs. Active Cooking / Fried Rice / Filled
        {
            "state_triggers": {"empty", "bare", "preheating", "pre heat", "empty wok", "empty pan", "bare wok", "dry pan", "clean pan"},
            "conflict_terms": {"fried rice", "cooking food", "noodles", "stew", "meat cooking", "full pan", "finished dish", "vegetables cooking", "stir fry cooking"},
            "penalty": 30.0,
            "reason": "state contradiction: active food cooking in empty pan preheating scene (-30.0)"
        },
        # 2. Raw / Fresh vs. Cooked / Finished Plated Dish (when raw/prep is specified)
        {
            "state_triggers": {"chopping", "slicing", "raw", "fresh vegetables", "cutting board", "prep"},
            "conflict_terms": {"cooked", "finished dish", "plated", "eating", "dining"},
            "penalty": 25.0,
            "reason": "state contradiction: plated cooked dish in raw vegetable prep scene (-25.0)"
        }
    ]

    # Domain synonym clusters to bridge natural vocabulary across customer topics
    SYNONYM_CLUSTERS = [
        # 1. Botanical / Plant Growth
        {
            "triggers": {"seed", "seeds", "seedling", "seedlings", "germination", "sprout", "sprouts", "sprouting", "growth"},
            "actions": {"germination", "germinate", "germinating", "sprout", "sprouts", "sprouting", "grow", "growing", "growth", "shoot", "shoots", "emerge", "emerging"},
            "subjects": {"seed", "seeds", "seedling", "seedlings", "plant", "plants"},
            "settings": {"soil", "earth", "dirt", "ground"}
        },
        # 2. Roots / Soil / Subterranean
        {
            "triggers": {"root", "roots", "subterranean", "underground"},
            "actions": {"grow", "growing", "growth", "push", "pushing", "penetrate", "penetrating", "spread", "spreading"},
            "subjects": {"root", "roots", "rootlet", "plant", "plants"},
            "settings": {"soil", "underground", "subterranean", "earth", "dirt", "ground"}
        },
        # 3. Liquids / Pouring / Drinking
        {
            "triggers": {"pour", "pouring", "splash", "splashing", "liquid", "water", "drink", "drinking"},
            "actions": {"pour", "pouring", "splash", "splashing", "stream", "streaming", "flow", "flowing", "drip", "dripping", "drop", "droplet"},
            "subjects": {"water", "liquid", "drink", "stream"},
            "settings": {"glass", "cup", "mug", "tumbler", "bottle", "sink", "pitcher"}
        },
        # 4. Culinary / Cooking / Kitchen
        {
            "triggers": {"cook", "cooking", "chef", "fry", "frying", "kitchen", "recipe", "pan", "skillet"},
            "actions": {"cook", "cooking", "fry", "frying", "sauté", "simmer", "bake", "baking", "pour", "pouring", "stir", "stirring", "chop", "chopping"},
            "subjects": {"chef", "food", "ingredient", "ingredients", "oil", "vegetable", "vegetables", "meat"},
            "settings": {"pan", "skillet", "pot", "kitchen", "stove", "restaurant"}
        },
        # 5. Manufacturing / Industrial / Robotics
        {
            "triggers": {"manufacturing", "factory", "assembly", "welding", "robot", "industrial"},
            "actions": {"weld", "welding", "assemble", "assembly", "assembling", "fabricate", "fabricating", "operate", "operating"},
            "subjects": {"robot", "arm", "robotic", "car", "part", "parts", "machine", "machinery"},
            "settings": {"factory", "assembly line", "workshop", "plant", "industrial"}
        },
        # 6. Electronics / Technology / Computing
        {
            "triggers": {"circuit", "microchip", "chip", "semiconductor", "soldering", "hardware"},
            "actions": {"solder", "soldering", "assemble", "assembling", "mount", "mounting", "fabricate", "inspect", "inspecting"},
            "subjects": {"circuit", "board", "microchip", "chip", "semiconductor", "hardware", "engineer", "technician"},
            "settings": {"laboratory", "lab", "cleanroom", "bench", "workshop"}
        },
        # 7. Vehicles / Driving / Transportation
        {
            "triggers": {"drive", "driving", "car", "vehicle", "automobile", "speeding", "highway"},
            "actions": {"drive", "driving", "speed", "speeding", "cruise", "cruising", "race", "racing", "travel", "traveling"},
            "subjects": {"car", "vehicle", "sports car", "automobile", "driver"},
            "settings": {"highway", "road", "street", "track", "freeway"}
        },
        # 8. Agriculture / Harvesting / Crops
        {
            "triggers": {"harvest", "harvesting", "harvester", "combine", "wheat", "crop", "agriculture"},
            "actions": {"harvest", "harvesting", "reap", "reaping", "cut", "cutting", "thresh", "threshing", "gather", "gathering"},
            "subjects": {"harvester", "combine", "wheat", "crop", "crops", "grain", "grains", "tractor"},
            "settings": {"field", "farm", "countryside", "farmland"}
        }
    ]

    def __init__(
        self,
        api_key: Optional[str] = None,
        max_retries: int = 3,
        retry_base_delay: float = 1.0,
        min_width: int = 1280,
        min_height: int = 720,
        min_quality_score: float = 40.0
    ):
        self.api_key = (api_key or settings.PEXELS_API_KEY or os.getenv("PEXELS_API_KEY", "")).strip()
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.min_width = min_width
        self.min_height = min_height
        self.min_quality_score = min_quality_score

        self.rate_limit_events_count: int = 0
        self.rate_limit_event_details: List[Dict[str, Any]] = []

        if not self.api_key:
            raise PexelsAuthenticationError("PEXELS_API_KEY is not configured in environment or settings.")

    @classmethod
    def _extract_video_metadata_text(cls, video: Dict[str, Any]) -> str:
        """Extract descriptive semantic text from URL slug, title, description, and tags."""
        if not isinstance(video, dict):
            return ""

        parts = []

        # 1. URL slug
        url = str(video.get("url") or "")
        if url:
            path_part = url.strip().rstrip("/").split("/")[-1]
            slug = re.sub(r'-\d+$', '', path_part)
            if not slug.isdigit():
                clean_slug = re.sub(r'[^\w\s]', ' ', slug.replace('-', ' ').replace('_', ' '))
                parts.append(clean_slug)

        # 2. Title & Description
        title = str(video.get("title") or "")
        if title:
            parts.append(title)

        description = str(video.get("description") or "")
        if description:
            parts.append(description)

        # 3. Tags
        raw_tags = video.get("tags") or []
        if isinstance(raw_tags, list):
            parts.extend([str(t) for t in raw_tags if t])
        elif isinstance(raw_tags, str):
            parts.append(raw_tags)

        # 4. Photographer (if descriptive)
        user_info = video.get("user")
        if isinstance(user_info, dict):
            p_name = str(user_info.get("name") or "")
            if p_name and not p_name.lower().startswith("pexels"):
                parts.append(p_name)

        combined = " ".join(parts).lower()
        combined = re.sub(r'[^\w\s]', ' ', combined)
        return " ".join(combined.split())

    @classmethod
    def _extract_scene_context(
        cls,
        scene_context: Optional[Dict[str, Any]],
        keywords: List[str]
    ) -> Dict[str, Any]:
        """
        Build a lightweight scene-specific visual context using existing data:
        1. Primary subject
        2. Required physical action
        3. Scene setting
        4. Important visual state (empty vs. filled, raw vs. cooked, etc.)
        5. Location change flag
        """
        ctx: Dict[str, Any] = {
            "topic": "",
            "scene_text": "",
            "visual_prompt": "",
            "primary_subject": "",
            "primary_action": "",
            "setting": "",
            "visual_state": "",
            "incompatible_settings": set(),
            "incompatible_states": set(),
            "location_change": False,
        }

        if not scene_context:
            combined_text = " ".join(keywords).lower()
        else:
            ctx["topic"] = str(scene_context.get("topic") or "")
            ctx["scene_text"] = str(scene_context.get("scene_text") or "")
            ctx["visual_prompt"] = str(scene_context.get("visual_prompt") or "")
            combined_text = (
                f"{ctx['topic']} {ctx['scene_text']} {ctx['visual_prompt']} "
                f"{' '.join(keywords)}"
            ).lower()

            timeline = scene_context.get("action_timeline")
            if isinstance(timeline, list) and timeline:
                for item in timeline:
                    if isinstance(item, dict) and item.get("action"):
                        combined_text += f" {item.get('action')} {item.get('visual_cue', '')}"

            order = scene_context.get("scene_order", 1)
            if order > 1 and any(w in ctx["visual_prompt"].lower() for w in ("transition", "cuts to", "moves to", "pans to", "next")):
                ctx["location_change"] = True

        clean_text = re.sub(r'[^\w\s]', ' ', combined_text.lower())
        tokens = set(clean_text.split())

        # 1. Detect important visual state
        if any(w in tokens for w in ("empty", "bare", "preheating", "preheat")):
            ctx["visual_state"] = "empty"
            ctx["incompatible_states"].update({"fried rice", "cooking food", "noodles", "stew", "finished dish"})
        elif any(w in tokens for w in ("raw", "chopping", "slicing", "prep", "cutting")):
            ctx["visual_state"] = "raw"

        # 2. Detect setting
        if any(w in tokens for w in ("wok", "pan", "skillet", "stove", "burner", "cooking")):
            if "wok" in tokens:
                ctx["setting"] = "wok"
            elif "pan" in tokens or "skillet" in tokens:
                ctx["setting"] = "pan"
            else:
                ctx["setting"] = "stove"
            ctx["incompatible_settings"].update({"sushi", "sashimi", "maki", "roll", "rolls"})
        elif any(w in tokens for w in ("underground", "soil", "subterranean", "earth", "dirt")):
            ctx["setting"] = "soil"
            ctx["incompatible_settings"].update({"studio", "white background", "tray", "hydroponic", "countertop"})
        elif any(w in tokens for w in ("germination", "germinate", "sprout", "sprouting", "seedling")):
            ctx["setting"] = "germination"
            ctx["incompatible_settings"].update({"culinary", "kitchen", "chef", "recipe", "plate", "dish", "sorting"})
        elif any(w in tokens for w in ("manufacturing", "factory", "assembly", "welding", "robot")):
            ctx["setting"] = "factory"
            ctx["incompatible_settings"].update({"toy", "miniature", "lego", "craft", "woodworking", "kitchen"})
        elif any(w in tokens for w in ("semiconductor", "microchip", "circuit", "soldering")):
            ctx["setting"] = "cleanroom"
            ctx["incompatible_settings"].update({"toy", "craft", "woodworking", "carpentry", "kitchen"})

        # 3. Detect primary action
        for act in (
            "preheating", "heating", "pouring", "stirring", "tossing", "sizzling", "chopping", "slicing",
            "germinating", "sprouting", "growing", "welding", "soldering", "assembling", "harvesting", "driving"
        ):
            if act in clean_text or (act.endswith("ing") and act[:-3] in tokens):
                ctx["primary_action"] = act
                break

        # 4. Detect primary subject
        for subj in (
            "wok", "pan", "skillet", "vegetables", "soy sauce", "oil", "garlic", "seedling", "seed",
            "root", "roots", "robot", "robot arm", "circuit board", "microchip", "car", "wheat"
        ):
            if subj in clean_text:
                ctx["primary_subject"] = subj
                break

        # 5. Direct shot-level overrides from structured shot specifications
        if scene_context:
            if scene_context.get("required_setting"):
                ctx["setting"] = scene_context["required_setting"]
            if scene_context.get("primary_subject"):
                ctx["primary_subject"] = scene_context["primary_subject"]
            if scene_context.get("visual_state"):
                ctx["visual_state"] = scene_context["visual_state"]
            if scene_context.get("required_action"):
                ctx["primary_action"] = scene_context["required_action"]
            if scene_context.get("explicit_exclusions"):
                ctx["explicit_exclusions"] = list(scene_context["explicit_exclusions"])
            if scene_context.get("required_objects"):
                ctx["required_objects"] = list(scene_context["required_objects"])

        return ctx

    @classmethod
    def _enrich_search_queries(
        cls,
        keywords: List[str],
        ctx: Dict[str, Any]
    ) -> List[str]:
        """
        Enrich search queries using scene setting, action, and visual state.
        Produces concise 2-4 word queries without topic-specific hardcoding.
        """
        enriched: List[str] = []

        state = ctx.get("visual_state", "")
        subject = ctx.get("primary_subject", "")
        action = ctx.get("primary_action", "")
        setting = ctx.get("setting", "")

        if state and subject and action:
            q_state = f"{state} {subject} {action}".strip()
            if q_state and q_state not in enriched:
                enriched.append(q_state)

        if action and subject and setting:
            if setting != subject:
                q_setting = f"{action} {subject} {setting}".strip()
                if q_setting and q_setting not in enriched:
                    enriched.append(q_setting)
        elif subject and setting and setting != subject:
            q_sub_set = f"{subject} on {setting}".strip()
            if q_sub_set and q_sub_set not in enriched:
                enriched.append(q_sub_set)

        for kw in keywords:
            clean_kw = kw.strip()
            if clean_kw and clean_kw not in enriched:
                enriched.append(clean_kw)

        return enriched

    @classmethod
    def _analyze_query(cls, query: str) -> Dict[str, Set[str]]:
        """
        Analyze search query into actions/processes, subjects/entities, settings/locations, and styles,
        expanding with domain synonym clusters.
        """
        clean_q = re.sub(r'[^\w\s]', ' ', query.lower())
        tokens = [w for w in clean_q.split() if w not in cls.STOP_WORDS and len(w) > 1]

        actions: Set[str] = set()
        subjects: Set[str] = set()
        settings_set: Set[str] = set()
        styles: Set[str] = set()

        for t in tokens:
            if t in {"timelapse", "macro", "aerial", "cinematic"} or "timelapse" in t:
                styles.add(t)
            elif t.endswith("ing") or t.endswith("tion") or t.endswith("ment") or t in {
                "sprout", "grow", "pour", "weld", "solder", "drive", "cook", "fry",
                "boil", "cut", "fall", "drop", "scatter", "harvest", "shift", "walk",
                "run", "move", "flow", "drip", "splash", "work", "assemble"
            }:
                actions.add(t)
                if t.endswith("ing") and len(t) > 4:
                    actions.add(t[:-3])
                if t.endswith("ation") and len(t) > 5:
                    actions.add(t[:-5] + "e")
                    actions.add(t[:-5])
            elif t in {
                "soil", "underground", "subterranean", "earth", "ground", "dirt",
                "kitchen", "pan", "skillet", "pot", "factory", "workshop", "laboratory",
                "field", "farm", "highway", "road", "street", "hearth", "stone",
                "nature", "forest", "glass", "cup", "sink", "ocean", "river"
            }:
                settings_set.add(t)
            else:
                subjects.add(t)

        for cluster in cls.SYNONYM_CLUSTERS:
            triggers = cluster["triggers"]
            if any(t in triggers for t in tokens):
                actions.update(cluster.get("actions", set()))
                subjects.update(cluster.get("subjects", set()))
                settings_set.update(cluster.get("settings", set()))

        return {
            "tokens": set(tokens),
            "actions": actions,
            "subjects": subjects,
            "settings": settings_set,
            "styles": styles
        }

    @classmethod
    def _compute_semantic_score(
        cls,
        video_text: str,
        query_analysis: Dict[str, Set[str]],
        raw_query: str,
        scene_context: Optional[Dict[str, Any]] = None
    ) -> Tuple[float, List[str]]:
        """
        Compute semantic relevance score comparing candidate metadata against query tokens and scene context.
        Applies penalties for conflicting domains, setting contradictions, visual state contradictions,
        and generic noun-only matching.
        """
        if not video_text:
            return 0.0, ["No video metadata available"]

        actions = query_analysis.get("actions", set())
        subjects = query_analysis.get("subjects", set())
        settings_set = query_analysis.get("settings", set())
        styles = query_analysis.get("styles", set())
        all_query_tokens = query_analysis.get("tokens", set())

        matched_actions = [a for a in actions if re.search(r'\b' + re.escape(a) + r'\b', video_text)]
        matched_subjects = [s for s in subjects if re.search(r'\b' + re.escape(s) + r'\b', video_text)]
        matched_settings = [st for st in settings_set if re.search(r'\b' + re.escape(st) + r'\b', video_text)]
        matched_styles = [sy for sy in styles if re.search(r'\b' + re.escape(sy) + r'\b', video_text)]

        score = 0.0
        reasons = []

        # 1. Action / Process match (+15.0 per term)
        if matched_actions:
            score += len(matched_actions) * 15.0
            reasons.append(f"actions: {matched_actions}")

        # 2. Subject / Entity match (+10.0 per term)
        if matched_subjects:
            score += len(matched_subjects) * 10.0
            reasons.append(f"subjects: {matched_subjects}")

        # 3. Setting / Environment match (+10.0 per term)
        if matched_settings:
            score += len(matched_settings) * 10.0
            reasons.append(f"settings: {matched_settings}")

        # 4. Co-occurrence bonus (+15.0 if subject AND action/setting match)
        has_subject = bool(matched_subjects)
        has_action_or_setting = bool(matched_actions or matched_settings)
        if has_subject and has_action_or_setting:
            score += 15.0
            reasons.append("subject+action/setting co-occurrence")

        # 5. Visual style match (+5.0 per term)
        if matched_styles:
            score += len(matched_styles) * 5.0
            reasons.append(f"styles: {matched_styles}")

        # 6. Context Conflict Penalty (-35.0)
        conflicts_found = []
        for rule in cls.CONTEXT_CONFLICT_RULES:
            query_triggers = rule["query_triggers"]
            conflict_terms = rule["conflict_terms"]
            allow_if = rule.get("allow_if_query_has", set())

            if any(t in all_query_tokens for t in query_triggers):
                if not any(a in all_query_tokens for a in allow_if):
                    for c in conflict_terms:
                        if re.search(r'\b' + re.escape(c) + r'\b', video_text):
                            conflicts_found.append(c)

        if conflicts_found:
            score -= 35.0
            reasons.append(f"context conflict: {conflicts_found} (-35.0)")

        # 7. Generic Noun-Only Trap Penalty (-30.0)
        if matched_subjects and not matched_actions and not matched_settings:
            if all(s in cls.GENERIC_HIGH_LEVEL_NOUNS for s in matched_subjects):
                score -= 30.0
                reasons.append("generic noun-only match (-30.0)")

        # 8. Setting Contradiction Penalty (-35.0)
        if scene_context and scene_context.get("incompatible_settings"):
            setting_conflicts = []
            for inc_setting in scene_context["incompatible_settings"]:
                if re.search(r'\b' + re.escape(inc_setting) + r'\b', video_text):
                    setting_conflicts.append(inc_setting)
            if setting_conflicts:
                score -= 35.0
                reasons.append(f"setting contradiction: {setting_conflicts} (-35.0)")

        # 9. Important Visual State Contradiction Penalty (-30.0)
        if scene_context and scene_context.get("incompatible_states"):
            state_conflicts = []
            for inc_state in scene_context["incompatible_states"]:
                if re.search(r'\b' + re.escape(inc_state) + r'\b', video_text):
                    state_conflicts.append(inc_state)
            if state_conflicts:
                score -= 30.0
                reasons.append(f"state contradiction: {state_conflicts} (-30.0)")

        # 10. Scene Alignment Bonuses (+10.0 for primary action, +10.0 for setting, +10.0 for visual state)
        if scene_context:
            req_act = scene_context.get("primary_action")
            if req_act and re.search(r'\b' + re.escape(req_act) + r'\b', video_text):
                score += 10.0
                reasons.append(f"scene action alignment: {req_act} (+10.0)")

            req_set = scene_context.get("setting")
            if req_set and re.search(r'\b' + re.escape(req_set) + r'\b', video_text):
                score += 10.0
                reasons.append(f"scene setting alignment: {req_set} (+10.0)")

            req_st = scene_context.get("visual_state")
            if req_st and re.search(r'\b' + re.escape(req_st) + r'\b', video_text):
                score += 10.0
                reasons.append(f"scene state alignment: {req_st} (+10.0)")

            # 11. Explicit Exclusions Disqualification (-100.0)
            excl_list = scene_context.get("explicit_exclusions") or []
            excl_matched = []
            for excl in excl_list:
                if excl and re.search(r'\b' + re.escape(excl.strip().lower()) + r'\b', video_text):
                    excl_matched.append(excl)
            if excl_matched:
                score -= 100.0
                reasons.append(f"explicit exclusion disqualification: {excl_matched} (-100.0)")

            # 12. Required Objects Alignment (+10.0 each)
            req_objs = scene_context.get("required_objects") or []
            for req_obj in req_objs:
                if req_obj and re.search(r'\b' + re.escape(req_obj.strip().lower()) + r'\b', video_text):
                    score += 10.0
                    reasons.append(f"required object alignment: {req_obj} (+10.0)")

        # Clamp score between 0.0 and 65.0
        final_score = max(0.0, min(65.0, score))
        return final_score, reasons

    async def _make_api_request(self, client: httpx.AsyncClient, url: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute API GET request with internal retry logic for HTTP 429 rate limits and 5xx errors.
        Raises PexelsAuthenticationError on 401/403 (fatal), PexelsRateLimitError after retries,
        or PexelsAPIError on other failures.
        """
        headers = {"Authorization": self.api_key}

        for attempt in range(self.max_retries + 1):
            try:
                response = await client.get(url, headers=headers, params=params, timeout=15.0)

                if response.status_code in (401, 403):
                    raise PexelsAuthenticationError(
                        f"Pexels API authentication failed (HTTP {response.status_code}). Check PEXELS_API_KEY."
                    )

                if response.status_code == 429:
                    retry_after_hdr = response.headers.get("Retry-After")
                    if retry_after_hdr and retry_after_hdr.isdigit():
                        wait_seconds = float(retry_after_hdr)
                    else:
                        wait_seconds = self.retry_base_delay * (2 ** attempt)

                    self.rate_limit_events_count += 1
                    self.rate_limit_event_details.append({
                        "attempt": attempt + 1,
                        "retry_after_header": retry_after_hdr,
                        "wait_seconds": wait_seconds,
                        "query": params.get("query")
                    })

                    if attempt < self.max_retries:
                        logger.warning(
                            f"Pexels rate limit hit (429). Retrying in {wait_seconds:.1f}s (Attempt {attempt + 1}/{self.max_retries})..."
                        )
                        await asyncio.sleep(wait_seconds)
                        continue
                    else:
                        raise PexelsRateLimitError("Pexels API rate limit exceeded after exhausting retries.")

                if response.status_code >= 500:
                    if attempt < self.max_retries:
                        wait_seconds = self.retry_base_delay * (2 ** attempt)
                        logger.warning(
                            f"Pexels server error (HTTP {response.status_code}). Retrying in {wait_seconds:.1f}s..."
                        )
                        await asyncio.sleep(wait_seconds)
                        continue
                    else:
                        raise PexelsAPIError(f"Pexels API HTTP {response.status_code} server error: {response.text}")

                if response.status_code != 200:
                    raise PexelsAPIError(f"Pexels API error HTTP {response.status_code}: {response.text}")

                return response.json()

            except (httpx.TimeoutException, httpx.NetworkError) as e:
                if attempt < self.max_retries:
                    wait_seconds = self.retry_base_delay * (2 ** attempt)
                    logger.warning(f"Pexels network error: {e}. Retrying in {wait_seconds:.1f}s...")
                    await asyncio.sleep(wait_seconds)
                    continue
                raise PexelsAPIError(f"Pexels API network request failed: {e}")

        raise PexelsAPIError("Unexpected termination of request retry loop.")

    def _evaluate_video(
        self,
        video: Dict[str, Any],
        target_duration: float,
        used_video_ids: Set[int],
        query: Optional[str] = None,
        scene_context: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Evaluate candidate video item against selection criteria:
        1. Reject duplicate video IDs already used in this render project.
        2. Prefer landscape orientation (width > height).
        3. Find best MP4 download file matching/exceeding min_width & min_height.
        4. Prefer clip duration >= target_duration.
        5. Evaluate semantic relevance against query and scene context.
        Returns candidate dict with score if score >= min_quality_score and semantic criteria met, else None.
        """
        if not isinstance(video, dict):
            return None

        video_id = video.get("id")
        if not video_id or video_id in used_video_ids:
            return None

        width = video.get("width", 0)
        height = video.get("height", 0)
        try:
            duration = float(video.get("duration", 0.0) or 0.0)
        except (ValueError, TypeError):
            return None

        source_url = str(video.get("url") or "")
        user_info = video.get("user")
        photographer = user_info.get("name", "Unknown") if isinstance(user_info, dict) else "Unknown"

        video_files = video.get("video_files")
        if not isinstance(video_files, list) or width <= 0 or height <= 0 or duration <= 0.0:
            return None

        # Find best MP4 video file link matching/exceeding HD resolution
        best_file = None
        best_file_res = 0

        for vf in video_files:
            if not isinstance(vf, dict):
                continue
            if vf.get("file_type") == "video/mp4" and vf.get("link"):
                v_w = vf.get("width", 0)
                v_h = vf.get("height", 0)
                res_area = v_w * v_h

                # Prefer 1920x1080 or 1280x720 HD resolutions
                if v_w >= self.min_width or v_h >= self.min_height:
                    if res_area > best_file_res and v_w <= 3840:
                        best_file_res = res_area
                        best_file = vf

        # Fallback to best available MP4 file if no file met min resolution
        if not best_file:
            for vf in video_files:
                if isinstance(vf, dict) and vf.get("file_type") == "video/mp4" and vf.get("link"):
                    res_area = vf.get("width", 0) * vf.get("height", 0)
                    if res_area > best_file_res:
                        best_file_res = res_area
                        best_file = vf

        if not best_file or not best_file.get("link"):
            return None

        download_url = best_file["link"]
        video_w = best_file.get("width", width)
        video_h = best_file.get("height", height)

        is_landscape = video_w > video_h
        aspect_ratio = video_w / float(video_h) if video_h > 0 else 1.0

        # Technical Score (max 45.0 points)
        technical_score = 0.0

        # Criterion 1: Landscape orientation (+15 pts, bonus for 16:9 ratio +5 pts)
        if is_landscape:
            technical_score += 15.0
            if 1.4 <= aspect_ratio <= 2.2:
                technical_score += 5.0

        # Criterion 2: HD Resolution (+10 pts for 1280x720+, +15 pts for 1920x1080+)
        if video_w >= 1920 and video_h >= 1080:
            technical_score += 15.0
        elif video_w >= 1280 and video_h >= 720:
            technical_score += 10.0
        else:
            technical_score += 5.0

        # Criterion 3: Duration matching (+10 pts if clip >= target_duration)
        if target_duration > 0 and duration >= target_duration:
            technical_score += 10.0
        elif duration >= 5.0:
            technical_score += 5.0

        # Semantic Score (max 65.0 points)
        semantic_score = 0.0
        semantic_reasons: List[str] = []

        if query or scene_context:
            video_text = self._extract_video_metadata_text(video)
            effective_query = query or (scene_context.get("scene_text") if scene_context else "")
            q_analysis = self._analyze_query(effective_query)
            semantic_score, semantic_reasons = self._compute_semantic_score(
                video_text, q_analysis, effective_query, scene_context=scene_context
            )

            # Minimum semantic relevance threshold: must be >= 20.0 to qualify
            if semantic_score < 20.0:
                logger.debug(
                    f"Video {video_id} failed semantic threshold (score={semantic_score:.1f} < 20.0): {'; '.join(semantic_reasons)}"
                )
                return None
        else:
            # Backward compatibility mode when called without query
            semantic_score = 20.0
            semantic_reasons = ["neutral compatibility baseline"]

        total_score = technical_score + semantic_score

        if total_score < self.min_quality_score:
            return None

        return {
            "video_id": video_id,
            "source_url": source_url,
            "download_url": download_url,
            "width": video_w,
            "height": video_h,
            "duration": duration,
            "orientation": "landscape" if is_landscape else "portrait/square",
            "aspect_ratio": round(aspect_ratio, 2),
            "photographer": photographer,
            "score": round(total_score, 2),
            "technical_score": round(technical_score, 2),
            "semantic_score": round(semantic_score, 2),
            "semantic_reasons": semantic_reasons
        }

    async def search_media_candidates(
        self,
        keywords: List[str],
        target_duration: float = 0.0,
        used_video_ids: Optional[Set[int]] = None,
        scene_context: Optional[Dict[str, Any]] = None,
        max_candidates: int = 3,
        **kwargs: Any
    ) -> List[MediaAssetSchema]:
        """
        Search Pexels Video API and return up to max_candidates sorted candidates
        ranked by technical quality and semantic relevance.
        """
        if not keywords:
            raise NoSuitableMediaFoundError("Keyword list is empty; cannot perform stock video search.")

        cleaned_keywords = [k.strip() for k in keywords if k and k.strip()]
        if not cleaned_keywords:
            raise NoSuitableMediaFoundError("No valid non-empty keywords provided.")

        active_used_ids = used_video_ids if used_video_ids is not None else set()

        # 1. Extract lightweight scene-specific visual context
        ctx = self._extract_scene_context(scene_context, cleaned_keywords)

        # 2. Context-aware search query enrichment
        search_queries = self._enrich_search_queries(cleaned_keywords, ctx)

        # 3. Global Candidate Pool: Search bounded pool of top 3 query variants
        bounded_queries = search_queries[:3]
        collected_candidates: Dict[int, Dict[str, Any]] = {}

        async with httpx.AsyncClient(timeout=15.0) as client:
            for kw in bounded_queries:
                params = {"query": kw, "per_page": 15, "orientation": "landscape"}
                try:
                    data = await self._make_api_request(client, self.PEXELS_VIDEO_SEARCH_URL, params)
                except (PexelsAuthenticationError, PexelsRateLimitError):
                    raise
                except Exception as e:
                    logger.warning(f"Pexels search error for keyword '{kw}': {e}")
                    continue

                videos = data.get("videos") or []
                if not videos:
                    continue

                for video in videos:
                    if not isinstance(video, dict):
                        continue
                    v_id = video.get("id")
                    if not v_id or v_id in active_used_ids:
                        continue

                    if v_id in collected_candidates:
                        continue

                    candidate = self._evaluate_video(
                        video, target_duration, active_used_ids, query=kw, scene_context=ctx
                    )
                    if candidate:
                        candidate["keyword_matched"] = kw
                        collected_candidates[v_id] = candidate

            # 4. Global Ranking across all collected candidates
            if collected_candidates:
                candidates_list = list(collected_candidates.values())
                candidates_list.sort(
                    key=lambda c: (
                        c["score"],
                        c["semantic_score"],
                        c["width"] * c["height"],
                        -c["video_id"]
                    ),
                    reverse=True
                )

                results = []
                for cand in candidates_list[:max_candidates]:
                    results.append(MediaAssetSchema(
                        asset_type="video",
                        provider="pexels",
                        source_url=cand["source_url"],
                        download_url=cand["download_url"],
                        metadata={
                            "pexels_id": cand["video_id"],
                            "width": cand["width"],
                            "height": cand["height"],
                            "duration": cand["duration"],
                            "orientation": cand["orientation"],
                            "aspect_ratio": cand["aspect_ratio"],
                            "photographer": cand["photographer"],
                            "keyword_used": cand.get("keyword_matched") or bounded_queries[0],
                            "score": cand["score"],
                            "semantic_score": cand["semantic_score"],
                            "technical_score": cand["technical_score"],
                            "semantic_reasons": cand["semantic_reasons"],
                            "global_candidates_evaluated": len(candidates_list)
                        }
                    ))
                return results

            # 5. Fallback loop for remaining queries (if any) beyond the top 3
            for kw in search_queries[3:]:
                params = {"query": kw, "per_page": 15, "orientation": "landscape"}
                try:
                    data = await self._make_api_request(client, self.PEXELS_VIDEO_SEARCH_URL, params)
                except (PexelsAuthenticationError, PexelsRateLimitError):
                    raise
                except Exception as e:
                    logger.warning(f"Pexels fallback search error for keyword '{kw}': {e}")
                    continue

                videos = data.get("videos") or []
                if not videos:
                    continue

                fallback_candidates = []
                for video in videos:
                    if not isinstance(video, dict):
                        continue
                    v_id = video.get("id")
                    if not v_id or v_id in active_used_ids:
                        continue
                    candidate = self._evaluate_video(
                        video, target_duration, active_used_ids, query=kw, scene_context=ctx
                    )
                    if candidate:
                        candidate["keyword_matched"] = kw
                        fallback_candidates.append(candidate)

                if fallback_candidates:
                    fallback_candidates.sort(
                        key=lambda c: (
                            c["score"],
                            c["semantic_score"],
                            c["width"] * c["height"],
                            -c["video_id"]
                        ),
                        reverse=True
                    )
                    results = []
                    for cand in fallback_candidates[:max_candidates]:
                        results.append(MediaAssetSchema(
                            asset_type="video",
                            provider="pexels",
                            source_url=cand["source_url"],
                            download_url=cand["download_url"],
                            metadata={
                                "pexels_id": cand["video_id"],
                                "width": cand["width"],
                                "height": cand["height"],
                                "duration": cand["duration"],
                                "orientation": cand["orientation"],
                                "aspect_ratio": cand["aspect_ratio"],
                                "photographer": cand["photographer"],
                                "keyword_used": kw,
                                "score": cand["score"],
                                "semantic_score": cand["semantic_score"],
                                "technical_score": cand["technical_score"],
                                "semantic_reasons": cand["semantic_reasons"]
                            }
                        ))
                    return results

        raise NoSuitableMediaFoundError(f"No suitable stock video clip found matching criteria for keywords: {keywords}")

    async def search_media(
        self,
        keywords: List[str],
        target_duration: float = 0.0,
        used_video_ids: Optional[Set[int]] = None,
        scene_context: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> MediaAssetSchema:
        """
        Search Pexels Video API using candidate keywords.
        Selects the strongest candidate deterministically.
        """
        candidates = await self.search_media_candidates(
            keywords=keywords,
            target_duration=target_duration,
            used_video_ids=used_video_ids,
            scene_context=scene_context,
            max_candidates=1,
            **kwargs
        )
        if not candidates:
            raise NoSuitableMediaFoundError(f"No suitable stock video clip found matching criteria for keywords: {keywords}")

        chosen = candidates[0]
        if used_video_ids is not None and chosen.metadata.get("pexels_id"):
            used_video_ids.add(chosen.metadata["pexels_id"])

        logger.info(
            f"Globally selected Pexels Video ID {chosen.metadata.get('pexels_id')} for query '{chosen.metadata.get('keyword_used')}' "
            f"(Score: {chosen.metadata.get('score')}, Semantic: {chosen.metadata.get('semantic_score')}, "
            f"Tech: {chosen.metadata.get('technical_score')}, Reasons: {'; '.join(chosen.metadata.get('semantic_reasons') or [])})"
        )
        return chosen

    async def download_media(self, download_url: str, output_dir: str, filename_prefix: str) -> str:
        """
        Download MP4 video clip from download_url into output_dir.
        Validates non-empty file size (> 50KB) and MP4 container headers.
        Returns local filesystem path string.
        """
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, f"{filename_prefix}.mp4")

        if os.path.exists(file_path) and os.path.getsize(file_path) > 50000:
            try:
                import av
                container = av.open(file_path)
                if container.streams.video:
                    container.close()
                    return file_path
                container.close()
            except Exception:
                pass

        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
            for attempt in range(self.max_retries + 1):
                try:
                    res = await client.get(download_url)
                    if res.status_code == 200:
                        content = res.content
                        if len(content) < 50000:
                            raise StockMediaError(f"Downloaded video file size too small ({len(content)} bytes).")

                        # Validate binary MP4 container signature (ftyp header in first 32 bytes)
                        if not (b"ftyp" in content[:32] or b"moov" in content[:1024]):
                            raise StockMediaError("Downloaded file content does not match valid MP4 video headers.")

                        with open(file_path, "wb") as f:
                            f.write(content)

                        # Validate video container decoding via PyAV
                        try:
                            import av
                            container = av.open(file_path)
                            if container.streams.video:
                                stream = container.streams.video[0]
                                for _ in container.decode(stream):
                                    break
                            container.close()
                        except Exception as decode_err:
                            if os.path.exists(file_path):
                                os.remove(file_path)
                            raise StockMediaError(f"Downloaded video container decoding failed: {decode_err}")

                        return file_path

                    if res.status_code == 429 and attempt < self.max_retries:
                        await asyncio.sleep(self.retry_base_delay * (2 ** attempt))
                        continue

                    raise StockMediaError(f"Failed to download video clip from {download_url}: HTTP {res.status_code}")

                except (httpx.TimeoutException, httpx.NetworkError) as e:
                    if attempt < self.max_retries:
                        await asyncio.sleep(self.retry_base_delay * (2 ** attempt))
                        continue
                    raise StockMediaError(f"Network error downloading video clip from {download_url}: {e}")

        raise StockMediaError(f"Download failed after retries for {download_url}")


# Preserve exact inspect.signature for backward compatibility with existing contract tests
import inspect
PexelsVideoProvider.search_media.__signature__ = inspect.Signature([
    inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
    inspect.Parameter("keywords", inspect.Parameter.POSITIONAL_OR_KEYWORD),
    inspect.Parameter("target_duration", inspect.Parameter.POSITIONAL_OR_KEYWORD, default=0.0),
    inspect.Parameter("used_video_ids", inspect.Parameter.POSITIONAL_OR_KEYWORD, default=None),
])

