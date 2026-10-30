import re
import json
import random
import asyncio
import logging
import httpx
from typing import Optional
from pydantic import ValidationError

from app.config import settings
from app.schemas.scene import SceneItemSchema, SceneListSchema
from app.schemas.visual_spec import VisualSpecificationSchema
from app.services.providers.base import ScriptProvider
from app.services.providers.exceptions import (
    OllamaUnavailableError,
    ScriptGenerationTimeoutError,
    InvalidScriptResponseError,
    ScriptProviderError
)

logger = logging.getLogger(__name__)

class GroqScriptProvider(ScriptProvider):
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        words_per_minute: Optional[int] = None,
        max_retries: Optional[int] = None,
        base_delay: Optional[float] = None,
        timeout_seconds: float = 120.0
    ):
        self.api_key = api_key or settings.GROQ_API_KEY
        self.model = model or settings.GROQ_MODEL
        # Calibrated narration speed: 130 WPM calibrated to EdgeTTS (en-US-ChristopherNeural)
        if words_per_minute is not None:
            self.words_per_minute = words_per_minute
        elif hasattr(settings, "CALIBRATED_NARRATION_WPM") and settings.CALIBRATED_NARRATION_WPM:
            self.words_per_minute = settings.CALIBRATED_NARRATION_WPM
        elif hasattr(settings, "NARRATION_WPM") and settings.NARRATION_WPM and settings.NARRATION_WPM != 150:
            self.words_per_minute = settings.NARRATION_WPM
        else:
            self.words_per_minute = 130
        self.max_retries = max_retries or settings.GROQ_MAX_RETRIES
        self.base_delay = base_delay or settings.GROQ_RETRY_BASE_DELAY
        self.timeout_seconds = timeout_seconds
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        self._internal_client: Optional[httpx.AsyncClient] = None

        self.last_expansion_occurred = False
        self.last_trimming_occurred = False
        self.last_retry_count = 0
        self.last_shot_plan_fallback_occurred = False
        self.last_shot_plan_error: Optional[str] = None
        self.fallback_to_single_shot: bool = True
        self.last_json_recovery_attempted: bool = False
        self.last_json_recovery_failed: bool = False

    async def __aenter__(self):
        self._internal_client = httpx.AsyncClient(timeout=httpx.Timeout(self.timeout_seconds))
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._internal_client:
            await self._internal_client.aclose()
            self._internal_client = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._internal_client is None or self._internal_client.is_closed:
            self._internal_client = httpx.AsyncClient(timeout=httpx.Timeout(self.timeout_seconds))
        return self._internal_client

    # Measured on two real runs: edge-tts speaks ~150 words/min (142-165 observed) and every scene
    # adds ~1.1s of trailing silence. Total length ~= words / 150 * 60 + 1.1 * scene_count, so the
    # word budget must leave room for the pauses of the scenes we expect (~1 scene per 5 seconds).
    SHORT_VIDEO_MAX_SECONDS = 30
    SHORT_VIDEO_SPEECH_WPM = 150.0
    EXPECTED_SECONDS_PER_SCENE = 5.0

    def calculate_target_words(self, duration_seconds: int) -> int:
        if duration_seconds <= self.SHORT_VIDEO_MAX_SECONDS:
            from app.services.audio_trim import pause_seconds_per_scene
            est_scenes = max(2, int(round(duration_seconds / self.EXPECTED_SECONDS_PER_SCENE)))
            speaking_seconds = max(duration_seconds - est_scenes * pause_seconds_per_scene(),
                                   duration_seconds * 0.5)
            return int(round(speaking_seconds * self.SHORT_VIDEO_SPEECH_WPM / 60.0))
        return int(round(duration_seconds * self.words_per_minute / 60.0))

    def clean_script_text(self, text: str) -> str:
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'\((?:Narrator|Voiceover|Scene|Music|SFX)[^)]*\)', '', text, flags=re.IGNORECASE)
        text = re.sub(r'^#+\s+.*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'\*+([^*]+)\*+', r'\1', text)
        text = re.sub(r'_+([^_]+)_+', r'\1', text)
        text = re.sub(r'^\s*[*•-]\s*', '', text, flags=re.MULTILINE)
        text = text.replace('—', ' - ').replace('–', ' - ').replace('’', "'").replace('“', '"').replace('”', '"')
        text = re.sub(r'^(Here is|Here\'s|Sure, here is|Below is)[^:\n]*:\s*', '', text, flags=re.IGNORECASE)
        # Clean any malformed double punctuation like ",.", ";.", ":.", "—.", or "-."
        # When a clause delimiter is followed by [.!?] and a lowercase word, preserve the clause delimiter.
        text = re.sub(r'([,;:—–-])\s*[.!?]+\s+(?=[a-z])', r'\1 ', text)
        text = re.sub(r'[,;:—–-]+\s*([.!?])', r'\1', text)
        text = re.sub(r'\s+([.!?])', r'\1', text)
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        cleaned = " ".join(lines)
        return cleaned.strip()

    def _extract_json_payload(self, raw_response: str) -> dict:
        text = raw_response.strip()

        match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL | re.IGNORECASE)
        if match:
            text = match.group(1).strip()
        elif text.startswith("```"):
            text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.IGNORECASE)
            text = re.sub(r'\s*```$', '', text)
            text = text.strip()

        first_brace = text.find('{')
        last_brace = text.rfind('}')
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            text = text[first_brace:last_brace + 1]
        elif first_brace != -1:
            text = text[first_brace:]

        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            last_obj_end = text.rfind('}')
            if last_obj_end != -1 and last_obj_end > first_brace:
                candidate = text[:last_obj_end + 1].rstrip(',') + "\n  ]\n}"
                try:
                    data = json.loads(candidate)
                except json.JSONDecodeError:
                    raise InvalidScriptResponseError(
                        f"Groq returned malformed JSON: {str(e)}"
                    ) from e
            else:
                raise InvalidScriptResponseError(
                    f"Groq returned malformed JSON: {str(e)}"
                ) from e

        if not isinstance(data, dict):
            raise InvalidScriptResponseError(
                f"Expected JSON object (dict), but got {type(data).__name__}."
            )

        return data

    async def _post_groq(
        self,
        prompt: str,
        is_json: bool = False,
        temperature: float = 0.7,
        operation: Optional[str] = None
    ) -> str:
        if not self.api_key:
            raise ScriptProviderError(
                "GROQ_API_KEY environment variable is not set. Cannot invoke Groq API."
            )

        if operation:
            op_name = operation
        elif is_json:
            if '"scenes"' in prompt:
                op_name = "split_into_scenes"
            elif '"visual_prompt"' in prompt:
                op_name = "generate_visual_specification"
            else:
                op_name = "json_completion"
        else:
            op_name = "script_generation"

        self.last_json_recovery_attempted = False
        self.last_json_recovery_failed = False

        client = self._get_client()
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature
        }
        if is_json:
            payload["response_format"] = {"type": "json_object"}

        for attempt in range(self.max_retries):
            try:
                response = await client.post(self.base_url, json=payload, headers=headers)
            except httpx.TimeoutException as e:
                if attempt == self.max_retries - 1:
                    raise ScriptGenerationTimeoutError(
                        f"Groq request timed out after {self.timeout_seconds} seconds ({op_name})."
                    ) from e
                await asyncio.sleep(self.base_delay * (2 ** attempt))
                continue
            except httpx.HTTPError as e:
                if attempt == self.max_retries - 1:
                    raise ScriptProviderError(
                        f"HTTP connection error connecting to Groq API ({op_name})."
                    ) from e
                await asyncio.sleep(self.base_delay * (2 ** attempt))
                continue

            if response.status_code == 200:
                data = response.json()
                try:
                    return data["choices"][0]["message"]["content"]
                except (KeyError, IndexError) as e:
                    raise InvalidScriptResponseError(
                        f"Groq response payload missing choices array ({op_name})."
                    ) from e

            elif response.status_code == 429:
                self.last_retry_count += 1
                retry_after_hdr = response.headers.get("Retry-After") or response.headers.get("retry-after")
                if retry_after_hdr:
                    try:
                        delay = float(retry_after_hdr)
                    except ValueError:
                        delay = self.base_delay * (2 ** attempt) + random.uniform(0.1, 0.5)
                else:
                    delay = self.base_delay * (2 ** attempt) + random.uniform(0.1, 0.5)

                if attempt == self.max_retries - 1:
                    raise ScriptProviderError(
                        f"Groq API rate limit exceeded after {self.max_retries} attempts (HTTP 429, {op_name})."
                    )
                await asyncio.sleep(delay)

            else:
                # Extract structured error details if present
                error_code = None
                try:
                    err_json = response.json()
                    if isinstance(err_json, dict) and "error" in err_json:
                        err_obj = err_json["error"]
                        if isinstance(err_obj, dict):
                            error_code = err_obj.get("code")
                except Exception:
                    pass

                # Targeted recovery only when:
                # 1. The request explicitly used JSON response mode.
                # 2. Groq returned HTTP 400.
                # 3. The structured provider error code is json_validate_failed.
                if is_json and response.status_code == 400 and error_code == "json_validate_failed":
                    logger.warning(
                        "Groq returned HTTP 400 with code '%s' for %s. Attempting targeted single-shot recovery without response_format.",
                        error_code,
                        op_name
                    )
                    self.last_json_recovery_attempted = True

                    recovery_payload = {
                        "model": self.model,
                        "messages": [
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": temperature
                    }

                    try:
                        recovery_resp = await client.post(self.base_url, json=recovery_payload, headers=headers)
                    except httpx.TimeoutException as te:
                        self.last_json_recovery_failed = True
                        raise ScriptGenerationTimeoutError(
                            f"Groq request timed out during JSON recovery ({op_name}) after {self.timeout_seconds} seconds."
                        ) from te
                    except httpx.HTTPError as he:
                        self.last_json_recovery_failed = True
                        raise ScriptProviderError(
                            f"HTTP connection error during Groq JSON recovery ({op_name})."
                        ) from he

                    if recovery_resp.status_code == 200:
                        try:
                            rec_data = recovery_resp.json()
                            content = rec_data["choices"][0]["message"]["content"]
                            self.last_json_recovery_failed = False
                            return content
                        except (KeyError, IndexError) as e:
                            self.last_json_recovery_failed = True
                            raise InvalidScriptResponseError(
                                f"Groq recovery response payload missing choices array ({op_name})."
                            ) from e
                    else:
                        self.last_json_recovery_failed = True
                        rec_err_code = None
                        try:
                            rec_err_json = recovery_resp.json()
                            if isinstance(rec_err_json, dict) and "error" in rec_err_json:
                                rec_err_code = rec_err_json["error"].get("code")
                        except Exception:
                            pass
                        err_detail = f", code: '{rec_err_code}'" if rec_err_code else ""
                        raise ScriptProviderError(
                            f"Groq API returned HTTP error status {recovery_resp.status_code}{err_detail} "
                            f"(operation: '{op_name}', json_recovery_attempted: True, fallback_failed: True)."
                        )

                # Unrecovered or unrelated HTTP error
                err_detail = f", code: '{error_code}'" if error_code else ""
                raise ScriptProviderError(
                    f"Groq API returned HTTP error status {response.status_code}{err_detail} "
                    f"(operation: '{op_name}', json_recovery_attempted: False)."
                )

        raise ScriptProviderError(
            f"Groq API request retries exhausted after {self.max_retries} attempts ({op_name})."
        )

    DANGLING_END_WORDS = {
        "a", "an", "the",
        "and", "or", "but", "nor", "so", "yet",
        "while", "as", "because", "if", "when", "although", "since", "though", "whereas",
        "that", "which", "who", "whom", "whose", "where",
        "with", "of", "to", "in", "on", "at", "by", "for", "from", "into", "through",
        "during", "before", "after", "above", "below", "between", "under", "over",
        "about", "against", "among", "toward", "towards", "upon",
        "then", "also"
    }

    @classmethod
    def _clean_dangling_ending(cls, text: str) -> str:
        words = text.split()
        while len(words) > 3:
            clean_word = re.sub(r'^[^\w]+|[^\w]+$', '', words[-1]).lower()
            if not clean_word or clean_word in cls.DANGLING_END_WORDS or words[-1] in {',', ';', ':', '-', '—', '–'}:
                words.pop()
            else:
                break
        result = " ".join(words)
        result = re.sub(r'[,;:—–\s-]+$', '', result)
        return result

    def trim_to_sentence_boundary(self, text: str, target_words: int) -> tuple[str, bool]:
        words = text.split()
        current_word_count = len(words)

        # Calibrate slack based on duration:
        # Short-form videos (target_words <= 65, representing <= 30 seconds) use conservative ~105% upper bound
        # Longer videos (target_words > 65) preserve 115% slack.
        if target_words <= 65:
            max_words = max(target_words + 1, int(round(target_words * 1.05)))
            min_words = int(round(target_words * 0.90))
        else:
            max_words = int(target_words * 1.15)
            min_words = int(target_words * 0.90)

        if current_word_count <= max_words:
            return text, False

        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        if not sentences:
            trimmed_text = " ".join(words[:max_words]).strip()
            trimmed_text = self._clean_dangling_ending(trimmed_text)
            trimmed_text = re.sub(r'[,;:—–-]+\s*([.!?])', r'\1', trimmed_text)
            trimmed_text = re.sub(r'[,;:—–\s-]+$', '', trimmed_text)
            trimmed_text = re.sub(r'\s+([.!?])', r'\1', trimmed_text)
            if trimmed_text and not trimmed_text.endswith(('.', '!', '?')):
                trimmed_text += '.'
            return trimmed_text, True

        trimmed_sentences = []
        accumulated_words = 0

        for sentence in sentences:
            sentence_words = len(sentence.split())
            if accumulated_words + sentence_words <= max_words:
                trimmed_sentences.append(sentence)
                accumulated_words += sentence_words
                if accumulated_words >= target_words:
                    break
            else:
                if accumulated_words >= min_words:
                    break
                else:
                    clauses = [c.strip() for c in re.split(r'(?<=[,;:—–])\s+|(?<=\s[-—–])\s+', sentence) if c.strip()]
                    added_clause = False
                    for clause in clauses:
                        clause_words = len(clause.split())
                        if accumulated_words + clause_words <= max_words:
                            trimmed_sentences.append(clause)
                            accumulated_words += clause_words
                            added_clause = True
                            if accumulated_words >= min_words:
                                break
                        else:
                            if accumulated_words >= min_words:
                                break
                            break

                    if not added_clause and accumulated_words < min_words:
                        if accumulated_words == 0:
                            rem = max_words - accumulated_words
                            clause_words_list = sentence.split()[:rem]
                            if clause_words_list:
                                partial = " ".join(clause_words_list)
                                partial = self._clean_dangling_ending(partial)
                                if partial:
                                    trimmed_sentences.append(partial)
                                    accumulated_words += len(partial.split())
                        else:
                            # Preserve complete grammatical sentences; do not chop words mid-sentence merely to satisfy word limit
                            pass
                    break

        trimmed_text = " ".join(trimmed_sentences).strip()
        trimmed_text = self._clean_dangling_ending(trimmed_text)
        trimmed_text = re.sub(r'[,;:—–-]+\s*([.!?])', r'\1', trimmed_text)
        trimmed_text = re.sub(r'[,;:—–\s-]+$', '', trimmed_text)
        trimmed_text = re.sub(r'\s+([.!?])', r'\1', trimmed_text)
        if trimmed_text and not trimmed_text.endswith(('.', '!', '?')):
            trimmed_text += '.'

        return trimmed_text, True

    def _ensure_narration_preservation(self, script: str, scenes: list[SceneItemSchema]) -> list[SceneItemSchema]:
        if not scenes:
            return scenes

        combined_text = " ".join(s.scene_text for s in scenes)
        script_words = len(script.split())
        combined_words = len(combined_text.split())

        if combined_words >= int(script_words * 0.90):
            return scenes

        script_sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', script) if s.strip()]
        combined_norm = re.sub(r'[^\w\s]', '', combined_text.lower())
        missing_sentences = []

        for s in script_sentences:
            s_norm = re.sub(r'[^\w\s]', '', s.lower())
            if s_norm and s_norm not in combined_norm:
                missing_sentences.append(s)

        if missing_sentences:
            appended_text = " " + " ".join(missing_sentences)
            scenes[-1].scene_text = (scenes[-1].scene_text + appended_text).strip()

        return scenes

    @classmethod
    def _classify_topic(cls, topic: str) -> str:
        t_lower = topic.lower()

        # 1. Historical topics that EXPLICITLY ask for history (strong signals only).
        # Weak era words such as "ancient" or "century" are handled after the
        # science/cooking/technology checks, so "ancient seed germination" is science.
        history_keywords = [
            "history", "historical", "origin of", "origins of",
            "empire", "dynasty", "civilization", "revolution",
            "war of", "world war", "battle of",
            "archaeology", "historical origins"
        ]
        weak_history_keywords = [
            "ancient", "century", "medieval", "renaissance", "antiquity"
        ]
        if any(re.search(r'\b' + re.escape(kw) + r'\b', t_lower) for kw in history_keywords):
            return "history"

        # 2. Scientific & educational processes (natural phenomena, biology, physics, astronomy, geology)
        science_keywords = [
            "seed", "seeds", "germinat", "plant", "plants", "photosynthesis",
            "biology", "biological", "physics", "physical process", "chemistry",
            "chemical", "water cycle", "gravity", "ecosystem", "cellular",
            "cell division", "organism", "metamorphosis", "geology", "volcano",
            "astronomy", "planet", "solar system", "glacier", "evolution",
            "anatomy", "pollination", "scientific"
        ]
        if any(re.search(r'\b' + re.escape(kw) + r'\b', t_lower) for kw in science_keywords):
            return "science"

        # 3. Cooking & culinary topics (recipes, ingredients, food preparation techniques)
        cooking_keywords = [
            "cook", "cooking", "bake", "baking", "recipe", "recipes", "kitchen",
            "culinary", "ingredient", "ingredients", "brew", "brewing", "roast",
            "roasting", "food", "dish", "dishes", "pasta", "bread", "soup",
            "steak", "sushi", "espresso", "coffee", "tea", "sear", "sauté", "knead",
            "fry", "frying", "stir-fry", "stir fry", "vegetable", "vegetables", "wok", "skillet"
        ]
        if any(re.search(r'\b' + re.escape(kw) + r'\b', t_lower) for kw in cooking_keywords):
            return "cooking"

        # 4. Technology & engineering topics (computing, mechanics, electronics, systems, manufacturing)
        tech_keywords = [
            "technology", "tech", "computer", "computers", "computing", "software", "hardware",
            "algorithm", "artificial intelligence", "ai", "machine learning",
            "robot", "robotics", "robotic", "automated", "automation", "manufacturing",
            "weld", "welding", "automotive", "assembly", "factory", "machinery",
            "engine", "internet", "coding", "programming",
            "microchip", "microchips", "semiconductor", "processor", "processors",
            "microprocessor", "microprocessors", "network", "blockchain",
            "quantum computing", "electric vehicle", "aerospace", "satellite"
        ]
        if any(re.search(r'\b' + re.escape(kw) + r'\b', t_lower) for kw in tech_keywords):
            return "technology"

        # 4b. Weak history signals only win when no subject-matter category matched above
        if any(re.search(r'\b' + re.escape(kw) + r'\b', t_lower) for kw in weak_history_keywords):
            return "history"

        # 5. Creative storytelling / narrative fiction
        story_keywords = [
            "story", "stories", "tale", "tales", "fable", "fables", "adventure",
            "adventures", "legend", "legends", "myth", "myths", "narrative",
            "fiction", "chronicle"
        ]
        if any(re.search(r'\b' + re.escape(kw) + r'\b', t_lower) for kw in story_keywords):
            return "story"

        # Process-oriented phrases ("how a ... works / happens / develops")
        if re.search(r'\bhow\s+(?:a|an|the|do|does|it)\b', t_lower):
            return "science"

        return "general"

    @classmethod
    def _build_topic_content_instructions(cls, topic: str) -> str:
        category = cls._classify_topic(topic)

        if category == "science":
            return (
                "- For this scientific and educational process topic, explain the actual physical and biological process directly and sequentially from the very first sentence.\n"
                "- Immediately discuss observable physical mechanisms, sequential developmental stages, cause-and-effect transformations, and natural outcomes.\n"
                "- For biological growth or lifecycle topics, directly describe the sequential physical progression (e.g. hydration, germination, structural development, root anchoring, and shoot emergence).\n"
                "- STRICTLY FORBIDDEN: Do NOT introduce ancient farmers, cultural ceremonies, historical origins, or invented backstories. Focus purely on the physical and biological process itself."
            )
        elif category == "cooking":
            return (
                "- For this culinary topic, focus directly on the actual cooking and preparation steps, ingredients, physical techniques, and sensory transformations (e.g. kneading, searing, textures, heat, aromas).\n"
                "- Walk through the preparation method in a clear, engaging sequence without detouring into ancient origins, historical anecdotes, or cultural backstories unless explicitly requested."
            )
        elif category == "technology":
            return (
                "- For this technology and engineering topic, explain the actual operational mechanisms, system components, engineering principles, and physical functionality directly.\n"
                "- Focus on how the components work together and the practical operation without irrelevant historical preamble."
            )
        elif category == "history":
            return (
                "- For this historical topic, provide an authentic, engaging historical narrative focusing on chronological context, key developments, pivotal figures or moments, and their lasting significance.\n"
                "- Ground the narration in concrete historical facts and storytelling without generic filler."
            )
        elif category == "story":
            return (
                "- For this storytelling topic, deliver an engaging, creative narrative with a compelling arc, vivid imagery, character journey, and evocative dramatic flow.\n"
                "- Maintain a natural spoken storytelling rhythm without academic lecturing or dry explanations."
            )
        else:
            return (
                "- Focus directly on the core subject matter requested, explaining key concepts, observable details, and engaging insights from the very first sentence.\n"
                "- Avoid forcing rigid historical origins, ancient backstories, or cultural digressions unless the topic explicitly requests history."
            )

    @classmethod
    def _build_expansion_instructions(cls, topic: str, words_needed: int) -> str:
        category = cls._classify_topic(topic)
        if category == "science":
            focus = "explaining the next sequential physical stages, detailed biological/physical mechanisms, and observable natural transformations (strictly avoiding historical anecdotes or ancient backstories)"
        elif category == "cooking":
            focus = "describing the subsequent culinary preparation steps, cooking techniques, sensory observations, and texture or flavor developments (avoiding historical detours)"
        elif category == "technology":
            focus = "elaborating on technical mechanisms, component interactions, engineering principles, and practical operational details"
        elif category == "history":
            focus = "adding deeper historical context, chronological developments, pivotal events, and narrative continuity"
        elif category == "story":
            focus = "deepening the creative narrative arc, character journey, sensory details, and atmospheric storytelling"
        else:
            focus = "providing concrete details, specific observable examples, and seamless narrative continuation"

        return (
            f"1. Write a seamless continuation by {focus} to reach an additional {words_needed} words.\n"
            f"2. DO NOT repeat the current script.\n"
            f"3. Return ONLY spoken narration text without Markdown, headings, bullet points, or stage directions."
        )

    async def generate_script(self, topic: str, desired_duration_seconds: int) -> str:
        self.last_expansion_occurred = False
        self.last_trimming_occurred = False
        target_words = self.calculate_target_words(desired_duration_seconds)
        content_instructions = self._build_topic_content_instructions(topic)

        # Calibrate slack based on duration:
        # For short-form videos (<= 30s), tighten upper bound slack to ~105% (e.g. 45 words for 43 target)
        # For longer videos (> 30s), preserve 115% slack.
        if desired_duration_seconds <= 30:
            max_target_words = max(target_words + 1, int(round(target_words * 1.05)))
            min_target_words = int(round(target_words * 0.90))
        else:
            max_target_words = int(target_words * 1.15)
            min_target_words = int(target_words * 0.90)

        prompt = (
            f"You are a clear, practical scriptwriter writing plain spoken narration for a short explainer video.\n"
            f"Write a full narration script for a video about: \"{topic}\".\n\n"
            f"TARGET DURATION: {desired_duration_seconds} seconds.\n"
            f"TARGET LENGTH: AT LEAST {target_words} WORDS (approx. {target_words} to {max_target_words} words).\n\n"
            f"CONTENT INSTRUCTIONS:\n"
            f"{content_instructions}\n"
            f"- The text must be long enough to reach the target word count of AT LEAST {target_words} words.\n\n"
            f"STRICT FORMATTING RULES:\n"
            f"1. Return ONLY the spoken narration text. Do not include intro/outro conversational remarks.\n"
            f"2. Do NOT include any titles, headers, or section names.\n"
            f"3. Do NOT include any bullet points or numbered lists.\n"
            f"4. Do NOT include any Markdown formatting.\n"
            f"5. Do NOT include any stage directions, narrator tags, or visual cues (e.g. no '[Music]', '(Narrator:)', 'Scene 1:').\n"
            f"6. Make the narration continuous and naturally spoken. Reach the word target by adding concrete facts, steps and cause-and-effect, never decorative language.\n"
            f"7. Write in direct, concrete sentences about what happens and why. Do NOT write poetic or ornate prose, do NOT use flowery adjectives or metaphors (e.g. 'tender', 'whispers', 'dances'), and do NOT describe camera views or how things look on screen.\n\n"
            f"Spoken Narration Script:"
        )

        raw_response = await self._post_groq(prompt, is_json=False, temperature=0.7)
        cleaned_text = self.clean_script_text(raw_response)

        if not cleaned_text:
            raise InvalidScriptResponseError("Groq returned an empty script response.")

        word_count = len(cleaned_text.split())

        if word_count < min_target_words:
            words_needed = target_words - word_count
            expansion_instructions = self._build_expansion_instructions(topic, words_needed)
            expansion_prompt = (
                f"The following video narration script on \"{topic}\" is currently {word_count} words long, but needs to be expanded by approximately {words_needed} more words to reach the target length of {target_words} words.\n\n"
                f"Current Script:\n\"{cleaned_text}\"\n\n"
                f"EXPANSION INSTRUCTIONS:\n"
                f"{expansion_instructions}\n\n"
                f"Continuation Spoken Narration:"
            )
            raw_expansion = await self._post_groq(expansion_prompt, is_json=False, temperature=0.7)
            cleaned_expansion = self.clean_script_text(raw_expansion)
            if cleaned_expansion:
                cleaned_text = f"{cleaned_text} {cleaned_expansion}".strip()
                word_count = len(cleaned_text.split())
                self.last_expansion_occurred = True

        if word_count > max_target_words:
            trimmed_text, did_trim = self.trim_to_sentence_boundary(cleaned_text, target_words)
            if did_trim:
                cleaned_text = trimmed_text
                word_count = len(cleaned_text.split())
                self.last_trimming_occurred = True

        final_threshold = int(target_words * 0.75)
        if word_count < final_threshold:
            raise ScriptProviderError(
                f"Script generation produced only {word_count} words, which is substantially below target of {target_words} words."
            )

        return cleaned_text

    async def split_into_scenes(
        self,
        script: str,
        desired_duration_seconds: int
    ) -> list[dict]:
        if not script or not script.strip():
            raise InvalidScriptResponseError("Cannot split an empty narration script into scenes.")

        # Target roughly 3 to 5 seconds per visual beat (approx. 8 to 14 words per beat)
        # For a 15-second narration, this aims for approximately 3 to 4 visual beats
        target_scenes = max(3, round(desired_duration_seconds / 4.2))
        prompt = (
            f"You are a master cinematic documentary director and visual editor.\n"
            f"Partition the following narration script into sequential, action-focused visual scene beats for a {desired_duration_seconds}-second video.\n\n"
            f"FULL SCRIPT NARRATION:\n\"\"\"\n{script.strip()}\n\"\"\"\n\n"
            f"PARTITIONING RULES:\n"
            f"1. Target approximately 3 to 5 seconds of narration per visual beat (approx. 8 to 14 words per beat). Aim for roughly {target_scenes} visual beats for this {desired_duration_seconds}-second video when the narrative content supports them.\n"
            f"2. Each scene beat must depict one clearly identifiable, concrete visual action.\n"
            f"3. You MUST preserve the narration's exact original words and sequential order verbatim. DO NOT summarize, omit, rephrase, or invent any words.\n"
            f"4. You MAY split long compound or complex sentences across consecutive scene beats at natural clause or phrase boundaries (such as commas, semicolons, or conjunctions like 'while', 'as', 'and').\n"
            f"5. DO NOT split mid-phrase or separate individual words unnaturally; ensure every scene beat forms a meaningful, coherent spoken segment.\n"
            f"6. 'scene_order' MUST start at 1 and increment sequentially (1, 2, 3, ..., N).\n"
            f"7. Every word of the original narration must be accounted for across the sequential 'scene_text' values without duplication or omission.\n\n"
            f"CRITICAL FORMATTING REQUIREMENT:\n"
            f"Do NOT write any introductory remarks, notes, or explanations before or after the JSON. Return ONLY a valid JSON object starting with '{{' and ending with '}}' matching this exact structure:\n"
            f"{{\n"
            f"  \"scenes\": [\n"
            f"    {{\"scene_order\": 1, \"scene_text\": \"...\"}},\n"
            f"    {{\"scene_order\": 2, \"scene_text\": \"...\"}}\n"
            f"  ]\n"
            f"}}\n"
        )

        raw_response = await self._post_groq(prompt, is_json=True, temperature=0.1)
        json_data = self._extract_json_payload(raw_response)

        try:
            validated = SceneListSchema.model_validate(json_data)
        except ValidationError as e:
            raise InvalidScriptResponseError(
                f"Pydantic scene list validation failed: {str(e)}"
            ) from e

        validated.scenes = self._ensure_narration_preservation(script, validated.scenes)
        return [scene.model_dump() for scene in validated.scenes]

    async def generate_visual_specification(
        self,
        scene_text: str,
        context: Optional[str] = None
    ) -> dict:
        if not scene_text or not scene_text.strip():
            raise InvalidScriptResponseError("Cannot generate visual specification for empty scene text.")

        word_count = len(scene_text.split())
        est_duration = round(max(3.0, word_count * 60.0 / self.words_per_minute), 1)

        context_block = ""
        if context and context.strip():
            context_block = (
                "VIDEO CONTEXT (use ONLY to understand what this beat refers to, for example what 'they' or 'it' "
                "means. The visual must depict the real subject of THIS video, never an unrelated object):\n"
                f"{context.strip()}\n\n"
            )

        prompt = (
            f"You are a visual cinematography director for high-end documentary video production.\n"
            f"Craft a concrete, action-focused visual specification for the following spoken narration scene beat.\n\n"
            f"{context_block}"
            f"SCENE NARRATION:\n\"{scene_text.strip()}\"\n"
            f"PROVISIONAL SCENE DURATION: approx. {est_duration} seconds.\n\n"
            f"INSTRUCTIONS:\n"
            f"1. 'primary_subject': The single concrete physical entity or object filmed in this scene (e.g. 'dry seed in soil', 'emerging root tip', 'cooking oil in pan', 'water simmering in kettle'). Must be specific to this exact scene beat, NOT the overall video title.\n"
            f"2. 'visual_prompt': Describe a single, concrete, observable physical action happening right now on screen. Specify the exact primary subject, physical motion, camera angle/framing (e.g., extreme close-up, macro shot, low angle, aerial view), setting, and natural lighting. Focus strictly on visible physical actions. Do NOT use abstract concepts, metaphors, or broad thematic summaries.\n"
            f"3. 'search_keywords': Provide a list of 3 to 5 highly specific stock video search phrases (2 to 4 words each) strictly tailored for video libraries like Pexels.\n"
            f"   - Prioritize concrete subjects, visible physical motions, and camera perspectives (e.g., 'macro timelapse', 'close up', 'underground').\n"
            f"   - Every keyword MUST directly depict the literal subject and action described in this narration beat.\n"
            f"   - STRICTLY FORBIDDEN: Do NOT include broad thematic concepts, historical/cultural associations, human activities not mentioned in the beat, or generic meta-tags (e.g., FORBIDDEN: 'ancient farmers', 'agricultural history', 'cultural ceremony', 'nature documentary', 'cinematic landscape').\n"
            f"   - Do NOT invent unrelated characters or events.\n"
            f"4. SCIENTIFIC & BIOLOGICAL ACCURACY CONSTRAINTS:\n"
            f"   - PLANT ROOT & SHOOT GROWTH DIRECTION (GRAVITROPISM & PHOTOTROPISM):\n"
            f"     * ROOTS (radicle, root tip, taproot): Plant roots exhibit positive gravitropism. Roots ALWAYS grow and push DOWNWARD into soil/substrate or dark earth. NEVER describe roots growing upward or toward the surface/sky.\n"
            f"     * SHOOTS & STEMS (hypocotyl, epicotyl, plumule): Stems and leaves exhibit positive phototropism and negative gravitropism. Shoots ALWAYS grow UPWARD toward the light/sun/sky.\n"
            f"   - DEVELOPMENTAL STAGE INTEGRITY:\n"
            f"     * A dormant, dry, or imbibing seed must be depicted as a seed (e.g., dry seed resting in soil, swelling seed coat). STRICTLY FORBIDDEN: Do NOT depict an already-grown green seedling, leafy plant, or nursery tray of sprouts when the narration describes a dry seed absorbing water or resting in soil.\n"
            f"     * Underground root emergence occurs beneath or at the soil surface; young roots are pale, translucent, or whitish. Do NOT depict green leafy foliage for subterranean root emergence.\n"
            f"   - BIOCHEMICAL, MOLECULAR & UNFILMABLE INTERNAL PROCESSES:\n"
            f"     * Internal biochemical processes (enzymes activating, food breakdown, starch converting to glucose, cellular respiration, ATP production, DNA replication, molecular binding) happen at the cellular/molecular level and CANNOT be filmed with a normal camera in stock footage.\n"
            f"     * DO NOT invent fake kitchen or laboratory physical actions (FORBIDDEN: pouring water into drinking glasses, mixing chemicals in glass bowls, laboratory beakers) to represent internal cellular or plant biology!\n"
            f"     * If the narration describes an internal biochemical process: specify a truthful, macroscopic observable physical manifestation (e.g. 'Observable macro timelapse of seed coat swelling and softening as water saturates the dormant seed in soil') with concrete search keywords (e.g. ['seed coat swelling', 'seed absorbing water timelapse']). If no truthful observable manifestation exists, specify the subject accurately and note that an internal diagram is required.\n"
            f"5. 'action_timeline': Provide a chronological list of 1 to 2 specific action events for this beat.\n"
            f"   Each action event must contain:\n"
            f"   - 'action_description': clear physical action visible in the shot.\n"
            f"   - 'visual_start_seconds': provisional start time relative to scene start (starting at 0.0).\n"
            f"   - 'visual_end_seconds': provisional end time relative to scene start (e.g. 2.0 to 4.0).\n"
            f"   - 'sound_type': choose 'action' (audible physical action like impact or pour), 'ambience' (continuous natural background like rain or wind), or 'none' (naturally silent actions).\n"
            f"   - 'sfx_query': realistic sound description query (e.g. 'water droplets falling'), or null if sound_type is 'none'.\n"
            f"   - 'sfx_start_seconds': offset when sound begins (or null if sound_type is 'none').\n"
            f"   - 'sfx_duration_seconds': duration of sound effect in seconds (or null if sound_type is 'none').\n"
            f"   - 'sfx_volume': relative volume between 0.1 and 1.0 (or null if sound_type is 'none').\n"
            f"   SOUND REALISM RULES:\n"
            f"   - Distinguish observable physical actions from audible acoustic events.\n"
            f"   - Naturally silent actions (e.g., seeds resting in soil, roots pushing downward underground, cell division, leaves gently unfolding) DO NOT generate audible impacts! Set sound_type='none' and sfx_query=null for naturally silent biological growth.\n"
            f"   - STRICTLY FORBIDDEN: Do NOT invent dramatic cracking, explosions, crunching, or fictional sci-fi sounds for silent biological processes.\n"
            f"   - Use sound_type='ambience' only when an environmental background sound naturally accompanies the shot (e.g. soft rain falling, gentle wind rustle).\n"
            f"6. 'shot_plan': For scenes containing multiple distinct physical actions, provide a sequential list of sub-scene camera shots (maximum 4 shots).\n"
            f"   SHOT PLANNING RULES:\n"
            f"   - SINGLE-ACTION SCENES: If the scene narration describes only one observable physical action or continuous state, set \"shot_plan\": null (or omit it). Do NOT split a single continuous action into redundant shots.\n"
            f"   - MULTI-ACTION SCENES: If the narration contains two distinct sequential physical actions (e.g., \"Pour oil into the hot pan, then add sliced carrots\"), create 2 shots. If it contains three distinct observable actions and duration permits, create 3 shots.\n"
            f"   - NEVER create extra shots solely to increase shot count.\n"
            f"   - NEVER invent actions absent from the spoken narration or supported scene context.\n"
            f"   - Preserve the original scene narration meaning and chronological sequence exactly.\n"
            f"   - Prefer actions that can realistically be represented by stock footage (e.g. pouring oil, dropping sliced carrots, robotic welding, circuit board soldering, seed sprouting, water dripping).\n"
            f"   - Maintain logical action order and continuity.\n\n"
            f"   EACH SHOT IN 'shot_plan' MUST CONTAIN:\n"
            f"   - 'shot_index': 1-indexed sequential integer starting at 1 (1, 2, 3, ...).\n"
            f"   - 'required_visible_action': concrete observable physical action on screen (non-empty string).\n"
            f"   - 'primary_subject': the main physical entity or object filmed (non-empty string).\n"
            f"   - 'required_setting': specific environment or setting (non-empty string).\n"
            f"   - 'visual_state': physical visual state (e.g. 'empty heating', 'sizzling', 'raw', 'clean', 'sprouting').\n"
            f"   - 'camera_framing': camera perspective (e.g. 'close-up', 'macro close-up', 'extreme close-up', 'overhead', 'medium shot').\n"
            f"   - 'search_keywords': list of 2 to 5 specific stock video search queries tailored for stock libraries (2 to 4 words each).\n"
            f"   - 'required_objects': list of 1 to 4 concrete physical items that must be visible in this shot (e.g. ['pan', 'cooking oil stream']).\n"
            f"   - 'explicit_exclusions': list of 2 to 5 contradictory items or settings that must NOT appear in this shot (e.g. ['meat', 'chicken', 'bathtub', 'raw fish']).\n"
            f"   - 'provisional_start_seconds': non-negative float timestamp relative to scene start (starting at 0.0 for shot 1).\n"
            f"   - 'provisional_end_seconds': float timestamp strictly greater than provisional_start_seconds.\n"
            f"   - 'expected_duration': float duration in seconds (provisional_end_seconds - provisional_start_seconds).\n"
            f"   - 'action_event_index': zero-based integer index pointing to the corresponding event in 'action_timeline' if this shot directly depicts that action event; or null if it does not correspond to an action event.\n\n"
            f"   PROVISIONAL TIMING & ACTION LINKAGE RULES:\n"
            f"   - Chronological and contiguous: shot 1 starts at 0.0s. Subsequent shots start at or immediately after the previous shot ends (no overlaps: shot[i].provisional_end_seconds <= shot[i+1].provisional_start_seconds).\n"
            f"   - Shot durations must be positive and practical (typically 1.5s to 3.5s per shot).\n"
            f"   - Total provisional shot duration should sum to approximately the scene duration ({est_duration}s).\n"
            f"   - Never reference an absent action event: action_event_index must be between 0 and len(action_timeline) - 1, or null.\n\n"
            f"CRITICAL FORMATTING REQUIREMENT:\n"
            f"Do NOT write any introductory remarks, notes, or explanations before or after the JSON. Return ONLY a valid JSON object starting with '{{' and ending with '}}' matching one of these structures:\n\n"
            f"Multi-action scene example (2 distinct actions):\n"
            f"{{\n"
            f"  \"primary_subject\": \"cooking oil in pan\",\n"
            f"  \"visual_prompt\": \"Cinematic macro close-up of cooking oil poured into a hot pan, followed by crisp sliced carrots added to sizzle.\",\n"
            f"  \"search_keywords\": [\"pouring oil pan\", \"cooking carrots wok\", \"vegetable stir fry skillet\"],\n"
            f"  \"action_timeline\": [\n"
            f"    {{\n"
            f"      \"action_description\": \"Pouring cooking oil into hot pan\",\n"
            f"      \"visual_start_seconds\": 0.0,\n"
            f"      \"visual_end_seconds\": 2.0,\n"
            f"      \"sound_type\": \"action\",\n"
            f"      \"sfx_query\": \"oil sizzling hot pan\",\n"
            f"      \"sfx_start_seconds\": 0.5,\n"
            f"      \"sfx_duration_seconds\": 1.5,\n"
            f"      \"sfx_volume\": 0.35\n"
            f"    }},\n"
            f"    {{\n"
            f"      \"action_description\": \"Adding sliced carrots into sizzling pan\",\n"
            f"      \"visual_start_seconds\": 2.0,\n"
            f"      \"visual_end_seconds\": 4.0,\n"
            f"      \"sound_type\": \"action\",\n"
            f"      \"sfx_query\": \"vegetables dropping hot oil\",\n"
            f"      \"sfx_start_seconds\": 2.2,\n"
            f"      \"sfx_duration_seconds\": 1.5,\n"
            f"      \"sfx_volume\": 0.4\n"
            f"    }}\n"
            f"  ],\n"
            f"  \"shot_plan\": [\n"
            f"    {{\n"
            f"      \"shot_index\": 1,\n"
            f"      \"required_visible_action\": \"Pouring golden cooking oil into hot pan\",\n"
            f"      \"primary_subject\": \"Cooking oil streaming into pan\",\n"
            f"      \"required_setting\": \"Stovetop skillet\",\n"
            f"      \"visual_state\": \"oil heating shimmering\",\n"
            f"      \"camera_framing\": \"close-up\",\n"
            f"      \"search_keywords\": [\"pouring oil pan\", \"hot skillet oil shimmering\"],\n"
            f"      \"required_objects\": [\"skillet\", \"cooking oil bottle\"],\n"
            f"      \"explicit_exclusions\": [\"meat\", \"chicken\", \"bathtub\", \"salad bowl\"],\n"
            f"      \"provisional_start_seconds\": 0.0,\n"
            f"      \"provisional_end_seconds\": 2.0,\n"
            f"      \"expected_duration\": 2.0,\n"
            f"      \"action_event_index\": 0\n"
            f"    }},\n"
            f"    {{\n"
            f"      \"shot_index\": 2,\n"
            f"      \"required_visible_action\": \"Dropping fresh sliced carrots into hot oil\",\n"
            f"      \"primary_subject\": \"Sliced orange carrots\",\n"
            f"      \"required_setting\": \"Stovetop skillet\",\n"
            f"      \"visual_state\": \"carrots hitting hot oil sizzling\",\n"
            f"      \"camera_framing\": \"macro close-up\",\n"
            f"      \"search_keywords\": [\"adding carrots wok\", \"sliced carrots stir fry\"],\n"
            f"      \"required_objects\": [\"sliced carrots\", \"skillet\"],\n"
            f"      \"explicit_exclusions\": [\"meat\", \"seafood\", \"crab\", \"bathtub\"],\n"
            f"      \"provisional_start_seconds\": 2.0,\n"
            f"      \"provisional_end_seconds\": 4.0,\n"
            f"      \"expected_duration\": 2.0,\n"
            f"      \"action_event_index\": 1\n"
            f"    }}\n"
            f"  ]\n"
            f"}}\n\n"
            f"Single-action scene example:\n"
            f"{{\n"
            f"  \"primary_subject\": \"simmering water in kettle\",\n"
            f"  \"visual_prompt\": \"Cinematic macro close-up of water simmering in a stainless steel kettle\",\n"
            f"  \"search_keywords\": [\"water simmering kettle\", \"boiling water steam close up\"],\n"
            f"  \"action_timeline\": [\n"
            f"    {{\n"
            f"      \"action_description\": \"Water heating and bubbling gently in kettle\",\n"
            f"      \"visual_start_seconds\": 0.0,\n"
            f"      \"visual_end_seconds\": 4.0,\n"
            f"      \"sound_type\": \"action\",\n"
            f"      \"sfx_query\": \"water simmering bubbling\",\n"
            f"      \"sfx_start_seconds\": 0.5,\n"
            f"      \"sfx_duration_seconds\": 3.0,\n"
            f"      \"sfx_volume\": 0.35\n"
            f"    }}\n"
            f"  ],\n"
            f"  \"shot_plan\": null\n"
            f"}}\n"
        )

        raw_response = await self._post_groq(prompt, is_json=True, temperature=0.1)
        json_data = self._extract_json_payload(raw_response)

        try:
            validated = VisualSpecificationSchema.model_validate(json_data)
            self.last_shot_plan_fallback_occurred = False
            self.last_shot_plan_error = None
            return validated.model_dump()
        except ValidationError as e:
            # Check if shot_plan was provided and caused the validation failure
            has_shot_plan = bool(json_data.get("shot_plan"))
            if has_shot_plan and self.fallback_to_single_shot:
                # Attempt safe fallback by validating base visual spec without shot_plan
                fallback_data = dict(json_data)
                fallback_data["shot_plan"] = None
                try:
                    base_validated = VisualSpecificationSchema.model_validate(fallback_data)
                    self.last_shot_plan_fallback_occurred = True
                    self.last_shot_plan_error = str(e)
                    logger.warning(
                        "Groq shot_plan validation failed: %s. Safely falling back to single-shot visual specification (shot_plan=None).",
                        e
                    )
                    return base_validated.model_dump()
                except ValidationError:
                    # Even base specification is invalid
                    pass

            raise InvalidScriptResponseError(
                f"Pydantic visual specification validation failed: {str(e)}"
            ) from e
