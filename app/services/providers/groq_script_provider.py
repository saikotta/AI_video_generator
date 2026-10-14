import re
import json
import random
import asyncio
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
        self.words_per_minute = words_per_minute or settings.NARRATION_WPM
        self.max_retries = max_retries or settings.GROQ_MAX_RETRIES
        self.base_delay = base_delay or settings.GROQ_RETRY_BASE_DELAY
        self.timeout_seconds = timeout_seconds
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        self._internal_client: Optional[httpx.AsyncClient] = None

        self.last_expansion_occurred = False
        self.last_trimming_occurred = False
        self.last_retry_count = 0

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

    def calculate_target_words(self, duration_seconds: int) -> int:
        return int(duration_seconds * self.words_per_minute / 60)

    def clean_script_text(self, text: str) -> str:
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'\((?:Narrator|Voiceover|Scene|Music|SFX)[^)]*\)', '', text, flags=re.IGNORECASE)
        text = re.sub(r'^#+\s+.*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'\*+([^*]+)\*+', r'\1', text)
        text = re.sub(r'_+([^_]+)_+', r'\1', text)
        text = re.sub(r'^\s*[*•-]\s*', '', text, flags=re.MULTILINE)
        text = text.replace('—', ' - ').replace('–', ' - ').replace('’', "'").replace('“', '"').replace('”', '"')
        text = re.sub(r'^(Here is|Here\'s|Sure, here is|Below is)[^:\n]*:\s*', '', text, flags=re.IGNORECASE)
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

    async def _post_groq(self, prompt: str, is_json: bool = False, temperature: float = 0.7) -> str:
        if not self.api_key:
            raise ScriptProviderError(
                "GROQ_API_KEY environment variable is not set. Cannot invoke Groq API."
            )

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
                        f"Groq request timed out after {self.timeout_seconds} seconds."
                    ) from e
                await asyncio.sleep(self.base_delay * (2 ** attempt))
                continue
            except httpx.HTTPError as e:
                if attempt == self.max_retries - 1:
                    raise ScriptProviderError(f"HTTP connection error connecting to Groq API.") from e
                await asyncio.sleep(self.base_delay * (2 ** attempt))
                continue

            if response.status_code == 200:
                data = response.json()
                try:
                    return data["choices"][0]["message"]["content"]
                except (KeyError, IndexError) as e:
                    raise InvalidScriptResponseError("Groq response payload missing choices array.") from e

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
                        f"Groq API rate limit exceeded after {self.max_retries} attempts (HTTP 429)."
                    )
                await asyncio.sleep(delay)

            else:
                raise ScriptProviderError(
                    f"Groq API returned HTTP error status {response.status_code}."
                )

        raise ScriptProviderError(f"Groq API request retries exhausted after {self.max_retries} attempts.")

    def trim_to_sentence_boundary(self, text: str, target_words: int) -> tuple[str, bool]:
        words = text.split()
        current_word_count = len(words)
        max_words = int(target_words * 1.15)
        min_words = int(target_words * 0.90)

        if current_word_count <= max_words:
            return text, False

        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        if not sentences:
            trimmed_text = " ".join(words[:max_words]).strip()
            if not trimmed_text.endswith(('.', '!', '?')):
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
                    clauses = [c.strip() for c in re.split(r'(?<=[,;:—–])\s+', sentence) if c.strip()]
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
                        rem = max_words - accumulated_words
                        clause_words_list = sentence.split()[:rem]
                        if clause_words_list:
                            trimmed_sentences.append(" ".join(clause_words_list))
                            accumulated_words += len(clause_words_list)
                    break

        trimmed_text = " ".join(trimmed_sentences).strip()
        if trimmed_text and not trimmed_text[-1] in '.!?':
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

    async def generate_script(self, topic: str, desired_duration_seconds: int) -> str:
        self.last_expansion_occurred = False
        self.last_trimming_occurred = False
        target_words = self.calculate_target_words(desired_duration_seconds)

        prompt = (
            f"You are a master documentary scriptwriter crafting a rich, immersive spoken narration.\n"
            f"Write a full narration script for a video about: \"{topic}\".\n\n"
            f"TARGET DURATION: {desired_duration_seconds} seconds.\n"
            f"TARGET LENGTH: AT LEAST {target_words} WORDS (approx. {target_words} to {int(target_words * 1.15)} words).\n\n"
            f"CONTENT INSTRUCTIONS:\n"
            f"- Provide a comprehensive, detailed story covering origins, key historical moments, cultural significance, anecdotes, major developments, and a strong conclusion.\n"
            f"- It is CRITICAL that the text is detailed and lengthy enough to reach the target word count of AT LEAST {target_words} words.\n\n"
            f"STRICT FORMATTING RULES:\n"
            f"1. Return ONLY the spoken narration text. Do not include intro/outro conversational remarks.\n"
            f"2. Do NOT include any titles, headers, or section names.\n"
            f"3. Do NOT include any bullet points or numbered lists.\n"
            f"4. Do NOT include any Markdown formatting.\n"
            f"5. Do NOT include any stage directions, narrator tags, or visual cues (e.g. no '[Music]', '(Narrator:)', 'Scene 1:').\n"
            f"6. Make the narration continuous, naturally spoken, and rich in detail to fulfill the {target_words} words target.\n\n"
            f"Spoken Narration Script:"
        )

        raw_response = await self._post_groq(prompt, is_json=False, temperature=0.7)
        cleaned_text = self.clean_script_text(raw_response)

        if not cleaned_text:
            raise InvalidScriptResponseError("Groq returned an empty script response.")

        word_count = len(cleaned_text.split())
        min_target_words = int(target_words * 0.90)

        if word_count < min_target_words:
            words_needed = target_words - word_count
            expansion_prompt = (
                f"The following video narration script on \"{topic}\" is currently {word_count} words long, but needs to be expanded by approximately {words_needed} more words to reach the target length of {target_words} words.\n\n"
                f"Current Script:\n\"{cleaned_text}\"\n\n"
                f"EXPANSION INSTRUCTIONS:\n"
                f"1. Write a seamless continuation adding more historical context, specific examples, fascinating details, and deeper narrative flow to reach an additional {words_needed} words.\n"
                f"2. DO NOT repeat the current script.\n"
                f"3. Return ONLY spoken narration text without Markdown, headings, bullet points, or stage directions.\n\n"
                f"Continuation Spoken Narration:"
            )
            raw_expansion = await self._post_groq(expansion_prompt, is_json=False, temperature=0.7)
            cleaned_expansion = self.clean_script_text(raw_expansion)
            if cleaned_expansion:
                cleaned_text = f"{cleaned_text} {cleaned_expansion}".strip()
                word_count = len(cleaned_text.split())
                self.last_expansion_occurred = True

        max_target_words = int(target_words * 1.15)
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

        target_scenes = max(3, round(desired_duration_seconds / 11.5))
        prompt = (
            f"You are a master documentary video director.\n"
            f"Partition the following full narration script into sequential scene beats for a {desired_duration_seconds}-second video.\n\n"
            f"FULL SCRIPT NARRATION:\n\"\"\"\n{script.strip()}\n\"\"\"\n\n"
            f"PARTITIONING RULES:\n"
            f"1. You MUST partition the script into sequential scene beats. Every sentence from the script MUST appear in one of the scenes.\n"
            f"2. DO NOT omit, summarize, truncate, or rewrite any sentences from the script.\n"
            f"3. Target roughly 8 to 15 seconds of narration per scene (approx. 20-35 words per scene beat).\n"
            f"4. Aim for roughly {target_scenes} scenes naturally following the narrative flow of the script.\n"
            f"5. 'scene_order' MUST start at 1 and increment sequentially (1, 2, 3, ..., N).\n"
            f"6. Each 'scene_text' must contain the verbatim narration sentences for that beat.\n\n"
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
        scene_text: str
    ) -> dict:
        if not scene_text or not scene_text.strip():
            raise InvalidScriptResponseError("Cannot generate visual specification for empty scene text.")

        prompt = (
            f"You are a visual cinematography director for a documentary video.\n"
            f"Craft a concrete visual specification for the following spoken narration scene beat.\n\n"
            f"SCENE NARRATION:\n\"{scene_text.strip()}\"\n\n"
            f"INSTRUCTIONS:\n"
            f"1. 'visual_prompt': Write a short, concrete visual description of observable subjects, people, actions, setting, lighting, and camera composition. NO abstract concepts or vague metaphors.\n"
            f"2. 'search_keywords': Provide a list of 3 to 5 concrete search terms suitable for finding stock footage on video platforms like Pexels.\n\n"
            f"CRITICAL FORMATTING REQUIREMENT:\n"
            f"Do NOT write any introductory remarks, notes, or explanations before or after the JSON. Return ONLY a valid JSON object starting with '{{' and ending with '}}' matching this exact structure:\n"
            f"{{\n"
            f"  \"visual_prompt\": \"Cinematic close-up of ...\",\n"
            f"  \"search_keywords\": [\"keyword1\", \"keyword2\", \"keyword3\"]\n"
            f"}}\n"
        )

        raw_response = await self._post_groq(prompt, is_json=True, temperature=0.1)
        json_data = self._extract_json_payload(raw_response)

        try:
            validated = VisualSpecificationSchema.model_validate(json_data)
        except ValidationError as e:
            raise InvalidScriptResponseError(
                f"Pydantic visual specification validation failed: {str(e)}"
            ) from e

        return validated.model_dump()
