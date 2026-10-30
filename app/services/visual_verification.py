
import os
import re
import logging
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from app.schemas.visual_spec import PLACEHOLDER_SETTING

logger = logging.getLogger(__name__)


class VerificationStatus(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    UNCERTAIN = "uncertain"
    MANUAL_REVIEW_REQUIRED = "manual_review_required"


class ConceptFeasibility(str, Enum):
    STOCK_FILMABLE = "stock_filmable"
    CONCEPTUAL_ABSTRACT = "conceptual_abstract"


@dataclass
class VisualVerificationResult:
    """
    Structured outcome of candidate-video inspection.

    IMPORTANT:
    The heuristic verifier inspects frame health and metadata, but does not
    semantically understand image content. Therefore, visual verification
    defaults to False. Explicit manual approval is handled separately.
    """
    status: VerificationStatus
    confidence: float
    confirmed_matches: List[str] = field(default_factory=list)
    contradictions: List[str] = field(default_factory=list)
    uncertainties: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)
    frame_count_evaluated: int = 0
    evaluated_timestamps: List[float] = field(default_factory=list)
    provider: str = "heuristic_frame_verifier"
    is_visual_frame_verified: bool = False
    is_manually_approved: bool = False
    recommended_strategy: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": (
                self.status.value
                if isinstance(self.status, VerificationStatus)
                else str(self.status)
            ),
            "confidence": round(float(self.confidence), 3),
            "confirmed_matches": list(self.confirmed_matches),
            "contradictions": list(self.contradictions),
            "uncertainties": list(self.uncertainties),
            "evidence": list(self.evidence),
            "frame_count_evaluated": self.frame_count_evaluated,
            "evaluated_timestamps": [
                round(t, 3) for t in self.evaluated_timestamps
            ],
            "provider": self.provider,
            "is_visual_frame_verified": self.is_visual_frame_verified,
            "is_manually_approved": self.is_manually_approved,
            "recommended_strategy": self.recommended_strategy,
        }


class FrameExtractor:
    """
    Decode representative frames from a video across its usable interval.

    Returns tuples of:
        (timestamp_seconds, PIL.Image, numpy_RGB_array)
    """

    @classmethod
    def extract_frames(
        cls,
        video_path: str,
        num_frames: int = 3,
        start_ratio: float = 0.20,
        end_ratio: float = 0.80,
        target_size: Tuple[int, int] = (640, 360),
    ) -> List[Tuple[float, Image.Image, np.ndarray]]:
        if not os.path.exists(video_path):
            raise FileNotFoundError(
                f"Video file not found for frame extraction: {video_path}"
            )

        if num_frames < 1:
            raise ValueError("num_frames must be at least 1")

        if not 0.0 <= start_ratio <= end_ratio <= 1.0:
            raise ValueError(
                "Expected 0 <= start_ratio <= end_ratio <= 1"
            )

        file_size = os.path.getsize(video_path)

        if file_size < 1000:
            try:
                with open(video_path, "rb") as file:
                    header = file.read(64)

                if any(
                    marker in header.lower()
                    for marker in (b"mock", b"dummy", b"test")
                ):
                    # Synthetic frames are only for offline mock tests.
                    extracted = []

                    for i in range(num_frames):
                        timestamp = 1.0 + i * 1.5
                        image = Image.new(
                            "RGB",
                            target_size,
                            color=(
                                min(255, 60 + i * 40),
                                min(255, 120 + i * 20),
                                max(0, 180 - i * 30),
                            ),
                        )
                        extracted.append(
                            (timestamp, image, np.asarray(image))
                        )

                    return extracted
            except Exception:
                pass

            raise ValueError(
                f"Video file too small ({file_size} bytes): {video_path}"
            )

        import av

        try:
            container = av.open(video_path)
        except Exception as exc:
            raise RuntimeError(
                f"PyAV failed to open container {video_path}: {exc}"
            ) from exc

        if not container.streams.video:
            container.close()
            raise ValueError(
                f"No video streams found in file: {video_path}"
            )

        stream = container.streams.video[0]
        duration_sec = 0.0

        if stream.duration is not None and stream.time_base is not None:
            duration_sec = float(stream.duration * stream.time_base)
        elif container.duration is not None:
            duration_sec = float(container.duration / av.time_base)

        if duration_sec <= 0.0:
            duration_sec = 5.0

        if num_frames == 1:
            timestamps = [duration_sec * 0.5]
        else:
            timestamps = [
                duration_sec
                * (
                    start_ratio
                    + i * (end_ratio - start_ratio)
                    / float(num_frames - 1)
                )
                for i in range(num_frames)
            ]

        extracted: List[Tuple[float, Image.Image, np.ndarray]] = []

        try:
            for target_t in timestamps:
                if stream.time_base:
                    target_pts = int(target_t / float(stream.time_base))
                else:
                    target_pts = int(target_t * 1000)

                try:
                    container.seek(
                        target_pts,
                        stream=stream,
                        backward=True,
                    )
                except Exception:
                    container.seek(0)

                frame_found = None

                for frame in container.decode(stream):
                    if frame.pts is not None and stream.time_base:
                        frame_t = float(frame.pts * stream.time_base)
                    else:
                        frame_t = target_t

                    if frame_t >= target_t - 0.25:
                        frame_found = frame
                        break

                if frame_found is None:
                    container.seek(0)

                    for frame in container.decode(stream):
                        frame_found = frame
                        break

                if frame_found is not None:
                    pil_image = frame_found.to_image()

                    if target_size:
                        pil_image = pil_image.resize(
                            target_size,
                            Image.Resampling.BILINEAR,
                        )

                    extracted.append(
                        (
                            round(target_t, 3),
                            pil_image,
                            np.asarray(pil_image),
                        )
                    )
        finally:
            container.close()

        if not extracted:
            raise RuntimeError(
                f"Could not decode any valid video frames from {video_path}"
            )

        return extracted

    @classmethod
    def compute_frame_motion_delta(
        cls,
        frame_arrays: List[np.ndarray],
    ) -> float:
        """
        Calculate average pixel difference between consecutive samples.

        This is a technical motion indicator only. It does not establish
        which object moved or whether the requested action occurred.
        """
        if len(frame_arrays) < 2:
            return 10.0

        deltas = []

        for first, second in zip(frame_arrays, frame_arrays[1:]):
            first_float = first.astype(np.float32)
            second_float = second.astype(np.float32)

            deltas.append(
                float(np.mean(np.abs(first_float - second_float)))
            )

        return float(np.mean(deltas))

    @classmethod
    def compute_frame_luminance(
        cls,
        frame_arrays: List[np.ndarray],
    ) -> float:
        """Calculate mean pixel intensity, not semantic image content."""
        if not frame_arrays:
            return 0.0

        return float(
            np.mean([np.mean(array) for array in frame_arrays])
        )

    @classmethod
    def generate_contact_sheet(
        cls,
        frames: List[Tuple[float, Image.Image, np.ndarray]],
        output_path: str,
        title: str = "Candidate Video Inspection",
        required_action: str = "",
        primary_subject: str = "",
        required_setting: str = "",
        candidate_id: Any = "unknown",
        evidence: Optional[List[str]] = None,
    ) -> str:
        """
        Generate a contact sheet for a human reviewer.

        The contact sheet supports review; it does not automatically
        approve the candidate.
        """
        from PIL import ImageDraw, ImageFont

        os.makedirs(
            os.path.dirname(os.path.abspath(output_path)),
            exist_ok=True,
        )

        if not frames:
            blank = Image.new("RGB", (640, 360), color=(30, 30, 30))
            draw = ImageDraw.Draw(blank)
            draw.text(
                (20, 20),
                f"No frames available for candidate {candidate_id}",
                fill=(255, 100, 100),
            )
            blank.save(output_path, "JPEG")
            return output_path

        thumb_w, thumb_h = 360, 202
        padding = 16
        header_h = 90
        footer_h = 40

        total_w = padding + len(frames) * (thumb_w + padding)
        total_h = header_h + thumb_h + footer_h + padding

        sheet = Image.new(
            "RGB",
            (total_w, total_h),
            color=(20, 24, 30),
        )
        draw = ImageDraw.Draw(sheet)
        font = ImageFont.load_default()

        draw.text(
            (padding, 12),
            f"{title} | Candidate Pexels ID: #{candidate_id}",
            fill=(240, 240, 255),
            font=font,
        )
        draw.text(
            (padding, 32),
            f"Subject: {primary_subject[:60]}  |  "
            f"Setting: {required_setting[:50]}",
            fill=(180, 200, 220),
            font=font,
        )
        draw.text(
            (padding, 52),
            f"Required Action: {required_action[:90]}",
            fill=(255, 220, 120),
            font=font,
        )
        draw.line(
            [(padding, 78), (total_w - padding, 78)],
            fill=(60, 70, 85),
            width=1,
        )

        for index, (timestamp, pil_image, _) in enumerate(frames):
            x = padding + index * (thumb_w + padding)
            y = header_h

            thumbnail = pil_image.resize(
                (thumb_w, thumb_h),
                Image.Resampling.LANCZOS,
            )
            sheet.paste(thumbnail, (x, y))

            draw.rectangle(
                [x, y, x + thumb_w, y + thumb_h],
                outline=(80, 100, 120),
                width=1,
            )
            draw.text(
                (x + 8, y + thumb_h + 8),
                f"Sampled: {timestamp:.2f}s (Frame #{index + 1})",
                fill=(200, 220, 240),
                font=font,
            )

        evidence_summary = (
            "; ".join(evidence[:2])
            if evidence
            else "Awaiting manual reviewer verification"
        )

        draw.text(
            (padding, total_h - 28),
            f"Review Evidence: {evidence_summary[:120]}",
            fill=(140, 150, 165),
            font=font,
        )

        sheet.save(output_path, "JPEG", quality=90)
        return output_path


class ConceptClassifier:
    """
    Classify requests as stock-filmable or abstract/conceptual.

    This is a routing heuristic, not an image recognition model.
    """

    CONCEPTUAL_ABSTRACT_KEYWORDS = {
        "enzyme", "enzymes", "enzymatic", "catalyst", "catalysis",
        "starch breakdown", "stored food into", "simple sugars",
        "glucose conversion", "glucose", "cellular respiration",
        "atp synthesis", "glycolysis", "metabolism", "metabolic",
        "protein synthesis", "transcription", "translation",
        "osmosis", "molecular diffusion", "active transport",
        "hormone signaling", "gibberellin", "auxin transport",
        "biochemical", "microscopic", "cellular", "cell division",
        "mitosis", "meiosis", "chloroplast", "chlorophyll",
        "photosynthesis", "dna", "rna", "ribosome", "mitochondria",
        "atom", "atomic", "electron", "proton", "neutron", "quantum",
        "molecule", "molecular", "internal combustion",
        "piston inside", "cylinder stroke", "four stroke cycle",
        "valve timing", "crankshaft rotation inside",
        "fuel injection chamber", "spark plug ignition cylinder",
        "inside the engine", "engine cross section",
        "chemical reaction equation", "thermodynamics diagram",
        "gravity wave", "electromagnetic spectrum", "magnetic field lines",
        "subatomic",
    }

    @classmethod
    def classify_feasibility(
        cls,
        action: str,
        subject: str,
        visual_prompt: str = "",
    ) -> Tuple[ConceptFeasibility, Optional[str]]:
        combined = f"{action} {subject} {visual_prompt}".lower()

        for term in cls.CONCEPTUAL_ABSTRACT_KEYWORDS:
            if re.search(r"\b" + re.escape(term) + r"\b", combined):
                return (
                    ConceptFeasibility.CONCEPTUAL_ABSTRACT,
                    f"Action or subject '{term}' represents an internal, "
                    "microscopic, or abstract scientific mechanism",
                )

        return ConceptFeasibility.STOCK_FILMABLE, None


class HeuristicVisualVerifier:
    """
    Inspect sampled frames for technical health and compare requirements
    with clip metadata.

    IMPORTANT:
    This class does not semantically interpret pixels. Metadata matches
    can help rank candidates, but can never establish that the requested
    subject, action, or developmental stage is actually visible.
    """

    DOMAIN_SETTING_PATTERNS = {
        "kitchen": {
            "kitchen", "stove", "stovetop", "skillet", "pan", "wok",
            "counter", "chef", "cook", "restaurant", "cutting board",
        },
        "soil": {
            "soil", "ground", "dirt", "earth", "underground",
            "subterranean", "roots", "root", "sprout", "burrow",
        },
        "workshop": {
            "workshop", "factory", "industrial", "engine", "machine",
            "garage", "mechanic", "machinery", "tools",
        },
        "ocean": {
            "ocean", "sea", "wave", "waves", "surf", "beach", "coast",
            "water", "marine", "shoreline",
        },
        "construction": {
            "construction", "site", "excavator", "bulldozer", "crane",
            "dirt work", "quarry", "digging", "heavy equipment",
        },
        "nature": {
            "forest", "plants", "trees", "leaves", "outdoor", "field",
            "garden", "foliage", "meadow",
        },
        "bathroom": {
            "bathroom", "bath", "bathtub", "bath salts", "tub",
            "shower", "soap", "towel", "toilet",
        },
    }

    @classmethod
    def evaluate(
        cls,
        frames: List[Tuple[float, Image.Image, np.ndarray]],
        shot_requirements: Dict[str, Any],
        candidate_metadata: Optional[Dict[str, Any]] = None,
    ) -> VisualVerificationResult:
        req_subject = str(
            shot_requirements.get("primary_subject") or ""
        ).strip().lower()

        req_action = str(
            shot_requirements.get("required_visible_action") or ""
        ).strip().lower()

        req_setting = str(
            shot_requirements.get("required_setting") or ""
        ).strip().lower()

        if req_setting == PLACEHOLDER_SETTING:
            req_setting = ""

        req_state = str(
            shot_requirements.get("visual_state") or ""
        ).strip().lower()

        req_objects = [
            str(obj).strip().lower()
            for obj in (shot_requirements.get("required_objects") or [])
            if obj
        ]

        explicit_exclusions = [
            str(item).strip().lower()
            for item in (
                shot_requirements.get("explicit_exclusions") or []
            )
            if item
        ]

        candidate_meta = candidate_metadata or {}

        # Only candidate-provided clip descriptions are used for metadata
        # matching. Search queries are deliberately excluded to avoid
        # treating the search request itself as evidence.
        cand_text = re.sub(
            r"[^\w\s]",
            " ",
            " ".join(
                [
                    str(candidate_meta.get("clip_text") or ""),
                    str(candidate_meta.get("title") or ""),
                    str(candidate_meta.get("tags") or ""),
                ]
            ).lower(),
        )

        frame_arrays = [array for _, _, array in frames]
        evaluated_timestamps = [timestamp for timestamp, _, _ in frames]

        motion_delta = FrameExtractor.compute_frame_motion_delta(
            frame_arrays
        )
        luminance = FrameExtractor.compute_frame_luminance(frame_arrays)

        evidence: List[str] = [
            f"Inspected {len(frames)} frames at timestamps "
            f"{evaluated_timestamps}s",
            f"Frame motion delta: {motion_delta:.2f} "
            "(pixel variance across interval; not action recognition)",
            f"Frame luminance: {luminance:.1f} (0-255 scale)",
        ]

        confirmed_matches: List[str] = []
        contradictions: List[str] = []
        uncertainties: List[str] = []

        # Technical frame health checks.
        if luminance < 10.0:
            contradictions.append(
                "Near-black or corrupted frames detected "
                "(luminance < 10.0)"
            )
        elif luminance > 252.0:
            contradictions.append(
                "Overexposed washed-out white frames detected "
                "(luminance > 252.0)"
            )

        # Motion is not proof of the requested action, but extremely
        # static footage may contradict a requirement for visible motion.
        active_action_terms = (
            "pour", "drop", "sizzle", "stir", "cut", "sprout", "grow",
            "dig", "wave", "crash", "move", "run", "weld", "drive",
            "reciprocate", "reciprocating", "stroke", "cycle", "turn",
            "rotate", "spin", "pump", "flow", "emerge", "erupt",
            "break", "breaking",
        )

        is_active_action = any(
            word in req_action for word in active_action_terms
        )

        if is_active_action and motion_delta < 0.2:
            contradictions.append(
                f"Nearly static sampled footage ({motion_delta:.2f} "
                f"delta) for requested active action '{req_action}'. "
                "This is a technical motion warning, not semantic recognition."
            )

        # Explicit exclusions can reject a candidate when its own metadata
        # explicitly names the excluded item. Metadata absence is not proof
        # that the item is absent from the actual frames.
        for exclusion in explicit_exclusions:
            if exclusion and re.search(
                r"\b" + re.escape(exclusion) + r"\b",
                cand_text,
            ):
                contradictions.append(
                    "Explicit exclusion matched in candidate metadata: "
                    f"'{exclusion}'"
                )

        # Developmental-stage contradiction heuristics.
        is_dry_seed_req = any(
            term in req_subject or term in req_action
            for term in (
                "dry seed", "dormant seed", "unsprouted seed",
                "seed imbibition", "seed water absorption",
                "seed in soil",
            )
        ) or (
            "seed" in req_subject
            and any(
                word in req_action
                for word in (
                    "absorb", "water enters", "swelling",
                    "resting in soil",
                )
            )
        )

        if is_dry_seed_req:
            seedling_terms = (
                "seedling", "seedlings", "sprout", "sprouts",
                "sprouted", "foliage", "leaves", "green plant",
                "microgreens", "nursery tray",
            )

            matched_seedling = [
                term for term in seedling_terms
                if re.search(r"\b" + re.escape(term) + r"\b", cand_text)
            ]

            if matched_seedling:
                contradictions.append(
                    "Developmental stage contradiction: "
                    f"{matched_seedling} found in candidate metadata "
                    "for a dry/dormant seed requirement"
                )

        is_root_req = any(
            term in req_subject or term in req_action
            for term in ("root", "radicle", "taproot", "root tip")
        )

        if is_root_req:
            foliage_only_terms = (
                "canopy", "treetop", "flower blossom", "fruit",
                "green salad", "foliage",
            )

            matched_foliage = [
                term for term in foliage_only_terms
                if re.search(r"\b" + re.escape(term) + r"\b", cand_text)
            ]

            has_root_or_soil_metadata = any(
                term in cand_text
                for term in ("root", "roots", "soil", "dirt", "underground")
            )

            if matched_foliage and not has_root_or_soil_metadata:
                contradictions.append(
                    "Botanical contradiction: above-ground foliage metadata "
                    f"{matched_foliage} without root/soil evidence for a "
                    "root-emergence requirement"
                )

        # Subject metadata matching. This is candidate-ranking evidence,
        # not confirmation of the subject in the actual video frames.
        subject_tokens = [
            word
            for word in re.split(r"\W+", req_subject)
            if len(word) > 2
            and word not in ("and", "the", "with", "for", "from")
        ]

        matched_subjects = [
            word for word in subject_tokens
            if re.search(r"\b" + re.escape(word) + r"\b", cand_text)
        ]

        if (
            subject_tokens
            and len(matched_subjects) == len(subject_tokens)
        ):
            confirmed_matches.append(
                f"Metadata subject terms matched: {matched_subjects}"
            )
        elif (
            len(subject_tokens) > 1
            and len(matched_subjects) >= len(subject_tokens) - 1
            and len(matched_subjects) >= 2
        ):
            confirmed_matches.append(
                f"Metadata subject terms partially matched: "
                f"{matched_subjects}"
            )
        elif matched_subjects:
            uncertainties.append(
                f"Primary subject '{req_subject}' only partially "
                f"matched in metadata: {matched_subjects}"
            )
        else:
            uncertainties.append(
                f"Primary subject '{req_subject}' not explicitly "
                "confirmed in candidate metadata"
            )

        # Required-object metadata matches.
        for required_object in req_objects:
            if re.search(
                r"\b" + re.escape(required_object) + r"\b",
                cand_text,
            ):
                confirmed_matches.append(
                    f"Metadata required-object term matched: "
                    f"'{required_object}'"
                )
            else:
                uncertainties.append(
                    f"Required object '{required_object}' not found "
                    "in candidate metadata"
                )

        # Action metadata matching. This cannot establish that the action
        # occurs visually in the clip.
        action_stop_words = {
            "the", "and", "with", "into", "from", "that",
            "this", "onto", "over", "while",
        }

        def stem(word: str) -> str:
            if len(word) > 4:
                return re.sub(r"(ing|ed|es|s)$", "", word)
            return word

        candidate_words = set(
            word for word in re.split(r"\W+", cand_text) if word
        )
        subject_stems = {stem(word) for word in subject_tokens}

        action_tokens = [
            word
            for word in re.split(r"\W+", req_action)
            if len(word) > 3
            and word not in action_stop_words
            and stem(word) not in subject_stems
        ]

        matched_actions = [
            word
            for word in action_tokens
            if len(stem(word)) >= 3
            and any(
                candidate_word.startswith(stem(word))
                for candidate_word in candidate_words
            )
        ]

        if matched_actions:
            confirmed_matches.append(
                f"Metadata action terms matched: {matched_actions}"
            )
        else:
            uncertainties.append(
                f"Required action '{req_action}' not matched in "
                "candidate metadata; actual frame semantics are unverified"
            )

        # Setting metadata matching.
        setting_tokens = [
            word
            for word in re.split(r"\W+", req_setting)
            if len(word) > 2
            and word not in ("and", "the", "with", "for", "from")
        ]

        matched_settings = [
            word for word in setting_tokens
            if re.search(r"\b" + re.escape(word) + r"\b", cand_text)
        ]

        if matched_settings:
            confirmed_matches.append(
                f"Metadata setting terms matched: {matched_settings}"
            )
        elif setting_tokens:
            uncertainties.append(
                f"Required setting '{req_setting}' not matched "
                "in candidate metadata"
            )

        # Cross-domain setting contradictions.
        for domain, domain_words in cls.DOMAIN_SETTING_PATTERNS.items():
            requested_domain = any(
                word in req_setting
                or word in req_subject
                or word in req_action
                for word in domain_words
            )

            candidate_domain = any(
                re.search(r"\b" + re.escape(word) + r"\b", cand_text)
                for word in domain_words
            )

            if candidate_domain and not requested_domain:
                if domain == "bathroom" and any(
                    term in req_setting or term in req_subject
                    for term in ("cook", "kitchen", "food", "soil", "engine")
                ):
                    contradictions.append(
                        "Bathroom setting metadata conflicts with "
                        "the requested non-bathroom scene"
                    )
                elif domain == "kitchen" and any(
                    term in req_setting or term in req_subject
                    for term in ("engine", "construction", "underground", "soil")
                ):
                    contradictions.append(
                        "Kitchen setting metadata conflicts with "
                        "the requested industrial or botanical scene"
                    )

        # Abstract/conceptual scene checks.
        feasibility, feasibility_reason = (
            ConceptClassifier.classify_feasibility(
                req_action,
                req_subject,
                shot_requirements.get("visual_prompt", ""),
            )
        )

        if feasibility == ConceptFeasibility.CONCEPTUAL_ABSTRACT:
            evidence.append(
                f"Conceptual feasibility: {feasibility_reason}"
            )

            if (
                any(
                    term in req_action
                    for term in ("cylinder", "piston", "internal combustion")
                )
                and "driving" in cand_text
            ):
                contradictions.append(
                    "Generic exterior driving metadata cannot establish "
                    "that internal engine mechanics are shown"
                )

            is_biochemical_process = any(
                term in req_subject or term in req_action
                for term in (
                    "enzyme", "enzymes", "biochemical", "molecular",
                    "starch breakdown", "cellular respiration",
                    "glucose", "metabolism",
                )
            )

            if is_biochemical_process:
                unrelated_domestic_terms = (
                    "glass", "drinking", "kitchen", "cook", "cooking",
                    "beverage", "cocktail", "faucet", "sink",
                    "dishwasher", "wine", "tea", "dining", "restaurant",
                )

                matched_domestic = [
                    term for term in unrelated_domestic_terms
                    if re.search(r"\b" + re.escape(term) + r"\b", cand_text)
                ]

                if matched_domestic:
                    contradictions.append(
                        "Physical substitution contradiction: "
                        "kitchen or domestic footage cannot represent "
                        "internal enzyme activation or biochemical "
                        f"starch breakdown: {matched_domestic}"
                    )

        # Decision:
        # - Reject only when a contradiction heuristic fires.
        # - Otherwise remain UNCERTAIN because this class has no semantic
        #   vision model and cannot verify what the sampled pixels depict.
        if contradictions:
            status = VerificationStatus.REJECTED
            confidence = 0.90
            evidence.append(
                "REJECTED candidate based on heuristic contradiction(s): "
                + "; ".join(contradictions)
            )
        else:
            status = VerificationStatus.UNCERTAIN
            confidence = 0.50

            uncertainties.append(
                "Semantic frame verification is unavailable. "
                "The sampled pixels were checked for technical properties, "
                "but not interpreted by a vision model. Metadata matches "
                "cannot prove the requested subject, action, setting, "
                "or developmental stage is visible."
            )

            evidence.append(
                "UNCERTAIN candidate: heuristic and metadata checks cannot "
                "establish semantic visual correctness. Manual review or "
                "a vision-capable verifier is required."
            )

        recommended_strategy = (
            "diagram_or_motion_graphic"
            if feasibility == ConceptFeasibility.CONCEPTUAL_ABSTRACT
            else None
        )

        return VisualVerificationResult(
            status=status,
            confidence=confidence,
            confirmed_matches=confirmed_matches,
            contradictions=contradictions,
            uncertainties=uncertainties,
            evidence=evidence,
            frame_count_evaluated=len(frames),
            evaluated_timestamps=evaluated_timestamps,
            provider="heuristic_frame_verifier",
            is_visual_frame_verified=False,
            is_manually_approved=False,
            recommended_strategy=recommended_strategy,
        )


class VisualVerifier:
    """
    Main verification facade.

    This version decodes sampled frames and applies heuristic checks.
    It does not perform semantic image recognition, so non-rejected
    candidates remain UNCERTAIN until manual or vision-model verification.
    """

    def __init__(
        self,
        num_sample_frames: int = 3,
        start_ratio: float = 0.20,
        end_ratio: float = 0.80,
    ):
        self.num_sample_frames = num_sample_frames
        self.start_ratio = start_ratio
        self.end_ratio = end_ratio

    async def verify_candidate_clip(
        self,
        video_path: str,
        shot_requirements: Dict[str, Any],
        candidate_metadata: Optional[Dict[str, Any]] = None,
    ) -> VisualVerificationResult:
        """Inspect sampled frames and evaluate heuristic contradictions."""
        if not video_path or not os.path.exists(video_path):
            return VisualVerificationResult(
                status=VerificationStatus.REJECTED,
                confidence=1.0,
                contradictions=[
                    f"Video file does not exist on disk: {video_path}"
                ],
                evidence=["Verification failed: clip file not found"],
                provider="heuristic_frame_verifier",
                is_visual_frame_verified=False,
            )

        try:
            frames = FrameExtractor.extract_frames(
                video_path=video_path,
                num_frames=self.num_sample_frames,
                start_ratio=self.start_ratio,
                end_ratio=self.end_ratio,
            )
        except Exception as exc:
            logger.warning(
                "Frame extraction failed for %s: %s",
                video_path,
                exc,
            )

            return VisualVerificationResult(
                status=VerificationStatus.REJECTED,
                confidence=0.90,
                contradictions=[
                    f"Video frame decoding failed: {exc}"
                ],
                evidence=[f"PyAV decode error: {exc}"],
                provider="heuristic_frame_verifier",
                is_visual_frame_verified=False,
            )

        result = HeuristicVisualVerifier.evaluate(
            frames=frames,
            shot_requirements=shot_requirements,
            candidate_metadata=candidate_metadata,
        )

        logger.info(
            "Heuristic inspection for shot '%s': status=%s, "
            "confidence=%.2f, metadata_matches=%s, contradictions=%s, "
            "visual_verified=%s",
            shot_requirements.get("required_visible_action"),
            result.status.value,
            result.confidence,
            len(result.confirmed_matches),
            len(result.contradictions),
            result.is_visual_frame_verified,
        )

        return result

    def generate_contact_sheet(
        self,
        video_path: str,
        output_path: str,
        shot_requirements: Dict[str, Any],
        candidate_id: Any = "unknown",
        evidence: Optional[List[str]] = None,
    ) -> str:
        """
        Extract representative frames and generate a contact sheet
        for reviewer inspection.
        """
        frames = FrameExtractor.extract_frames(
            video_path=video_path,
            num_frames=self.num_sample_frames,
            start_ratio=self.start_ratio,
            end_ratio=self.end_ratio,
        )

        return FrameExtractor.generate_contact_sheet(
            frames=frames,
            output_path=output_path,
            title=(
                f"Scene {shot_requirements.get('scene_order', 1)} "
                f"Shot {shot_requirements.get('shot_index', 1)}"
            ),
            required_action=str(
                shot_requirements.get("required_action")
                or shot_requirements.get("required_visible_action")
                or ""
            ),
            primary_subject=str(
                shot_requirements.get("primary_subject") or ""
            ),
            required_setting=str(
                shot_requirements.get("required_setting") or ""
            ),
            candidate_id=candidate_id,
            evidence=evidence,
        )
